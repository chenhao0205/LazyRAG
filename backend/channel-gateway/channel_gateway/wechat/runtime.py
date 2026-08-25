from __future__ import annotations

import base64
from collections import OrderedDict
import hashlib
import json
import logging
import math
import mimetypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from channel_gateway.common.domain.channel import (
    InboundEnvelope,
    ReceiverCheckpoint,
)
from channel_gateway.common.domain.chat import (
    ChannelAttachment,
    ChannelExecutionContext,
)
from channel_gateway.common.ports.providers import (
    ReceiverRepository,
    RuntimeCredentialStore,
    RuntimeLease,
)
from channel_gateway.wechat.domain import (
    WeChatAddressFactory,
    WeChatConfig,
    WeChatError,
)
from channel_gateway.wechat.ports import WeChatReceiverClient


_logger = logging.getLogger(__name__)
_MIN_POLL_TIMEOUT_MS = 5_000
_MAX_POLL_TIMEOUT_MS = 60_000
_MAX_INBOUND_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_INBOUND_FILE_BYTES = 100 * 1024 * 1024
_MAX_TOTAL_INBOUND_DOWNLOAD_BYTES = 100 * 1024 * 1024
_MAX_INBOUND_ATTACHMENTS = 10
_FILE_KEY_CACHE_SIZE = 128
_ATTACHMENT_ONLY_PROMPT = '请分析附件内容。'
_REF_SHAPE_DEBUG_ENV = 'WECHAT_REF_SHAPE_DEBUG'


@dataclass(slots=True)
class _AccountWorker:
    account_id: str
    revision: int
    stop_event: threading.Event
    thread: threading.Thread | None = None
    lease: RuntimeLease | None = None


def _message_key(message: dict[str, Any]) -> str:
    message_id = message.get('message_id')
    if message_id is not None and str(message_id).strip():
        raw = str(message_id).strip()
    else:
        raw = json.dumps(
            message,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
        )
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _message_item_ids(message: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for candidate in (message.get('message_id'), message.get('msg_id')):
        value = str(candidate or '').strip()
        if value:
            values.append(value)
    for item in message.get('item_list') or []:
        if isinstance(item, dict):
            value = str(item.get('msg_id') or '').strip()
            if value:
                values.append(value)
    return list(dict.fromkeys(values))


def _reference_message_id(
    ref: dict[str, Any],
    ref_item: dict[str, Any] | None,
) -> str:
    for candidate in (
        ref_item.get('msg_id') if ref_item else None,
        ref.get('msg_id'),
        ref.get('message_id'),
    ):
        value = str(candidate or '').strip()
        if value:
            return value
    return ''


def _message_type_name(value: Any) -> str:
    return {1: 'text', 2: 'image', 4: 'file'}.get(value, 'unknown')


def _has_inline_reference_media(item: dict[str, Any] | None) -> bool:
    if not isinstance(item, dict):
        return False
    if item.get('type') == 2:
        return isinstance(item.get('image_item'), dict)
    if item.get('type') == 4:
        return isinstance(item.get('file_item'), dict)
    return False


def _ref_shape_logging_enabled() -> bool:
    return str(os.getenv(_REF_SHAPE_DEBUG_ENV) or '').strip().lower() in {
        '1', 'true', 'yes', 'on',
    }


def _shape_keys(value: Any) -> list[str]:
    if not isinstance(value, dict):
        return []
    return sorted(str(key) for key in value.keys())


def _log_inbound_message_shape(message: dict[str, Any]) -> None:
    """Temporary E2E diagnostics: log field names only, never field values."""
    if not _ref_shape_logging_enabled():
        return
    _logger.info(
        'wechat_e2e_inbound_shape message_keys=%s item_list_type=%s',
        _shape_keys(message),
        type(message.get('item_list')).__name__,
    )
    items = message.get('item_list')
    if not isinstance(items, list):
        return
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            _logger.info(
                'wechat_e2e_inbound_item_shape index=%s item_type=<non-dict> '
                'python_type=%s', index, type(item).__name__,
            )
            continue
        ref = item.get('ref_msg')
        ref_item = ref.get('message_item') if isinstance(ref, dict) else None
        _logger.info(
            'wechat_e2e_inbound_item_shape index=%s item_type=%s item_keys=%s '
            'text_item_keys=%s image_item_keys=%s file_item_keys=%s '
            'video_item_keys=%s voice_item_keys=%s ref_msg_keys=%s '
            'message_item_keys=%s',
            index,
            item.get('type'),
            _shape_keys(item),
            _shape_keys(item.get('text_item')),
            _shape_keys(item.get('image_item')),
            _shape_keys(item.get('file_item')),
            _shape_keys(item.get('video_item')),
            _shape_keys(item.get('voice_item')),
            _shape_keys(ref),
            _shape_keys(ref_item),
        )


def _message_text(message: dict[str, Any]) -> str:
    values: list[str] = []
    for item in message.get('item_list') or []:
        if not isinstance(item, dict):
            continue
        if item.get('type') == 1:
            text_item = item.get('text_item') or {}
            if isinstance(text_item, dict) and text_item.get('text') is not None:
                text = str(text_item['text']).strip()
                if text:
                    values.append(text)
        if item.get('type') == 3:
            voice_item = item.get('voice_item') or {}
            if isinstance(voice_item, dict) and voice_item.get('text'):
                text = str(voice_item['text']).strip()
                if text:
                    values.append(text)
    return '\n'.join(values)


def _item_text(item: dict[str, Any]) -> str:
    if item.get('type') != 1:
        return ''
    text_item = item.get('text_item') or {}
    if not isinstance(text_item, dict):
        return ''
    return str(text_item.get('text') or '').strip()


def _parse_reference(
    ref: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Normalize the public, non-sensitive reference fields from iLink."""
    title = str(ref.get('title') or '').strip()
    ref_item = ref.get('message_item')
    if not isinstance(ref_item, dict):
        ref_item = None
    parts: list[str] = []
    for value in (title, _item_text(ref_item) if ref_item else ''):
        if value and value not in parts:
            parts.append(value)
    record: dict[str, Any] = {
        'has_title': bool(title),
        'title_length': len(title),
        'has_message_item': ref_item is not None,
    }
    if ref_item is not None:
        record['item_type'] = ref_item.get('type')
    if parts:
        record['text'] = '\n'.join(parts)
    return record, ref_item


def _log_reference_shape(
    ref: dict[str, Any],
    ref_item: dict[str, Any] | None,
) -> None:
    """Emit debug-only payload shape diagnostics without logging content/secrets."""
    _logger.debug(
        'wechat_ref_msg_shape has_ref_msg=true has_title=%s title_length=%s '
        'has_message_item=%s message_item_type=%s has_text_item=%s '
        'has_image_item=%s has_file_item=%s has_video_item=%s '
        'has_voice_item=%s',
        bool(str(ref.get('title') or '').strip()),
        len(str(ref.get('title') or '').strip()),
        ref_item is not None,
        ref_item.get('type') if ref_item else None,
        bool(ref_item and isinstance(ref_item.get('text_item'), dict)),
        bool(ref_item and isinstance(ref_item.get('image_item'), dict)),
        bool(ref_item and isinstance(ref_item.get('file_item'), dict)),
        bool(ref_item and isinstance(ref_item.get('video_item'), dict)),
        bool(ref_item and isinstance(ref_item.get('voice_item'), dict)),
    )


def _image_data_url(content: bytes) -> str | None:
    if content.startswith(b'\x89PNG\r\n\x1a\n'):
        media_type = 'image/png'
    elif content.startswith((b'GIF87a', b'GIF89a')):
        media_type = 'image/gif'
    elif content.startswith(b'RIFF') and content[8:12] == b'WEBP':
        media_type = 'image/webp'
    elif content.startswith(b'\xff\xd8\xff'):
        media_type = 'image/jpeg'
    else:
        return None
    return f'data:{media_type};base64,{base64.b64encode(content).decode("ascii")}'


def _file_data_url(content: bytes, filename: str) -> str:
    media_type = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
    return f'data:{media_type};base64,{base64.b64encode(content).decode("ascii")}'


class WeChatRuntime:
    """Receives iLink events and durably enqueues them without calling Core."""

    def __init__(
        self,
        *,
        config: WeChatConfig,
        store: ReceiverRepository,
        credentials: RuntimeCredentialStore,
        client: WeChatReceiverClient,
        addresses: WeChatAddressFactory,
    ):
        self._config = config
        self._store = store
        self._credentials = credentials
        self._client = client
        self._addresses = addresses
        self._shutdown = threading.Event()
        self._lock = threading.Lock()
        self._workers: dict[str, _AccountWorker] = {}
        self._file_aes_keys: OrderedDict[tuple[str, int], str] = OrderedDict()
        self._file_key_lock = threading.Lock()

    def reconcile_accounts(
        self,
        accounts: list[dict[str, Any]],
    ) -> None:
        desired = {
            str(account['id']): int(account['credential_revision'])
            for account in accounts
        }
        with self._lock:
            current = {
                account_id: worker.revision
                for account_id, worker in self._workers.items()
            }
        for account_id in current.keys() - desired.keys():
            self.stop_account(account_id)
        for account_id, revision in desired.items():
            if current.get(account_id) != revision:
                self.start_account(account_id, revision=revision)

    def stop(self) -> None:
        self._shutdown.set()
        with self._lock:
            workers = list(self._workers.values())
        for worker in workers:
            worker.stop_event.set()
            if worker.lease:
                try:
                    self._store.set_runtime_status(
                        worker.account_id,
                        'stopped',
                        runtime_fence=worker.lease.fence,
                    )
                except Exception:
                    _logger.exception(
                        'wechat_runtime_stop_status_failed '
                        'account_id=%s',
                        worker.account_id,
                    )
                worker.lease.close()
        for worker in workers:
            if worker.thread:
                worker.thread.join(timeout=1.0)

    def start_account(
        self,
        account_id: str,
        *,
        revision: int = 0,
    ) -> None:
        old_worker = None
        with self._lock:
            existing = self._workers.get(account_id)
            if (
                existing
                and existing.thread
                and existing.thread.is_alive()
                and (revision == 0 or existing.revision == revision)
            ):
                return
            if existing:
                self._workers.pop(account_id, None)
                existing.stop_event.set()
                if existing.lease:
                    existing.lease.close()
                old_worker = existing
            stop_event = threading.Event()
            worker = _AccountWorker(
                account_id=account_id,
                revision=revision,
                stop_event=stop_event,
            )
            thread = threading.Thread(
                target=self._run_account,
                args=(worker,),
                name=f'channel-wechat-receiver-{account_id[-8:]}',
                daemon=True,
            )
            worker.thread = thread
            self._workers[account_id] = worker
            thread.start()
        if old_worker and old_worker.thread:
            old_worker.thread.join(timeout=1.0)

    def restart_account(self, account_id: str) -> None:
        try:
            account = self._credentials.load_runtime_account(account_id)
        except Exception:
            _logger.exception(
                'wechat_account_reload_failed account_id=%s',
                account_id,
            )
            return
        self.start_account(
            account_id,
            revision=int(account['credential_revision']),
        )

    def stop_account(self, account_id: str) -> None:
        with self._lock:
            worker = self._workers.pop(account_id, None)
        if not worker:
            return
        worker.stop_event.set()
        if worker.lease:
            worker.lease.close()
        if worker.thread:
            worker.thread.join(timeout=1.0)

    def _run_account(
        self,
        worker: _AccountWorker,
    ) -> None:
        account_id = worker.account_id
        stop_event = worker.stop_event
        failures = 0
        try:
            while not self._shutdown.is_set() and not stop_event.is_set():
                lease = None
                try:
                    lease = self._store.acquire_runtime_lease(account_id)
                    if lease is None:
                        stop_event.wait(5)
                        continue
                    with self._lock:
                        worker.lease = lease
                    account = self._credentials.load_runtime_account(account_id)
                    if account.get('status') != 'connected':
                        return
                    credentials = dict(account['credentials'])
                    self._store.set_runtime_status(
                        account_id,
                        'starting',
                        runtime_fence=lease.fence,
                    )
                    self._notify_start(account_id, credentials)
                    failures = 0
                    self._poll(account, credentials, stop_event, lease)
                except Exception as exc:
                    failures += 1
                    delay = min(30, 2 ** min(failures, 5))
                    _logger.exception(
                        'wechat_receiver_failed account_id=%s retry_in=%s',
                        account_id,
                        delay,
                    )
                    if lease is not None:
                        try:
                            self._store.set_runtime_status(
                                account_id,
                                'failed',
                                str(exc)[:500],
                                runtime_fence=lease.fence,
                            )
                        except Exception:
                            pass
                    stop_event.wait(delay)
                finally:
                    if lease is not None:
                        lease.close()
                        with self._lock:
                            if worker.lease is lease:
                                worker.lease = None
                if not self._shutdown.is_set() and not stop_event.is_set():
                    stop_event.wait(2)
        finally:
            with self._lock:
                current = self._workers.get(account_id)
                if current is worker:
                    self._workers.pop(account_id, None)

    def _poll(
        self,
        account: dict[str, Any],
        credentials: dict[str, str],
        stop_event: threading.Event,
        lease: RuntimeLease,
    ) -> None:
        account_id = str(account['id'])
        checkpoint = self._store.get_checkpoint(account_id)
        cursor = str(checkpoint.get('cursor') or '')
        timeout_ms = int(checkpoint.get('longpoll_timeout_ms') or 35000)
        failures = 0
        self._store.set_runtime_status(
            account_id,
            'running',
            runtime_fence=lease.fence,
        )
        _logger.info('wechat_receiver_started account_id=%s', account_id)

        while not self._shutdown.is_set() and not stop_event.is_set():
            lease.keepalive()
            try:
                result = self._client.get_updates(
                    base_url=credentials['base_url'],
                    token=credentials['token'],
                    cursor=cursor,
                    timeout_ms=timeout_ms,
                )
                if self._shutdown.is_set() or stop_event.is_set():
                    return
                lease.keepalive()
                failures = 0
                timeout_ms = self._next_timeout(result, timeout_ms)
                next_cursor = str(result.get('get_updates_buf') or cursor)
                envelopes = [
                    envelope
                    for message in (result.get('msgs') or [])
                    if isinstance(message, dict)
                    for envelope in [
                        self._normalize(account, credentials, message)
                    ]
                    if envelope is not None
                ]
                self._store.ingest_batch(
                    account_id,
                    envelopes,
                    ReceiverCheckpoint(
                        cursor=next_cursor,
                        metadata={'longpoll_timeout_ms': timeout_ms},
                    ),
                    lease.fence,
                )
                cursor = next_cursor
            except WeChatError as exc:
                failures += 1
                delay = (
                    30
                    if failures
                    >= self._config.max_consecutive_errors
                    else 2
                )
                self._store.set_runtime_status(
                    account_id,
                    'degraded',
                    f'{exc.__class__.__name__}: {exc}'[:500],
                    lease.fence,
                )
                _logger.warning(
                    'wechat_getupdates_failed account_id=%s attempt=%s retry_in=%s',
                    account_id,
                    failures,
                    delay,
                )
                stop_event.wait(delay)

    def _normalize(
        self,
        account: dict[str, Any],
        credentials: dict[str, str],
        message: dict[str, Any],
    ) -> InboundEnvelope | None:
        _log_inbound_message_shape(message)
        if message.get('message_type') not in (None, 1):
            return None
        sender_id = str(message.get('from_user_id') or '')
        if sender_id != credentials['authorized_user_id']:
            return None
        account_id = str(account['id'])
        message_key = _message_key(message)
        context_token = str(message.get('context_token') or '')
        text = _message_text(message)
        attachments, attachment_metadata, remaining_download_bytes = (
            self._attachments_from_items(
            message.get('item_list') or [],
            source='message',
            budget=_MAX_INBOUND_ATTACHMENTS,
            remaining_download_bytes=_MAX_TOTAL_INBOUND_DOWNLOAD_BYTES,
        )
        )
        references, reference_attachments, _remaining_download_bytes = self._references(
            message,
            account_id=account_id,
            recipient_id=sender_id,
            budget=_MAX_INBOUND_ATTACHMENTS - len(attachments),
            remaining_download_bytes=remaining_download_bytes,
        )
        attachments = attachments + reference_attachments
        quoted_text = '\n'.join(
            item['text'] for item in references if item.get('text')
        )
        if references:
            fallback_quote = (
                '当前引用消息无法解析，请重新引用或重新发送。'
                if any(item.get('resolved') is False for item in references)
                else '（包含引用附件）'
            )
            text = (
                f'[引用消息]\n{quoted_text or fallback_quote}\n[/引用消息]\n\n'
                f'[当前消息]\n{text or "请结合引用消息处理。"}\n[/当前消息]'
            )
        if not text and attachments:
            text = _ATTACHMENT_ONLY_PROMPT
        if not sender_id or not context_token or not text:
            return None
        address_hash = self._addresses.direct(
            account_id,
            sender_id,
        ).route_hash
        execution = ChannelExecutionContext(attachments=attachments)
        provider_context: dict[str, Any] = {
            'context_token': context_token,
            'session_id': str(message.get('session_id') or ''),
            'wechat_message_ids': _message_item_ids(message),
            'channel_execution': execution.to_dict(),
        }
        if attachment_metadata:
            provider_context['wechat_attachments'] = attachment_metadata
        if references:
            provider_context['wechat_ref_msg'] = references
        if attachments:
            provider_context['command_action'] = {
                'schema_version': '1',
                'command': 'chat',
                'parameters': {'message': text},
            }
        return InboundEnvelope(
            provider='wechat',
            account_id=account_id,
            message_key=message_key,
            order_key=address_hash,
            external_address_hash=address_hash,
            owner_user_id=str(account['owner_user_id']),
            recipient_id=sender_id,
            text=text,
            provider_context=provider_context,
        )

    def _references(
        self,
        message: dict[str, Any],
        *,
        account_id: str,
        recipient_id: str,
        budget: int,
        remaining_download_bytes: int,
    ) -> tuple[list[dict[str, Any]], tuple[ChannelAttachment, ...], int]:
        references: list[dict[str, Any]] = []
        attachments: list[ChannelAttachment] = []
        ref_sources: list[dict[str, Any]] = []
        for item in message.get('item_list') or []:
            if not isinstance(item, dict):
                continue
            ref = item.get('ref_msg')
            if not isinstance(ref, dict):
                continue
            ref_sources.append(ref)
        # Some iLink payload variants expose the reference beside item_list.
        # Prefer item-level references, which are associated with their message
        # item, and use the top-level shape only as a compatibility fallback.
        if not ref_sources and isinstance(message.get('ref_msg'), dict):
            ref_sources.append(message['ref_msg'])

        for ref in ref_sources:
            record, ref_item = _parse_reference(ref)
            _log_reference_shape(ref, ref_item)
            message_id = _reference_message_id(ref, ref_item)
            inline = bool(record.get('text')) or _has_inline_reference_media(ref_item)
            if inline:
                record.update({
                    'resolved': True,
                    'message_id': message_id,
                    'source': 'inline',
                    'type': _message_type_name(
                        ref_item.get('type') if ref_item else None,
                    ),
                })
            elif message_id:
                resolved = self._resolve_persisted_reference(
                    account_id=account_id,
                    recipient_id=recipient_id,
                    message_id=message_id,
                )
                if resolved is None:
                    record.update({
                        'resolved': False,
                        'message_id': message_id,
                        'reason': 'message_not_found',
                    })
                else:
                    resolved_text = str(resolved.get('text') or '').strip()
                    resolved_context = resolved.get('provider_context')
                    execution = ChannelExecutionContext.from_provider_context(
                        resolved_context if isinstance(resolved_context, dict) else None,
                    )
                    resolved_attachments = execution.attachments
                    record = {
                        'resolved': True,
                        'message_id': message_id,
                        'source': 'db',
                        'type': _message_type_name(
                            ref_item.get('type') if ref_item else None,
                        ),
                    }
                    if resolved_text:
                        record['text'] = resolved_text
                    if resolved_attachments:
                        record['attachments'] = [
                            {'source': 'db', 'input_type': attachment.input_type}
                            for attachment in resolved_attachments
                        ]
                        attachments.extend(
                            resolved_attachments[:max(0, budget - len(attachments))]
                        )
            elif (
                not record.get('has_title')
                and (ref_item is None or ref_item.get('type') is None)
            ):
                continue
            else:
                record.update({
                    'resolved': False,
                    'message_id': '',
                    'reason': 'message_id_missing',
                })
            if inline and isinstance(ref_item, dict):
                ref_attachments, metadata, remaining_download_bytes = (
                    self._attachments_from_items(
                    [ref_item],
                    source='ref_msg',
                    budget=budget - len(attachments),
                    remaining_download_bytes=remaining_download_bytes,
                )
                )
                attachments.extend(ref_attachments)
                if metadata:
                    record['attachments'] = metadata
            _logger.info(
                'wechat_reference_resolver resolved=%s source=%s message_id=%s '
                'reference_type=%s reason=%s',
                record.get('resolved') is True,
                record.get('source') or 'unresolved',
                message_id,
                record.get('type') or _message_type_name(
                    ref_item.get('type') if ref_item else None,
                ),
                record.get('reason') or '',
            )
            if (
                record.get('text')
                or record.get('attachments')
                or record.get('resolved') is False
            ):
                references.append(record)
        return references, tuple(attachments), remaining_download_bytes

    def _resolve_persisted_reference(
        self,
        *,
        account_id: str,
        recipient_id: str,
        message_id: str,
    ) -> dict[str, Any] | None:
        lookup = getattr(self._store, 'find_inbound_by_provider_message_id', None)
        if not callable(lookup):
            return None
        try:
            result = lookup(
                provider='wechat',
                account_id=account_id,
                recipient_id=recipient_id,
                message_id=message_id,
            )
        except Exception:
            _logger.exception(
                'wechat_reference_resolver_lookup_failed message_id=%s',
                message_id,
            )
            return None
        return result if isinstance(result, dict) else None

    def _attachments_from_items(
        self,
        items: list[Any],
        *,
        source: str,
        budget: int,
        remaining_download_bytes: int,
    ) -> tuple[tuple[ChannelAttachment, ...], list[dict[str, Any]], int]:
        attachments: list[ChannelAttachment] = []
        metadata: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or len(attachments) >= budget:
                continue
            item_type = item.get('type')
            if item_type == 2:
                image = item.get('image_item')
                input_type = 'image'
                filename = ''
                max_bytes = _MAX_INBOUND_IMAGE_BYTES
                image_aeskey = (
                    str(image.get('aeskey') or '')
                    if isinstance(image, dict)
                    else ''
                )
                media = image.get('media') if isinstance(image, dict) else None
            elif item_type == 4:
                file_item = item.get('file_item')
                input_type = 'file'
                filename = (
                    str(file_item.get('file_name') or '').strip()
                    if isinstance(file_item, dict)
                    else ''
                )
                max_bytes = _MAX_INBOUND_FILE_BYTES
                image_aeskey = ''
                media = (
                    file_item.get('media') if isinstance(file_item, dict) else None
                )
                integrity, metadata_error = self._file_integrity(file_item)
                if integrity is None:
                    _logger.warning(
                        'wechat_inbound_file_metadata_invalid reason=%s',
                        metadata_error,
                    )
                    continue
                if integrity[1] > remaining_download_bytes:
                    _logger.warning(
                        'wechat_inbound_media_budget_declared_exceeded '
                        'type=file declared=%s remaining=%s',
                        integrity[1],
                        remaining_download_bytes,
                    )
                    continue
            else:
                continue
            if item_type == 2:
                integrity = None
            if not isinstance(media, dict):
                _logger.warning('wechat_inbound_media_missing type=%s', item_type)
                continue
            if remaining_download_bytes <= 0:
                _logger.warning('wechat_inbound_media_budget_exhausted type=%s', item_type)
                continue
            effective_limit = min(max_bytes, remaining_download_bytes)
            def _consume_download_bytes(chunk_size: int) -> None:
                nonlocal remaining_download_bytes
                remaining_download_bytes = max(
                    0,
                    remaining_download_bytes - chunk_size,
                )

            try:
                fallback_aes_keys = self._file_aes_key_candidates(integrity)
                content, used_aes_key = self._client.download_media(
                    media,
                    image_aeskey=image_aeskey,
                    max_bytes=effective_limit,
                    max_download_bytes=effective_limit,
                    fallback_aes_keys=fallback_aes_keys,
                    validate_plaintext=(
                        lambda value, integrity=integrity:
                        self._file_matches_integrity(value, integrity)
                    ) if integrity is not None else None,
                    on_download_bytes=_consume_download_bytes,
                )
            except WeChatError as exc:
                _logger.warning(
                    'wechat_inbound_media_download_failed type=%s error=%s',
                    item_type,
                    exc,
                )
                continue
            if item_type == 4 and integrity is not None:
                self._remember_file_aes_key(integrity, used_aes_key)
            if not content:
                _logger.warning('wechat_inbound_media_empty type=%s', item_type)
                continue
            encoded = (
                _image_data_url(content)
                if input_type == 'image'
                else _file_data_url(content, filename)
            )
            if not encoded:
                _logger.warning('wechat_inbound_image_type_unsupported')
                continue
            attachment = ChannelAttachment.from_dict({
                'input_type': input_type,
                'input_base64': encoded,
            })
            if attachment is None:
                continue
            attachments.append(attachment)
            entry: dict[str, Any] = {
                'source': source,
                'input_type': input_type,
            }
            if filename:
                entry['filename'] = filename
            for key in ('len', 'md5'):
                if item_type == 4 and isinstance(file_item, dict):
                    value = file_item.get(key)
                    if value is not None:
                        entry[key] = str(value)
            metadata.append(entry)
        return tuple(attachments), metadata, remaining_download_bytes

    def _file_aes_key_candidates(
        self,
        integrity: tuple[str, int] | None,
    ) -> tuple[str, ...]:
        if integrity is None:
            return ()
        with self._file_key_lock:
            cached = self._file_aes_keys.get(integrity)
            if not cached:
                return ()
            self._file_aes_keys.move_to_end(integrity)
            return (cached,)

    def _remember_file_aes_key(
        self,
        integrity: tuple[str, int],
        aes_key: str,
    ) -> None:
        if not aes_key:
            return
        with self._file_key_lock:
            self._file_aes_keys[integrity] = aes_key
            self._file_aes_keys.move_to_end(integrity)
            while len(self._file_aes_keys) > _FILE_KEY_CACHE_SIZE:
                self._file_aes_keys.popitem(last=False)

    @staticmethod
    def _file_integrity(
        file_item: Any,
    ) -> tuple[tuple[str, int] | None, str]:
        if not isinstance(file_item, dict):
            return None, 'file_item_missing'
        raw_md5 = file_item.get('md5')
        raw_length = file_item.get('len')
        if raw_length is None or not str(raw_length).strip():
            return None, 'len_missing'
        if raw_md5 is None or not str(raw_md5).strip():
            return None, 'md5_missing'
        expected_md5 = str(raw_md5).strip().lower()
        try:
            expected_length = int(str(raw_length).strip())
        except (TypeError, ValueError):
            return None, 'len_invalid'
        if expected_length < 0:
            return None, 'len_negative'
        if len(expected_md5) != 32 or any(
            char not in '0123456789abcdef' for char in expected_md5
        ):
            return None, 'md5_invalid'
        return (expected_md5, expected_length), ''

    @staticmethod
    def _file_matches_integrity(
        content: bytes,
        integrity: tuple[str, int],
    ) -> bool:
        expected_md5, expected_length = integrity
        if len(content) != expected_length:
            _logger.warning(
                'wechat_inbound_file_length_mismatch expected=%s actual=%s',
                expected_length,
                len(content),
            )
            return False
        actual_md5 = hashlib.md5(content, usedforsecurity=False).hexdigest()
        if actual_md5 != expected_md5:
            _logger.warning('wechat_inbound_file_md5_mismatch')
            return False
        return True

    def _notify_start(
        self,
        account_id: str,
        credentials: dict[str, str],
    ) -> None:
        try:
            self._client.notify_start(
                base_url=credentials['base_url'],
                token=credentials['token'],
            )
        except WeChatError:
            _logger.warning(
                'wechat_notify_start_failed account_id=%s',
                account_id,
            )

    @staticmethod
    def _next_timeout(result: dict[str, Any], current: int) -> int:
        suggested = result.get('longpolling_timeout_ms')
        if not isinstance(suggested, (int, float)):
            return current
        if not math.isfinite(suggested):
            return current
        return min(
            _MAX_POLL_TIMEOUT_MS,
            max(_MIN_POLL_TIMEOUT_MS, int(suggested)),
        )
