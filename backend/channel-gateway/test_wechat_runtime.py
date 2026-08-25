import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from channel_gateway.wechat.client import WeChatClient
from channel_gateway.common.application.routing import ChannelCommandRouter
from channel_gateway.common.domain.commands import ChatCommand
from channel_gateway.common.domain.channel import InboundEnvelope
from channel_gateway.common.infrastructure.sqlite import SQLiteGatewayStore
from channel_gateway.wechat.domain import (
    WeChatAddressFactory,
    WeChatConfig,
    WeChatError,
)
from channel_gateway.wechat.runtime import (
    WeChatRuntime,
    _image_data_url,
    _log_inbound_message_shape,
)


class _StreamResponse:
    def __init__(self, chunks, *, headers=None):
        self._chunks = chunks
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self, **_kwargs):
        return iter(self._chunks)


class _ReceiverClient:
    def __init__(self):
        self.downloads = []
        self.contents = {}
        self.fail = False

    def download_media(
        self,
        media,
        *,
        image_aeskey='',
        max_bytes,
        max_download_bytes,
        fallback_aes_keys=(),
        validate_plaintext=None,
        on_download_bytes=None,
    ):
        self.downloads.append((media, image_aeskey, max_bytes, fallback_aes_keys))
        token = media.get('encrypt_query_param')
        content = self.contents.get(token)
        if content is None:
            content = b'\x89PNG\r\n\x1a\nimage' if image_aeskey else b'file body'
        read_size = int(media.get('bytes_read') or len(content))
        if on_download_bytes is not None:
            on_download_bytes(read_size)
        if self.fail:
            raise WeChatError('bad media')
        if media.get('aes_key') == 'new-invalid-key':
            if not fallback_aes_keys:
                raise WeChatError('bad padding')
            if validate_plaintext is not None and not validate_plaintext(content):
                raise WeChatError('integrity validation failed')
            return content, fallback_aes_keys[0]
        if media.get('aes_key') == 'padding-valid-wrong-key':
            wrong_content = b'wrong content'
            if validate_plaintext is None or validate_plaintext(wrong_content):
                return wrong_content, 'padding-valid-wrong-key'
            if fallback_aes_keys and validate_plaintext(content):
                return content, fallback_aes_keys[0]
            raise WeChatError('integrity validation failed')
        if validate_plaintext is not None and not validate_plaintext(content):
            raise WeChatError('integrity validation failed')
        return content, image_aeskey or str(media.get('aes_key') or '')


class _ReferenceStore:
    def __init__(self):
        self.records = {}

    def remember(self, envelope):
        for message_id in envelope.provider_context.get('wechat_message_ids') or []:
            self.records[(envelope.account_id, envelope.recipient_id, message_id)] = {
                'text': envelope.text,
                'provider_context': envelope.provider_context,
            }

    def find_inbound_by_provider_message_id(
        self,
        *,
        provider,
        account_id,
        recipient_id,
        message_id,
    ):
        if provider != 'wechat':
            return None
        return self.records.get((account_id, recipient_id, message_id))


class WeChatRuntimeNormalizeTest(unittest.TestCase):
    def setUp(self):
        self.client = _ReceiverClient()
        self.store = _ReferenceStore()
        self.runtime = WeChatRuntime(
            config=WeChatConfig(
                '', 480, 40, 3, 1800, '/tmp', 100 * 1024 * 1024,
            ),
            store=self.store,
            credentials=object(),
            client=self.client,
            addresses=WeChatAddressFactory(),
        )
        self.account = {'id': 'account-1', 'owner_user_id': 'owner-1'}
        self.credentials = {'authorized_user_id': 'user-1'}
        # The WeChat branch returns before using these collaborators. This
        # exercises the production router without restoring the removed
        # intent-classifier API.
        self.router = ChannelCommandRouter(
            store=None, shortcuts=None, catalog=None,
        )

    def _message(self, items, **extra):
        return {
            'message_id': 'message-1',
            'message_type': 1,
            'from_user_id': 'user-1',
            'context_token': 'context-1',
            'item_list': items,
            **extra,
        }

    def _normalize(self, items, **extra):
        return self.runtime._normalize(
            self.account,
            self.credentials,
            self._message(items, **extra),
        )

    def _route_wechat(self, envelope):
        return self.router.route(
            provider='wechat',
            account_id=envelope.account_id,
            external_address_hash=envelope.external_address_hash,
            owner_user_id=envelope.owner_user_id,
            text=envelope.text,
            request_id='wechat-routing-contract',
            provider_context=envelope.provider_context,
        )

    @staticmethod
    def _file(token, content, *, key='current-key', filename='notes.txt'):
        return {
            'type': 4,
            'file_item': {
                'file_name': filename,
                'len': str(len(content)),
                'md5': hashlib.md5(content, usedforsecurity=False).hexdigest(),
                'media': {'encrypt_query_param': token, 'aes_key': key},
            },
        }

    @staticmethod
    def _real_ref_message_item(msg_id, item_type):
        # Field shape captured from the 15:57 iLink E2E payload. It deliberately
        # has no title, text_item, image_item, or file_item content.
        return {
            'at_bot_username_list': [],
            'button_item_list': [],
            'create_time_ms': 0,
            'is_completed': True,
            'msg_id': msg_id,
            'type': item_type,
            'update_time_ms': 0,
        }

    def test_text_and_voice_messages_are_unchanged(self):
        text = self._normalize([{'type': 1, 'text_item': {'text': ' hello '}}])
        voice = self._normalize([{'type': 3, 'voice_item': {'text': '语音转写'}}])

        self.assertEqual(text.text, 'hello')
        self.assertEqual(voice.text, '语音转写')

    def test_e2e_shape_logging_records_only_field_structure(self):
        message = self._message([{
            'type': 1,
            'text_item': {'text': 'PRIVATE-TEXT'},
            'ref_msg': {
                'title': 'PRIVATE-REF',
                'message_item': {'type': 4, 'file_item': {
                    'media': {'encrypt_query_param': 'PRIVATE-CDN-QUERY'},
                }},
            },
        }], context_token='PRIVATE-CONTEXT-TOKEN')
        with patch.dict('os.environ', {'WECHAT_REF_SHAPE_DEBUG': '1'}), patch(
            'channel_gateway.wechat.runtime._logger',
        ) as logger:
            _log_inbound_message_shape(message)

        logged = '\n'.join(str(call) for call in logger.info.call_args_list)
        self.assertIn('message_keys', logged)
        self.assertIn('ref_msg_keys', logged)
        self.assertIn('message_item_keys', logged)
        self.assertNotIn('PRIVATE-TEXT', logged)
        self.assertNotIn('PRIVATE-REF', logged)
        self.assertNotIn('PRIVATE-CDN-QUERY', logged)
        self.assertNotIn('PRIVATE-CONTEXT-TOKEN', logged)

    def test_image_only_and_text_image_are_delivered(self):
        image = {'type': 2, 'image_item': {
            'aeskey': '0' * 32,
            'media': {'encrypt_query_param': 'image-token'},
        }}
        image_only = self._normalize([image])
        text_image = self._normalize([
            {'type': 1, 'text_item': {'text': '看看这张图'}}, image,
        ])

        self.assertEqual(image_only.text, '请分析附件内容。')
        self.assertEqual(text_image.text, '看看这张图')
        self.assertEqual(
            image_only.provider_context['channel_execution']['attachments'][0]['input_type'],
            'image',
        )

    def test_file_integrity_mismatch_rejects_only_attachment(self):
        content = b'valid file'
        self.client.contents['bad-md5'] = content
        self.client.contents['bad-len'] = content
        bad_md5 = self._file('bad-md5', content)
        bad_md5['file_item']['md5'] = '0' * 32
        bad_len = self._file('bad-len', content)
        bad_len['file_item']['len'] = str(len(content) + 1)

        md5_envelope = self._normalize([
            {'type': 1, 'text_item': {'text': '继续'}}, bad_md5,
        ])
        len_envelope = self._normalize([
            {'type': 1, 'text_item': {'text': '继续'}}, bad_len,
        ])

        self.assertEqual(md5_envelope.text, '继续')
        self.assertEqual(len_envelope.text, '继续')
        self.assertEqual(
            md5_envelope.provider_context['channel_execution']['attachments'], []
        )
        self.assertEqual(
            len_envelope.provider_context['channel_execution']['attachments'], []
        )

    def test_file_metadata_is_strictly_fail_closed(self):
        content = b'valid file'
        invalid_items = []
        for token, field, value in (
            ('missing-len', 'len', None),
            ('invalid-len', 'len', 'not-a-number'),
            ('negative-len', 'len', '-1'),
            ('missing-md5', 'md5', None),
            ('invalid-md5', 'md5', 'not-an-md5'),
        ):
            self.client.contents[token] = content
            item = self._file(token, content)
            item['file_item'].pop(field) if value is None else item['file_item'].__setitem__(field, value)
            invalid_items.append(item)

        envelope = self._normalize([
            {'type': 1, 'text_item': {'text': '继续'}}, *invalid_items,
        ])

        self.assertEqual(
            envelope.provider_context['channel_execution']['attachments'], []
        )
        self.assertEqual(self.client.downloads, [])

    def test_repeated_file_uses_verified_cached_key(self):
        content = b'repeated file'
        self.client.contents['first'] = content
        self.client.contents['second'] = content
        first = self._normalize([self._file('first', content, key='old-key')])
        second = self._normalize([
            self._file('second', content, key='new-invalid-key'),
        ])

        self.assertEqual(len(first.provider_context['channel_execution']['attachments']), 1)
        self.assertEqual(len(second.provider_context['channel_execution']['attachments']), 1)
        self.assertEqual(self.client.downloads[-1][3], ('old-key',))

    def test_padding_valid_current_key_still_uses_cached_key_after_integrity_failure(self):
        content = b'repeated file'
        self.client.contents['first'] = content
        self.client.contents['second'] = content
        self._normalize([self._file('first', content, key='old-key')])

        envelope = self._normalize([
            self._file('second', content, key='padding-valid-wrong-key'),
        ])

        self.assertEqual(len(envelope.provider_context['channel_execution']['attachments']), 1)
        self.assertEqual(self.client.downloads[-1][3], ('old-key',))

    def test_cached_key_with_bad_integrity_is_rejected(self):
        content = b'repeated file'
        self.client.contents['first'] = content
        self.client.contents['second'] = content
        self._normalize([self._file('first', content, key='old-key')])
        rejected = self._file('second', content, key='new-invalid-key')
        rejected['file_item']['md5'] = '0' * 32
        self.runtime._remember_file_aes_key(
            ('0' * 32, len(content)),
            'old-key',
        )

        envelope = self._normalize([
            {'type': 1, 'text_item': {'text': '继续'}}, rejected,
        ])

        self.assertEqual(self.client.downloads[-1][3], ('old-key',))
        self.assertEqual(
            envelope.provider_context['channel_execution']['attachments'], []
        )

    def test_ref_text_file_and_delimiters_are_delivered(self):
        content = b'referenced file'
        self.client.contents['ref-file'] = content
        ref_file = self._file('ref-file', content)['file_item']
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': '请总结'},
            'ref_msg': {
                'message_item': {'type': 4, 'file_item': ref_file},
            },
        }])

        self.assertEqual(
            envelope.text,
            '[引用消息]\n（包含引用附件）\n[/引用消息]\n\n'
            '[当前消息]\n请总结\n[/当前消息]',
        )
        self.assertEqual(len(envelope.provider_context['channel_execution']['attachments']), 1)
        self.assertEqual(
            envelope.provider_context['wechat_ref_msg'][0]['attachments'][0]['source'],
            'ref_msg',
        )

    def test_ref_text_has_explicit_delimiters(self):
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': '请总结'},
            'ref_msg': {'message_item': {
                'type': 1,
                'text_item': {'text': '被引用的内容'},
            }},
        }])

        self.assertEqual(
            envelope.text,
            '[引用消息]\n被引用的内容\n[/引用消息]\n\n'
            '[当前消息]\n请总结\n[/当前消息]',
        )

    def test_ref_title_only_is_delivered_with_explicit_delimiters(self):
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': 'CASE_REF_TEXT_RETEST：只返回引用口令'},
            'ref_msg': {'title': 'REF-TEXT-SECRET-92731'},
        }])

        self.assertEqual(
            envelope.text,
            '[引用消息]\nREF-TEXT-SECRET-92731\n[/引用消息]\n\n'
            '[当前消息]\nCASE_REF_TEXT_RETEST：只返回引用口令\n[/当前消息]',
        )
        reference = envelope.provider_context['wechat_ref_msg'][0]
        self.assertEqual(reference['text'], 'REF-TEXT-SECRET-92731')
        self.assertTrue(reference['has_title'])
        self.assertFalse(reference['has_message_item'])
        self.assertEqual(reference['title_length'], len('REF-TEXT-SECRET-92731'))

    def test_real_shape_text_ref_resolves_persisted_message_id(self):
        original = self._normalize([{
            'type': 1,
            'msg_id': '100',
            'text_item': {'text': 'REF-TEXT-INTEGRATION-31415'},
        }], message_id='parent-100')
        self.store.remember(original)

        envelope = self._normalize([{
            'type': 1,
            'msg_id': '101',
            'text_item': {'text': '只返回我引用消息中的口令。'},
            'ref_msg': {
                'message_item': self._real_ref_message_item('100', 1),
            },
        }], message_id='parent-101')

        self.assertEqual(
            envelope.text,
            '[引用消息]\nREF-TEXT-INTEGRATION-31415\n[/引用消息]\n\n'
            '[当前消息]\n只返回我引用消息中的口令。\n[/当前消息]',
        )
        reference = envelope.provider_context['wechat_ref_msg'][0]
        self.assertTrue(reference['resolved'])
        self.assertEqual(reference['source'], 'db')
        self.assertEqual(reference['message_id'], '100')
        self.assertIn('100', original.provider_context['wechat_message_ids'])

        routed = self._route_wechat(envelope)
        self.assertIsInstance(routed.command, ChatCommand)
        self.assertEqual(routed.source, 'wechat_chat')
        self.assertEqual(routed.command.parameters.message, envelope.text)

    def test_real_shape_file_and_image_refs_reuse_persisted_attachments(self):
        file_content = b'current reference file only'
        self.client.contents['original-file'] = file_content
        original_file = self._file('original-file', file_content)
        original_file['msg_id'] = '200'
        self.store.remember(self._normalize([original_file], message_id='parent-200'))

        image = {'type': 2, 'msg_id': '300', 'image_item': {
            'aeskey': '0' * 32,
            'media': {'encrypt_query_param': 'original-image'},
        }}
        self.client.contents['original-image'] = b'\x89PNG\r\n\x1a\nimage'
        self.store.remember(self._normalize([image], message_id='parent-300'))
        downloads_after_originals = len(self.client.downloads)

        file_ref = self._normalize([{
            'type': 1,
            'text_item': {'text': '读取当前引用文件'},
            'ref_msg': {
                'message_item': self._real_ref_message_item('200', 4),
            },
        }])
        image_ref = self._normalize([{
            'type': 1,
            'text_item': {'text': '分析当前引用图片'},
            'ref_msg': {
                'message_item': self._real_ref_message_item('300', 2),
            },
        }])

        self.assertEqual(len(self.client.downloads), downloads_after_originals)
        for envelope, input_type, message_id in (
            (file_ref, 'file', '200'), (image_ref, 'image', '300'),
        ):
            attachments = envelope.provider_context['channel_execution']['attachments']
            self.assertEqual([item['input_type'] for item in attachments], [input_type])
            reference = envelope.provider_context['wechat_ref_msg'][0]
            self.assertTrue(reference['resolved'])
            self.assertEqual(reference['source'], 'db')
            self.assertEqual(reference['message_id'], message_id)

    def test_unresolved_real_shape_ref_is_explicit_and_never_attaches_history(self):
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': '只处理当前引用'},
            'ref_msg': {
                'message_item': self._real_ref_message_item('999999', 4),
            },
        }])

        self.assertIn('当前引用消息无法解析，请重新引用或重新发送。', envelope.text)
        reference = envelope.provider_context['wechat_ref_msg'][0]
        self.assertFalse(reference['resolved'])
        self.assertEqual(reference['message_id'], '999999')
        self.assertEqual(reference['reason'], 'message_not_found')
        self.assertEqual(
            envelope.provider_context['channel_execution']['attachments'], [],
        )
        routed = self._route_wechat(envelope)
        self.assertIsInstance(routed.command, ChatCommand)
        self.assertEqual(routed.command.parameters.message, envelope.text)

    def test_sqlite_inbox_lookup_persists_message_id_across_runtime_cache_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteGatewayStore(
                f'sqlite:///{Path(directory) / "gateway.db"}',
            )
            store.initialize()
            with store._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO channel_accounts(
                        id, owner_user_id, provider, external_id_hash, label,
                        status, credentials_ciphertext
                    ) VALUES(%s, %s, %s, %s, %s, %s, %s)
                    """,
                    ('account-1', 'owner-1', 'wechat', 'hash', 'test',
                     'connected', 'ciphertext'),
                )
            envelope = InboundEnvelope(
                provider='wechat', account_id='account-1', message_key='key-100',
                order_key='order', external_address_hash='hash',
                owner_user_id='owner-1', recipient_id='user-1',
                text='REF-TEXT-INTEGRATION-31415',
                provider_context={
                    'wechat_message_ids': ['100'],
                    'channel_execution': {'attachments': []},
                },
            )
            store.ingest_batch('account-1', [envelope], None)
            resolved = store.find_inbound_by_provider_message_id(
                provider='wechat', account_id='account-1',
                recipient_id='user-1', message_id='100',
            )

        self.assertEqual(resolved['text'], 'REF-TEXT-INTEGRATION-31415')

    def test_ref_title_and_message_text_merge_without_duplicates(self):
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': '当前问题'},
            'ref_msg': {
                'title': '标题引用',
                'message_item': {
                    'type': 1,
                    'text_item': {'text': '消息体引用'},
                },
            },
        }])
        self.assertIn('[引用消息]\n标题引用\n消息体引用\n[/引用消息]', envelope.text)

        duplicate = self._normalize([{
            'type': 1,
            'text_item': {'text': '当前问题'},
            'ref_msg': {
                'title': '相同引用',
                'message_item': {
                    'type': 1,
                    'text_item': {'text': '相同引用'},
                },
            },
        }])
        self.assertEqual(
            duplicate.provider_context['wechat_ref_msg'][0]['text'],
            '相同引用',
        )

    def test_top_level_ref_msg_fallback_is_delivered_and_serializable(self):
        envelope = self._normalize(
            [{'type': 1, 'text_item': {'text': '当前问题'}}],
            ref_msg={'title': '顶层引用'},
        )
        self.assertIn('顶层引用', envelope.text)
        encoded = json.dumps(
            envelope.provider_context['wechat_ref_msg'], ensure_ascii=False,
        )
        self.assertNotIn('aes_key', encoded)
        self.assertNotIn('context_token', encoded)

    def test_empty_ref_msg_does_not_change_plain_text(self):
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': '普通消息'},
            'ref_msg': {'title': '', 'message_item': {}},
        }])
        self.assertEqual(envelope.text, '普通消息')
        self.assertNotIn('wechat_ref_msg', envelope.provider_context)
        routed = self._route_wechat(envelope)
        self.assertIsInstance(routed.command, ChatCommand)
        self.assertEqual(routed.command.parameters.message, '普通消息')

    def test_multiple_ref_titles_keep_item_order(self):
        envelope = self._normalize([{
            'type': 1,
            'text_item': {'text': '当前消息'},
            'ref_msg': {'title': '第一条引用'},
        }, {
            'type': 1,
            'ref_msg': {'title': '第二条引用'},
        }])
        self.assertEqual(
            [item['text'] for item in envelope.provider_context['wechat_ref_msg']],
            ['第一条引用', '第二条引用'],
        )

    def test_attachment_budget_applies_before_ref_downloads(self):
        content = b'file'
        self.client.contents['current'] = content
        current = [self._file('current', content) for _ in range(10)]
        refs = [{
            'type': 1,
            'text_item': {'text': 'question'},
            'ref_msg': {'message_item': self._file('ref', content)['file_item']},
        }]

        envelope = self._normalize(current + refs)

        self.assertEqual(len(envelope.provider_context['channel_execution']['attachments']), 10)
        self.assertEqual(len(self.client.downloads), 10)

    def test_total_media_budget_skips_downloads_and_keeps_ref_text(self):
        content = b'file'
        self.client.contents['first'] = content
        self.client.contents['second'] = content
        second = self._file('second', content)
        envelope_context = patch(
            'channel_gateway.wechat.runtime._MAX_TOTAL_INBOUND_DOWNLOAD_BYTES',
            len(content),
        )
        with envelope_context:
            envelope = self._normalize([
                self._file('first', content),
                second,
                {
                    'type': 1,
                    'text_item': {'text': 'question'},
                    'ref_msg': {'message_item': {
                        'type': 1,
                        'text_item': {'text': '仍需保留的引用文本'},
                    }},
                },
            ])

        self.assertEqual(len(self.client.downloads), 1)
        self.assertEqual(len(envelope.provider_context['channel_execution']['attachments']), 1)
        self.assertIn('仍需保留的引用文本', envelope.text)

    def test_total_media_budget_is_shared_with_ref_attachments(self):
        content = b'file'
        self.client.contents['current'] = content
        self.client.contents['ref'] = content
        with patch(
            'channel_gateway.wechat.runtime._MAX_TOTAL_INBOUND_DOWNLOAD_BYTES',
            len(content),
        ):
            envelope = self._normalize([
                self._file('current', content),
                {
                    'type': 1,
                    'text_item': {'text': 'question'},
                    'ref_msg': {'message_item': self._file('ref', content)['file_item']},
                },
            ])

        self.assertEqual(len(self.client.downloads), 1)
        self.assertEqual(len(envelope.provider_context['channel_execution']['attachments']), 1)

    def test_failed_file_download_consumes_message_budget(self):
        content = b'file'
        self.client.contents['failed'] = content
        self.client.contents['next'] = content
        failed = self._file('failed', content)
        failed['file_item']['md5'] = '0' * 32
        failed['file_item']['media']['bytes_read'] = 8
        with patch(
            'channel_gateway.wechat.runtime._MAX_TOTAL_INBOUND_DOWNLOAD_BYTES',
            10,
        ):
            envelope = self._normalize([
                {'type': 1, 'text_item': {'text': '继续'}},
                failed,
                self._file('next', content),
            ])

        self.assertEqual(len(self.client.downloads), 1)
        self.assertEqual(
            envelope.provider_context['channel_execution']['attachments'], []
        )

    def test_unknown_image_mime_download_consumes_message_budget(self):
        content = b'file'
        self.client.contents['unknown-image'] = b'not an image'
        self.client.contents['next'] = content
        image = {
            'type': 2,
            'image_item': {
                'aeskey': '0' * 32,
                'media': {
                    'encrypt_query_param': 'unknown-image',
                    'bytes_read': 8,
                },
            },
        }
        with patch(
            'channel_gateway.wechat.runtime._MAX_TOTAL_INBOUND_DOWNLOAD_BYTES',
            10,
        ):
            envelope = self._normalize([
                {'type': 1, 'text_item': {'text': '继续'}},
                image,
                self._file('next', content),
            ])

        self.assertEqual(len(self.client.downloads), 1)
        self.assertEqual(
            envelope.provider_context['channel_execution']['attachments'], []
        )

    def test_failed_current_media_and_ref_media_share_download_budget(self):
        content = b'file'
        self.client.fail = True
        image = {
            'type': 2,
            'image_item': {
                'aeskey': '0' * 32,
                'media': {'encrypt_query_param': 'failed', 'bytes_read': 10},
            },
        }
        with patch(
            'channel_gateway.wechat.runtime._MAX_TOTAL_INBOUND_DOWNLOAD_BYTES',
            10,
        ):
            envelope = self._normalize([
                image,
                {
                    'type': 1,
                    'text_item': {'text': 'question'},
                    'ref_msg': {
                        'message_item': self._file('ref', content)['file_item'],
                        'title': '引用文本仍应保留',
                    },
                },
            ])

        self.assertEqual(len(self.client.downloads), 1)
        self.assertIn('引用文本仍应保留', envelope.text)

    def test_invalid_media_and_original_filters_do_not_crash(self):
        self.client.fail = True
        with_text = self._normalize([
            {'type': 1, 'text_item': {'text': '继续'}},
            {'type': 2, 'image_item': {'media': {'encrypt_query_param': 'bad'}}},
        ])

        self.assertEqual(with_text.text, '继续')
        self.assertIsNone(self._normalize([], from_user_id='other'))
        self.assertIsNone(self._normalize([], context_token=''))
        self.assertIsNone(self._normalize([], message_type=2))


class WeChatClientMediaDownloadTest(unittest.TestCase):
    _KEY = 'AAECAwQFBgcICQoLDA0ODw=='
    _CIPHERTEXT = bytes.fromhex('0dd3124e2a7a76f0c6bacd8ef54c8f60')

    def _download(self, chunks, **kwargs):
        client = WeChatClient('https://ilinkai.weixin.qq.com', 40)
        headers = kwargs.pop('headers', None)
        with patch(
            'channel_gateway.wechat.client.httpx.stream',
            return_value=_StreamResponse(chunks, headers=headers),
        ):
            return client.download_media(
                {
                    'full_url': 'https://novac2c.cdn.weixin.qq.com/c2c/download',
                    'aes_key': self._KEY,
                },
                max_bytes=kwargs.pop('max_bytes', 1024),
                max_download_bytes=kwargs.pop('max_download_bytes', 1024),
                **kwargs,
            )

    def test_fixed_independent_aes_ecb_pkcs7_vector(self):
        # Generated independently with OpenSSL AES-128-ECB + PKCS#7.
        plaintext, key = self._download([self._CIPHERTEXT])

        self.assertEqual(plaintext, b'fixed media')
        self.assertEqual(key, self._KEY)

    def test_streaming_limit_rejects_missing_or_false_content_length(self):
        cases = (
            ([b'x' * 17], None),
            ([b'x' * 8, b'x' * 9], {'content-length': '1'}),
        )
        for chunks, headers in cases:
            with self.assertRaisesRegex(WeChatError, 'size limit'):
                self._download(chunks, max_bytes=1, headers=headers)

    def test_stream_exception_path_reports_downloaded_bytes(self):
        charged = []
        with self.assertRaisesRegex(WeChatError, 'size limit'):
            self._download(
                [b'x' * 17],
                max_bytes=1,
                max_download_bytes=1,
                on_download_bytes=charged.append,
            )
        self.assertEqual(charged, [17])

    def test_bad_padding_and_non_wechat_url_are_rejected(self):
        with self.assertRaisesRegex(WeChatError, 'decryption or integrity'):
            self._download([b'\0' * 16])
        client = WeChatClient('https://ilinkai.weixin.qq.com', 40)
        with self.assertRaisesRegex(WeChatError, 'not an iLink CDN URL'):
            client.download_media(
                {'full_url': 'https://example.com/file', 'aes_key': self._KEY},
                max_bytes=1024,
                max_download_bytes=1024,
            )


class WeChatImageMimeTest(unittest.TestCase):
    def test_known_image_magic_and_unknown_bytes(self):
        cases = {
            b'\xff\xd8\xff\xe0': 'image/jpeg',
            b'\x89PNG\r\n\x1a\n': 'image/png',
            b'GIF89a': 'image/gif',
            b'RIFFxxxxWEBP': 'image/webp',
        }
        for content, media_type in cases.items():
            self.assertTrue(_image_data_url(content).startswith(f'data:{media_type};'))
        self.assertIsNone(_image_data_url(b'not an image'))


if __name__ == '__main__':
    unittest.main()
