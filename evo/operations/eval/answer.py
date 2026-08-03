from __future__ import annotations

import os
import re
import math
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from evo.operations.route.chat_router import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_WAIT_MAX_SECONDS,
    RouterChatRequest,
    call_router_chat,
)

HEX = re.compile(r'^[0-9a-fA-F]+$')
DEFAULT_CASE_DEADLINE_SECONDS = 300.0
DEFAULT_FIRST_FRAME_TIMEOUT_SECONDS = 60.0


def answer_case(case: Mapping[str, Any], target_config: Mapping[str, Any]) -> dict[str, Any]:
    kb_id = case_kb_id(case, target_config)
    target = {
        'router_chat_url': str(target_config.get('router_chat_url') or ''),
        'router_admin_url': str(target_config.get('router_admin_url') or ''),
        'algorithm_id': str(target_config.get('algorithm_id') or ''),
        'kb_id': kb_id,
    }
    if not kb_id:
        return failed_rag_answer(case, {}, target, 'dataset_contract_error',
                                 'case routing metadata missing kb_id')
    if not _has_role(target_config.get('llm_config'), 'llm'):
        return failed_rag_answer(case, {}, target, 'chat_config_error',
                                 'eval.target_config.llm_config.llm missing; '
                                 'eval must be launched through core model-config injection')
    if not target['router_admin_url']:
        return failed_rag_answer(case, {}, target, 'chat_config_error',
                                 'eval.target_config.router_admin_url missing')
    if not target['algorithm_id']:
        return failed_rag_answer(case, {}, target, 'chat_config_error',
                                 'eval.target_config.algorithm_id missing')
    return call_chat_answer(case, target_config, kb_id)


def case_kb_id(case: Mapping[str, Any], target_config: Mapping[str, Any]) -> str:
    case_id = str(case.get('id') or '')
    by_case = target_config.get('case_metadata_by_id')
    preparation = case.get('source_preparation') if isinstance(case.get('source_preparation'), Mapping) else {}
    case_source = preparation.get('case_source') if isinstance(preparation.get('case_source'), Mapping) else {}
    if isinstance(by_case, Mapping) and isinstance(by_case.get(case_id), Mapping):
        text = _kb_ids_text(by_case[case_id].get('kb_ids') or by_case[case_id].get('kb_id'))
        if text:
            return text
    for value in (
        preparation.get('kb_ids'),
        preparation.get('kb_id'),
        case_source.get('kb_ids'),
        case_source.get('kb_id'),
        target_config.get('kb_ids'),
        target_config.get('kb_id'),
    ):
        text = _kb_ids_text(value)
        if text:
            return text
    return ''


def _kb_ids_text(value: object) -> str:
    if isinstance(value, list | tuple | set):
        return ';'.join(dict.fromkeys(str(item).strip() for item in value if str(item or '').strip()))
    return str(value or '').strip()


def call_chat_answer(case: Mapping[str, Any], target_config: Mapping[str, Any], kb_id: str) -> dict[str, Any]:
    try:
        kb_ids = tuple(dict.fromkeys(item.strip() for item in str(kb_id or '').split(';') if item.strip()))
        session_id = str(target_config.get('session_id') or uuid4().hex).strip().lower()
        session_id = session_id if HEX.fullmatch(session_id) else uuid4().hex
        target = _target(target_config, kb_ids, session_id)
        if not kb_ids:
            return failed_rag_answer(case, {}, target, 'dataset_contract_error',
                                     'case routing metadata missing kb_id')

        result = call_router_chat(RouterChatRequest(
            router_chat_url=target['router_chat_url'],
            router_admin_url=target['router_admin_url'],
            algorithm_id=target['algorithm_id'],
            query=str(case.get('question') or ''),
            kb_ids=kb_ids,
            trace_id=session_id,
            conversation_id=target['conversation_id'],
            user_id=target['user_id'],
            llm_config=target_config.get('llm_config') if isinstance(target_config.get('llm_config'), Mapping) else None,
            connect_timeout_seconds=_number(target_config.get('connect_timeout_seconds'), 5.0),
            write_timeout_seconds=_number(target_config.get('write_timeout_seconds'), 60.0),
            pool_timeout_seconds=_number(target_config.get('pool_timeout_seconds'), 5.0),
            case_deadline_seconds=_number(
                target_config.get('case_deadline_seconds') or os.getenv('LAZYMIND_EVO_CHAT_CASE_DEADLINE_SECONDS'),
                DEFAULT_CASE_DEADLINE_SECONDS,
            ),
            first_frame_timeout_seconds=_number(
                target_config.get('first_frame_timeout_seconds')
                or os.getenv('LAZYMIND_EVO_CHAT_FIRST_FRAME_TIMEOUT_SECONDS'),
                DEFAULT_FIRST_FRAME_TIMEOUT_SECONDS,
            ),
            max_attempts=_configured_value(
                target_config,
                'chat_max_attempts',
                'LAZYMIND_EVO_CHAT_MAX_ATTEMPTS',
                DEFAULT_MAX_ATTEMPTS,
            ),
            retry_wait_max_seconds=_configured_value(
                target_config,
                'chat_retry_wait_max_seconds',
                'LAZYMIND_EVO_CHAT_RETRY_WAIT_MAX_SECONDS',
                DEFAULT_RETRY_WAIT_MAX_SECONDS,
            ),
        ))
    except (TypeError, ValueError) as exc:
        target = _raw_target(target_config, kb_id)
        return failed_rag_answer(case, {}, target, 'chat_config_error', str(exc))
    return _with_case(case, result)


def failed_rag_answer(
    case: Mapping[str, Any],
    stream: Mapping[str, Any],
    target: Mapping[str, Any],
    error_type: str,
    message: str,
) -> dict[str, Any]:
    return _answer_base(case, stream, target) | {
        'status': 'failed',
        'chat_error': {'type': error_type, 'message': message},
        'evidence_status': 'failed',
    }


def _with_case(case: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    stream = {'answer': result.get('answer') or '', 'frames': result.get('frames') or []}
    answer = _answer_base(case, stream, result.get('target') if isinstance(result.get('target'), Mapping) else {})
    answer.update(dict(result))
    _align_evidence_refs(answer)
    answer['case_id'] = str(case.get('id') or answer.get('case_id') or '')
    answer['question'] = str(case.get('question') or '')
    answer['evidence_status'] = _evidence_status(answer)
    return answer


def _target(target_config: Mapping[str, Any], kb_ids: tuple[str, ...], session_id: str) -> dict[str, str]:
    target = {
        'router_chat_url': str(target_config.get('router_chat_url') or '').strip(),
        'router_admin_url': str(target_config.get('router_admin_url') or '').strip(),
        'algorithm_id': str(target_config.get('algorithm_id') or '').strip(),
        'kb_id': ';'.join(kb_ids),
        'trace_id': session_id,
        'conversation_id': str(target_config.get('conversation_id') or session_id).strip(),
        'user_id': str(target_config.get('user_id') or '0').strip() or '0',
    }
    target.update({
        key: str(target_config[key])
        for key in ('target_id', 'target_kind', 'target_label')
        if target_config.get(key)
    })
    if not target['router_admin_url']:
        raise ValueError('router_admin_url is required')
    if not target['algorithm_id']:
        raise ValueError('algorithm_id is required')
    return target


def _raw_target(target_config: Mapping[str, Any], kb_id: str) -> dict[str, str]:
    return {
        'router_chat_url': str(target_config.get('router_chat_url') or ''),
        'router_admin_url': str(target_config.get('router_admin_url') or ''),
        'algorithm_id': str(target_config.get('algorithm_id') or ''),
        'kb_id': str(kb_id or ''),
    }


def _answer_base(case: Mapping[str, Any], stream: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'case_id': str(case.get('id') or ''),
        'question': str(case.get('question') or ''),
        'answer': str(stream.get('answer') or ''),
        'tool_errors': [],
        'contexts': [],
        'doc_ids': [],
        'chunk_ids': [],
        'trace_id': str(target.get('trace_id') or ''),
        'target': dict(target),
    }


def _align_evidence_refs(answer: dict[str, Any]) -> None:
    refs = _source_refs(answer.get('sources'), answer.get('target') if isinstance(answer.get('target'), Mapping) else {})
    if not refs:
        refs = _zipped_refs(answer.get('contexts'), answer.get('doc_ids'), answer.get('chunk_ids'))
    if not refs:
        return
    answer['contexts'] = refs
    answer['doc_ids'] = _unique(ref.get('doc_id') for ref in refs)
    answer['chunk_ids'] = _unique(ref.get('chunk_id') for ref in refs)


def _source_refs(sources: Any, target: Mapping[str, Any]) -> list[dict[str, str]]:
    values = sources if isinstance(sources, list | tuple) else []
    target_kbs = [item for item in str(target.get('kb_id') or '').split(';') if item]
    fallback_kb = target_kbs[0] if len(target_kbs) == 1 else ''
    refs = []
    seen: set[tuple[str, str, str]] = set()
    for source in values:
        if not isinstance(source, Mapping):
            continue
        metadata = source.get('global_metadata') if isinstance(source.get('global_metadata'), Mapping) else {}
        kb_id = str(source.get('kb_id') or source.get('dataset_id') or metadata.get('kb_id')
                    or metadata.get('dataset_id') or fallback_kb).strip()
        doc = str(source.get('doc_id') or source.get('docid') or source.get('document_id')
                  or metadata.get('docid') or metadata.get('core_document_id') or '').strip()
        chunk = str(source.get('chunk_id') or source.get('chunkid') or source.get('segment_id')
                    or source.get('segement_id') or source.get('uid') or source.get('id') or '').strip()
        doc_ref = doc if ':' in doc else f'{kb_id}:{doc}' if kb_id and doc else doc
        chunk_ref = chunk if ':' in chunk else f'{doc_ref}:{chunk}' if doc_ref and chunk else chunk
        content = str(source.get('content') or source.get('text') or source.get('chunk') or '').strip()
        key = (content, doc_ref, chunk_ref)
        if (not any(key)) or key in seen:
            continue
        seen.add(key)
        refs.append({
            'content': content,
            'doc_id': doc_ref,
            'doc_name': _source_doc_name(source, metadata, doc_ref),
            'chunk_id': chunk_ref,
        })
    return refs


def _source_doc_name(source: Mapping[str, Any], metadata: Mapping[str, Any], doc_id: str) -> str:
    for key in ('doc_name', 'document_name', 'title', 'file_name', 'filename', 'name'):
        text = str(source.get(key) or metadata.get(key) or '').strip()
        if text:
            return text
    return doc_id


def _zipped_refs(contexts: Any, doc_ids: Any, chunk_ids: Any) -> list[dict[str, str]]:
    context_values = contexts if isinstance(contexts, list | tuple) else [contexts] if contexts else []
    doc_values = list(_ordered_values(doc_ids))
    chunk_values = list(_ordered_values(chunk_ids))
    refs = []
    seen: set[tuple[str, str, str]] = set()
    for index, context in enumerate(context_values):
        if isinstance(context, Mapping):
            content = str(context.get('content') or context.get('text') or context.get('context') or '').strip()
            doc_id = str(context.get('doc_id') or '').strip()
            chunk_id = str(context.get('chunk_id') or context.get('id') or '').strip()
            doc_name = str(
                context.get('doc_name') or context.get('document_name') or context.get('title')
                or context.get('file_name') or context.get('filename') or context.get('name') or doc_id
            ).strip()
        else:
            content = str(context or '').strip()
            doc_id = ''
            chunk_id = ''
            doc_name = ''
        if index < len(doc_values) and not doc_id:
            doc_id = doc_values[index]
        if index < len(chunk_values) and not chunk_id:
            chunk_id = chunk_values[index]
        if not doc_name:
            doc_name = doc_id
        key = (content, doc_id, chunk_id)
        if (not any(key)) or key in seen:
            continue
        seen.add(key)
        refs.append({'content': content, 'doc_id': doc_id, 'doc_name': doc_name, 'chunk_id': chunk_id})
    return refs


def _ordered_values(value: Any) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    return [str(item).strip() for item in values if str(item or '').strip()]


def _unique(values: Any) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value or '').strip()))


def _evidence_status(answer: Mapping[str, Any]) -> str:
    if answer.get('status') != 'ok':
        return 'failed'
    return 'found' if answer.get('contexts') or answer.get('doc_ids') or answer.get('chunk_ids') else 'empty'


def _number(value: Any, default: float) -> float:
    result = float(default if value in (None, '') else value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('timeout values must be positive finite numbers')
    return result


def _configured_value(config: Mapping[str, Any], key: str, env_name: str, default: Any) -> Any:
    if key in config:
        value = config[key]
        return default if value in (None, '') else value
    value = os.getenv(env_name)
    return default if value in (None, '') else value


def _has_role(value: object, role_name: str) -> bool:
    return isinstance(value, Mapping) and isinstance(value.get(role_name), Mapping) and bool(value[role_name])
