from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .kb_client import KnowledgeBaseClient


REQUIRED_COLUMNS = {'question', 'question_type', 'ground_truth', 'grading_guidance'}
OPTIONAL_COLUMNS = {
    'case_id', 'difficulty', 'key_points', 'forbidden_claims', 'reference_context',
    'reference_doc', 'reference_doc_ids', 'reference_chunk_ids', 'generate_reason', 'is_deleted',
}
DIFFICULTIES = ('easy', 'medium', 'hard')
QUESTION_TYPES = ('precision', 'reasoning')


def import_cases(ctx: Any, inputs: Mapping[str, object], kb_client: KnowledgeBaseClient | None = None) -> dict[str, object]:
    del ctx
    config = _mapping(inputs.get('source_config'), 'source_config')
    kb_ids = _ids(config.get('kb_ids'), 'kb_ids')
    configured_target = _positive(config.get('target_case_count'), 'target_case_count')
    sources = _sources(config)
    inline_cases = config.get('imported_cases', [])
    if not isinstance(inline_cases, list):
        raise ValueError('imported_cases must be a list')
    if not sources and not inline_cases:
        return {'import_cases_manifest': _manifest([], configured_target, [], imported=False)}

    client = kb_client or KnowledgeBaseClient()
    csv_payloads: list[tuple[dict[str, str], list[dict[str, str]]]] = []
    source_records: list[dict[str, object]] = []
    for source in sources:
        path = source['path']
        try:
            raw = Path(path).read_bytes()
            reader = csv.DictReader(raw.decode('utf-8-sig').splitlines(), strict=True)
            headers = set(reader.fieldnames or ())
            rows = list(reader)
        except Exception as exc:
            raise ValueError(f'csv_path is unreadable: {path}') from exc
        if not REQUIRED_COLUMNS <= headers or headers - REQUIRED_COLUMNS - OPTIONAL_COLUMNS:
            raise ValueError(f'csv header is invalid: {path}')
        source_records.append({
            'kb_id': source['kb_id'], 'csv_path': path,
            'csv_sha256': hashlib.sha256(raw).hexdigest(), 'csv_size_bytes': len(raw),
        })
        csv_payloads.append((source, rows))

    chunk_ids: list[str] = []
    for index, raw_row in enumerate(inline_cases):
        row = _mapping(raw_row, f'imported_cases[{index}]')
        chunk_ids.extend(_peek_chunk_ids(row))
    for _, rows in csv_payloads:
        for row in rows:
            chunk_ids.extend(_peek_chunk_ids(row))
    chunks, documents = _knowledge_index(client, kb_ids, chunk_ids)

    details: list[dict[str, object]] = []
    questions: set[str] = set()
    case_ids: set[str] = set()
    source_row_number = 0
    for row_number, raw_row in enumerate(inline_cases, 1):
        row = _mapping(raw_row, f'imported_cases[{row_number - 1}]')
        detail: dict[str, object] = {
            'source_row_number': row_number,
            'source_id': str(row.get('case_id') or '').strip(),
        }
        _append_imported_detail(
            details, detail, row, chunks, documents, questions, case_ids, 'imported_eval_set',
        )
    source_row_number = len(inline_cases)
    for source, rows in csv_payloads:
        path = source['path']
        for csv_row_number, row in enumerate(rows, 2):
            source_row_number += 1
            detail = {
                'source_row_number': source_row_number,
                'csv_row_number': csv_row_number,
                'source_kb_id': source['kb_id'],
                'source_path': path,
                'source_id': str(row.get('case_id') or '').strip(),
            }
            _append_imported_detail(
                details, detail, row, chunks, documents, questions, case_ids, 'imported_csv',
                extra_source={'kb_id': str(detail.get('source_kb_id') or ''),
                              'csv_path': str(detail.get('source_path') or '')},
            )

    if not any(item['load_status'] == 'loaded' for item in details):
        raise ValueError('no valid imported cases')
    return {'import_cases_manifest': _manifest(
        source_records,
        configured_target,
        details,
        imported=True,
        supplement=bool(config.get('supplement_existing_eval_set', False)),
    )}


def _append_imported_detail(
    details: list[dict[str, object]],
    detail: dict[str, object],
    row: Mapping[str, object],
    chunks: Mapping[str, dict[str, str]],
    documents: Mapping[str, set[str]],
    questions: set[str],
    case_ids: set[str],
    source: str,
    extra_source: Mapping[str, str] | None = None,
) -> None:
    try:
        case, deleted = _case(row, chunks, documents, questions, case_ids)
    except ValueError as exc:
        details.append({**detail, 'load_status': 'invalid', 'error': {
            'code': str(exc).split(':', 1)[0], 'reason': str(exc),
        }})
        return
    if deleted:
        details.append({**detail, 'load_status': 'deleted'})
        return
    case_id = str(case.get('id') or '')
    if not case_id:
        case_id = _next_case_id(case_ids)
        case['id'] = case_id
        case_ids.add(case_id)
    preparation = dict(_mapping(case.get('source_preparation'), 'case.source_preparation'))
    case_source = {
        'final_id': case_id,
        'original_id': str(detail.get('source_id') or case_id),
        'source': source,
    }
    if extra_source:
        case_source.update(extra_source)
    preparation['case_source'] = case_source
    case['source_preparation'] = preparation
    details.append({**detail, 'load_status': 'loaded', 'case_id': case_id, 'case': case})


def _manifest(sources: list[dict[str, object]], target: int, details: list[dict[str, object]], *,
              imported: bool, supplement: bool = False) -> dict[str, object]:
    loaded = [item for item in details if item['load_status'] == 'loaded']
    invalid = [item for item in details if item['load_status'] == 'invalid']
    deleted = [item for item in details if item['load_status'] == 'deleted']
    if imported:
        assignments = {
            str(item['case_id']): {'mode': 'imported', 'source_row_number': item['source_row_number']}
            for item in loaded
        }
        if supplement:
            target = max(target, len(loaded))
            automatic = target - len(loaded)
            case_ids = set(assignments)
            for _ in range(automatic):
                case_id = _next_case_id(case_ids)
                case_ids.add(case_id)
                assignments[case_id] = {'mode': 'generated'}
        else:
            target = len(loaded)
            automatic = 0
    else:
        assignments = {f'case_{index:04d}': {'mode': 'generated'} for index in range(1, target + 1)}
        automatic = target
    return {
        'source': {'csv_sources': sources},
        'stats': {
            'csv_reading': {
                'total_row_count': len(details),
                'valid_row_count': len(loaded),
                'loaded_row_count': len(loaded),
                'invalid_row_count': len(invalid),
                'deleted_row_count': len(deleted),
                'truncated_row_count': 0,
            },
            'case_allocation': {
                'target_case_count': target,
                'import_case_count': len(loaded),
                'auto_case_count': automatic,
                'assignments': assignments,
            },
        },
        'details': details,
    }


def _sources(config: Mapping[str, object]) -> list[dict[str, str]]:
    raw_sources = config.get('csv_sources')
    if raw_sources is None:
        path = str(config.get('csv_path') or '').strip()
        raw_sources = [] if not path else [{'kb_id': _ids(config.get('kb_ids'), 'kb_ids')[0], 'path': path}]
    if not isinstance(raw_sources, list):
        raise ValueError('csv_sources must be a list')
    sources = []
    for index, raw in enumerate(raw_sources):
        source = _mapping(raw, f'csv_sources[{index}]')
        sources.append({
            'kb_id': _text(source.get('kb_id'), f'csv_sources[{index}].kb_id'),
            'path': _text(source.get('path'), f'csv_sources[{index}].path'),
        })
    return sources


def _case(row: Mapping[str, str], chunks: Mapping[str, dict[str, str]], documents: Mapping[str, set[str]],
          questions: set[str], case_ids: set[str]) -> tuple[dict[str, object], bool]:
    question = _text(row.get('question'), 'question')
    answer = _text(row.get('ground_truth'), 'ground_truth')
    guidance = _text(row.get('grading_guidance'), 'grading_guidance')
    question_type = _choice(row.get('question_type'), QUESTION_TYPES, 'question_type')
    deleted = _boolean(row.get('is_deleted'), 'is_deleted')
    if deleted:
        return {}, True

    normalized_question = ' '.join(question.lower().split())
    if normalized_question in questions:
        raise ValueError('duplicate_question: question duplicates an earlier row')
    source_id = _optional_text(row.get('case_id'))
    if source_id and source_id in case_ids:
        raise ValueError('duplicate_case_id: case_id duplicates an earlier row')

    reference_doc_ids = _string_list(row.get('reference_doc_ids'), 'reference_doc_ids')
    reference_chunk_ids = _string_list(row.get('reference_chunk_ids'), 'reference_chunk_ids')
    _unique(reference_doc_ids, 'duplicate_reference_doc_id')
    _unique(reference_chunk_ids, 'duplicate_reference_chunk_id')
    for doc_id in reference_doc_ids:
        if doc_id not in documents:
            raise ValueError('reference_doc_not_found: document id cannot be resolved')
    resolved_references: list[dict[str, str]] = []
    for chunk_id in reference_chunk_ids:
        found = chunks.get(chunk_id)
        if not found:
            raise ValueError('reference_chunk_not_found: chunk id cannot be resolved')
        if reference_doc_ids and found['doc_id'] not in reference_doc_ids:
            raise ValueError('reference_document_mismatch: chunk does not belong to a referenced document')
        resolved_references.append({'chunk_id': chunk_id, **found})

    difficulty = _optional_text(row.get('difficulty'))
    if difficulty and difficulty not in DIFFICULTIES:
        raise ValueError('invalid_difficulty: unsupported value')
    if not difficulty:
        difficulty = _difficulty(len(reference_chunk_ids))

    key_points = _json_array(row.get('key_points'), 'key_points')
    forbidden_claims = _json_array(row.get('forbidden_claims'), 'forbidden_claims')
    source_preparation = {
        'dataset_mode': 'imported',
        'kb_ids': list(dict.fromkeys(item['kb_id'] for item in resolved_references)),
    }
    questions.add(normalized_question)
    if source_id:
        case_ids.add(source_id)
    return {
        'id': source_id,
        'question': question,
        'answer': answer,
        'question_type': question_type,
        'difficulty': difficulty,
        'grading_guidance': guidance,
        'key_points': key_points,
        'forbidden_claims': forbidden_claims,
        'reference_context': _context(row.get('reference_context')),
        'reference_doc': _string_list(row.get('reference_doc'), 'reference_doc'),
        'reference_chunk_ids': reference_chunk_ids,
        'reference_doc_ids': reference_doc_ids,
        'references': resolved_references,
        'generate_reason': _optional_text(row.get('generate_reason')),
        'is_deleted': False,
        'source_preparation': source_preparation,
    }, False


def _knowledge_index(
    client: KnowledgeBaseClient,
    kb_ids: list[str],
    chunk_ids: list[str] | None = None,
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    chunks: dict[str, dict[str, str]] = {}
    documents: dict[str, set[str]] = {}
    lookup = getattr(client, 'lookup_chunks', None)
    wanted = list(dict.fromkeys(chunk_ids or ()))
    for kb_id in kb_ids:
        kb_docs = []
        for doc in client.list_documents(kb_id):
            doc_id = str(doc['doc_id'])
            documents.setdefault(doc_id, set()).add(kb_id)
            kb_docs.append(doc_id)
        if lookup is not None:
            for chunk_id, payload in lookup(kb_id, wanted).items():
                _index_chunk(chunks, chunk_id, payload)
            continue
        for doc_id in kb_docs:
            for batch in client.iter_chunks(kb_id, [doc_id], ['block', 'line'], 200, require_embeddings=False):
                for node in batch:
                    chunk_id = str(getattr(node, 'uid', '') or getattr(node, '_uid', '') or '')
                    _index_chunk(chunks, chunk_id, {
                        'kb_id': kb_id,
                        'doc_id': doc_id,
                        'text': str(getattr(node, 'text', '') or ''),
                    })
    return chunks, documents


def _index_chunk(
    chunks: dict[str, dict[str, str]],
    chunk_id: str,
    payload: Mapping[str, str],
) -> None:
    if not chunk_id:
        return
    existing = chunks.get(chunk_id)
    if existing is None:
        chunks[chunk_id] = dict(payload)
        return
    if existing.get('doc_id') != payload.get('doc_id'):
        raise ValueError(f'ambiguous_chunk_id: {chunk_id}')


def _peek_chunk_ids(row: Mapping[str, object]) -> list[str]:
    value = row.get('reference_chunk_ids')
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    try:
        return _string_list(value, 'reference_chunk_ids')
    except ValueError:
        return []


def _next_case_id(case_ids: set[str]) -> str:
    index = 1
    while f'case_{index:04d}' in case_ids:
        index += 1
    return f'case_{index:04d}'


def _difficulty(reference_count: int) -> str:
    if reference_count == 1:
        return 'easy'
    if reference_count == 2:
        return 'medium'
    if reference_count >= 3:
        return 'hard'
    return ''


def _context(value: object) -> object:
    text = _optional_text(value)
    if not text:
        return ''
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _json_array(value: object, name: str) -> list[object]:
    text = _optional_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'invalid_{name}: {name} must be JSON') from exc
    if not isinstance(parsed, list):
        raise ValueError(f'invalid_{name}: {name} must be an array')
    return parsed


def _string_list(value: object, name: str) -> list[str]:
    text = _optional_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = [part.strip() for part in text.split(',')]
    if not isinstance(parsed, list) or any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise ValueError(f'invalid_{name}: {name} must be a string list')
    return [item.strip() for item in parsed]


def _boolean(value: object, name: str) -> bool:
    text = _optional_text(value).lower()
    if not text:
        return False
    if text in {'true', '1'}:
        return True
    if text in {'false', '0'}:
        return False
    raise ValueError(f'invalid_{name}: {name} must be boolean')


def _unique(values: list[str], code: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f'{code}: values must be unique')


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _ids(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must be a non-empty list')
    values = [_text(item, name) for item in value]
    if len(set(values)) != len(values):
        raise ValueError(f'{name} must be unique')
    return values


def _optional_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ''


def _text(value: object, name: str) -> str:
    text = _optional_text(value)
    if not text:
        raise ValueError(f'{name} is required')
    return text


def _choice(value: object, values: tuple[str, ...], name: str) -> str:
    item = _text(value, name)
    if item not in values:
        raise ValueError(f'invalid_{name}: unsupported value')
    return item


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be positive')
    return value
