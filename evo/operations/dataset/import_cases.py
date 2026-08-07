from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .kb_client import KnowledgeBaseClient


REQUIRED_COLUMNS = {'question', 'answer', 'question_type', 'difficulty', 'grading_guidance', 'reference_context'}
OPTIONAL_COLUMNS = {'id'}
REFERENCE_COUNTS = {'easy': 1, 'medium': 2, 'hard': 3}


def import_cases(
    ctx: Any,
    inputs: Mapping[str, object],
    kb_client: KnowledgeBaseClient | None = None,
) -> dict[str, object]:
    config = _mapping(inputs.get('source_config'), 'source_config')
    kb_ids = _ids(config.get('kb_ids'), 'kb_ids')
    target = _positive(config.get('target_case_count'), 'target_case_count')
    sources = _sources(config)
    if not sources:
        return {'import_cases_manifest': _manifest([], target, [], [])}

    index = _chunk_index(kb_client or KnowledgeBaseClient(), kb_ids)
    details: list[dict[str, object]] = []
    valid: list[dict[str, object]] = []
    questions: set[str] = set()
    source_records: list[dict[str, object]] = []
    source_row_number = 0
    for source in sources:
        path = source['path']
        file_path = Path(path)
        try:
            raw = file_path.read_bytes()
            reader = csv.DictReader(raw.decode('utf-8-sig').splitlines(), strict=True)
            headers = set(reader.fieldnames or ())
            rows = list(reader)
        except Exception as exc:
            raise ValueError(f'csv_path is unreadable: {path}') from exc
        if not REQUIRED_COLUMNS <= headers or headers - REQUIRED_COLUMNS - OPTIONAL_COLUMNS:
            raise ValueError(f'csv header is invalid: {path}')
        source_records.append({
            'kb_id': source['kb_id'],
            'csv_path': path,
            'csv_sha256': hashlib.sha256(raw).hexdigest(),
            'csv_size_bytes': len(raw),
        })
        for csv_row_number, row in enumerate(rows, 2):
            source_row_number += 1
            detail = {
                'source_row_number': source_row_number,
                'csv_row_number': csv_row_number,
                'source_kb_id': source['kb_id'],
                'source_path': path,
                'source_id': str(row.get('id') or '').strip(),
            }
            try:
                case = _case(row, index, questions)
            except ValueError as exc:
                details.append({
                    **detail,
                    'load_status': 'invalid',
                    'error': {'code': str(exc).split(':', 1)[0], 'reason': str(exc)},
                })
            else:
                valid.append({**detail, 'case': case})
                details.append({**detail, 'load_status': 'pending', 'case': case})
    valid_index = 0
    for detail in details:
        if detail['load_status'] != 'pending':
            continue
        if valid_index < target:
            case_id = f'case_{valid_index + 1:04d}'
            case = dict(detail['case'])
            case['id'] = case_id
            preparation = dict(_mapping(case.get('source_preparation'), 'case.source_preparation'))
            preparation['case_source'] = {
                'final_id': case_id,
                'original_id': str(detail.get('source_id') or case_id),
                'source': 'imported_csv',
                'kb_id': str(detail.get('source_kb_id') or ''),
                'csv_path': str(detail.get('source_path') or ''),
            }
            case['source_preparation'] = preparation
            detail.update({'load_status': 'loaded', 'case_id': case_id, 'case': case})
        else:
            detail.pop('case', None)
            detail['load_status'] = 'truncated'
        valid_index += 1
    return {'import_cases_manifest': _manifest(source_records, target, details, valid)}


def _manifest(
    sources: list[dict[str, object]],
    target: int,
    details: list[dict[str, object]],
    valid: list[dict[str, object]],
) -> dict[str, object]:
    loaded = [item for item in details if item['load_status'] == 'loaded']
    assignments = {
        item['case_id']: {'mode': 'imported', 'source_row_number': item['source_row_number']}
        for item in loaded
    }
    assignments |= {f'case_{i:04d}': {'mode': 'generated'} for i in range(len(loaded) + 1, target + 1)}
    return {
        'source': {'csv_sources': sources},
        'stats': {
            'csv_reading': {
                'total_row_count': len(details),
                'valid_row_count': len(valid),
                'loaded_row_count': len(loaded),
            },
            'case_allocation': {
                'target_case_count': target,
                'import_case_count': len(loaded),
                'auto_case_count': target - len(loaded),
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


def _case(row: Mapping[str, str], index: Mapping[str, dict[str, str]], questions: set[str]) -> dict[str, object]:
    question, answer, guidance = (_text(row.get(name), name) for name in ('question', 'answer', 'grading_guidance'))
    normalized_question = ' '.join(question.lower().split())
    if normalized_question in questions:
        raise ValueError('duplicate_question: question duplicates an earlier row')
    question_type = _choice(row.get('question_type'), ('precision', 'reasoning'), 'question_type')
    difficulty = _choice(row.get('difficulty'), tuple(REFERENCE_COUNTS), 'difficulty')
    try:
        references = json.loads(_text(row.get('reference_context'), 'reference_context'))
    except json.JSONDecodeError as exc:
        raise ValueError('invalid_reference_context: reference_context must be JSON') from exc
    if not isinstance(references, list) or len(references) != REFERENCE_COUNTS[difficulty]:
        raise ValueError('invalid_reference_count: reference count does not match difficulty')
    result, seen = [], set()
    for reference in references:
        if not isinstance(reference, Mapping):
            raise ValueError('invalid_reference_context: reference must be an object')
        chunk_id = _text(reference.get('chunk_id'), 'chunk_id')
        if chunk_id in seen:
            raise ValueError('duplicate_reference_chunk_id: duplicate chunk id')
        seen.add(chunk_id)
        found = index.get(chunk_id)
        if not found:
            raise ValueError('reference_chunk_not_found: chunk id cannot be resolved')
        text = _clean(_text(reference.get('text'), 'reference text'))
        if text != _clean(found['text']):
            raise ValueError('reference_text_mismatch: csv text differs from knowledge base')
        result.append({'chunk_id': chunk_id, 'text': text, **found})
    questions.add(normalized_question)
    return {
        'id': '',
        'question': question,
        'answer': answer,
        'question_type': question_type,
        'difficulty': difficulty,
        'grading_guidance': guidance,
        'reference_context': [
            {'chunk_id': item['chunk_id'], 'text': item['text']}
            for item in result
        ],
        'reference_chunk_ids': [item['chunk_id'] for item in result],
        'reference_doc_ids': list(dict.fromkeys(item['doc_id'] for item in result)),
        'source_preparation': {
            'kb_ids': list(dict.fromkeys(item['kb_id'] for item in result)),
        },
    }


def _chunk_index(client: KnowledgeBaseClient, kb_ids: list[str]) -> dict[str, dict[str, str]]:
    values: dict[str, dict[str, str]] = {}
    for kb_id in kb_ids:
        for doc in client.list_documents(kb_id):
            doc_id = str(doc['doc_id'])
            for batch in client.iter_chunks(
                kb_id, [doc_id], ['block', 'line'], 200, require_embeddings=False,
            ):
                for node in batch:
                    chunk_id = str(getattr(node, 'uid', '') or getattr(node, '_uid', '') or '')
                    if chunk_id in values:
                        raise ValueError(f'ambiguous_chunk_id: {chunk_id}')
                    values[chunk_id] = {
                        'kb_id': kb_id,
                        'doc_id': doc_id,
                        'text': str(getattr(node, 'text', '') or ''),
                    }
    return values


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


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} is required')
    return value.strip()


def _choice(value: object, values: tuple[str, ...], name: str) -> str:
    value = _text(value, name)
    if value not in values:
        raise ValueError(f'invalid_{name}: unsupported value')
    return value


def _positive(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be positive')
    return value


def _clean(value: str) -> str:
    return value.replace('\r\n', '\n').replace('\r', '\n').strip()
