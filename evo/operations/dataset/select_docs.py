from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ... import validate_id
from .kb_client import KnowledgeBaseClient


@dataclass(frozen=True)
class SelectDocsParams:
    kb_ids: list[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'SelectDocsParams':
        raw_ids = data.get('kb_ids')
        if not isinstance(raw_ids, list) or not raw_ids:
            raise ValueError('kb_ids must be a non-empty list')
        if not all(isinstance(value, str) for value in raw_ids):
            raise ValueError('kb_ids must contain only strings')
        kb_ids = [validate_id(value.strip(), 'kb_id') for value in raw_ids]
        if len(set(kb_ids)) != len(kb_ids):
            raise ValueError('kb_ids must be unique')
        if 'max_docs' in data:
            raise ValueError('max_docs is not supported')
        return cls(kb_ids=kb_ids)

    def to_dict(self) -> dict[str, Any]:
        return {'kb_ids': list(self.kb_ids)}


def select_docs(
    ctx: Any,
    inputs: Mapping[str, object],
    kb_client: KnowledgeBaseClient | None = None,
) -> Mapping[str, object]:
    source_config = inputs.get('source_config')
    if not isinstance(source_config, Mapping):
        raise ValueError('source_config must be a mapping')
    params = SelectDocsParams.from_dict(source_config)
    allocation = _allocation(inputs.get('import_cases_manifest'))
    auto_case_count = allocation['auto_case_count']
    if auto_case_count == 0:
        return {'selected_docs': _payload(params, [], {}, auto_case_count)}

    client = kb_client or KnowledgeBaseClient()
    rows_by_kb = {kb_id: list(client.list_documents(kb_id)) for kb_id in params.kb_ids}
    if not any(rows_by_kb.values()):
        raise ValueError('dataset.select_docs discovered no documents')
    docs = [
        {'kb_id': kb_id, **_doc_payload(row)}
        for kb_id in params.kb_ids
        for row in rows_by_kb[kb_id]
    ]
    return {'selected_docs': _payload(
        params, docs,
        {kb_id: len(rows) for kb_id, rows in rows_by_kb.items()},
        auto_case_count,
    )}


def _payload(
    params: SelectDocsParams,
    docs: list[dict[str, Any]],
    discovered_by_kb: dict[str, int],
    auto_case_count: int,
) -> dict[str, Any]:
    return {
        'kb_ids': list(params.kb_ids),
        'docs': docs,
        'stats': {
            'discovered_by_kb': discovered_by_kb,
            'discovered': len(docs),
        },
        'params': {**params.to_dict(), 'auto_case_count': auto_case_count},
    }


def _allocation(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError('import_cases_manifest must be a mapping')
    stats = value.get('stats')
    if not isinstance(stats, Mapping):
        raise ValueError('import_cases_manifest.stats must be a mapping')
    allocation = stats.get('case_allocation')
    if not isinstance(allocation, Mapping):
        raise ValueError('import_cases_manifest.stats.case_allocation must be a mapping')
    auto_case_count = allocation.get('auto_case_count')
    if isinstance(auto_case_count, bool) or not isinstance(auto_case_count, int) or auto_case_count < 0:
        raise ValueError('import_cases_manifest.stats.case_allocation.auto_case_count must be non-negative')
    return {'auto_case_count': auto_case_count}


def _doc_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'doc_id': str(row.get('doc_id') or ''),
        'filename': str(row.get('filename') or row.get('display_name') or row.get('doc_id') or ''),
        'file_type': str(row.get('file_type') or ''),
        'status': str(row.get('status') or row.get('upload_status') or ''),
    }
