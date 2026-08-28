from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ... import validate_id
from .kb_client import KnowledgeBaseClient


@dataclass(frozen=True)
class SelectDocsParams:
    kb_ids: list[str]
    knowledge_bases: dict[str, bool]
    excluded_docs: list[dict[str, str]]

    @classmethod
    def from_dict(cls, source: Mapping[str, Any], raw: Mapping[str, Any] | None = None) -> 'SelectDocsParams':
        values = source.get('kb_ids')
        if not isinstance(values, list) or not values:
            raise ValueError('kb_ids must be a non-empty list')
        try:
            kb_ids = [validate_id(str(value).strip(), 'kb_id') for value in values]
        except ValueError as exc:
            raise ValueError('kb_ids contains an invalid value') from exc
        if len(set(kb_ids)) != len(kb_ids):
            raise ValueError('kb_ids must be unique')
        data = raw or {}
        knowledge_bases = _knowledge_base_inclusion(data.get('knowledge_bases'), kb_ids)
        values = data.get('excluded_docs', [])
        if not isinstance(values, list):
            raise ValueError('excluded_docs must be a list')
        excluded, seen = [], set()
        for index, value in enumerate(values):
            if not isinstance(value, Mapping):
                raise ValueError(f'excluded_docs[{index}] must be a mapping')
            try:
                item = {
                    'kb_id': validate_id(str(value.get('kb_id') or '').strip(), 'excluded_docs.kb_id'),
                    'doc_id': validate_id(str(value.get('doc_id') or '').strip(), 'excluded_docs.doc_id'),
                }
            except ValueError as exc:
                raise ValueError(f'excluded_docs[{index}] contains an invalid reference') from exc
            key = item['kb_id'], item['doc_id']
            if key in seen:
                raise ValueError('excluded_docs must contain unique (kb_id, doc_id) references')
            seen.add(key)
            excluded.append(item)
        return cls(kb_ids, knowledge_bases, excluded)


def select_docs(ctx: Any, inputs: Mapping[str, object], kb_client: KnowledgeBaseClient | None = None) -> Mapping[str, object]:
    source = inputs.get('source_config')
    if not isinstance(source, Mapping):
        raise ValueError('source_config must be a mapping')
    raw = inputs.get('select_docs_params', {})
    if not isinstance(raw, Mapping):
        raise ValueError('select_docs_params must be a mapping')
    params = SelectDocsParams.from_dict(source, raw)
    allocation = _allocation(inputs.get('import_cases_manifest'))
    if allocation['auto_case_count'] == 0:
        return {'selected_docs': {'documents': [], 'stats': {
            'discovered_count': 0, 'included_count': 0, 'excluded_count': 0,
        }}}
    excluded = {(item['kb_id'], item['doc_id']) for item in params.excluded_docs}
    client = kb_client or KnowledgeBaseClient()
    names = _knowledge_base_names(source, params.kb_ids)
    documents = []
    for kb_id in params.kb_ids:
        for row in client.list_documents(kb_id):
            doc_id = str(row.get('doc_id') or '').strip()
            if not doc_id:
                continue
            documents.append({
                'kb_id': kb_id,
                'doc_id': doc_id,
                'filename': str(row.get('filename') or row.get('display_name') or doc_id),
                'knowledge_base_name': names[kb_id],
                'file_type': str(row.get('file_type') or ''),
                'status': str(row.get('status') or row.get('upload_status') or ''),
                'included': params.knowledge_bases[kb_id] and (kb_id, doc_id) not in excluded,
                'discovery_index': len(documents),
            })
    included = sum(item['included'] for item in documents)
    return {'selected_docs': {'documents': documents, 'stats': {
        'discovered_count': len(documents), 'included_count': included, 'excluded_count': len(documents) - included,
    }}}


def _allocation(value: object) -> Mapping[str, object]:
    manifest = _mapping(value, 'import_cases_manifest')
    stats = _mapping(manifest.get('stats'), 'import_cases_manifest.stats')
    allocation = _mapping(stats.get('case_allocation'), 'import_cases_manifest.stats.case_allocation')
    auto_case_count = allocation.get('auto_case_count')
    if isinstance(auto_case_count, bool) or not isinstance(auto_case_count, int) or auto_case_count < 0:
        raise ValueError('import_cases_manifest.stats.case_allocation.auto_case_count must be non-negative')
    return allocation


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _knowledge_base_inclusion(value: object, kb_ids: list[str]) -> dict[str, bool]:
    if not isinstance(value, list):
        raise ValueError('knowledge_bases must be a list')
    configured: dict[str, bool] = {}
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f'knowledge_bases[{index}] must be a mapping')
        try:
            kb_id = validate_id(str(item.get('kb_id') or '').strip(), 'knowledge_bases.kb_id')
        except ValueError as exc:
            raise ValueError(f'knowledge_bases[{index}] contains an invalid reference') from exc
        included = item.get('included')
        if not isinstance(included, bool):
            raise ValueError(f'knowledge_bases[{index}].included must be a boolean')
        if kb_id in configured:
            raise ValueError('knowledge_bases must contain unique kb_id values')
        configured[kb_id] = included
    if set(configured) != set(kb_ids):
        raise ValueError('knowledge_bases must exactly match source_config.kb_ids')
    return configured


def _knowledge_base_names(source: Mapping[str, object], kb_ids: list[str]) -> dict[str, str]:
    raw = source.get('knowledge_base_names', {})
    if not isinstance(raw, Mapping):
        raise ValueError('knowledge_base_names must be a mapping')
    names = {}
    for kb_id in kb_ids:
        value = raw.get(kb_id, kb_id)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'knowledge_base_names.{kb_id} must be a non-empty string')
        names[kb_id] = value.strip()
    return names
