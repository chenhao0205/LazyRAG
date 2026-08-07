from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import re

from ... import validate_id
from .kb_client import KnowledgeBaseClient
from .models import chunk_from_docnode

DEFAULT_ALLOWED_TYPES = ('text', 'paragraph', 'table', 'formula', 'equation', 'unknown')
DEFAULT_MAX_SCAN_DOCS_PER_KB = 10_000
DEFAULT_MAX_SCAN_CHUNKS = 100_000
CHUNK_PARTITION_PATTERN = re.compile(r'^chunk_\d{4,}$')


@dataclass(frozen=True)
class BuildChunksParams:
    groups: list[str]
    allowed_types: list[str]
    max_scan_docs_per_kb: int
    max_scan_chunks: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'BuildChunksParams':
        groups = _string_list(data['groups'], 'groups') if 'groups' in data else ['block']
        for group in groups:
            validate_id(group, 'group')
        allowed_types = (
            normalized_types(data['allowed_types'])
            if 'allowed_types' in data
            else list(DEFAULT_ALLOWED_TYPES)
        )
        return cls(
            groups=groups,
            allowed_types=allowed_types,
            max_scan_docs_per_kb=_positive_int(
                data.get('max_scan_docs_per_kb', DEFAULT_MAX_SCAN_DOCS_PER_KB), 'max_scan_docs_per_kb',
            ),
            max_scan_chunks=_positive_int(
                data.get('max_scan_chunks', DEFAULT_MAX_SCAN_CHUNKS), 'max_scan_chunks',
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            'groups': list(self.groups),
            'allowed_types': list(self.allowed_types),
            'max_scan_docs_per_kb': self.max_scan_docs_per_kb,
            'max_scan_chunks': self.max_scan_chunks,
        }


def build_chunk_candidates(
    ctx: Any,
    inputs: Mapping[str, object],
    kb_client: KnowledgeBaseClient | None = None,
) -> Mapping[str, object]:
    selected = _mapping(inputs.get('selected_docs'), 'selected_docs')
    params = BuildChunksParams.from_dict(_mapping(inputs.get('build_chunks_params'), 'build_chunks_params'))
    allocation = _mapping(
        _mapping(inputs.get('import_cases_manifest'), 'import_cases_manifest').get('stats'),
        'import_cases_manifest.stats',
    )
    case_allocation = _mapping(allocation.get('case_allocation'), 'import_cases_manifest.stats.case_allocation')
    auto_case_count = _non_negative_int(case_allocation.get('auto_case_count'), 'auto_case_count')
    candidate_limit = (auto_case_count * 3 + 1) // 2
    if candidate_limit == 0:
        return {'build_chunk_candidates': {
            'chunks': [],
            'selection_stats': _empty_selection_stats(),
            'params': params.to_dict(),
        }}
    docs = _docs(selected)
    kb_ids = _string_list(selected.get('kb_ids'), 'selected_docs.kb_ids')
    docs_by_kb = {kb_id: [doc for doc in docs if str(doc.get('kb_id') or '') == kb_id] for kb_id in kb_ids}
    for kb_id, items in docs_by_kb.items():
        if len(items) > params.max_scan_docs_per_kb:
            raise ValueError(
                f'max_scan_docs_per_kb exceeded for {kb_id}: {len(items)} > {params.max_scan_docs_per_kb}'
            )

    client = kb_client or KnowledgeBaseClient()
    counts = {
        kb_id: client.count_valid_chunks(
            kb_id,
            [str(doc.get('doc_id') or '') for doc in docs_by_kb[kb_id]],
            params.groups,
            params.allowed_types,
            params.max_scan_chunks,
        )
        for kb_id in kb_ids
    }
    scanned_count = sum(_non_negative_int(item.get('scanned_count'), 'scanned_count') for item in counts.values())
    if scanned_count > params.max_scan_chunks:
        raise ValueError(f'max_scan_chunks exceeded: {scanned_count} > {params.max_scan_chunks}')

    chunks, allocation_stats = _allocate_candidates(
        client, docs_by_kb, kb_ids, counts, params, candidate_limit,
    )
    effective_count = sum(_non_negative_int(item.get('effective_count'), 'effective_count') for item in counts.values())
    selection_stats = {
        'target': {
            'candidate_limit': candidate_limit,
            'selected_count': len(chunks),
            'shortfall_count': max(candidate_limit - len(chunks), 0),
        },
        'scan': {'doc_count': len(docs), 'chunk_count': scanned_count},
        'eligibility': {
            'effective_count': effective_count,
            'filtered_count_by_type': _sum_count_maps(counts.values(), 'filtered_count_by_type'),
            'invalid_count_by_reason': _sum_count_maps(counts.values(), 'invalid_count_by_reason'),
        },
        'allocation': allocation_stats,
    }
    return {'build_chunk_candidates': {
        'chunks': chunks,
        'selection_stats': selection_stats,
        'params': params.to_dict(),
    }}


def build_chunks(
    ctx: Any,
    inputs: Mapping[str, object],
    *,
    partition_key: str | None = None,
) -> Mapping[str, object]:
    candidates = _mapping(inputs.get('build_chunk_candidates'), 'build_chunk_candidates')
    chunks = _candidate_chunks(candidates)
    params = BuildChunksParams.from_dict(_mapping(candidates.get('params'), 'build_chunk_candidates.params'))
    partition = _output_partition(ctx, partition_key)
    index = _slot_index(partition)
    payload = (
        dict(chunks[index])
        if index < len(chunks)
        else unavailable_chunk_payload(partition, params.groups[0])
    )
    return {'chunk': payload}


def build_chunks_manifest(
    ctx: Any,
    inputs: Mapping[str, object],
    *,
    partition_keys: tuple[str, ...] | None = None,
) -> Mapping[str, object]:
    selected = _mapping(inputs.get('selected_docs'), 'selected_docs')
    allocation = _case_allocation(inputs.get('import_cases_manifest'))
    candidates = _mapping(inputs.get('build_chunk_candidates'), 'build_chunk_candidates')
    params = BuildChunksParams.from_dict(_mapping(candidates.get('params'), 'build_chunk_candidates.params'))
    chunks = _chunk_tuple(inputs.get('chunk'))
    partitions = tuple(partition_keys or ())
    if len(partitions) != len(chunks):
        raise ValueError('dataset.build_chunks_manifest runtime partitions do not match chunk tuple')

    selection_stats = normalize_selection_stats(
        _mapping(candidates.get('selection_stats'), 'build_chunk_candidates.selection_stats')
    )
    fallback_used = bool(_mapping(selection_stats['allocation'], 'selection_stats.allocation').get('fallback_used'))
    warnings = build_warnings(sum(1 for chunk in chunks if chunk.get('available')), len(partitions), fallback_used)
    return {
        'build_chunks_manifest': built_chunks_payload(
            selected, chunks, partitions, allocation['auto_case_count'], warnings, params, selection_stats,
        )
    }


def _allocate_candidates(
    client: KnowledgeBaseClient,
    docs_by_kb: dict[str, list[Mapping[str, Any]]],
    kb_ids: list[str],
    counts: dict[str, Mapping[str, Any]],
    params: BuildChunksParams,
    candidate_limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    group_stats = []
    remaining = candidate_limit
    seen_chunk_ids: set[str] = set()

    for group in params.groups:
        kb_capacities = {
            kb_id: sum(_group_doc_capacities(counts[kb_id], group).values())
            for kb_id in kb_ids
        }
        group_capacity = sum(kb_capacities.values())
        group_quota = min(remaining, group_capacity)
        kb_quotas = largest_remainder(group_quota, kb_capacities, kb_ids)
        kb_stats = []
        group_selected = 0

        for kb_id in kb_ids:
            doc_order = [str(doc.get('doc_id') or '') for doc in docs_by_kb[kb_id]]
            doc_capacities = _group_doc_capacities(counts[kb_id], group)
            doc_quotas = largest_remainder(kb_quotas[kb_id], doc_capacities, doc_order)
            docs_by_id = {str(doc.get('doc_id') or ''): dict(doc) for doc in docs_by_kb[kb_id]}
            doc_stats = []
            kb_selected = 0

            for doc_id in doc_order:
                quota = doc_quotas[doc_id]
                nodes = client.fetch_valid_chunks(
                    kb_id, doc_id, group, params.allowed_types, quota, order_by='stable_chunk_id_hash',
                ) if quota else []
                nodes = sorted(nodes, key=lambda node: (getattr(node, 'number', 0), str(getattr(node, 'uid', ''))))
                for node in nodes:
                    chunk = chunk_payload(node, kb_id, doc_id, group, docs_by_id[doc_id])
                    chunk_id = chunk['chunk_id']
                    if chunk_id in seen_chunk_ids:
                        raise ValueError(f'duplicate chunk_id: {chunk_id}')
                    seen_chunk_ids.add(chunk_id)
                    chunks.append(chunk)
                selected_count = len(nodes)
                kb_selected += selected_count
                doc_stats.append({
                    'doc_id': doc_id,
                    'effective_count': doc_capacities.get(doc_id, 0),
                    'quota': quota,
                    'selected_count': selected_count,
                })

            group_selected += kb_selected
            kb_stats.append({
                'kb_id': kb_id,
                'effective_count': kb_capacities[kb_id],
                'quota': kb_quotas[kb_id],
                'selected_count': kb_selected,
                'documents': doc_stats,
            })

        group_stats.append({
            'group': group,
            'effective_count': group_capacity,
            'quota': group_quota,
            'selected_count': group_selected,
            'knowledge_bases': kb_stats,
        })
        remaining -= group_quota

    return chunks, {
        'fallback_used': any(item['selected_count'] > 0 for item in group_stats[1:]),
        'groups': group_stats,
    }


def built_chunks_payload(
    selected: Mapping[str, Any],
    chunks: tuple[Mapping[str, Any], ...],
    partitions: tuple[str, ...],
    auto_case_count: int,
    warnings: list[str],
    params: BuildChunksParams,
    selection_stats: dict[str, Any],
) -> dict[str, Any]:
    manifest_chunks = [
        {
            'available': bool(chunk.get('available')),
            'kb_id': str(chunk.get('kb_id') or ''),
            'chunk_id': str(chunk.get('chunk_id') or ''),
            'doc_id': str(chunk.get('doc_id') or ''),
            'filename': str(chunk.get('filename') or ''),
            'group': str(chunk.get('group') or ''),
            'type': str(chunk.get('type') or ''),
            'partition': partition,
        }
        for partition, chunk in zip(partitions, chunks, strict=True)
    ]
    coverage = chunk_stats(manifest_chunks)
    stats = {
        'auto_case_count': auto_case_count,
        'slots': {
            'total_count': len(manifest_chunks),
            'available_count': sum(1 for chunk in manifest_chunks if chunk['available']),
            'placeholder_count': sum(1 for chunk in manifest_chunks if not chunk['available']),
        },
        'candidate_selection': selection_stats,
        'source_coverage': coverage,
        'warnings': list(warnings),
    }
    source = {'kb_ids': list(selected.get('kb_ids') or [])}
    return {
        'source': source,
        'chunks': manifest_chunks,
        'stats': stats,
        'params': params.to_dict(),
    }


def chunk_payload(node: Any, kb_id: str, doc_id: str, group: str, doc: dict[str, Any]) -> dict[str, Any]:
    chunk = chunk_from_docnode(node, kb_id=kb_id, doc_id=doc_id, group=group, doc=doc)
    return {
        'available': True,
        'kb_id': kb_id,
        'chunk_id': chunk.chunk_id,
        'doc_id': chunk.source.doc_id,
        'filename': chunk.source.filename,
        'group': chunk.group,
        'type': normalized_type(chunk.type),
        'text': chunk.text,
        'embedding': json_value(chunk.embedding),
        'metadata': json_value(chunk.source.metadata),
    }


def unavailable_chunk_payload(partition: str, group: str) -> dict[str, Any]:
    return {
        'available': False,
        'chunk_id': f'unavailable:{partition}',
        'doc_id': '__unavailable__',
        'filename': '',
        'group': group,
        'type': 'placeholder',
        'text': 'Unavailable chunk placeholder.',
        'embedding': {'model': '', 'vector': []},
        'metadata': {'partition': partition, 'available': False},
    }


def largest_remainder(total: int, capacities: dict[str, int], order: list[str]) -> dict[str, int]:
    capacities = {key: max(int(capacities.get(key, 0)), 0) for key in order}
    quota = min(max(total, 0), sum(capacities.values()))
    if quota == 0:
        return dict.fromkeys(order, 0)

    capacity = sum(capacities.values())
    raw = {key: quota * capacities[key] / capacity for key in order}
    quotas = {key: int(raw[key]) for key in order}
    remaining = quota - sum(quotas.values())
    ranked = sorted(order, key=lambda key: (-(raw[key] - quotas[key]), order.index(key)))
    for key in ranked:
        if remaining == 0:
            break
        if quotas[key] < capacities[key]:
            quotas[key] += 1
            remaining -= 1
    return quotas


def _group_doc_capacities(result: Mapping[str, Any], group: str) -> dict[str, int]:
    capacities = _mapping(result.get('capacities'), 'count_valid_chunks.capacities')
    values = _mapping(capacities.get(group, {}), f'count_valid_chunks.capacities.{group}')
    return {str(doc_id): _non_negative_int(count, 'effective_count') for doc_id, count in values.items()}


def _sum_count_maps(results: Any, key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for result in results:
        values = _mapping(result.get(key), key)
        counts.update({str(name): _non_negative_int(count, key) for name, count in values.items()})
    return dict(counts)


def _empty_selection_stats() -> dict[str, Any]:
    return {
        'target': {'candidate_limit': 0, 'selected_count': 0, 'shortfall_count': 0},
        'scan': {'doc_count': 0, 'chunk_count': 0},
        'eligibility': {
            'effective_count': 0,
            'filtered_count_by_type': {},
            'invalid_count_by_reason': {},
        },
        'allocation': {'fallback_used': False, 'groups': []},
    }


def build_warnings(chunk_count: int, target: int, fallback_used: bool) -> list[str]:
    warnings = []
    if chunk_count < target:
        warnings.append(f'chunk build produced {chunk_count} chunks, below target {target}; continuing')
    if fallback_used:
        warnings.append('fallback group sampling was used')
    return warnings


def chunk_stats(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    available_chunks = [chunk for chunk in chunks if chunk.get('available')]
    group_counts = Counter(str(chunk.get('group') or '') for chunk in available_chunks)
    doc_groups: dict[str, Counter] = {}
    filenames: dict[str, str] = {}
    for chunk in available_chunks:
        doc_id = str(chunk.get('doc_id') or '')
        doc_groups.setdefault(doc_id, Counter())[str(chunk.get('group') or '')] += 1
        filenames.setdefault(doc_id, str(chunk.get('filename') or ''))
    return {
        'doc_count': len(doc_groups),
        'group_counts': dict(group_counts),
        'documents': [
            {'doc_id': doc_id, 'filename': filenames.get(doc_id, ''),
             'total_count': sum(groups.values()), 'group_counts': dict(groups)}
            for doc_id, groups in sorted(doc_groups.items())
        ],
    }


def normalize_selection_stats(value: Mapping[str, Any]) -> dict[str, Any]:
    for key in ('target', 'scan', 'eligibility', 'allocation'):
        _mapping(value.get(key), f'selection_stats.{key}')
    return json_value(value)


def json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def normalized_types(value: Any) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ValueError('allowed_types must be a non-empty list of strings')
    normalized = [normalized_type(item) for item in value]
    if any(not item for item in normalized):
        raise ValueError('allowed_types must not contain empty values')
    return list(dict.fromkeys(normalized))


def normalized_type(value: Any) -> str:
    normalized = str(value or '').strip().lower()
    return 'unknown' if normalized in {'', 'unknown'} else normalized


def _docs(selected: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    docs = selected.get('docs')
    if not isinstance(docs, list) or not docs:
        raise ValueError('selected_docs.docs must be a non-empty list')
    return [doc for doc in docs if isinstance(doc, Mapping)]


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _output_partition(ctx: Any, partition_key: str | None) -> str:
    partition = partition_key or getattr(ctx, 'partition_key', '') or 'chunk_0001'
    _validate_chunk_partition(partition)
    return partition


def _slot_index(partition: str) -> int:
    return int(partition.rsplit('_', 1)[-1]) - 1


def _chunk_tuple(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, tuple):
        raise ValueError('chunk input must be a partitioned tuple')
    return tuple(_mapping(item, 'chunk[]') for item in value)


def _candidate_chunks(value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    chunks = value.get('chunks')
    if not isinstance(chunks, list):
        raise ValueError('build_chunk_candidates.chunks must be a list')
    return tuple(_mapping(chunk, 'build_chunk_candidates.chunks[]') for chunk in chunks)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must be a non-empty list')
    values = [str(item or '').strip() for item in value]
    if any(not item for item in values) or len(set(values)) != len(values):
        raise ValueError(f'{name} must contain unique non-empty strings')
    return values


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def _case_allocation(value: object) -> Mapping[str, Any]:
    manifest = _mapping(value, 'import_cases_manifest')
    stats = _mapping(manifest.get('stats'), 'import_cases_manifest.stats')
    allocation = _mapping(stats.get('case_allocation'), 'import_cases_manifest.stats.case_allocation')
    _non_negative_int(allocation.get('auto_case_count'), 'auto_case_count')
    return allocation


def _validate_chunk_partition(partition: str) -> None:
    if not CHUNK_PARTITION_PATTERN.fullmatch(partition):
        raise ValueError(f'invalid chunk partition: {partition!r}')
