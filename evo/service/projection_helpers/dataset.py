from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping
from typing import Any

from evo.artifact_runtime import ArtifactKey, ArtifactRef
from evo.operations.dataset.qaplan import LANES, _lane_counts

from ..contracts import ServiceError


def _topic_filters(question_type: str, min_chunk_count: int | None,
                   max_chunk_count: int | None) -> dict[str, object]:
    if question_type not in ('', 'precision', 'reasoning'):
        raise ServiceError(400, 'question_type must be precision or reasoning')
    for name, value in (('min_chunk_count', min_chunk_count), ('max_chunk_count', max_chunk_count)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ServiceError(400, f'{name} must be a non-negative integer')
    if min_chunk_count is not None and max_chunk_count is not None and min_chunk_count > max_chunk_count:
        raise ServiceError(400, 'min_chunk_count must not exceed max_chunk_count')
    return {
        **({'question_type': question_type} if question_type else {}),
        **({'min_chunk_count': min_chunk_count} if min_chunk_count is not None else {}),
        **({'max_chunk_count': max_chunk_count} if max_chunk_count is not None else {}),
    }


_TOPIC_OPTION_MIN_CHUNKS = {'easy': 1, 'medium': 2, 'hard': 3}


def _project_capability_directory(capabilities: Mapping[str, Any], source_ids: list[str], enabled: Mapping[str, bool],
                                  field: str, selected: object, *, priority: bool) -> list[dict[str, object]]:
    configured = list(selected) if isinstance(selected, (list, tuple)) else []
    active_sources = [kb_id for kb_id in source_ids if enabled.get(kb_id) is True]
    rows: dict[str, str] = {}
    for kb_id in source_ids:
        capability = capabilities.get(kb_id, {})
        values = capability.get(field, ()) if isinstance(capability, Mapping) else ()
        if not isinstance(values, (list, tuple)):
            raise ServiceError(503, 'parser capabilities are invalid')
        for item in values:
            if not isinstance(item, Mapping) or not isinstance(item.get('id'), str) or not item.get('id'):
                raise ServiceError(503, 'parser capabilities are invalid')
            identifier = item['id']
            rows.setdefault(identifier, str(item.get('name') or identifier))
    result = []
    for identifier in sorted(rows):
        supported = bool(active_sources) and all(
            any(isinstance(item, Mapping) and item.get('id') == identifier
                for item in (capabilities.get(kb_id, {}) or {}).get(field, ()))
            for kb_id in active_sources
        )
        enabled_now = identifier in configured
        row: dict[str, object] = {
            'id': identifier, 'name': rows[identifier], 'supported': supported, 'enabled': enabled_now,
        }
        if priority:
            row['priority'] = configured.index(identifier) + 1 if enabled_now else None
        result.append(row)
    return result


def _mapping_value(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ServiceError(503, f'{name} projection is invalid')
    return value


def _source_knowledge_base_ids(source: Mapping[str, object]) -> list[str]:
    raw_ids = source.get('kb_ids', source.get('kb_id', ()))
    ids = [item for item in raw_ids if isinstance(item, str)] if isinstance(raw_ids, list) else []
    raw_csv = source.get('csv_sources', source.get('csv_data', ()))
    if isinstance(raw_csv, list):
        for item in raw_csv:
            if isinstance(item, Mapping):
                csv_id = item.get('kb_id') if 'kb_id' in item else next(iter(item), None) if len(item) == 1 else None
                if isinstance(csv_id, str) and csv_id not in ids:
                    ids.append(csv_id)
    return ids


def _required_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ServiceError(400, f'{name} must be a non-empty string')
    return value


def _document_detail_filters(knowledge_base_id: str, document_id: str, selected: bool | None,
                             split_rule: str) -> dict[str, object]:
    if selected is not None and not isinstance(selected, bool):
        raise ServiceError(400, 'selected must be a boolean')
    if not isinstance(split_rule, str) or (split_rule and not split_rule.strip()):
        raise ServiceError(400, 'split_rule must not be blank')
    return {
        'knowledge_base_id': _required_id(knowledge_base_id, 'knowledge_base_id'),
        'document_id': _required_id(document_id, 'document_id'),
        **({'selected': selected} if selected is not None else {}),
        **({'split_rule': split_rule} if split_rule else {}),
    }


def _document_candidate_chunks(value: object, knowledge_base_id: str, document_id: str) -> list[Mapping[str, object]]:
    source = value if isinstance(value, Mapping) else {}
    chunks = source.get('chunks', ()) if isinstance(source, Mapping) else ()
    return [
        item for item in chunks
        if isinstance(item, Mapping)
        and item.get('kb_id') == knowledge_base_id
        and item.get('doc_id') == document_id
    ]


def _document_quotas(value: object, chunks: list[Mapping[str, object]], knowledge_base_id: str,
                     document_id: str) -> list[dict[str, object]]:
    source = value if isinstance(value, Mapping) else {}
    quotas = source.get('quotas', ()) if isinstance(source, Mapping) else ()
    selected_by_group: dict[str, int] = {}
    for chunk in chunks:
        group = str(chunk.get('group') or '')
        selected_by_group[group] = selected_by_group.get(group, 0) + int(chunk.get('selected') is True)
    rows = [
        {
            'split_rule': str(quota.get('group') or ''),
            'required': quota.get('required'),
            'selected': selected_by_group.get(str(quota.get('group') or ''), 0),
        }
        for quota in quotas
        if isinstance(quota, Mapping)
        and quota.get('kb_id') == knowledge_base_id
        and quota.get('doc_id') == document_id
    ]
    return sorted(rows, key=lambda item: str(item['split_rule']))


def _document_chunk_matches(chunk: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    return (
        ('selected' not in filters or chunk.get('selected') is filters['selected'])
        and ('split_rule' not in filters or chunk.get('group') == filters['split_rule'])
    )


def _document_chunk_dto(chunk: Mapping[str, object]) -> dict[str, object]:
    discovery_index = chunk.get('discovery_index')
    return {
        'chunk_id': str(chunk.get('chunk_id') or ''),
        'split_rule': str(chunk.get('group') or ''),
        'layout_type': str(chunk.get('type') or ''),
        'text': str(chunk.get('text') or ''),
        'selected': chunk.get('selected') is True,
        '_discovery_index': discovery_index if isinstance(discovery_index, int) else 0,
    }


def _topic_chunk_ids(topic: Mapping[str, object]) -> tuple[str, ...]:
    raw = topic.get('chunk_ids')
    if not isinstance(raw, list) or not raw:
        raise ServiceError(503, 'topic chunk_ids are invalid')
    values = tuple(_required_id(value, 'topic.chunk_id') for value in raw)
    if len(set(values)) != len(values):
        raise ServiceError(503, 'topic chunk_ids are duplicated')
    return values


def _document_names(value: object) -> dict[tuple[str, str], tuple[str, str]]:
    source = value if isinstance(value, Mapping) else {}
    documents = source.get('documents', ()) if isinstance(source, Mapping) else ()
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for document in documents:
        if not isinstance(document, Mapping):
            continue
        kb_id, doc_id = document.get('kb_id'), document.get('doc_id')
        if isinstance(kb_id, str) and kb_id and isinstance(doc_id, str) and doc_id:
            result[kb_id, doc_id] = (
                str(document.get('knowledge_base_name') or kb_id),
                str(document.get('filename') or doc_id),
            )
    return result


def _topic_chunk_dto(chunk: object, document_names: Mapping[tuple[str, str], tuple[str, str]]) -> dict[str, object]:
    if not isinstance(chunk, Mapping):
        raise ServiceError(503, 'topic chunk artifact is invalid')
    kb_id, doc_id = chunk.get('kb_id'), chunk.get('doc_id')
    if not isinstance(kb_id, str) or not kb_id or not isinstance(doc_id, str) or not doc_id:
        raise ServiceError(503, 'topic chunk source is invalid')
    names = document_names.get((kb_id, doc_id))
    if names is None:
        raise ServiceError(503, 'topic chunk source document is unavailable')
    return {
        'chunk_id': str(chunk.get('chunk_id') or ''),
        'knowledge_base': {'id': kb_id, 'name': names[0]},
        'document': {'id': doc_id, 'name': names[1]},
        'split_rule': str(chunk.get('group') or ''),
        'layout_type': str(chunk.get('type') or ''),
        'text': str(chunk.get('text') or ''),
    }


def _topic_matches(topic: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    chunk_count = topic.get('chunk_count')
    if not isinstance(chunk_count, int) or isinstance(chunk_count, bool):
        return False
    return (
        (not filters.get('question_type') or topic.get('question_type') == filters['question_type'])
        and ('min_chunk_count' not in filters or chunk_count >= filters['min_chunk_count'])
        and ('max_chunk_count' not in filters or chunk_count <= filters['max_chunk_count'])
    )


def _document_filters(included: bool | None, knowledge_base_id: str) -> dict[str, object]:
    if included is not None and not isinstance(included, bool):
        raise ServiceError(400, 'included must be a boolean')
    if not isinstance(knowledge_base_id, str) or (knowledge_base_id and not knowledge_base_id.strip()):
        raise ServiceError(400, 'knowledge_base_id must not be blank')
    return {
        **({'included': included} if included is not None else {}),
        **({'knowledge_base_id': knowledge_base_id} if knowledge_base_id else {}),
    }


def _document_matches(document: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    return (
        ('included' not in filters or document.get('included') is filters['included'])
        and ('knowledge_base_id' not in filters or document.get('kb_id') == filters['knowledge_base_id'])
    )


def _document_chunk_counts(value: object) -> dict[tuple[str, str], dict[str, int]]:
    source = value if isinstance(value, Mapping) else {}
    chunks = source.get('chunks', ()) if isinstance(source, Mapping) else ()
    result: dict[tuple[str, str], dict[str, int]] = {}
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        kb_id, doc_id = chunk.get('kb_id'), chunk.get('doc_id')
        if not isinstance(kb_id, str) or not isinstance(doc_id, str):
            continue
        counts = result.setdefault((kb_id, doc_id), {'effective': 0, 'selected': 0})
        counts['effective'] += 1
        counts['selected'] += int(chunk.get('selected') is True)
    return result


def _document_dto(document: Mapping[str, object], counts: Mapping[tuple[str, str], Mapping[str, int]], *,
                  has_candidates: bool) -> dict[str, object]:
    kb_id = str(document.get('kb_id') or '')
    doc_id = str(document.get('doc_id') or '')
    included = document.get('included') is True
    count = counts.get((kb_id, doc_id))
    chunks: dict[str, object] | None = None
    if included and has_candidates:
        effective = int((count or {}).get('effective', 0))
        selected = int((count or {}).get('selected', 0))
        chunks = {
            'effective': effective,
            'selected': selected,
            'selection_rate': selected / effective if effective else None,
        }
    discovery_index = document.get('discovery_index')
    return {
        'document_id': doc_id,
        'name': str(document.get('filename') or doc_id),
        'included': included,
        'knowledge_base': {'id': kb_id, 'name': str(document.get('knowledge_base_name') or kb_id)},
        'chunks': chunks,
        '_discovery_index': discovery_index if isinstance(discovery_index, int) else 0,
    }


def _public_artifact_ref(record: object) -> ArtifactRef:
    if not isinstance(record, Mapping):
        raise ServiceError(503, 'artifact record projection is invalid')
    ref = record.get('ref')
    if not isinstance(ref, Mapping) or not isinstance(ref.get('key'), Mapping):
        raise ServiceError(503, 'artifact record projection is invalid')
    key = ref['key']
    try:
        return ArtifactRef(
            ArtifactKey(key['artifact_id'], key.get('partition_key', '')),
            ref['version'],
        )
    except (KeyError, TypeError, ValueError):
        raise ServiceError(503, 'artifact record projection is invalid') from None


_DATASET_RESULT_FIELDS = (
    'case_id', 'question', 'question_type', 'difficulty', 'ground_truth', 'grading_guidance',
    'key_points', 'forbidden_claims', 'reference_context', 'reference_doc', 'reference_doc_ids',
    'reference_chunk_ids', 'generate_reason', 'is_deleted',
)


def _dataset_result_case(value: Mapping[str, Any]) -> dict[str, Any]:
    preparation = value.get('source_preparation') if isinstance(value.get('source_preparation'), Mapping) else {}
    enhancement = preparation.get('dataset_enhancement') \
        if isinstance(preparation.get('dataset_enhancement'), Mapping) else {}
    return {
        'case_id': str(value.get('case_id') or value.get('id') or ''),
        'question': str(value.get('question') or ''),
        'question_type': str(value.get('question_type') or ''),
        'difficulty': str(value.get('difficulty') or ''),
        'ground_truth': value.get('ground_truth', value.get('answer', '')),
        'grading_guidance': str(value.get('grading_guidance') or ''),
        'key_points': list(value.get('key_points') or enhancement.get('key_points') or []),
        'forbidden_claims': list(value.get('forbidden_claims') or enhancement.get('forbidden_claims') or []),
        'reference_context': value.get('reference_context') or [],
        'reference_doc': list(value.get('reference_doc') or []),
        'reference_doc_ids': list(value.get('reference_doc_ids') or []),
        'reference_chunk_ids': list(value.get('reference_chunk_ids') or []),
        'generate_reason': str(value.get('generate_reason') or ''),
        'is_deleted': bool(value.get('is_deleted', False)),
    }


def _dataset_result_csv(rows: list[dict[str, Any]]) -> bytes:
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=_DATASET_RESULT_FIELDS)
    writer.writeheader()
    for row in rows:
        serialized = dict(row)
        for field in ('key_points', 'forbidden_claims'):
            serialized[field] = json.dumps(serialized[field], ensure_ascii=False, separators=(',', ':'))
        context = serialized['reference_context']
        serialized['reference_context'] = (
            json.dumps(context, ensure_ascii=False, separators=(',', ':'))
            if isinstance(context, (list, dict)) else str(context or '')
        )
        for field in ('reference_doc', 'reference_doc_ids', 'reference_chunk_ids'):
            serialized[field] = ','.join(str(item) for item in serialized[field])
        serialized['is_deleted'] = 'true' if serialized['is_deleted'] else 'false'
        writer.writerow(serialized)
    return ('\ufeff' + output.getvalue()).encode('utf-8')


def _overview_stage_status(snapshot: Mapping[str, Any]) -> str:
    value = snapshot.get('snapshot')
    progress = value.get('progress') if isinstance(value, Mapping) else None
    status = progress.get('status') if isinstance(progress, Mapping) else None
    if not isinstance(status, str):
        raise ServiceError(409, 'stage snapshot has no status')
    return status


def _overview_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ServiceError(409, f'{name} is invalid')
    return value


def _material_target_minimum(source: Mapping[str, object]) -> int:
    if source.get('supplement_existing_eval_set') is not True:
        return 1
    imported_cases = source.get('imported_cases', ())
    if not isinstance(imported_cases, list):
        return 1
    return max(1, sum(
        isinstance(case, Mapping) and case.get('is_deleted') is not True
        for case in imported_cases
    ))


def _overview_count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ServiceError(409, f'{name} must be a non-negative integer')
    return value


def _overview_warnings(value: object) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ServiceError(409, 'materials overview.warnings is invalid')
    return list(value)


async def _optional_overview_artifact(service: Any, thread_id: str,
                                      artifact_id: str) -> dict[str, Any] | None:
    try:
        return await service.artifact(thread_id, artifact_id)
    except ServiceError as error:
        if error.status_code != 404:
            raise
        return None


def _empty_cases_overview(thread_id: str, status: str, *, revision: str | None,
                          execution_revision: str) -> dict[str, Any]:
    return {
        'thread_id': thread_id,
        'revision': revision,
        'execution_revision': execution_revision,
        'status': status,
        'stages': _pending_case_stage_overview(),
        'automatic_plan': None,
    }


def _pending_case_stage_overview(total: int | None = None) -> dict[str, Any]:
    return {
        name: {
            'status': 'pending', 'completed': 0 if total is not None else None, 'total': total,
            'status_counts': None if total is None else {
                'pending': total, 'running': 0, 'completed': 0, 'failed': 0, 'canceled': 0,
            },
        }
        for name in ('plan', 'generate', 'grading')
    }


def _partition_count(value: object | None) -> int | None:
    if value is None:
        return None
    return len(_case_ids_from_partition_set(value))


def _topic_artifact_stats(value: object | None) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    stats = value.get('stats')
    return stats if isinstance(stats, Mapping) else {}


def _topic_stats_count(stats: Mapping[str, Any], key: str) -> int:
    value = stats.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _topic_entity_failed(value: object | None) -> int:
    return _topic_stats_count(_topic_artifact_stats(value), 'placeholder_count')


def _topic_semantic_failed(value: object | None) -> int:
    stats = _topic_artifact_stats(value)
    candidate = _topic_stats_count(stats, 'candidate_count')
    labeled = _topic_stats_count(stats, 'labeled_cluster_count')
    return max(0, candidate - labeled)


def _topic_stage_entry(total: int | None, failed: int, *, settled: bool) -> dict[str, object]:
    failed_count = max(0, failed)
    if total is not None:
        failed_count = min(failed_count, total)
    if not settled:
        return {'status': 'pending', 'completed': 0, 'total': total, 'failed': 0}
    return {
        'status': 'completed',
        'completed': max(0, (total or 0) - failed_count),
        'total': total,
        'failed': failed_count,
    }


def _topic_execution_stages(status: str, entity_total: int | None, semantic_total: int | None,
                            topic_total: int | None, *, entity_failed: int = 0,
                            semantic_failed: int = 0, topic_failed: int = 0) -> dict[str, dict[str, object]]:
    # Published topic_manifest is the settle signal. Stage may still be paused
    # at the continue gate; rings must not stay at 0/N pending. completed is
    # successes only so partition failures remain visible after the flow finishes.
    settled = topic_total is not None or status in {'completed', 'succeeded'}
    return {
        'entities': _topic_stage_entry(entity_total, entity_failed, settled=settled),
        'semantic': _topic_stage_entry(semantic_total, semantic_failed, settled=settled),
        'topics': _topic_stage_entry(topic_total, topic_failed, settled=settled),
    }


def _planned_case_count(value: object | None) -> int | None:
    if value is None:
        return None
    allocation = _overview_mapping(
        _overview_mapping(_overview_mapping(value, 'case import manifest').get('stats'), 'case import manifest.stats').get('case_allocation'),
        'case import manifest.case_allocation',
    )
    target = allocation.get('target_case_count')
    return target if isinstance(target, int) and not isinstance(target, bool) and target >= 0 else None


def _import_case_plan(value: object) -> dict[str, int]:
    allocation = _overview_mapping(
        _overview_mapping(_overview_mapping(value, 'case import manifest').get('stats'), 'case import manifest.stats').get('case_allocation'),
        'case import manifest.case_allocation',
    )
    return {
        'target': _overview_count(allocation.get('target_case_count'), 'case import target'),
        'imported': _overview_count(allocation.get('import_case_count'), 'case import imported'),
        'automatic': _overview_count(allocation.get('auto_case_count'), 'case import automatic'),
    }


def _pending_case_statuses(case_ids: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {case_id: {
        'dataset.qaplan_spec': 'pending', 'dataset.generate_case': 'pending',
        'dataset.enhance_case': 'pending',
    } for case_id in case_ids}


def _initial_case_statuses(import_manifest_value: object) -> dict[str, dict[str, str]]:
    imported = _overview_mapping(import_manifest_value, 'case import manifest')
    allocation = _overview_mapping(
        _overview_mapping(imported.get('stats'), 'case import manifest.stats').get('case_allocation'),
        'case import manifest.case_allocation',
    )
    assignments = _overview_mapping(allocation.get('assignments'), 'case import manifest.assignments')
    statuses = _pending_case_statuses(tuple(assignments))
    for case_id, raw_assignment in assignments.items():
        assignment = _overview_mapping(raw_assignment, f'case assignment {case_id}')
        if assignment.get('mode') == 'imported':
            statuses[case_id] = {
                'dataset.qaplan_spec': 'completed',
                'dataset.generate_case': 'completed',
                'dataset.enhance_case': 'completed',
            }
    return statuses


def _with_imported_completed_placeholders(
    statuses_by_case: Mapping[str, Mapping[str, str]],
    import_manifest_value: object | None,
) -> dict[str, dict[str, str]]:
    """Keep imported cases completed until runtime actually starts that operation."""
    if import_manifest_value is None:
        return {case_id: dict(statuses) for case_id, statuses in statuses_by_case.items()}
    initial = _initial_case_statuses(import_manifest_value)
    merged: dict[str, dict[str, str]] = {}
    for case_id, statuses in statuses_by_case.items():
        imported = initial.get(case_id)
        if imported is None:
            merged[case_id] = dict(statuses)
            continue
        merged[case_id] = {
            operation_id: (
                imported[operation_id]
                if status == 'pending' and imported.get(operation_id) == 'completed'
                else status
            )
            for operation_id, status in statuses.items()
        }
    return merged


def _planned_case_rows(import_manifest_value: object, params_value: object) -> list[dict[str, Any]]:
    imported = _overview_mapping(import_manifest_value, 'case import manifest')
    allocation = _overview_mapping(
        _overview_mapping(imported.get('stats'), 'case import manifest.stats').get('case_allocation'),
        'case import manifest.case_allocation',
    )
    assignments = _overview_mapping(allocation.get('assignments'), 'case import manifest.assignments')
    details = imported.get('details')
    if not isinstance(details, list):
        raise ServiceError(503, 'case import manifest.details is invalid')
    imported_cases = {item.get('source_row_number'): item.get('case') for item in details if isinstance(item, Mapping)}
    generated_ids = [case_id for case_id, assignment in assignments.items()
                     if isinstance(assignment, Mapping) and assignment.get('mode') == 'generated']
    lane_counts = _lane_counts(params_value, len(generated_ids))
    generated_meta: dict[str, tuple[str, str]] = {}
    index = 0
    for lane, question_type, difficulty in LANES:
        for _ in range(lane_counts[lane]):
            generated_meta[generated_ids[index]] = (question_type, difficulty)
            index += 1
    statuses = _initial_case_statuses(import_manifest_value)
    rows = []
    for case_id, raw_assignment in assignments.items():
        assignment = _overview_mapping(raw_assignment, f'case assignment {case_id}')
        mode = assignment.get('mode')
        if mode == 'imported':
            case = _overview_mapping(imported_cases.get(assignment.get('source_row_number')), f'imported case {case_id}')
            question_type = _case_choice(case.get('question_type'), {'precision', 'reasoning'}, 'question_type')
            difficulty = _case_optional_choice(case.get('difficulty'), {'easy', 'medium', 'hard'}, 'difficulty')
        elif mode == 'generated' and case_id in generated_meta:
            question_type, difficulty = generated_meta[case_id]
        else:
            raise ServiceError(503, f'case assignment mode is invalid: {case_id}')
        rows.append({'case_id': case_id, 'stages': {'plan': statuses[case_id]['dataset.qaplan_spec'],
                     'generate': statuses[case_id]['dataset.generate_case'], 'grading': statuses[case_id]['dataset.enhance_case']},
                     'source': mode, 'question_type': question_type, 'difficulty': difficulty, 'topic': None})
    return rows


def _case_ids_from_partition_set(value: object) -> tuple[str, ...]:
    data = _overview_mapping(value, 'case partition set')
    raw = data.get('keys')
    if not isinstance(raw, list) or not all(isinstance(case_id, str) and case_id for case_id in raw):
        raise ServiceError(409, 'case partition set.keys is invalid')
    if len(set(raw)) != len(raw):
        raise ServiceError(409, 'case partition set.keys is invalid')
    return tuple(raw)


def _natural_case_id_key(case_id: str) -> tuple[object, ...]:
    """Sort case_0010 before case_0011 (and before case_0009 when reversed in storage)."""
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', case_id) if part)


def _ordered_case_rows(rows: list[dict[str, Any]], import_manifest_value: object) -> list[dict[str, Any]]:
    """Order imported cases by natural case_id, then generated cases the same way."""
    imported = _overview_mapping(import_manifest_value, 'case import manifest')
    allocation = _overview_mapping(
        _overview_mapping(imported.get('stats'), 'case import manifest.stats').get('case_allocation'),
        'case import manifest.case_allocation',
    )
    assignments = _overview_mapping(allocation.get('assignments'), 'case import manifest.assignments')

    def sort_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, tuple[object, ...], int]:
        index, row = item
        case_id = str(row.get('case_id') or '')
        raw_assignment = assignments.get(case_id)
        group = (
            0
            if isinstance(raw_assignment, Mapping) and raw_assignment.get('mode') == 'imported'
            else 1
        )
        return (group, _natural_case_id_key(case_id), index)

    return [row for _, row in sorted(enumerate(rows), key=sort_key)]


def _case_operation_summary(statuses: list[str]) -> dict[str, Any]:
    counts = {name: statuses.count(name) for name in ('pending', 'running', 'completed', 'failed', 'canceled')}
    if counts['failed']:
        status = 'failed'
    elif counts['running']:
        status = 'running'
    elif statuses and counts['completed'] == len(statuses):
        status = 'completed'
    elif statuses and counts['canceled'] == len(statuses):
        status = 'canceled'
    else:
        status = 'pending'
    return {
        'status': status,
        'completed': counts['completed'],
        'total': len(statuses),
        'status_counts': counts,
    }


def _case_filters(plan_status: str, generate_status: str, grading_status: str, source: str,
                  question_type: str, difficulty: str) -> dict[str, object]:
    values = {
        'plan_status': (plan_status, {'pending', 'running', 'completed', 'failed', 'canceled'}),
        'generate_status': (generate_status, {'pending', 'running', 'completed', 'failed', 'canceled'}),
        'grading_status': (grading_status, {'pending', 'running', 'completed', 'failed', 'canceled'}),
        'source': (source, {'imported', 'generated'}),
        'question_type': (question_type, {'precision', 'reasoning'}),
        'difficulty': (difficulty, {'easy', 'medium', 'hard'}),
    }
    result: dict[str, object] = {}
    for name, (value, allowed) in values.items():
        if not isinstance(value, str):
            raise ServiceError(400, f'{name} must be a string')
        if not value:
            continue
        if value not in allowed:
            raise ServiceError(400, f'{name} is invalid')
        result[name] = value
    return result


def _case_rows(case_ids: tuple[str, ...], import_manifest_value: object, plan_value: object,
               topic_manifest_value: object, statuses: Mapping[str, Mapping[str, str]],
               specifications: Mapping[str, object] | None = None) -> list[dict[str, Any]]:
    imported = _overview_mapping(import_manifest_value, 'case import manifest')
    allocation = _overview_mapping(
        _overview_mapping(imported.get('stats'), 'case import manifest.stats').get('case_allocation'),
        'case import manifest.case_allocation',
    )
    assignments = _overview_mapping(allocation.get('assignments'), 'case import manifest.assignments')
    details = imported.get('details')
    if not isinstance(details, list):
        raise ServiceError(503, 'case import manifest.details is invalid')
    imported_cases = {
        detail.get('source_row_number'): detail.get('case')
        for detail in details if isinstance(detail, Mapping)
    }
    plan = _overview_mapping(plan_value, 'qaplan plan')
    raw_plan_items = plan.get('items')
    if not isinstance(raw_plan_items, list):
        raise ServiceError(503, 'qaplan plan.items is invalid')
    plan_items = {
        item.get('case_id'): item
        for item in raw_plan_items if isinstance(item, Mapping)
    }
    topics = _overview_mapping(topic_manifest_value, 'topic manifest').get('topics')
    if not isinstance(topics, list):
        raise ServiceError(503, 'topic manifest.topics is invalid')
    topics_by_id = {
        item.get('topic_id'): item
        for item in topics if isinstance(item, Mapping)
    }
    rows = []
    for case_id in case_ids:
        assignment = _overview_mapping(assignments.get(case_id), f'case assignment {case_id}')
        mode = assignment.get('mode')
        case_statuses = statuses.get(case_id)
        if case_statuses is None:
            raise ServiceError(409, f'case snapshot is missing: {case_id}')
        if mode == 'imported':
            source_row = assignment.get('source_row_number')
            case = _overview_mapping(imported_cases.get(source_row), f'imported case {case_id}')
            question_type = _case_choice(case.get('question_type'), {'precision', 'reasoning'}, 'question_type')
            difficulty = _case_optional_choice(case.get('difficulty'), {'easy', 'medium', 'hard'}, 'difficulty')
            topic = None
        elif mode == 'generated':
            plan_item = _overview_mapping(plan_items.get(case_id), f'qaplan plan item {case_id}')
            question_type = _case_choice(plan_item.get('question_type'), {'precision', 'reasoning'}, 'question_type')
            difficulty = _case_choice(plan_item.get('difficulty'), {'easy', 'medium', 'hard'}, 'difficulty')
            specification = None if specifications is None else specifications.get(case_id)
            spec_topic = (
                _overview_mapping(specification, f'qaplan spec {case_id}').get('topic')
                if isinstance(specification, Mapping) else None
            )
            topic_id = (
                _overview_mapping(spec_topic, f'qaplan spec {case_id}.topic').get('topic_id')
                if isinstance(spec_topic, Mapping) else plan_item.get('topic_id')
            )
            topic_value = _overview_mapping(topics_by_id.get(topic_id), f'topic {topic_id}')
            topic = {'topic_id': topic_id, 'name': _required_id(topic_value.get('name'), 'topic.name')}
        else:
            raise ServiceError(503, f'case assignment mode is invalid: {case_id}')
        rows.append({
            'case_id': case_id,
            'stages': {
                'plan': case_statuses['dataset.qaplan_spec'],
                'generate': case_statuses['dataset.generate_case'],
                'grading': case_statuses['dataset.enhance_case'],
            },
            'source': mode,
            'question_type': question_type,
            'difficulty': difficulty,
            'topic': topic,
        })
    return rows


def _case_choice(value: object, allowed: set[str], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ServiceError(503, f'case {name} is invalid')
    return value


def _case_optional_choice(value: object, allowed: set[str], name: str) -> str | None:
    if value in (None, ''):
        return None
    return _case_choice(value, allowed, name)


def _case_matches(row: Mapping[str, object], filters: Mapping[str, object]) -> bool:
    stages = row.get('stages')
    if not isinstance(stages, Mapping):
        return False
    return (
        (not filters.get('plan_status') or stages.get('plan') == filters['plan_status'])
        and (not filters.get('generate_status') or stages.get('generate') == filters['generate_status'])
        and (not filters.get('grading_status') or stages.get('grading') == filters['grading_status'])
        and (not filters.get('source') or row.get('source') == filters['source'])
        and (not filters.get('question_type') or row.get('question_type') == filters['question_type'])
        and (not filters.get('difficulty') or row.get('difficulty') == filters['difficulty'])
    )


def _case_detail_topic(value: object, topic_manifest_value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    topic = _overview_mapping(value, 'case topic')
    topic_id = _required_id(topic.get('topic_id'), 'topic.topic_id')
    manifest = _overview_mapping(topic_manifest_value, 'topic manifest')
    topics = manifest.get('topics')
    if not isinstance(topics, list):
        raise ServiceError(503, 'topic manifest.topics is invalid')
    source = next(
        (item for item in topics if isinstance(item, Mapping) and item.get('topic_id') == topic_id),
        None,
    )
    source = _overview_mapping(source, f'topic {topic_id}')
    return {
        'topic_id': topic_id,
        'name': _required_id(source.get('name'), 'topic.name'),
        'chunk_count': _overview_count(source.get('chunk_count'), 'topic.chunk_count'),
    }


def _detail_case_references(spec: dict[str, Any] | None, draft: dict[str, Any] | None,
                            source: object) -> list[Mapping[str, object]]:
    if draft is not None:
        value = _overview_mapping(draft['value'], 'case draft').get('references')
    elif spec is not None:
        spec_value = _overview_mapping(spec['value'], 'qaplan spec')
        if source == 'imported':
            value = _overview_mapping(spec_value.get('imported_case'), 'imported qaplan spec').get('references')
        else:
            value = spec_value.get('references')
    else:
        value = []
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ServiceError(503, 'case references are invalid')
    return list(value)


def _selected_document_names(value: object) -> dict[tuple[str, str], tuple[str, str]]:
    documents = _overview_mapping(value, 'selected docs').get('documents')
    if not isinstance(documents, list):
        raise ServiceError(503, 'selected docs.documents is invalid')
    result: dict[tuple[str, str], tuple[str, str]] = {}
    for document in documents:
        item = _overview_mapping(document, 'selected docs.document')
        kb_id = _required_id(item.get('kb_id'), 'document.kb_id')
        doc_id = _required_id(item.get('doc_id'), 'document.doc_id')
        result[(kb_id, doc_id)] = (
            _required_id(item.get('knowledge_base_name'), 'document.knowledge_base_name'),
            _required_id(item.get('filename'), 'document.filename'),
        )
    return result


def _detail_reference_rows(references: list[Mapping[str, object]],
                           documents: Mapping[tuple[str, str], tuple[str, str]],
                           source: object) -> list[dict[str, Any]]:
    rows = []
    for reference in references:
        kb_id = _required_id(reference.get('kb_id'), 'reference.kb_id')
        doc_id = _required_id(reference.get('doc_id'), 'reference.doc_id')
        names = documents.get((kb_id, doc_id))
        if names is None:
            if source != 'imported':
                raise ServiceError(503, f'reference document is unavailable: {kb_id}/{doc_id}')
            names = (kb_id, doc_id)
        rows.append({
            'chunk_id': _required_id(reference.get('chunk_id'), 'reference.chunk_id'),
            'knowledge_base': {'id': kb_id, 'name': names[0]},
            'document': {'id': doc_id, 'name': names[1]},
            'text': _required_id(reference.get('text'), 'reference.text'),
        })
    return rows


def _detail_generate_stage(status: object, draft: dict[str, Any] | None) -> dict[str, Any]:
    result = {'status': status, 'question': None, 'answer': None, 'grading_guidance': None}
    if draft is None:
        return result
    value = _overview_mapping(draft['value'], 'case draft')
    result.update({
        field: _required_id(value.get(field), f'case draft.{field}')
        for field in ('question', 'answer', 'grading_guidance')
    })
    return result


def _detail_grading_stage(status: object, enhancement: dict[str, Any] | None) -> dict[str, Any]:
    result = {'status': status, 'key_points': None, 'forbidden_claims': None}
    if enhancement is None:
        return result
    value = _overview_mapping(enhancement['value'], 'case enhancement')
    key_points = value.get('key_points')
    forbidden_claims = value.get('forbidden_claims')
    if not isinstance(key_points, list) or not isinstance(forbidden_claims, list):
        raise ServiceError(503, 'case enhancement is invalid')
    result['key_points'] = [
        {
            'statement': _required_id(_overview_mapping(item, 'case key point').get('statement'), 'key_point.statement'),
            'evidence_chunk_ids': _detail_text_list(
                _overview_mapping(item, 'case key point').get('evidence_chunk_ids'), 'key_point.evidence_chunk_ids',
            ),
        }
        for item in key_points
    ]
    result['forbidden_claims'] = _detail_text_list(forbidden_claims, 'case forbidden_claims')
    return result


def _detail_text_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ServiceError(503, f'{name} is invalid')
    return list(value)
