from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from evo.operations.public_contracts import clean_text as _text, mapping_or_empty as _mapping

from .classify import classify_case
from .judge import judge_primary_summary
from .repair_groups import build_repair_group_queue


def build_analysis_detail(
    classifications: tuple[Mapping[str, Any], ...],
    clusters: Mapping[str, Any],
    *,
    sidecars: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in classifications if isinstance(row, Mapping)),
        key=lambda row: _text(row.get('case_id')),
    )
    if len(rows) != len(classifications):
        raise ValueError('analysis.summary classifications must all be mappings')
    cluster_items = _validated_cluster_items(rows, clusters)
    cluster_rows = {_text(row.get('case_id')): row for row in cluster_items}
    sidecar_by_case = _validate_sidecars(classifications, sidecars)
    for row in rows:
        cluster = cluster_rows[_text(row.get('case_id'))]
        row['cluster_id'] = _text(cluster.get('cluster_id'))
        row['outlier_score'] = float(cluster.get('outlier_score') or 0.0)
        row['diagnosis'] = _diagnosis_brief(
            sidecar_by_case.get(_text(row.get('case_id')), {}),
        )
    repair_queue = build_repair_group_queue(_repair_rows(rows, sidecar_by_case))
    repair_groups_by_case = _repair_groups_by_case(repair_queue)
    case_diagnostics = [
        _case_diagnostic(
            row,
            sidecar_by_case.get(_text(row.get('case_id')), {}),
            repair_groups_by_case.get(_text(row.get('case_id')), ()),
        )
        for row in rows
    ]
    root_cause_groups = _root_cause_groups(case_diagnostics)
    trace_report = trace_quality(rows)
    summary_rows = [_summary_row(row) for row in rows]
    actionable = [case_brief(row) for row in rows if row.get('actionable')]
    pending = [case_brief(row) for row in rows if row.get('pending_analysis')]
    runtime = [case_brief(row) for row in rows if row.get('issue_category') == 'runtime_infra']
    contract = [case_brief(row) for row in rows if row.get('issue_category') == 'contract']
    return {
        'id': 'analysis.summary',
        'case_ids': [_text(row.get('case_id')) for row in rows],
        'total': len(rows),
        'issue_category_counts': dict(Counter(_text(row.get('issue_category')) for row in rows)),
        'issue_type_counts': dict(Counter(_text(row.get('issue_type')) for row in rows)),
        'affected_block_counts': dict(Counter(_text(row.get('affected_block')) for row in rows)),
        'failure_mode_counts': dict(Counter(_text(row.get('failure_mode')) for row in rows)),
        'judge_primary_summary': judge_primary_summary(rows),
        'trace_quality': trace_report,
        'actionable_cases': actionable,
        'pending_cases': pending,
        'runtime_infra_cases': runtime,
        'contract_cases': contract,
        'top_failure_patterns': top_failure_patterns(rows, clusters),
        'diagnostic_overview': _diagnostic_overview(
            case_diagnostics,
            root_cause_groups,
            repair_queue,
            trace_report,
        ),
        'root_cause_groups': root_cause_groups,
        'case_diagnostics': case_diagnostics,
        'repair_group_queue': repair_queue,
        'clusters': list(clusters.get('clusters') or []),
        'rows': summary_rows,
        'checks': {
            'ready': True,
            'errors': [],
            'case_count_matches': len(rows) == int(clusters.get('total') or 0),
        },
    }


def build_analysis_summary(
    run_id: str,
    classifications: tuple[Mapping[str, Any], ...],
    clusters: Mapping[str, Any],
    *,
    sidecars: tuple[Mapping[str, Any], ...] = (),
) -> dict[str, Any]:
    detail = build_analysis_detail(classifications, clusters, sidecars=sidecars)
    return {
        **detail,
        **_build_analysis_summary_root(run_id, detail['rows']),
    }


def build_analysis_from_answers(
    cases: Mapping[str, Mapping[str, Any]],
    answers: Mapping[str, Mapping[str, Any]],
    judges: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    from .cluster import cluster_traces
    from .trace_summary import build_trace_summary

    classifications = []
    for case_id, case in cases.items():
        trace = build_trace_summary(case, answers[case_id])
        classifications.append(classify_case(case, answers[case_id], judges[case_id], trace))
    clusters = cluster_traces(tuple(classifications))
    return build_analysis_detail(tuple(classifications), clusters)


def _build_analysis_summary_root(
    run_id: str,
    rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    cases = []
    for row in rows:
        diagnosis = _mapping(row.get('diagnosis'))
        root = diagnosis.get('root_cause') if isinstance(diagnosis.get('root_cause'), Mapping) else {}
        cases.append({
            'case_id': _text(row.get('case_id')),
            'trace_id': _text(row.get('trace_id')),
            'source': _text(row.get('source')),
            'failure_type': _text(row.get('issue_type')),
            'reason': _text(root.get('mechanism_id') or row.get('root_cause_reason') or row.get('reason')),
            'diagnosis': diagnosis,
        })
    return {
        'run_id': _text(run_id),
        'case_num': len(rows),
        'algo_id': next((_text(row.get('algo_id')) for row in rows if _text(row.get('algo_id'))), ''),
        'type_count': dict(Counter(case['failure_type'] for case in cases)),
        'cases': cases,
    }


def _validate_sidecars(
    classifications: tuple[Mapping[str, Any], ...],
    sidecars: tuple[Mapping[str, Any], ...],
) -> dict[str, Mapping[str, Any]]:
    if not sidecars:
        return {}
    if any(not isinstance(item, Mapping) for item in sidecars):
        raise ValueError('analysis.summary sidecars must all be mappings')
    by_case = {_text(item.get('case_id')): item for item in sidecars}
    case_ids = {_text(item.get('case_id')) for item in classifications}
    if not all(by_case) or set(by_case) != case_ids or len(by_case) != len(sidecars):
        raise ValueError('analysis.summary sidecars must cover every classification')
    return by_case


def _diagnosis_brief(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    if not sidecar:
        return {}
    result = sidecar.get('diagnostic_result')
    result = result if isinstance(result, Mapping) else {}
    execution = sidecar.get('investigation_execution')
    execution = execution if isinstance(execution, Mapping) else {}
    review_batch = execution.get('semantic_review_batch')
    review_batch = review_batch if isinstance(review_batch, Mapping) else {}
    probe_batch = execution.get('probe_batch')
    probe_batch = probe_batch if isinstance(probe_batch, Mapping) else {}
    primary = (
        result.get('primary_mechanism')
        if isinstance(result.get('primary_mechanism'), Mapping)
        else {}
    )
    return {
        'status': _text(result.get('status') or result.get('actionability')),
        'actionability': _text(result.get('actionability')),
        'fully_resolved': bool(result.get('fully_resolved')),
        'root_cause': dict(primary),
        'target_count': sum(
            isinstance(item, Mapping) for item in sidecar.get('target_results') or ()
        ),
        'review_status': _text(review_batch.get('status')),
        'probe_status': _text(probe_batch.get('status')),
        'unavailable_probes': list(probe_batch.get('unavailable') or ()),
    }


def _repair_groups_by_case(
    repair_queue: list[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for group in repair_queue:
        group_id = _text(group.get('group_id'))
        for case_id in group.get('case_ids') or ():
            case_key = _text(case_id)
            if case_key and group_id:
                groups.setdefault(case_key, []).append(group_id)
    return {
        case_id: tuple(dict.fromkeys(group_ids))
        for case_id, group_ids in groups.items()
    }


def _case_diagnostic(
    row: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    repair_group_ids: tuple[str, ...],
) -> dict[str, Any]:
    diagnosis = _mapping(row.get('diagnosis'))
    trace = _mapping(row.get('trace_summary'))
    judge = _mapping(row.get('judge'))
    target_results = [
        item for item in sidecar.get('target_results') or ()
        if isinstance(item, Mapping)
    ]
    problems = [
        dict(item.get('problem'))
        for item in target_results
        if isinstance(item.get('problem'), Mapping)
    ]
    root = (
        dict(diagnosis.get('root_cause'))
        if isinstance(diagnosis.get('root_cause'), Mapping)
        else {}
    )
    evidence_records = [
        dict(record)
        for target in target_results
        for record in target.get('evidence_records') or ()
        if isinstance(record, Mapping)
    ][:8]
    missing_evidence = list(dict.fromkeys(
        _text(reason)
        for target in target_results
        for reason in target.get('missing_evidence') or ()
        if _text(reason)
    ))[:8]
    actionability = (
        _text(diagnosis.get('actionability'))
        or _fallback_actionability(row)
    )
    source_counts = Counter(
        _text(item.get('source')) or 'unknown'
        for item in evidence_records
    )
    evidence_levels = Counter(
        _text(item.get('evidence_level')) or 'unknown'
        for item in evidence_records
    )
    directions = list(dict.fromkeys(
        _text(item.get('investigation_direction'))
        for item in target_results
        if _text(item.get('investigation_direction'))
    ))
    return {
        'case_id': _text(row.get('case_id')),
        'trace_id': _text(row.get('trace_id') or trace.get('trace_id')),
        'cluster_id': _text(row.get('cluster_id')),
        'analysis_status': actionability,
        'problem': {
            'issue_type': _text(row.get('issue_type')),
            'affected_block': _text(row.get('affected_block')),
            'failure_mode': _text(row.get('failure_mode')),
            'judge_failure_type': _text(judge.get('failure_type')),
            'quality_label': _text(judge.get('quality_label')),
            'overall_score': judge.get('overall_score'),
            'target_count': len(target_results),
            'target_types': list(dict.fromkeys(
                _text(item.get('target_type'))
                for item in target_results
                if _text(item.get('target_type'))
            )),
            'statements': [
                _text(item.get('statement'))
                for item in problems[:4]
                if _text(item.get('statement'))
            ],
        },
        'investigation': {
            'directions': directions,
            'stage_sequence': [
                _text(item) for item in trace.get('diagnostic_stage_sequence') or ()
                if _text(item)
            ][:12],
            'route_signature': _text(trace.get('route_signature')),
            'checkpoint_stage': _text(root.get('stage')),
            'trace_complete': _trace_complete(trace),
            'review_status': _text(diagnosis.get('review_status')),
            'probe_status': _text(diagnosis.get('probe_status')),
            'unavailable_probes': [
                dict(item) if isinstance(item, Mapping) else _text(item)
                for item in diagnosis.get('unavailable_probes') or ()
            ][:4],
        },
        'root_cause': root,
        'evidence': {
            'count': len(evidence_records),
            'source_counts': dict(source_counts),
            'level_counts': dict(evidence_levels),
            'records': evidence_records,
            'missing': missing_evidence,
        },
        'repair': {
            'ready': actionability == 'repair_ready',
            'group_ids': list(repair_group_ids),
        },
    }


def _fallback_actionability(row: Mapping[str, Any]) -> str:
    if _text(row.get('issue_type')) == 'correct':
        return 'not_required'
    if row.get('pending_analysis'):
        return 'pending_analysis'
    if _text(row.get('issue_category')) in {'runtime_infra', 'contract'}:
        return 'guard_failure'
    return 'classified'


def _root_cause_groups(
    case_diagnostics: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for case in case_diagnostics:
        root = _mapping(case.get('root_cause'))
        mechanism_id = _text(root.get('mechanism_id'))
        affected_block = _text(root.get('affected_block'))
        if mechanism_id:
            groups.setdefault((mechanism_id, affected_block), []).append(case)
    result = []
    for (mechanism_id, affected_block), cases in groups.items():
        roots = [_mapping(case.get('root_cause')) for case in cases]
        repair_group_ids = list(dict.fromkeys(
            _text(group_id)
            for case in cases
            for group_id in _mapping(case.get('repair')).get('group_ids') or ()
            if _text(group_id)
        ))
        ordered = sorted(
            cases,
            key=lambda case: (
                not bool(_mapping(case.get('repair')).get('ready')),
                -_number(_mapping(case.get('root_cause')).get('confidence')),
                -int(_mapping(case.get('evidence')).get('count') or 0),
                _text(case.get('case_id')),
            ),
        )
        case_ids = [_text(case.get('case_id')) for case in ordered]
        route_signatures = list(dict.fromkeys(
            _text(_mapping(case.get('investigation')).get('route_signature'))
            for case in cases
            if _text(_mapping(case.get('investigation')).get('route_signature'))
        ))
        cluster_ids = list(dict.fromkeys(
            _text(case.get('cluster_id')) for case in cases
            if _text(case.get('cluster_id'))
        ))
        result.append({
            'group_id': f'root:{affected_block}:{mechanism_id}',
            'mechanism_id': mechanism_id,
            'affected_block': affected_block,
            'failure_mode': _text(roots[0].get('failure_mode')),
            'stage': _text(roots[0].get('stage')),
            'evidence_level': _text(roots[0].get('evidence_level')),
            'case_count': len(cases),
            'case_ids': case_ids[:100],
            'case_ids_truncated': len(case_ids) > 100,
            'representative_case_id': case_ids[0],
            'target_count': sum(
                int(_mapping(case.get('problem')).get('target_count') or 0)
                for case in cases
            ),
            'repair_ready_count': sum(
                bool(_mapping(case.get('repair')).get('ready'))
                for case in cases
            ),
            'average_confidence': round(
                sum(_number(root.get('confidence')) for root in roots) / len(roots),
                4,
            ),
            'route_signatures': route_signatures[:8],
            'route_signature_count': len(route_signatures),
            'cluster_ids': cluster_ids[:8],
            'cluster_count': len(cluster_ids),
            'repair_group_ids': repair_group_ids,
        })
    return sorted(
        result,
        key=lambda item: (
            -item['case_count'],
            -item['repair_ready_count'],
            -item['average_confidence'],
            item['group_id'],
        ),
    )


def _diagnostic_overview(
    case_diagnostics: list[Mapping[str, Any]],
    root_cause_groups: list[Mapping[str, Any]],
    repair_queue: list[Mapping[str, Any]],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(
        _text(case.get('analysis_status')) or 'unknown'
        for case in case_diagnostics
    )
    evidence_level_counts = Counter(
        _text(_mapping(case.get('root_cause')).get('evidence_level')) or 'unconfirmed'
        for case in case_diagnostics
    )
    total = len(case_diagnostics)
    root_confirmed = sum(bool(_mapping(case.get('root_cause'))) for case in case_diagnostics)
    evidence_backed = sum(
        _text(_mapping(case.get('root_cause')).get('evidence_level'))
        in {'trace_fact', 'controlled_probe', 'corroborated_semantic'}
        for case in case_diagnostics
    )
    repair_ready = int(status_counts.get('repair_ready', 0))
    return {
        'total_cases': total,
        'status_counts': dict(status_counts),
        'evidence_level_counts': dict(evidence_level_counts),
        'progress_counts': {
            'problem_observed': sum(
                _text(_mapping(case.get('problem')).get('issue_type')) != 'correct'
                for case in case_diagnostics
            ),
            'root_cause_confirmed': root_confirmed,
            'evidence_backed': evidence_backed,
            'repair_ready': repair_ready,
        },
        'root_cause_group_count': len(root_cause_groups),
        'repair_group_count': len(repair_queue),
        'trace_complete': int(trace.get('complete') or 0),
        'trace_incomplete': max(0, total - int(trace.get('complete') or 0)),
    }


def _repair_rows(
    rows: list[Mapping[str, Any]],
    sidecar_by_case: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not sidecar_by_case:
        return [dict(row) for row in rows]
    repair_rows = []
    for row in rows:
        case_id = _text(row.get('case_id'))
        sidecar = sidecar_by_case.get(case_id, {})
        for target in sidecar.get('target_results') or ():
            if not isinstance(target, Mapping) or target.get('repair_ready') is not True:
                continue
            mechanism = target.get('primary_mechanism')
            if not isinstance(mechanism, Mapping) or not mechanism:
                continue
            affected_block = _text(mechanism.get('affected_block'))
            confidence = _number(mechanism.get('confidence'))
            repair_rows.append({
                **dict(row),
                'issue_category': _root_issue_category(affected_block, row),
                'issue_type': _text(mechanism.get('mechanism_id')),
                'affected_block': affected_block,
                'failure_mode': _text(mechanism.get('failure_mode')),
                'confidence': confidence,
                'actionable': True,
                'pending_analysis': False,
                'root_cause_reason': _text(mechanism.get('mechanism_id')),
                'diagnosis_target_id': _text(target.get('target_id')),
                'diagnosis_target_statement': _text(target.get('statement')),
                'diagnosis_evidence': list(target.get('evidence') or ()),
                'diagnosis_evidence_level': _text(mechanism.get('evidence_level')),
            })
    return repair_rows


def _root_issue_category(affected_block: str, row: Mapping[str, Any]) -> str:
    if affected_block in {
        'query_rewrite',
        'retrieval',
        'rerank',
        'context_assembly',
        'prompt_build',
    }:
        return 'retrieval'
    if affected_block == 'llm_generation':
        return 'generation'
    if affected_block == 'tool_orchestration':
        return 'execution'
    return _text(row.get('issue_category'))


def _number(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _validated_cluster_items(
    rows: list[Mapping[str, Any]],
    clusters: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    cluster_items = [row for row in clusters.get('rows', ()) if isinstance(row, Mapping) and row.get('case_id')]
    cluster_ids = [_text(row.get('case_id')) for row in cluster_items]
    row_ids = [_text(row.get('case_id')) for row in rows]
    if len(set(cluster_ids)) != len(cluster_ids) or set(cluster_ids) != set(row_ids):
        raise ValueError('analysis.summary cluster rows must cover every classification')
    if int(clusters.get('total') or 0) != len(rows):
        raise ValueError('analysis.summary cluster total must match classifications')
    return cluster_items


def case_brief(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'case_id': _text(row.get('case_id')),
        'issue_type': _text(row.get('issue_type')),
        'affected_block': _text(row.get('affected_block')),
        'failure_mode': _text(row.get('failure_mode')),
        'confidence': _text(row.get('confidence')),
        'reason': _text(row.get('root_cause_reason')),
        'cluster_id': _text(row.get('cluster_id')),
        'outlier_score': float(row.get('outlier_score') or 0.0),
    }


def _summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **case_brief(row),
        'trace_id': _text(row.get('trace_id')),
        'source': _text(row.get('source')),
        'algo_id': _text(row.get('algo_id')),
        'diagnosis': dict(_mapping(row.get('diagnosis'))),
    }


def trace_quality(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    traces = [_mapping(row.get('trace_summary')) for row in rows]
    features = [_mapping(trace.get('features')) for trace in traces]
    missing = [
        _text(row.get('case_id'))
        for row in rows
        if not _trace_complete(_mapping(row.get('trace_summary')))
    ]
    total = len(rows)
    return {
        'total': total,
        'complete': max(0, total - len(missing)),
        'missing': missing,
        'stage_unknown': [_text(row.get('case_id')) for row in rows
                          if _mapping(row.get('trace_summary')).get('unknown_stage_count')],
        'metrics_missing': [_text(row.get('case_id')) for row in rows
                            if row.get('issue_type') == 'trace_metrics_missing'],
        'error_stage_present': [_text(row.get('case_id')) for row in rows
                                if _mapping(row.get('trace_summary')).get('error_stages')],
        'avg_node_count': _avg(item.get('node_count') for item in features),
        'avg_trace_latency_ms': _avg(item.get('trace_latency_ms') for item in features),
    }


def top_failure_patterns(rows: list[Mapping[str, Any]], clusters: Mapping[str, Any]) -> list[dict[str, Any]]:
    cluster_items = [item for item in clusters.get('clusters', ()) if isinstance(item, Mapping)]
    if cluster_items:
        patterns = [
            {
                'pattern': '/'.join(_text(item.get(key)) for key in (
                    'dominant_affected_block', 'dominant_failure_mode',
                )),
                'cluster_id': _text(item.get('cluster_id')),
                'case_count': int(item.get('size') or 0),
                'representative_case_id': _text(item.get('representative_case_id')),
            }
            for item in cluster_items
            if _text(item.get('dominant_issue_type')) != 'correct'
        ]
        return patterns[:10]
    counts = Counter(
        (_text(row.get('affected_block')), _text(row.get('failure_mode')))
        for row in rows
        if row.get('issue_type') != 'correct'
    )
    return [
        {'pattern': f'{block}/{mode}', 'cluster_id': '', 'case_count': count, 'representative_case_id': ''}
        for (block, mode), count in counts.most_common(10)
    ]


def _trace_complete(trace: Mapping[str, Any]) -> bool:
    return all(trace.get(field) for field in ('trace_id', 'trace_source', 'route_signature', 'stage_sequence'))


def _avg(values: Any) -> float:
    rows = []
    for value in values:
        try:
            rows.append(float(value or 0.0))
        except (TypeError, ValueError):
            pass
    return round(sum(rows) / len(rows), 4) if rows else 0.0
