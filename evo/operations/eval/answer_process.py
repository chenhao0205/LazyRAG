from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evo.operations.analysis.trace_summary import STAGE_RULES, build_trace_summary

HANDBOOK_CALL_CHAIN = (
    ('query_rewrite', 'Query Rewrite', 'query_rewrite'),
    ('retrieve', 'Retrieve', 'retrieve'),
    ('rerank', 'Rerank', 'rerank'),
    ('generate', 'Generate', 'llm_generate'),
)
STAGE_LABELS = {
    'query_rewrite': 'Query Rewrite',
    'retrieve': 'Retrieve',
    'rerank': 'Rerank',
    'context_assembly': 'Context Assembly',
    'prompt_build': 'Prompt Build',
    'tool_call': 'Tool Call',
    'llm_generate': 'Generate',
    'postprocess': 'Postprocess',
    'stream': 'Stream',
}
TRACE_HINT = 'use /threads/{thread_id}/results/traces/{trace_id} for raw trace detail'
TRACE_NOTE_OK = 'Trace loaded; durations use exclusive stage latency from the final attempt.'
TRACE_NOTE_HEURISTIC = (
    'Trace spans were unavailable; step status is inferred from evidence/answer and durations are null.'
)
TRACE_NOTE_PARTIAL = (
    'Trace loaded but some handbook stages have no duration; expand shows observed stages only.'
)
TRACE_NOTE_MISSING = 'No trace_id on the answer artifact.'


def build_answer_process_panel(
    row: Mapping[str, Any],
    *,
    load_trace: bool = True,
    attempts: int = 1,
    retry_seconds: float = 0.0,
) -> dict[str, Any]:
    stored = row.get('answer_process') if isinstance(row.get('answer_process'), Mapping) else None
    if _usable_stored(stored) and (_stored_has_latency(stored) or not load_trace):
        return _normalize_panel(stored, row)

    has_evidence = _has_evidence(row)
    has_answer = bool(str(row.get('rag_answer') or row.get('answer') or '').strip())
    trace_id = str(row.get('trace_id') or '').strip()
    summary = None
    if load_trace and trace_id:
        summary = build_trace_summary(
            {'id': str(row.get('case_id') or '')},
            {'case_id': str(row.get('case_id') or ''), 'trace_id': trace_id},
            attempts=attempts,
            retry_seconds=retry_seconds,
        )
        if not _usable_trace_summary(summary) and _usable_stored(stored):
            return _normalize_panel(stored, row)
    elif _usable_stored(stored):
        return _normalize_panel(stored, row)
    return compose_answer_process_panel(
        summary,
        trace_id=trace_id,
        has_evidence=has_evidence,
        has_answer=has_answer,
    )


def compose_answer_process_panel(
    summary: Mapping[str, Any] | None,
    *,
    trace_id: str = '',
    has_evidence: bool = False,
    has_answer: bool = False,
) -> dict[str, Any]:
    usable = _usable_trace_summary(summary)
    call_chain = _call_chain(summary if usable else None, has_evidence=has_evidence, has_answer=has_answer)
    latency_expand = _latency_expand(summary if usable else None)
    return {
        'call_chain': call_chain,
        'latency_expand': latency_expand,
        'trace': _trace_meta(
            trace_id=trace_id or str((summary or {}).get('trace_id') or ''),
            summary=summary if usable else None,
            call_chain=call_chain,
            latency_expand=latency_expand,
            source=str((summary or {}).get('trace_source') or ''),
        ),
    }


def _usable_stored(stored: Mapping[str, Any] | None) -> bool:
    if not isinstance(stored, Mapping):
        return False
    call_chain = stored.get('call_chain')
    return isinstance(call_chain, list) and bool(call_chain)


def _stored_has_latency(stored: Mapping[str, Any] | None) -> bool:
    if not isinstance(stored, Mapping):
        return False
    expand = stored.get('latency_expand') if isinstance(stored.get('latency_expand'), Mapping) else {}
    if expand.get('available') and expand.get('total_duration_ms') is not None:
        return True
    call_chain = stored.get('call_chain') if isinstance(stored.get('call_chain'), list) else []
    return any(
        isinstance(item, Mapping) and item.get('duration_ms') is not None
        for item in call_chain
    )


def _normalize_panel(stored: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    has_evidence = _has_evidence(row)
    has_answer = bool(str(row.get('rag_answer') or row.get('answer') or '').strip())
    call_chain = stored.get('call_chain')
    if not isinstance(call_chain, list) or not call_chain:
        call_chain = _call_chain(None, has_evidence=has_evidence, has_answer=has_answer)
    else:
        call_chain = [_normalize_chain_step(item) for item in call_chain if isinstance(item, Mapping)]
    latency_expand = stored.get('latency_expand')
    if not isinstance(latency_expand, Mapping):
        latency_expand = _latency_expand(None)
    else:
        latency_expand = _normalize_latency_expand(latency_expand, call_chain)
    trace = stored.get('trace') if isinstance(stored.get('trace'), Mapping) else {}
    trace_id = str(trace.get('trace_id') or row.get('trace_id') or '')
    return {
        'call_chain': call_chain,
        'latency_expand': latency_expand,
        'trace': _trace_meta(
            trace_id=trace_id,
            summary=None,
            call_chain=call_chain,
            latency_expand=latency_expand,
            source=str(trace.get('source') or 'row.answer_process'),
            status_override=str(trace.get('status') or ''),
            reason_override=str(trace.get('reason') or ''),
        ),
    }


def _usable_trace_summary(summary: Mapping[str, Any] | None) -> bool:
    if not isinstance(summary, Mapping):
        return False
    if str(summary.get('trace_status') or '') == 'unavailable':
        return False
    return bool(summary.get('stages') or summary.get('latency_by_stage') or summary.get('diagnostic_stage_sequence'))


def _trace_meta(
    *,
    trace_id: str,
    summary: Mapping[str, Any] | None,
    call_chain: list[dict[str, Any]],
    latency_expand: Mapping[str, Any],
    source: str = '',
    status_override: str = '',
    reason_override: str = '',
) -> dict[str, Any]:
    latency_available = bool(latency_expand.get('available')) and any(
        item.get('duration_ms') is not None for item in call_chain
    )
    incomplete = bool(latency_expand.get('incomplete'))
    if summary is not None and latency_available and not incomplete:
        status = 'ok'
        reason = ''
        note = TRACE_NOTE_OK
        resolved_source = source or str(summary.get('trace_source') or 'lazyllm.get_single_trace')
    elif summary is not None and latency_available:
        status = 'partial'
        reason = 'trace_partial_latency'
        note = TRACE_NOTE_PARTIAL
        resolved_source = source or str(summary.get('trace_source') or 'lazyllm.get_single_trace')
    elif summary is not None:
        status = 'partial'
        reason = 'trace_without_stage_latency'
        note = TRACE_NOTE_PARTIAL
        resolved_source = source or str(summary.get('trace_source') or 'lazyllm.get_single_trace')
    elif status_override in {'ok', 'partial'} and latency_available:
        status = status_override
        reason = reason_override or ('trace_partial_latency' if incomplete else '')
        note = TRACE_NOTE_PARTIAL if incomplete or status == 'partial' else TRACE_NOTE_OK
        resolved_source = source or 'row.answer_process'
    elif trace_id:
        status = status_override or 'unavailable'
        reason = reason_override or 'trace_spans_unavailable'
        note = TRACE_NOTE_HEURISTIC
        resolved_source = source or 'heuristic'
    else:
        status = 'missing'
        reason = 'trace_id_missing'
        note = TRACE_NOTE_MISSING
        resolved_source = source or 'heuristic'
    return {
        'trace_id': trace_id,
        'available': bool(trace_id),
        'readable': status in {'ok', 'partial'},
        'latency_available': latency_available,
        'source': resolved_source,
        'status': status,
        'reason': reason,
        'note': note,
        'hint': TRACE_HINT,
        'raw_entry': 'advanced',
    }


def _call_chain(
    summary: Mapping[str, Any] | None,
    *,
    has_evidence: bool,
    has_answer: bool,
) -> list[dict[str, Any]]:
    latency = summary.get('latency_by_stage') if isinstance(summary, Mapping) else {}
    if not isinstance(latency, Mapping):
        latency = {}
    stages = summary.get('stages') if isinstance(summary, Mapping) else []
    if not isinstance(stages, list):
        stages = []
    observed = {
        str(item.get('stage') or '')
        for item in stages
        if isinstance(item, Mapping) and item.get('stage')
    }
    observed.update(str(key) for key in latency)
    errors = {
        str(item.get('stage') or '')
        for item in ((summary or {}).get('error_stages') or [])
        if isinstance(item, Mapping) and item.get('stage')
    }
    result = []
    for step, label, stage in HANDBOOK_CALL_CHAIN:
        duration = _optional_ms(latency.get(stage))
        status = _chain_status(
            stage,
            observed=observed,
            errors=errors,
            has_trace=summary is not None,
            has_evidence=has_evidence,
            has_answer=has_answer,
            duration=duration,
        )
        stage_nodes = [item for item in stages if isinstance(item, Mapping) and item.get('stage') == stage]
        result.append({
            'step': step,
            'label': label,
            'status': status,
            'duration_ms': duration,
            'exclusive_duration_ms': duration,
            'span_count': len(stage_nodes),
            'names': [str(item.get('name') or '') for item in stage_nodes if item.get('name')][:8],
            'latency_note': _latency_note(status, duration, has_trace=summary is not None),
        })
    return result


def _latency_note(status: str, duration: float | None, *, has_trace: bool) -> str:
    if duration is not None:
        return ''
    if status == 'skipped':
        return 'Stage not observed in the final attempt trace.'
    if status == 'unknown':
        return 'No trace spans; status unknown.'
    if status == 'done' and not has_trace:
        return 'Inferred from answer evidence; duration unavailable.'
    if status == 'done':
        return 'Stage inferred or observed without exclusive latency.'
    if status == 'failed':
        return 'Stage reported error; duration may be incomplete.'
    return ''


def _chain_status(
    stage: str,
    *,
    observed: set[str],
    errors: set[str],
    has_trace: bool,
    has_evidence: bool,
    has_answer: bool,
    duration: float | None,
) -> str:
    if stage in errors:
        return 'failed'
    if stage in observed or duration is not None:
        return 'done'
    if stage == 'retrieve' and has_evidence:
        return 'done'
    if stage == 'llm_generate' and has_answer:
        return 'done'
    if has_trace:
        return 'skipped'
    return 'unknown'


def _latency_expand(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(summary, Mapping):
        return {
            'available': False,
            'incomplete': True,
            'total_duration_ms': None,
            'bottleneck_stage': '',
            'bottleneck_label': '',
            'route_signature': '',
            'missing_stages': [step for step, _, _ in HANDBOOK_CALL_CHAIN],
            'note': TRACE_NOTE_HEURISTIC,
            'stages': [],
        }
    latency = summary.get('latency_by_stage') if isinstance(summary.get('latency_by_stage'), Mapping) else {}
    stages = [item for item in (summary.get('stages') or []) if isinstance(item, Mapping)]
    ordered = _ordered_stages(summary, latency)
    errors = {
        str(item.get('stage') or '')
        for item in (summary.get('error_stages') or [])
        if isinstance(item, Mapping)
    }
    expand_stages = []
    for stage in ordered:
        nodes = [item for item in stages if item.get('stage') == stage]
        duration = _optional_ms(latency.get(stage))
        if duration is None and nodes:
            duration = round(sum(float(item.get('exclusive_latency_ms') or 0.0) for item in nodes), 4)
        expand_stages.append({
            'stage': stage,
            'label': STAGE_LABELS.get(stage, stage),
            'duration_ms': duration,
            'exclusive_duration_ms': duration,
            'status': 'failed' if stage in errors else ('done' if nodes or duration is not None else 'skipped'),
            'steps': [
                {
                    'id': str(item.get('id') or item.get('span_id') or ''),
                    'name': str(item.get('name') or ''),
                    'latency_ms': _optional_ms(item.get('latency_ms')),
                    'exclusive_latency_ms': _optional_ms(item.get('exclusive_latency_ms')),
                    'status': str(item.get('status') or ''),
                    'error': str(item.get('error') or ''),
                }
                for item in nodes
            ],
        })
    features = summary.get('features') if isinstance(summary.get('features'), Mapping) else {}
    total = _optional_ms(features.get('trace_latency_ms'))
    if total is None and expand_stages:
        values = [item['duration_ms'] for item in expand_stages if item['duration_ms'] is not None]
        total = round(sum(values), 4) if values else None
    observed = {item['stage'] for item in expand_stages if item.get('duration_ms') is not None}
    missing = [
        step for step, _, stage in HANDBOOK_CALL_CHAIN
        if stage not in observed
    ]
    bottleneck = str(summary.get('bottleneck_stage') or '')
    return {
        'available': bool(expand_stages),
        'incomplete': bool(missing) or total is None,
        'total_duration_ms': total,
        'bottleneck_stage': bottleneck,
        'bottleneck_label': STAGE_LABELS.get(bottleneck, bottleneck),
        'route_signature': str(summary.get('route_signature') or ''),
        'missing_stages': missing,
        'note': TRACE_NOTE_PARTIAL if missing else TRACE_NOTE_OK,
        'stages': expand_stages,
    }


def _normalize_latency_expand(
    latency_expand: Mapping[str, Any],
    call_chain: list[dict[str, Any]],
) -> dict[str, Any]:
    stages = [
        _normalize_expand_stage(item)
        for item in (latency_expand.get('stages') or [])
        if isinstance(item, Mapping)
    ]
    missing = latency_expand.get('missing_stages')
    if not isinstance(missing, list):
        observed = {
            item['step'] for item in call_chain
            if item.get('duration_ms') is not None
        }
        missing = [step for step, _, _ in HANDBOOK_CALL_CHAIN if step not in observed]
    bottleneck = str(latency_expand.get('bottleneck_stage') or '')
    available = bool(latency_expand.get('available') if 'available' in latency_expand else stages)
    return {
        'available': available,
        'incomplete': bool(latency_expand.get('incomplete', bool(missing) or not available)),
        'total_duration_ms': _optional_ms(latency_expand.get('total_duration_ms')),
        'bottleneck_stage': bottleneck,
        'bottleneck_label': str(
            latency_expand.get('bottleneck_label') or STAGE_LABELS.get(bottleneck, bottleneck)
        ),
        'route_signature': str(latency_expand.get('route_signature') or ''),
        'missing_stages': [str(item) for item in missing],
        'note': str(latency_expand.get('note') or (TRACE_NOTE_PARTIAL if missing else TRACE_NOTE_OK)),
        'stages': stages,
    }


def _ordered_stages(summary: Mapping[str, Any], latency: Mapping[str, Any]) -> list[str]:
    ordered: list[str] = []
    for value in summary.get('diagnostic_stage_sequence') or []:
        stage = str(value or '')
        if stage and stage not in ordered:
            ordered.append(stage)
    for stage, _ in STAGE_RULES:
        if stage in latency and stage not in ordered:
            ordered.append(stage)
    for stage in latency:
        key = str(stage)
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def _has_evidence(row: Mapping[str, Any]) -> bool:
    contexts = row.get('retrieve_contexts') or row.get('retrieved_contexts') or row.get('contexts') or []
    if isinstance(contexts, Sequence) and not isinstance(contexts, (str, bytes)) and contexts:
        return True
    return bool(
        row.get('retrieve_chunk_ids')
        or row.get('chunk_ids')
        or row.get('retrieve_doc_ids')
        or row.get('doc_ids')
    )


def _normalize_chain_step(item: Mapping[str, Any]) -> dict[str, Any]:
    step = str(item.get('step') or '')
    label = str(item.get('label') or STAGE_LABELS.get(step, step))
    duration = _optional_ms(item.get('duration_ms'))
    status = str(item.get('status') or 'unknown')
    return {
        'step': step,
        'label': label,
        'status': status,
        'duration_ms': duration,
        'exclusive_duration_ms': _optional_ms(
            item.get('exclusive_duration_ms') if item.get('exclusive_duration_ms') is not None
            else item.get('duration_ms')
        ),
        'span_count': int(item.get('span_count') or 0),
        'names': [str(name) for name in (item.get('names') or []) if str(name)][:8],
        'latency_note': str(item.get('latency_note') or _latency_note(status, duration, has_trace=True)),
    }


def _normalize_expand_stage(item: Mapping[str, Any]) -> dict[str, Any]:
    stage = str(item.get('stage') or '')
    return {
        'stage': stage,
        'label': str(item.get('label') or STAGE_LABELS.get(stage, stage)),
        'duration_ms': _optional_ms(item.get('duration_ms')),
        'exclusive_duration_ms': _optional_ms(
            item.get('exclusive_duration_ms') if item.get('exclusive_duration_ms') is not None
            else item.get('duration_ms')
        ),
        'status': str(item.get('status') or 'unknown'),
        'steps': [
            {
                'id': str(step.get('id') or ''),
                'name': str(step.get('name') or ''),
                'latency_ms': _optional_ms(step.get('latency_ms')),
                'exclusive_latency_ms': _optional_ms(step.get('exclusive_latency_ms')),
                'status': str(step.get('status') or ''),
                'error': str(step.get('error') or ''),
            }
            for step in (item.get('steps') or [])
            if isinstance(step, Mapping)
        ],
    }


def _optional_ms(value: object) -> float | None:
    if value is None or value == '':
        return None
    try:
        return round(max(0.0, float(value)), 4)
    except (TypeError, ValueError):
        return None
