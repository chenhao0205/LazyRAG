from __future__ import annotations

import importlib
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from evo.operations.public_contracts import clean_text as _text

from . import _as_list as _list, _stable_hash, _unique_text_values as _merge_text_values
from .mechanism_registry import PROBE_REGISTRY, registered_probes_for

ProbeHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]
DEFAULT_MAX_PROBE_CALLS = 4
PROBE_PROVIDER_ENV = 'LAZYMIND_ANALYSIS_PROBE_PROVIDERS'
PROBE_COST_NUMERIC_KEYS = (
    'duration_ms',
    'model_calls',
    'runtime_replays',
    'input_tokens',
    'output_tokens',
    'estimated_cost_usd',
)
_EXTERNAL_PROBE_HANDLERS: dict[str, ProbeHandler] = {}
_LOADED_PROVIDERS: set[str] = set()
_PROVIDER_LOCK = threading.Lock()
_PROBE_GATE = threading.BoundedSemaphore(8)
_PROBE_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix='analysis-probe')


def analysis_owned_probe_handlers() -> dict[str, ProbeHandler]:
    return {
        'rerank.selection_replay': _observed_rerank_diff,
        'context.selection_replay': _observed_context_diff,
    }


def register_probe_handler(probe_id: str, handler: ProbeHandler) -> None:
    probe_id = _text(probe_id)
    if probe_id not in PROBE_REGISTRY:
        raise ValueError(f'cannot register unknown analysis probe: {probe_id}')
    if not callable(handler):
        raise ValueError(f'analysis probe handler must be callable: {probe_id}')
    _EXTERNAL_PROBE_HANDLERS[probe_id] = handler


def registered_probe_handlers() -> dict[str, ProbeHandler]:
    configure_probe_handlers()
    return {
        **analysis_owned_probe_handlers(),
        **_EXTERNAL_PROBE_HANDLERS,
    }


def configure_probe_handlers(provider_refs: str | Sequence[str] | None = None) -> dict[str, str]:
    refs = _provider_refs(os.getenv(PROBE_PROVIDER_ENV, '') if provider_refs is None else provider_refs)
    with _PROVIDER_LOCK:
        for ref in refs:
            if ref in _LOADED_PROVIDERS:
                continue
            provider = _load_provider(ref)
            handlers = provider()
            if not isinstance(handlers, Mapping):
                raise ValueError(f'analysis probe provider must return a mapping: {ref}')
            for probe_id, handler in handlers.items():
                register_probe_handler(str(probe_id), handler)
            _LOADED_PROVIDERS.add(ref)
    return {
        probe_id: _handler_name(handler)
        for probe_id, handler in _EXTERNAL_PROBE_HANDLERS.items()
    }


def run_registered_probe(
    probe_id: str,
    params: Mapping[str, Any],
    *,
    handlers: Mapping[str, ProbeHandler],
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    probe = PROBE_REGISTRY.get(_text(probe_id))
    if probe is None:
        raise ValueError(f'unknown analysis probe: {probe_id}')
    if _text(probe.get('kind')) not in {'readonly_replay', 'readonly_diff'}:
        raise ValueError(f'analysis probe kind is not allowed: {probe.get("kind")}')
    handler = handlers.get(_text(probe_id))
    if handler is None:
        raise ValueError(f'analysis probe handler is not registered: {probe_id}')
    started = time.perf_counter()
    result = _probe_with_timeout(handler, dict(params), timeout_seconds)
    duration_ms = round((time.perf_counter() - started) * 1000.0, 4)
    if not isinstance(result, Mapping):
        raise ValueError(f'analysis probe handler must return a mapping: {probe_id}')
    mechanism_ids = _probe_mechanism_ids(probe_id, probe, result)
    target_ids = _merge_text_values(
        result.get('target_ids'),
        result.get('target_id'),
        params.get('target_ids'),
        params.get('target_id'),
    )
    decision = _probe_decision(result)
    confidence = _probe_confidence(result.get('confidence'), decision)
    return {
        'id': 'analysis.probe_observation',
        'probe_id': _text(probe_id),
        'kind': _text(probe.get('kind')),
        'mechanism_ids': mechanism_ids,
        'target_ids': target_ids,
        'decision': decision,
        'confidence': confidence,
        'evidence_refs': _merge_text_values(result.get('evidence_refs')),
        'controlled_variables': _merge_text_values(result.get('controlled_variables')),
        'baseline': result.get('baseline') if isinstance(result.get('baseline'), Mapping) else {},
        'treatment': result.get('treatment') if isinstance(result.get('treatment'), Mapping) else {},
        'cost': _probe_cost(result.get('cost'), duration_ms),
        'observation': _redact_sensitive(dict(result)),
        'provenance': {
            'input_hash': _stable_hash(params),
            'handler': _handler_name(handler),
        },
        'checks': {
            'ready': True,
            'errors': [],
            'registered_probe_only': True,
            'readonly_or_rollback': True,
            'decision_ready': decision in {'confirmed', 'ruled_out'},
        },
    }


def run_confirmation_probe_batch(
    confirmation_plan: Mapping[str, Any],
    *,
    handlers: Mapping[str, ProbeHandler],
    context: Mapping[str, Any] | None = None,
    eligible_target_ids: Sequence[str] | None = None,
    max_probe_calls: int = DEFAULT_MAX_PROBE_CALLS,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    if not isinstance(max_probe_calls, int) or isinstance(max_probe_calls, bool) or max_probe_calls < 0:
        raise ValueError('analysis probe max_probe_calls must be a non-negative integer')
    restrict_targets = eligible_target_ids is not None
    eligible = {_text(item) for item in eligible_target_ids or () if _text(item)}
    requested = [
        dict(step)
        for step in confirmation_plan.get('steps') or ()
        if isinstance(step, Mapping)
        and _text(step.get('probe_id'))
        and (
            not restrict_targets
            or bool(eligible.intersection(_merge_text_values(step.get('target_ids'))))
        )
    ]
    selected, unavailable_steps, delayed_steps = _schedule_probe_steps(
        requested,
        handlers,
        max_probe_calls,
    )
    observations: list[dict[str, Any]] = []
    unavailable = [
        {
            'step_id': _text(step.get('step_id')),
            'probe_id': _text(step.get('probe_id')),
            'target_ids': _merge_text_values(step.get('target_ids')),
            'reason': 'handler_not_registered',
        }
        for step in unavailable_steps
    ]
    failed: list[dict[str, Any]] = []
    for step in selected:
        probe_id = _text(step.get('probe_id'))
        step_id = _text(step.get('step_id'))
        params = {
            **dict(context or {}),
            'step_id': step_id,
            'target_ids': _merge_text_values(step.get('target_ids')),
            'mechanism_ids': _merge_text_values(step.get('mechanism_ids')),
            'fixed_variables': list(step.get('fixed_variables') or ()),
            'compare': list(step.get('compare') or ()),
        }
        try:
            observations.append(run_registered_probe(
                probe_id,
                params,
                handlers=handlers,
                timeout_seconds=timeout_seconds,
            ))
        except FutureTimeoutError:
            failed.append({
                'step_id': step_id,
                'probe_id': probe_id,
                'target_ids': params['target_ids'],
                'reason': 'handler_timeout',
                'error': f'probe timed out after {timeout_seconds:g}s',
            })
        except Exception as exc:
            failed.append({
                'step_id': step_id,
                'probe_id': probe_id,
                'target_ids': params['target_ids'],
                'reason': 'handler_failed',
                'error': _text(exc)[:500],
            })
    delayed = [
        {
            'step_id': _text(step.get('step_id')),
            'probe_id': _text(step.get('probe_id')),
            'target_ids': _merge_text_values(step.get('target_ids')),
            'reason': 'probe_budget_exhausted',
        }
        for step in delayed_steps
    ]
    if not requested:
        status = 'not_required'
    elif observations and not unavailable and not failed and not delayed:
        status = 'completed'
    elif observations:
        status = 'partial'
    else:
        status = 'unavailable'
    return {
        'id': 'analysis.probe_batch',
        'status': status,
        'observations': observations,
        'requested_steps': [_text(step.get('step_id')) for step in requested],
        'executed_steps': [_text(item.get('probe_id')) for item in observations],
        'unavailable': unavailable,
        'failed': failed,
        'delayed': delayed,
        'cost': _probe_batch_cost(observations),
        'checks': {
            'ready': True,
            'errors': [],
            'registered_probe_only': True,
            'readonly_or_rollback': True,
            'max_probe_calls': max_probe_calls,
            'requested_count': len(requested),
            'executed_count': len(observations),
            'all_handlers_available': not unavailable,
            'timeout_seconds': timeout_seconds,
        },
    }


def _probe_with_timeout(
    handler: ProbeHandler,
    params: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    if timeout_seconds <= 0:
        raise ValueError('analysis probe timeout_seconds must be positive')
    future = _PROBE_EXECUTOR.submit(_gated_probe, handler, params)
    try:
        return future.result(timeout=timeout_seconds)
    finally:
        future.cancel()


def _gated_probe(handler: ProbeHandler, params: Mapping[str, Any]) -> Mapping[str, Any]:
    with _PROBE_GATE:
        return handler(params)


def _handler_name(handler: ProbeHandler) -> str:
    return f'{getattr(handler, "__module__", "")}.{getattr(handler, "__qualname__", type(handler).__name__)}'.strip('.')


def _provider_refs(value: str | Sequence[str]) -> list[str]:
    raw = value.split(',') if isinstance(value, str) else list(value)
    return list(dict.fromkeys(_text(item) for item in raw if _text(item)))


def _load_provider(ref: str) -> Callable[[], Mapping[str, ProbeHandler]]:
    module_name, separator, attribute = ref.partition(':')
    module_name = _text(module_name)
    attribute = _text(attribute) if separator else 'analysis_probe_handlers'
    if not module_name or not attribute:
        raise ValueError(f'analysis probe provider is invalid: {ref}')
    provider = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(provider):
        raise ValueError(f'analysis probe provider is not callable: {ref}')
    return provider


def _probe_cost(value: Any, duration_ms: float) -> dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    cost: dict[str, Any] = {
        key: raw.get(key)
        for key in PROBE_COST_NUMERIC_KEYS
        if key in raw
    }
    source = _text(raw.get('source'))[:120]
    if source:
        cost['source'] = source
    cost['duration_ms'] = duration_ms
    return cost


def _probe_batch_cost(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {key: 0.0 for key in PROBE_COST_NUMERIC_KEYS}
    sources: list[str] = []
    for observation in observations:
        cost = observation.get('cost') if isinstance(observation.get('cost'), Mapping) else {}
        for key in PROBE_COST_NUMERIC_KEYS:
            try:
                totals[key] += float(cost.get(key) or 0.0)
            except (TypeError, ValueError):
                continue
        source = _text(cost.get('source'))
        if source and source not in sources:
            sources.append(source)
    return {
        **{key: round(value, 6) for key, value in totals.items()},
        'sources': sources,
    }


def _redact_sensitive(value: Any) -> Any:
    secret_key = re.compile(r'(?i)(authorization|api[_-]?key|token|password|secret)')
    secret_value = re.compile(
        r'(?i)((authorization|api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+'
    )
    if isinstance(value, Mapping):
        return {
            str(key): (
                '[REDACTED]'
                if secret_key.search(str(key))
                else _redact_sensitive(raw)
            )
            for key, raw in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return secret_value.sub(r'\1[REDACTED]', value)
    return value


def _observed_rerank_diff(params: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _single_probe_target(params)
    step = _latest_retrieval_step(params, 'rerank')
    reference_ids = _target_reference_ids(target)
    candidate_ids = set(_merge_text_values(
        step.get('candidate_doc_ids'),
        step.get('candidate_node_ids'),
    ))
    ranked = _merge_text_values(step.get('ranked_doc_ids'), step.get('ranked_node_ids'))
    ranked_ids = set(ranked)
    if not reference_ids or not candidate_ids:
        decision = 'insufficient_evidence'
    elif not reference_ids.intersection(candidate_ids):
        decision = 'ruled_out'
    elif not reference_ids.intersection(ranked_ids):
        decision = 'confirmed'
    else:
        topk = _positive_int(step.get('topk'))
        first_rank = min(
            index + 1
            for index, item in enumerate(ranked)
            if item in reference_ids
        )
        decision = 'confirmed' if topk and first_rank > topk else 'ruled_out'
    return {
        'mechanism_ids': ['rerank.relevant_candidate_demoted'],
        'decision': decision,
        'confidence': 0.98 if decision in {'confirmed', 'ruled_out'} else 0.0,
        'controlled_variables': ['recorded query', 'recorded candidate set', 'recorded reranker output'],
        'baseline': {
            'candidate_ids': sorted(candidate_ids),
            'required_candidate_ids': sorted(reference_ids.intersection(candidate_ids)),
        },
        'treatment': {
            'ranked_ids': ranked,
            'required_ranked_ids': sorted(reference_ids.intersection(ranked_ids)),
            'topk': _positive_int(step.get('topk')),
        },
        'evidence_refs': [
            f'trace.retrieval_steps[{_text(step.get("id")) or "rerank"}].candidate_doc_ids',
            f'trace.retrieval_steps[{_text(step.get("id")) or "rerank"}].ranked_doc_ids',
        ],
        'cost': {'model_calls': 0, 'runtime_replays': 0, 'source': 'recorded_trace_diff'},
    }


def _observed_context_diff(params: Mapping[str, Any]) -> Mapping[str, Any]:
    target = _single_probe_target(params)
    trace = params.get('trace') if isinstance(params.get('trace'), Mapping) else {}
    reference_ids = _target_reference_ids(target)
    rerank = _latest_retrieval_step(params, 'rerank')
    retrieve = _latest_retrieval_step(params, 'retrieve')
    upstream_ids = set(_merge_text_values(
        rerank.get('ranked_doc_ids'),
        rerank.get('ranked_node_ids'),
        retrieve.get('returned_node_ids'),
        retrieve.get('doc_ids'),
        retrieve.get('chunk_ids'),
    ))
    final_ids = set(_merge_text_values(
        trace.get('final_context_doc_ids'),
        trace.get('final_context_chunk_ids'),
    ))
    if not reference_ids or not upstream_ids:
        decision = 'insufficient_evidence'
    elif not reference_ids.intersection(upstream_ids):
        decision = 'ruled_out'
    elif reference_ids.intersection(final_ids):
        decision = 'ruled_out'
    else:
        decision = 'confirmed'
    return {
        'mechanism_ids': ['context.required_evidence_dropped'],
        'decision': decision,
        'confidence': 0.98 if decision in {'confirmed', 'ruled_out'} else 0.0,
        'controlled_variables': ['recorded upstream candidates', 'recorded final context', 'diagnosis target'],
        'baseline': {
            'upstream_ids': sorted(upstream_ids),
            'required_upstream_ids': sorted(reference_ids.intersection(upstream_ids)),
        },
        'treatment': {
            'final_context_ids': sorted(final_ids),
            'required_final_ids': sorted(reference_ids.intersection(final_ids)),
        },
        'evidence_refs': [
            'trace.retrieval_steps',
            'trace.final_context_doc_ids',
            'trace.final_context_chunk_ids',
        ],
        'cost': {'model_calls': 0, 'runtime_replays': 0, 'source': 'recorded_trace_diff'},
    }


def _single_probe_target(params: Mapping[str, Any]) -> Mapping[str, Any]:
    target_ids = _merge_text_values(params.get('target_ids'), params.get('target_id'))
    packet = params.get('evidence_packet')
    targets = packet.get('diagnosis_targets') if isinstance(packet, Mapping) else ()
    matches = [
        item for item in targets or ()
        if isinstance(item, Mapping) and _text(item.get('id')) in target_ids
    ]
    return matches[0] if len(matches) == 1 else {}


def _latest_retrieval_step(params: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    trace = params.get('trace') if isinstance(params.get('trace'), Mapping) else {}
    steps = [
        item for item in trace.get('retrieval_steps') or ()
        if isinstance(item, Mapping) and _text(item.get('stage')) == stage
    ]
    return steps[-1] if steps else {}


def _target_reference_ids(target: Mapping[str, Any]) -> set[str]:
    return set(_merge_text_values(
        target.get('reference_doc_ids'),
        target.get('reference_chunk_ids'),
    ))


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _probe_mechanism_ids(
    probe_id: str,
    probe: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[str]:
    registered = [_text(item) for item in probe.get('mechanisms') or () if _text(item)]
    requested = _merge_text_values(result.get('mechanism_ids'), result.get('mechanism_id'))
    if not requested:
        if len(registered) != 1:
            raise ValueError(
                f'analysis probe {probe_id} must identify one mechanism from: {", ".join(registered)}'
            )
        return registered
    unknown = [item for item in requested if item not in registered]
    if unknown:
        raise ValueError(
            f'analysis probe {probe_id} returned unregistered mechanism ids: {", ".join(unknown)}'
        )
    return requested


def _probe_decision(result: Mapping[str, Any]) -> str:
    value = _text(result.get('decision') or result.get('mechanism_status') or result.get('outcome'))
    aliases = {
        'supports_mechanism': 'confirmed',
        'confirmed': 'confirmed',
        'rules_out_mechanism': 'ruled_out',
        'ruled_out': 'ruled_out',
        'inconclusive': 'insufficient_evidence',
        'insufficient_evidence': 'insufficient_evidence',
        '': 'insufficient_evidence',
    }
    if value not in aliases:
        raise ValueError(f'analysis probe decision is invalid: {value}')
    return aliases[value]


def _probe_confidence(value: Any, decision: str) -> float:
    if value in (None, ''):
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ValueError('analysis probe confidence must be numeric') from None
    if not 0.0 <= confidence <= 1.0:
        raise ValueError('analysis probe confidence must be in [0, 1]')
    return confidence


def _schedule_probe_steps(
    requested: Sequence[Mapping[str, Any]],
    handlers: Mapping[str, ProbeHandler],
    max_probe_calls: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available = [
        dict(step)
        for step in requested
        if _text(step.get('probe_id')) in handlers
    ]
    unavailable = [
        dict(step)
        for step in requested
        if _text(step.get('probe_id')) not in handlers
    ]
    queues: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for step in available:
        target_ids = _merge_text_values(step.get('target_ids'))
        key = target_ids[0] if target_ids else '__global__'
        if key not in queues:
            queues[key] = []
            order.append(key)
        queues[key].append(step)
    selected: list[dict[str, Any]] = []
    while len(selected) < max_probe_calls and any(queues.values()):
        for key in order:
            if queues[key] and len(selected) < max_probe_calls:
                selected.append(queues[key].pop(0))
    delayed = [step for key in order for step in queues[key]]
    return selected, unavailable, delayed


def build_confirmation_plan(
    target_paths: Sequence[Mapping[str, Any]],
    agenda: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in target_paths:
        if path.get('investigation_direction') == 'evidence_backtrack':
            _append_backtrack_steps(steps, seen, path, trace)
        elif path.get('investigation_direction') == 'needs_review':
            _append_ambiguous_steps(steps, seen, path, trace)
    for item in agenda:
        if item.get('status') != 'needs_probe':
            continue
        for probe in item.get('probe_plan') or registered_probes_for(_text(item.get('mechanism_id'))):
            _append_probe_step(steps, seen, probe, item)
    return {
        'id': 'analysis.confirmation_plan',
        'strategy': 'target_round_robin',
        'steps': steps,
        'checks': {
            'ready': True,
            'errors': [],
            'readonly_first': True,
            'step_count': len(steps),
        },
    }


def _append_backtrack_steps(
    steps: list[dict[str, Any]],
    seen: set[str],
    path: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> None:
    target_id = _text(path.get('target_id'))
    _append_probe_step(steps, seen, _probe('context.selection_replay'), {
        'mechanism_id': 'context.required_evidence_dropped',
    }, target_id=target_id, tier='L1')
    if 'rerank' in _list(trace.get('diagnostic_stage_sequence')):
        _append_probe_step(steps, seen, _probe('rerank.selection_replay'), {
            'mechanism_id': 'rerank.relevant_candidate_demoted',
        }, target_id=target_id, tier='L1')
    _append_probe_step(steps, seen, _probe('retrieve.rank_expand_replay'), {
        'mechanism_id': 'retrieve.reference_absent',
    }, target_id=target_id, tier='L2')


def _append_ambiguous_steps(
    steps: list[dict[str, Any]],
    seen: set[str],
    path: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> None:
    target_id = _text(path.get('target_id'))
    _append_probe_step(steps, seen, _probe('retrieve.rank_expand_replay'), {
        'mechanism_id': 'retrieve.reference_absent',
    }, target_id=target_id, tier='L2')
    if 'query_rewrite' in _list(trace.get('diagnostic_stage_sequence')):
        _append_probe_step(steps, seen, _probe('query.retrieve_ab'), {
            'mechanism_id': 'query.intent_lost',
        }, target_id=target_id, tier='L2')
    _append_probe_step(steps, seen, _probe('index.presence_probe'), {
        'mechanism_id': 'retrieve.reference_absent',
    }, target_id=target_id, tier='L2')


def _append_probe_step(
    steps: list[dict[str, Any]],
    seen: set[str],
    probe: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    *,
    target_id: str = '',
    tier: str = 'L1',
) -> None:
    probe_id = _text(probe.get('probe_id'))
    if not probe_id:
        return
    _append_step(steps, seen, {
        'step_id': f'probe.{probe_id}:{target_id or _text(mechanism.get("mechanism_id"))}',
        'tier': tier,
        'kind': _text(probe.get('kind')) or 'readonly_replay',
        'probe_id': probe_id,
        'target_ids': [target_id] if target_id else [],
        'mechanism_ids': list(probe.get('mechanisms') or [_text(mechanism.get('mechanism_id'))]),
        'fixed_variables': list(probe.get('fixed_variables') or ('case target', 'recorded artifacts')),
        'compare': list(probe.get('compare') or ('before', 'after', 'material_effect')),
    })


def _append_step(steps: list[dict[str, Any]], seen: set[str], step: Mapping[str, Any]) -> None:
    step_id = _text(step.get('step_id'))
    if not step_id or step_id in seen:
        return
    seen.add(step_id)
    steps.append(dict(step))


def _probe(probe_id: str) -> Mapping[str, Any]:
    spec = PROBE_REGISTRY.get(probe_id)
    return {'probe_id': probe_id, **dict(spec or {})}
