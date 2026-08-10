from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from evo.operations.eval.judge import validate_judge_result
from evo.operations.public_contracts import clean_text as _text, mapping_or_empty as _mapping

from . import _evidence_record as _evidence

CASE_FIELDS = ('answer', 'id', 'question')
TRACE_FIELDS = (
    'case_id',
    'trace_id',
    'trace_source',
    'route_signature',
    'tree_text',
    'stage_sequence',
    'diagnostic_stage_sequence',
    'edges',
    'critical_path',
    'bottleneck_stage',
    'stages',
    'stage_counts',
    'latency_by_stage',
    'error_stages',
    'retrieval_steps',
    'retrieved_doc_ids',
    'retrieved_chunk_ids',
    'final_context_doc_ids',
    'final_context_chunk_ids',
    'semantic_metric_keys',
    'features',
)
PRIMARY_SCORE_FIELDS = (
    'overall_score',
    'answer_quality_score',
    'retrieval_quality_score',
    'quality_label',
    'failure_type',
    'retrieval_failure_type',
)
CORE_EXPLAINER_FIELDS = (
    'key_point_recall',
    'claim_support_rate',
    'answer_relevance',
    'retrieval_recall_at_k',
    'retrieval_mrr',
    'context_noise_rate',
)
DIAGNOSTIC_EVIDENCE_FIELDS = (
    'matched_key_points',
    'missing_points',
    'wrong_points',
    'extra_points',
    'claims',
    'unsupported_claims',
    'evidence_mapping',
)
OPTIONAL_DIAGNOSTIC_EVIDENCE_FIELDS = ('contradicted_claims',)
SPECIALIZED_METRIC_FIELDS = (
    'retrieval_ndcg',
    'retrieval_precision_at_k',
    'context_relevance_avg',
)
OPTIONAL_SPECIALIZED_METRIC_FIELDS = (
    'numeric_accuracy',
    'list_set_f1',
    'contradiction_rate',
)
COMPATIBILITY_METRIC_FIELDS = (
    'answer_correctness',
    'completeness',
    'groundedness',
    'format_compliance',
    'semantic_similarity',
    'chunk_recall',
    'chunk_precision',
    'doc_recall',
    'doc_precision',
    'context_recall',
    'context_precision',
    'retrieval_hit_at_k',
)
LAYER_FIELDS = {
    'primary_scores': PRIMARY_SCORE_FIELDS,
    'core_explainers': CORE_EXPLAINER_FIELDS,
    'diagnostic_evidence': DIAGNOSTIC_EVIDENCE_FIELDS,
    'specialized_metrics': SPECIALIZED_METRIC_FIELDS,
    'compatibility_metrics': COMPATIBILITY_METRIC_FIELDS,
}
NUMERIC_FIELDS = (
    'answer_correctness',
    'answer_relevance',
    'completeness',
    'groundedness',
    'format_compliance',
    'key_point_recall',
    'key_point_precision',
    'semantic_similarity',
    'claim_support_rate',
    'unsupported_claim_rate',
    'retrieval_hit_at_k',
    'retrieval_recall_at_k',
    'retrieval_precision_at_k',
    'retrieval_mrr',
    'retrieval_ndcg',
    'context_relevance_avg',
    'context_noise_rate',
    'chunk_recall',
    'chunk_precision',
    'doc_recall',
    'doc_precision',
    'context_recall',
    'context_precision',
    'answer_quality_score',
    'retrieval_quality_score',
    'overall_score',
)


def validate_classification_inputs(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> None:
    _validate_case(case, 'eval.case')
    _validate_answer(answer)
    validate_judge_result(judge)
    _validate_identity(case, answer, judge)
    _validate_optional_layers(judge)
    _validate_trace(case, answer, trace)


def layer_keys(judge: Mapping[str, Any], layer: str) -> tuple[str, ...]:
    layers = _mapping(judge.get('metric_layers'))
    declared = layers.get(layer)
    if isinstance(declared, (list, tuple)):
        return tuple(dict.fromkeys(_text(item) for item in declared if _text(item)))
    expected = list(LAYER_FIELDS.get(layer, ()))
    if layer == 'diagnostic_evidence':
        expected.extend(OPTIONAL_DIAGNOSTIC_EVIDENCE_FIELDS)
    elif layer == 'specialized_metrics':
        expected[0:0] = OPTIONAL_SPECIALIZED_METRIC_FIELDS
    return tuple(key for key in expected if key in judge)


def layer_values(judge: Mapping[str, Any], layer: str) -> dict[str, Any]:
    return {key: judge[key] for key in layer_keys(judge, layer) if key in judge}


def compact_features(
    judge: Mapping[str, Any],
    layer: str,
    *,
    keys: tuple[str, ...] | None = None,
) -> list[str]:
    allowed = set(keys or layer_keys(judge, layer))
    features = []
    for key, value in layer_values(judge, layer).items():
        if key not in allowed or value is None:
            continue
        if isinstance(value, (list, tuple, set, Mapping)):
            features.append(f'{key}_count={len(value)}')
        else:
            features.append(f'{key}={value}')
    return features


def diagnostic_count_features(
    judge: Mapping[str, Any],
    *,
    keys: tuple[str, ...] | None = None,
) -> list[str]:
    return compact_features(judge, 'diagnostic_evidence', keys=keys)


def eval_policy(judge: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(judge.get('eval_policy'))


def policy_number(judge: Mapping[str, Any], key: str, default: float) -> float:
    try:
        number = float(eval_policy(judge).get(key))
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def score_breakdown(judge: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(judge.get('score_breakdown'))


def judge_evidence(judge: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        _evidence(
            'judge_primary_scores',
            'eval.judge_result.primary_scores',
            layer_values(judge, 'primary_scores'),
        ),
        _evidence(
            'judge_core_explainers',
            'eval.judge_result.core_explainers',
            layer_values(judge, 'core_explainers'),
        ),
        _evidence(
            'judge_diagnostic_evidence',
            'eval.judge_result.diagnostic_evidence',
            layer_values(judge, 'diagnostic_evidence'),
        ),
        _evidence(
            'judge_specialized_metrics',
            'eval.judge_result.specialized_metrics',
            layer_values(judge, 'specialized_metrics'),
        ),
        _evidence(
            'judge_compatibility_metrics',
            'eval.judge_result.compatibility_metrics',
            layer_values(judge, 'compatibility_metrics'),
        ),
        _evidence(
            'judge_score_breakdown',
            'eval.judge_result.score_breakdown',
            dict(score_breakdown(judge)),
        ),
        _evidence('judge_reason', 'eval.judge_result.reason', _text(judge.get('reason'))[:300]),
    ]


def judge_primary_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    judges = [_mapping(row.get('judge')) for row in rows]
    scored = [judge for judge in judges if judge.get('quality_label') != 'infra_failure']
    return {
        'total': len(judges),
        'score_averages': {
            key: _average(judge.get(key) for judge in scored)
            for key in ('overall_score', 'answer_quality_score', 'retrieval_quality_score')
        },
        'quality_label_counts': dict(Counter(_text(judge.get('quality_label')) for judge in judges)),
        'failure_type_counts': dict(Counter(_text(judge.get('failure_type')) for judge in judges)),
        'retrieval_failure_type_counts': dict(
            Counter(_text(judge.get('retrieval_failure_type')) for judge in judges)
        ),
        'correct_rate': round(
            sum(1 for judge in scored if judge.get('is_correct') is True) / len(scored),
            4,
        ) if scored else 0.0,
    }


def _validate_case(case: Mapping[str, Any], path: str) -> None:
    missing = [field for field in CASE_FIELDS if field not in case]
    if missing:
        raise ValueError(f'{path} missing fields: ' + ', '.join(missing))
    empty = [field for field in CASE_FIELDS if _empty(case.get(field))]
    if empty:
        raise ValueError(f'{path} empty required fields: ' + ', '.join(empty))


def _validate_answer(answer: Mapping[str, Any]) -> None:
    if answer.get('status') not in {'ok', 'failed'}:
        raise ValueError('eval.rag_answer status must be ok or failed')
    if answer.get('status') == 'ok' and not _text(answer.get('trace_id')):
        raise ValueError('eval.rag_answer trace_id is required')


def _validate_identity(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
) -> None:
    case_id = _text(case.get('id'))
    trace_id = _text(answer.get('trace_id'))
    nested_case = _mapping(judge.get('case'))
    nested_answer = _mapping(judge.get('rag_answer'))
    _validate_case(nested_case, 'eval.judge_result.case')
    if _text(judge.get('case_id')) != case_id or _text(nested_case.get('id')) != case_id:
        raise ValueError('eval.judge_result case_id must match eval.case id')
    if _text(judge.get('trace_id')) != trace_id:
        raise ValueError('eval.judge_result trace_id must match eval.rag_answer trace_id')
    if _text(nested_answer.get('trace_id')) != trace_id:
        raise ValueError('eval.judge_result nested rag_answer trace_id must match eval.rag_answer trace_id')
    if _text(nested_answer.get('case_id')) not in {'', case_id}:
        raise ValueError('eval.judge_result nested rag_answer case_id must match eval.case id')
    _validate_target_consistency(answer, judge, nested_answer)


def _validate_optional_layers(judge: Mapping[str, Any]) -> None:
    if 'metric_layers' not in judge:
        return
    layers = judge.get('metric_layers')
    if not isinstance(layers, Mapping):
        raise ValueError('eval.judge_result metric_layers must be a mapping')
    for layer, declared in layers.items():
        if not isinstance(declared, list) or not declared:
            raise ValueError(f'eval.judge_result metric layer must be a non-empty list: {layer}')
        keys = [_text(item) for item in declared]
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError(f'eval.judge_result metric layer contains invalid fields: {layer}')
        missing = [key for key in keys if key not in judge]
        if missing:
            raise ValueError('eval.judge_result declared fields missing: ' + ', '.join(missing))


def _validate_target_consistency(
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    nested_answer: Mapping[str, Any],
) -> None:
    targets = (
        ('eval.rag_answer.target', _mapping(answer.get('target'))),
        ('eval.judge_result.target', _mapping(judge.get('target'))),
        ('eval.judge_result.rag_answer.target', _mapping(nested_answer.get('target'))),
    )
    for key in ('algorithm_id', 'kb_id', 'routed_algorithm_id'):
        observed = {
            _text(target.get(key))
            for _, target in targets
            if _text(target.get(key))
        }
        if len(observed) > 1:
            sources = ', '.join(name for name, target in targets if _text(target.get(key)))
            raise ValueError(f'eval target {key} must match across: {sources}')


def _validate_trace(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> None:
    missing = [field for field in TRACE_FIELDS if field not in trace]
    if missing:
        raise ValueError('analysis.trace_summary missing fields: ' + ', '.join(missing))
    if _text(trace.get('case_id')) != _text(case.get('id')):
        raise ValueError('analysis.trace_summary case_id must match eval.case id')
    synthetic_failed_trace = (
        trace.get('trace_source') == 'analysis.synthetic_failed_answer'
        and (answer.get('status') == 'failed' or answer.get('chat_error'))
    )
    if answer.get('trace_id') and trace.get('trace_id') != answer.get('trace_id'):
        raise ValueError('analysis.trace_summary trace_id must match eval.rag_answer trace_id')
    if not answer.get('trace_id') and not synthetic_failed_trace:
        raise ValueError('eval.rag_answer trace_id is required')
    if trace.get('trace_source') not in {
        'lazyllm.get_single_trace',
        'analysis.synthetic_failed_answer',
        'analysis.trace_unavailable',
    }:
        raise ValueError('analysis.trace_summary trace_source is unsupported')


def _average(values: Any) -> float:
    numbers = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return round(sum(numbers) / len(numbers), 4) if numbers else 0.0


def _empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple, set, dict)):
        return not value
    return False
