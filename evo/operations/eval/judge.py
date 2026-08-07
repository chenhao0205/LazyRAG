from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from difflib import SequenceMatcher
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evo.llm import LazyLLMClient, parse_json_object

QualityLabel = Literal['good', 'partial', 'bad', 'infra_failure']
FailureType = Literal[
    'none',
    'wrong_answer',
    'partial_answer',
    'question_not_answered',
    'format_error',
    'hallucination',
    'infra_failure',
    'judge_contract_error',
    'dataset_contract_error',
]
RetrievalFailureType = Literal['none', 'retrieval_miss', 'retrieval_partial', 'retrieval_noise', 'not_applicable']
SCORE_KEYS = ('answer_correctness', 'answer_relevance', 'completeness', 'groundedness', 'format_compliance')
DIAGNOSTIC_SCORE_KEYS = (
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
)
OPTIONAL_DIAGNOSTIC_SCORE_KEYS = (
    'numeric_accuracy',
    'list_set_f1',
    'contradiction_rate',
)
EXPLANATION_KEYS = (
    'matched_key_points',
    'missing_points',
    'wrong_points',
    'extra_points',
    'unsupported_claims',
    'evidence_mapping',
    'claims',
)
OPTIONAL_EXPLANATION_KEYS = (
    'contradicted_claims',
)
METRIC_LAYERS = {
    'primary_scores': (
        'overall_score',
        'answer_quality_score',
        'retrieval_quality_score',
        'quality_label',
        'failure_type',
        'retrieval_failure_type',
    ),
    'core_explainers': (
        'key_point_recall',
        'claim_support_rate',
        'answer_relevance',
        'retrieval_recall_at_k',
        'retrieval_mrr',
        'context_noise_rate',
    ),
    'diagnostic_evidence': (
        'matched_key_points',
        'missing_points',
        'wrong_points',
        'extra_points',
        'claims',
        'unsupported_claims',
        'evidence_mapping',
    ),
    'specialized_metrics': (
        'retrieval_ndcg',
        'retrieval_precision_at_k',
        'context_relevance_avg',
    ),
    'compatibility_metrics': (
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
    ),
}
ANSWER_QUALITY_WEIGHTS = {
    'answer_correctness': 0.30,
    'key_point_recall': 0.20,
    'completeness': 0.15,
    'claim_support_rate': 0.15,
    'answer_relevance': 0.10,
    'semantic_similarity': 0.05,
    'format_compliance': 0.05,
}
RETRIEVAL_QUALITY_WEIGHTS = {
    'retrieval_recall_at_k': 0.35,
    'retrieval_ndcg': 0.25,
    'retrieval_mrr': 0.20,
    'context_precision': 0.10,
    'context_relevance_avg': 0.10,
}
OVERALL_SCORE_WEIGHTS = {
    'answer_quality_score': 0.80,
    'retrieval_quality_score': 0.20,
}
NUMBER = re.compile(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?')


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra='allow')

    answer_correctness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    groundedness: float = Field(ge=0.0, le=1.0)
    format_compliance: float = Field(ge=0.0, le=1.0)
    failure_type: Literal[
        'none',
        'wrong_answer',
        'partial_answer',
        'question_not_answered',
        'format_error',
        'hallucination',
    ]
    reason: str
    defect: str
    key_point_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    key_point_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    numeric_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    list_set_f1: float = Field(default=0.0, ge=0.0, le=1.0)
    claim_support_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    contradiction_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    matched_key_points: list[Any] = Field(default_factory=list)
    missing_points: list[Any] = Field(default_factory=list)
    wrong_points: list[Any] = Field(default_factory=list)
    extra_points: list[Any] = Field(default_factory=list)
    unsupported_claims: list[Any] = Field(default_factory=list)
    contradicted_claims: list[Any] = Field(default_factory=list)
    evidence_mapping: list[Any] = Field(default_factory=list)
    claims: list[Any] = Field(default_factory=list)


class JudgeResult(JudgeScores):
    model_config = ConfigDict(extra='allow')

    case_id: str
    answer_quality_score: float = Field(ge=0.0, le=1.0)
    retrieval_quality_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    retrieval_failure_type: RetrievalFailureType
    quality_label: QualityLabel
    failure_type: FailureType
    is_correct: bool


def judge_case(case: Mapping[str, Any], rag_answer: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return validate_judge_result(judge_answer(case, rag_answer, policy))
    except Exception as exc:
        return validate_judge_result(judge_contract_error(case, rag_answer, policy, str(exc)))


def judge_answer(case: Mapping[str, Any], rag_answer: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    base = {
        'case_id': str(case.get('id') or rag_answer.get('case_id') or ''),
        'case': dict(case),
        'rag_answer': dict(rag_answer),
        'trace_id': str(rag_answer.get('trace_id') or ''),
        'target': dict(rag_answer.get('target') or {}),
        'tool_errors': list(rag_answer.get('tool_errors') or []),
        'eval_policy': {
            key: policy[key]
            for key in (
                'answer_good_threshold',
                'answer_partial_threshold',
                'answer_correctness_floor',
                'groundedness_floor',
                'answer_relevance_floor',
                'key_point_recall_floor',
                'contradiction_rate_ceiling',
                'unsupported_claim_rate_ceiling',
                'claim_support_rate_floor',
                'retrieval_top_k',
                'top_k',
                'judge_schema_version',
                'rubric',
                'judge_model',
            )
            if key in policy
        },
    }
    if rag_answer.get('status') != 'ok':
        error = rag_answer.get('chat_error') if isinstance(rag_answer.get('chat_error'), Mapping) else {}
        failure = 'dataset_contract_error' if error.get('type') == 'dataset_contract_error' else 'infra_failure'
        reason = str(error.get('message') or error.get('type') or 'chat_failed')
        return _failure(base, failure, f'{error.get("type")}: {reason}' if error.get('type') else reason)

    try:
        llm_config = policy.get('judge_llm_config') if isinstance(policy.get('judge_llm_config'), Mapping) else {}
        if not isinstance(llm_config.get('evo_llm'), Mapping):
            raise ValueError('eval.policy.judge_llm_config.evo_llm missing; eval must use core model-config injection')
        client = LazyLLMClient(llm_config=llm_config, model='evo_llm')
        prompt = _prompt(case, rag_answer, policy)
        error: Exception | None = None
        for structured_attempt in range(2):
            try:
                scores = JudgeScores.model_validate(parse_json_object(client(prompt, stream=False)))
                break
            except Exception as exc:
                error = exc
                if structured_attempt == 0:
                    prompt += '\nThe previous response was invalid. Return exactly one complete JSON object.'
        else:
            raise ValueError(f'judge did not return valid scores: {error}') from error
    except Exception as exc:
        return judge_contract_error(case, rag_answer, policy, str(exc))

    diagnostics = _diagnostics(case, rag_answer, scores, policy)
    score_payload = _score_payload(scores)
    score_payload.update(diagnostics)

    ref_chunks = {_id_tail(item) for item in _ids(case.get('reference_chunk_ids'))}
    got_chunks = {_id_tail(item) for item in _ids(rag_answer.get('chunk_ids'))}
    ref_docs = {_id_tail(item) for item in _ids(case.get('reference_doc_ids'))}
    got_docs = {_id_tail(item) for item in _ids(rag_answer.get('doc_ids'))}
    chunk_recall, chunk_precision = _overlap(ref_chunks, got_chunks)
    doc_recall, doc_precision = _overlap(ref_docs, got_docs)
    recall = chunk_recall if ref_chunks and got_chunks else doc_recall
    precision = chunk_precision if ref_chunks and got_chunks else doc_precision
    if not ref_chunks and not ref_docs:
        retrieval_failure, recall, precision = 'not_applicable', 1.0, 1.0
    elif recall == 0.0:
        retrieval_failure = 'retrieval_miss'
    elif recall < 1.0:
        retrieval_failure = 'retrieval_partial'
    elif precision < 0.5:
        retrieval_failure = 'retrieval_noise'
    else:
        retrieval_failure = 'none'

    contradiction_penalty = 0.20 * _payload_float(score_payload, 'contradiction_rate', 0.0)
    answer_quality = _score(
        0.30 * scores.answer_correctness
        + 0.20 * _payload_float(score_payload, 'key_point_recall', scores.answer_correctness)
        + 0.15 * scores.completeness
        + 0.15 * _payload_float(score_payload, 'claim_support_rate', scores.groundedness)
        + 0.10 * scores.answer_relevance
        + 0.05 * _payload_float(score_payload, 'semantic_similarity', scores.answer_correctness)
        + 0.05 * scores.format_compliance
        - contradiction_penalty
    )
    retrieval_quality = _score(
        0.35 * _payload_float(score_payload, 'retrieval_recall_at_k', recall)
        + 0.25 * _payload_float(score_payload, 'retrieval_ndcg', recall)
        + 0.20 * _payload_float(score_payload, 'retrieval_mrr', recall)
        + 0.10 * precision
        + 0.10 * _payload_float(score_payload, 'context_relevance_avg', recall)
    )
    overall = answer_quality if retrieval_failure == 'not_applicable' else _score(
        0.80 * answer_quality + 0.20 * retrieval_quality
    )
    failure = scores.failure_type
    if failure == 'none':
        failure = (
            'question_not_answered' if scores.answer_relevance < 0.4 else
            'hallucination' if scores.groundedness < 0.4 else
            'wrong_answer' if scores.answer_correctness < 0.5 else
            'partial_answer' if scores.completeness < 0.6 else
            'none'
        )
    if failure == 'none' and case.get('key_points') and _payload_float(score_payload, 'key_point_recall', 0.0) < _policy_float(
        policy, 'key_point_recall_floor', 0.8
    ):
        failure = 'partial_answer'
    if failure == 'none' and _payload_float(score_payload, 'contradiction_rate', 0.0) > _policy_float(
        policy, 'contradiction_rate_ceiling', 0.0
    ):
        failure = 'hallucination'
    unsupported_claim_rate = _payload_float(score_payload, 'unsupported_claim_rate', 0.0)
    claim_support_rate = _payload_float(score_payload, 'claim_support_rate', 1.0)
    has_claims = bool(score_payload.get('claims'))
    if failure == 'none' and has_claims and (
        unsupported_claim_rate > _policy_float(policy, 'unsupported_claim_rate_ceiling', 0.0)
        or claim_support_rate < _policy_float(policy, 'claim_support_rate_floor', 0.8)
    ):
        failure = 'partial_answer'
    if scores.format_compliance < 1.0:
        failure = 'format_error'

    gates_ok = (
        failure == 'none'
        and scores.answer_correctness >= _policy_float(policy, 'answer_correctness_floor', 0.6)
        and scores.groundedness >= _policy_float(policy, 'groundedness_floor', 0.6)
        and scores.answer_relevance >= _policy_float(policy, 'answer_relevance_floor', 0.6)
        and (
            not case.get('key_points')
            or _payload_float(score_payload, 'key_point_recall', 0.0) >= _policy_float(
                policy, 'key_point_recall_floor', 0.8
            )
        )
    )
    if gates_ok and overall >= _policy_float(policy, 'answer_good_threshold', 0.8):
        label, failure = 'good', 'none'
    elif overall >= _policy_float(policy, 'answer_partial_threshold', 0.5):
        label, failure = 'partial', failure if failure != 'none' else 'partial_answer'
    else:
        label, failure = 'bad', failure if failure != 'none' else 'wrong_answer'
    if retrieval_failure not in {'none', 'not_applicable'} and label == 'good':
        label = 'partial'
    if retrieval_failure not in {'none', 'not_applicable'} and failure == 'none':
        failure = 'partial_answer'

    return base | score_payload | {
        'chunk_recall': chunk_recall,
        'chunk_precision': chunk_precision,
        'doc_recall': doc_recall,
        'doc_precision': doc_precision,
        'context_recall': recall,
        'context_precision': precision,
        'retrieval_quality_score': retrieval_quality,
        'retrieval_failure_type': retrieval_failure,
        'answer_quality_score': answer_quality,
        'overall_score': overall,
        'metric_layers': _metric_layers(score_payload),
        'score_breakdown': _score_breakdown(retrieval_failure, score_payload),
        'quality_label': label,
        'failure_type': failure,
        'is_correct': label == 'good',
    }


def judge_contract_error(case: Mapping[str, Any], rag_answer: Mapping[str, Any], policy: Mapping[str, Any], reason: str
                         ) -> dict[str, Any]:
    base = {
        'case_id': str(case.get('id') or rag_answer.get('case_id') or ''),
        'case': dict(case),
        'rag_answer': dict(rag_answer),
        'trace_id': str(rag_answer.get('trace_id') or ''),
        'target': dict(rag_answer.get('target') or {}),
        'tool_errors': list(rag_answer.get('tool_errors') or []),
        'eval_policy': {'judge_schema_version': policy.get('judge_schema_version', 'v1')},
    }
    return _failure(base, 'judge_contract_error', reason)


def validate_judge_result(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        JudgeResult.model_validate(value)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return dict(value)


def _failure(base: Mapping[str, Any], failure_type: FailureType, reason: str) -> dict[str, Any]:
    return dict(base) | {
        **{key: 0.0 for key in SCORE_KEYS},
        **{key: 0.0 for key in DIAGNOSTIC_SCORE_KEYS},
        **{key: [] for key in EXPLANATION_KEYS},
        **{key: 0.0 for key in ('chunk_recall', 'chunk_precision', 'doc_recall', 'doc_precision',
                                'context_recall', 'context_precision')},
        'answer_quality_score': 0.0,
        'retrieval_quality_score': 0.0,
        'overall_score': 0.0,
        'metric_layers': _metric_layers({}),
        'score_breakdown': _score_breakdown('not_applicable', {}),
        'retrieval_failure_type': 'not_applicable',
        'quality_label': 'infra_failure',
        'failure_type': failure_type,
        'is_correct': False,
        'reason': reason,
        'defect': failure_type,
    }


def _prompt(case: Mapping[str, Any], rag_answer: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    reference_contexts = _evidence_contexts(
        case.get('reference_context'),
        'reference_context',
        case.get('reference_chunk_ids'),
    )
    retrieved_contexts = _evidence_contexts(
        rag_answer.get('contexts'),
        'retrieved_context',
        rag_answer.get('chunk_ids'),
    )
    payload = {
        'question': case.get('question'),
        'reference_answer': case.get('answer'),
        'key_points': case.get('key_points'),
        'grading_guidance': case.get('grading_guidance'),
        'reference_context': _prompt_contexts(reference_contexts),
        'rag_answer': rag_answer.get('answer'),
        'retrieved_contexts': _prompt_contexts(retrieved_contexts),
    }
    if case.get('answer_type'):
        payload['answer_type'] = case.get('answer_type')
    if case.get('forbidden_claims'):
        payload['forbidden_claims'] = case.get('forbidden_claims')
    return (
        'Judge one RAG answer. Return one JSON object only, no markdown. '
        f'Scores must be floats from 0 to 1 with keys: {", ".join(SCORE_KEYS)}. '
        'When possible, also return diagnostic scores: key_point_recall, key_point_precision, '
        'semantic_similarity, numeric_accuracy, list_set_f1, claim_support_rate, '
        'unsupported_claim_rate. Return contradiction_rate only when forbidden_claims are provided '
        'or a contradiction is explicitly detected. '
        'Also return arrays for matched_key_points, missing_points, wrong_points, extra_points, '
        'unsupported_claims, contradicted_claims, evidence_mapping, claims. '
        'Each evidence_mapping item should use claim, reference_support, retrieved_support, derivation. '
        'reference_support and retrieved_support are objects with evidence_chunk_id and score, or null. '
        'derivation must be llm. Do not include full evidence text. '
        'Return failure_type, reason, defect. failure_type must be one of none, wrong_answer, '
        'partial_answer, question_not_answered, format_error, hallucination. '
        'First compare rag_answer with key_points and reference_answer. Then check every factual claim '
        'against reference_context and retrieved_contexts. Judge correctness against reference_answer and '
        'grading_guidance; judge groundedness against reference_context and retrieved_contexts.\n'
        f'rubric: {policy.get("rubric") or "Use the provided references and grading guidance."}\n'
        f'case_json: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}'
    )


def _ids(value: Any) -> set[str]:
    items = [value] if isinstance(value, str) else list(value or [])
    return {str(item).strip() for item in items if str(item or '').strip()}


def _overlap(reference: set[str], retrieved: set[str]) -> tuple[float, float]:
    if not reference:
        return 0.0, 0.0
    hit = len(reference & retrieved)
    return _score(hit / len(reference)), _score(hit / len(retrieved)) if retrieved else 0.0


def _score(value: float) -> float:
    if not math.isfinite(float(value)):
        raise ValueError('score must be finite')
    return round(max(0.0, min(1.0, float(value))), 4)


def _payload_float(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key) if key in payload else default
    if value is None:
        value = default
    return float(value)


def _policy_float(policy: Mapping[str, Any], key: str, default: float) -> float:
    value = policy.get(key) if key in policy else default
    if value in (None, ''):
        value = default
    return float(value)


def _score_payload(scores: JudgeScores) -> dict[str, Any]:
    payload = {key: float(getattr(scores, key)) for key in SCORE_KEYS}
    payload.update({
        'failure_type': scores.failure_type,
        'reason': scores.reason,
        'defect': scores.defect,
    })
    return payload


def _metric_layers(payload: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    layers = {key: list(values) for key, values in METRIC_LAYERS.items()}
    present = set(payload or {})
    optional = [key for key in OPTIONAL_DIAGNOSTIC_SCORE_KEYS if key in present]
    if optional:
        layers['specialized_metrics'] = [*optional, *layers['specialized_metrics']]
    optional_explanations = [key for key in OPTIONAL_EXPLANATION_KEYS if key in present]
    if optional_explanations:
        layers['diagnostic_evidence'] = [*layers['diagnostic_evidence'], *optional_explanations]
    return layers


def _score_breakdown(retrieval_failure_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    overall = {'answer_quality_score': 1.0} if retrieval_failure_type == 'not_applicable' \
        else dict(OVERALL_SCORE_WEIGHTS)
    penalties = {'contradiction_rate': 0.20} if 'contradiction_rate' in payload else {}
    return {
        'answer_quality_score': {
            'weights': dict(ANSWER_QUALITY_WEIGHTS),
            'penalties': penalties,
        },
        'retrieval_quality_score': {
            'weights': dict(RETRIEVAL_QUALITY_WEIGHTS),
        },
        'overall_score': {
            'weights': overall,
            'retrieval_not_applicable': retrieval_failure_type == 'not_applicable',
        },
    }


def _diagnostics(
    case: Mapping[str, Any],
    rag_answer: Mapping[str, Any],
    scores: JudgeScores,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    answer = str(rag_answer.get('answer') or '')
    reference = str(case.get('answer') or '')
    contexts = _contexts(rag_answer.get('contexts'))
    reference_contexts = _contexts(case.get('reference_context'))
    retrieved_evidence = _evidence_contexts(
        rag_answer.get('contexts'),
        'retrieved_context',
        rag_answer.get('chunk_ids'),
    )
    reference_evidence = _evidence_contexts(
        case.get('reference_context'),
        'reference_context',
        case.get('reference_chunk_ids'),
    )
    key_point_diag = _key_point_diagnostics(case, answer, scores)
    claim_diag = _claim_diagnostics(case, answer, retrieved_evidence, reference_evidence, scores)
    numeric_accuracy = _numeric_accuracy(answer, reference)
    list_set_f1 = _list_set_f1(case, answer, reference)
    retrieval = _ranked_retrieval(case, rag_answer, policy, reference_contexts)
    payload = {
        **key_point_diag,
        **claim_diag,
        **retrieval,
        'semantic_similarity': _metric(scores, 'semantic_similarity', _similarity(answer, reference)),
        'wrong_points': _list_or_default(scores.wrong_points, []),
        'extra_points': _list_or_default(scores.extra_points, []),
    }
    if numeric_accuracy is not None:
        payload['numeric_accuracy'] = _metric(scores, 'numeric_accuracy', numeric_accuracy)
    if list_set_f1 is not None:
        payload['list_set_f1'] = _metric(scores, 'list_set_f1', list_set_f1)
    return payload


def _metric(scores: JudgeScores, field: str, default: float) -> float:
    if field in scores.model_fields_set:
        return _score(float(getattr(scores, field)))
    return _score(default)


def _key_point_diagnostics(case: Mapping[str, Any], answer: str, scores: JudgeScores) -> dict[str, Any]:
    key_points = _key_points(case.get('key_points'))
    if not key_points:
        return {
            'key_point_recall': _metric(scores, 'key_point_recall', scores.answer_correctness),
            'key_point_precision': _metric(scores, 'key_point_precision', scores.answer_correctness),
            'matched_key_points': _list_or_default(scores.matched_key_points, []),
            'missing_points': _list_or_default(scores.missing_points, []),
        }
    matched, missing = [], []
    for item in key_points:
        hit = _similarity(answer, item['statement']) >= 0.55
        (matched if hit else missing).append(item)
    recall = len(matched) / len(key_points) if key_points else scores.answer_correctness
    precision = len(matched) / len(key_points) if key_points else scores.answer_correctness
    return {
        'key_point_recall': _metric(scores, 'key_point_recall', recall),
        'key_point_precision': _metric(scores, 'key_point_precision', precision),
        'matched_key_points': _list_or_default(scores.matched_key_points, matched),
        'missing_points': _list_or_default(scores.missing_points, missing),
    }


def _claim_diagnostics(
    case: Mapping[str, Any],
    answer: str,
    retrieved_contexts: list[dict[str, str]],
    reference_contexts: list[dict[str, str]],
    scores: JudgeScores,
) -> dict[str, Any]:
    include_contradiction = bool(case.get('forbidden_claims')) or 'contradiction_rate' in scores.model_fields_set \
        or bool(scores.contradicted_claims)
    if scores.claims:
        claims = scores.claims
        claim_texts = [_claim_text(claim) for claim in claims]
        claim_texts = [claim for claim in claim_texts if claim]
        llm_mapping = _normal_evidence_mapping(scores.evidence_mapping, derivation='llm')
        fallback_mapping = _fallback_evidence_mapping(claim_texts, retrieved_contexts, reference_contexts)
        if _evidence_mapping_complete(llm_mapping, claim_texts):
            evidence_mapping = llm_mapping
        else:
            evidence_mapping = fallback_mapping
        supported_claims = {
            str(item.get('claim') or '')
            for item in evidence_mapping
            if _has_support(item)
        }
        unsupported = _list_or_default(scores.unsupported_claims, [
            claim for claim in claim_texts
            if claim not in supported_claims
        ])
        support = len(supported_claims) / len(claim_texts) if claim_texts else scores.groundedness
        payload = {
            'claims': claims,
            'unsupported_claims': unsupported,
            'evidence_mapping': evidence_mapping,
            'claim_support_rate': _metric(scores, 'claim_support_rate', support),
            'unsupported_claim_rate': _metric(scores, 'unsupported_claim_rate', 1.0 - support),
        }
        if include_contradiction:
            payload['contradiction_rate'] = _metric(scores, 'contradiction_rate', 0.0)
            payload['contradicted_claims'] = _list_or_default(scores.contradicted_claims, [])
        return payload
    claims = _sentences(answer)
    supported, unsupported = [], []
    mapping = _fallback_evidence_mapping(claims, retrieved_contexts, reference_contexts)
    supported_claims = {
        str(item.get('claim') or '')
        for item in mapping
        if _has_support(item)
    }
    for claim in claims:
        if claim in supported_claims:
            supported.append(claim)
        else:
            unsupported.append(claim)
    support_rate = len(supported) / len(claims) if claims else scores.groundedness
    payload = {
        'claims': [{'text': claim, 'supported': claim in supported} for claim in claims],
        'unsupported_claims': _list_or_default(scores.unsupported_claims, unsupported),
        'evidence_mapping': _normal_evidence_mapping(scores.evidence_mapping, derivation='llm') or mapping,
        'claim_support_rate': _metric(scores, 'claim_support_rate', support_rate),
        'unsupported_claim_rate': _metric(scores, 'unsupported_claim_rate', 1.0 - support_rate),
    }
    if include_contradiction:
        payload['contradiction_rate'] = _metric(scores, 'contradiction_rate', 0.0)
        payload['contradicted_claims'] = _list_or_default(scores.contradicted_claims, [])
    return payload


def _ranked_retrieval(
    case: Mapping[str, Any],
    rag_answer: Mapping[str, Any],
    policy: Mapping[str, Any],
    reference_contexts: list[str],
) -> dict[str, float]:
    top_k = int(policy.get('retrieval_top_k') or policy.get('top_k') or 5)
    ranked = _ranked_items(rag_answer)
    ref_chunks = {_id_tail(item) for item in _ids(case.get('reference_chunk_ids'))}
    ref_docs = {_id_tail(item) for item in _ids(case.get('reference_doc_ids'))}
    relevant_flags = []
    relevance_scores = []
    for item in ranked:
        chunk_id = _id_tail(str(item.get('chunk_id') or item.get('id') or ''))
        doc_id = _id_tail(str(item.get('doc_id') or ''))
        is_relevant = bool((chunk_id and chunk_id in ref_chunks) or (doc_id and doc_id in ref_docs))
        text = str(item.get('content') or item.get('text') or '')
        context_score = max((_similarity(text, ref) for ref in reference_contexts), default=0.0)
        relevant_flags.append(is_relevant)
        relevance_scores.append(context_score)
    if not ranked:
        return {
            'retrieval_hit_at_k': 0.0,
            'retrieval_recall_at_k': 0.0,
            'retrieval_precision_at_k': 0.0,
            'retrieval_mrr': 0.0,
            'retrieval_ndcg': 0.0,
            'context_relevance_avg': 0.0,
            'context_noise_rate': 0.0,
        }
    top_flags = relevant_flags[:top_k]
    reference_total = len(ref_chunks or ref_docs) or sum(1 for flag in relevant_flags if flag) or 1
    hit_at_k = 1.0 if any(top_flags) else 0.0
    recall_at_k = min(sum(1 for flag in top_flags if flag) / reference_total, 1.0)
    precision_at_k = sum(1 for flag in top_flags if flag) / min(top_k, len(ranked))
    first_rank = next((index + 1 for index, flag in enumerate(relevant_flags) if flag), 0)
    mrr = 1.0 / first_rank if first_rank else 0.0
    ndcg = _ndcg(relevant_flags[:top_k])
    context_relevance = sum(relevance_scores[:top_k]) / min(top_k, len(relevance_scores))
    noise = sum(1 for score in relevance_scores[:top_k] if score < 0.25) / min(top_k, len(relevance_scores))
    return {
        'retrieval_hit_at_k': _score(hit_at_k),
        'retrieval_recall_at_k': _score(recall_at_k),
        'retrieval_precision_at_k': _score(precision_at_k),
        'retrieval_mrr': _score(mrr),
        'retrieval_ndcg': _score(ndcg),
        'context_relevance_avg': _score(context_relevance),
        'context_noise_rate': _score(noise),
    }


def _key_points(value: Any) -> list[dict[str, Any]]:
    values = value if isinstance(value, list | tuple) else []
    result = []
    for index, item in enumerate(values, 1):
        if isinstance(item, Mapping):
            statement = str(item.get('statement') or item.get('text') or item.get('point') or '').strip()
            key = str(item.get('id') or f'kp_{index}')
            evidence_chunk_ids = [
                str(chunk_id).strip()
                for chunk_id in item.get('evidence_chunk_ids', [])
                if str(chunk_id or '').strip()
            ] if isinstance(item.get('evidence_chunk_ids'), list) else []
        else:
            statement = str(item or '').strip()
            key, evidence_chunk_ids = f'kp_{index}', []
        if statement:
            result.append({
                'id': key,
                'statement': statement,
                'evidence_chunk_ids': evidence_chunk_ids,
            })
    return result


def _ranked_items(rag_answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = rag_answer.get('retrieved_contexts') or rag_answer.get('contexts') or []
    values = raw if isinstance(raw, list | tuple) else []
    result = []
    chunk_ids = list(_ordered_values(rag_answer.get('chunk_ids')))
    doc_ids = list(_ordered_values(rag_answer.get('doc_ids')))
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            result.append(dict(item) | {'rank': _int(item.get('rank'), index + 1)})
        else:
            result.append({
                'content': str(item or ''),
                'chunk_id': chunk_ids[index] if index < len(chunk_ids) else '',
                'doc_id': doc_ids[index] if index < len(doc_ids) else '',
                'rank': index + 1,
            })
    if not result and (chunk_ids or doc_ids):
        for index, chunk_id in enumerate(chunk_ids or doc_ids):
            result.append({
                'chunk_id': chunk_id if chunk_ids else '',
                'doc_id': doc_ids[index] if index < len(doc_ids) else '',
                'rank': index + 1,
            })
    return sorted(result, key=lambda item: _int(item.get('rank'), 0))


def _ordered_values(value: Any) -> Iterable[str]:
    items = [value] if isinstance(value, str) else list(value or [])
    for item in items:
        text = str(item or '').strip()
        if text:
            yield text


def _contexts(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        direct = str(value.get('content') or value.get('text') or value.get('context') or '').strip()
        if direct:
            return [direct]
        joined = '\n'.join(str(item).strip() for item in value.values() if str(item or '').strip())
        return [joined] if joined else []
    values = value if isinstance(value, list | tuple) else [value] if value else []
    result = []
    for item in values:
        if isinstance(item, Mapping):
            direct = str(item.get('content') or item.get('text') or item.get('context') or '').strip()
            text = direct or '\n'.join(str(value).strip() for value in item.values() if str(value or '').strip())
        else:
            text = str(item or '').strip()
        if text:
            result.append(text)
    return result


def _prompt_contexts(values: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            'chunk_id': str(item.get('chunk_id') or ''),
            'content': str(item.get('content') or ''),
        }
        for item in values
        if str(item.get('content') or '').strip()
    ]


def _evidence_contexts(value: Any, source: str, fallback_chunk_ids: Any = None) -> list[dict[str, str]]:
    result = []
    if isinstance(value, Mapping):
        direct = str(value.get('content') or value.get('text') or value.get('context') or '').strip()
        if direct:
            return [{
                'source': source,
                'chunk_id': str(value.get('chunk_id') or value.get('id') or '').strip(),
                'content': direct,
            }]
        for key, item in value.items():
            text = str(item or '').strip()
            if text:
                result.append({'source': source, 'chunk_id': str(key).strip(), 'content': text})
        return result
    values = value if isinstance(value, list | tuple) else [value] if value else []
    chunk_ids = list(_ordered_values(fallback_chunk_ids))
    for index, item in enumerate(values):
        if isinstance(item, Mapping):
            text = str(item.get('content') or item.get('text') or item.get('context') or '').strip()
            chunk_id = str(
                item.get('chunk_id')
                or item.get('id')
                or (chunk_ids[index] if index < len(chunk_ids) else '')
            ).strip()
        else:
            text = str(item or '').strip()
            chunk_id = chunk_ids[index] if index < len(chunk_ids) else ''
        if text:
            result.append({'source': source, 'chunk_id': chunk_id, 'content': text})
    return result


def _normal_evidence_mapping(value: Any, derivation: str) -> list[dict[str, Any]]:
    values = value if isinstance(value, list | tuple) else []
    result = []
    for item in values:
        if not isinstance(item, Mapping):
            continue
        claim = str(item.get('claim') or item.get('text') or '').strip()
        if not claim:
            continue
        reference_support = _normal_support(item.get('reference_support'))
        retrieved_support = _normal_support(item.get('retrieved_support'))
        if not reference_support and not retrieved_support:
            source = str(item.get('evidence_source') or item.get('source') or '').strip()
            support = _normal_support(item)
            if source == 'reference_context':
                reference_support = support
            elif source == 'retrieved_context':
                retrieved_support = support
        if reference_support or retrieved_support:
            result.append({
                'claim': claim,
                'reference_support': reference_support,
                'retrieved_support': retrieved_support,
                'derivation': derivation,
            })
    return result


def _normal_support(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    chunk_id = str(value.get('evidence_chunk_id') or value.get('chunk_id') or value.get('id') or '').strip()
    if not chunk_id or value.get('score') is None:
        return None
    try:
        score = _score(float(value.get('score')))
    except (TypeError, ValueError):
        return None
    return {'evidence_chunk_id': chunk_id, 'score': score}


def _fallback_evidence_mapping(
    claims: list[str],
    retrieved_contexts: list[dict[str, str]],
    reference_contexts: list[dict[str, str]],
) -> list[dict[str, Any]]:
    result = []
    for claim in claims:
        entry = {
            'claim': claim,
            'reference_support': _best_support(claim, reference_contexts),
            'retrieved_support': _best_support(claim, retrieved_contexts),
            'derivation': 'fallback',
        }
        result.append(entry)
    return result


def _best_support(claim: str, contexts: list[dict[str, str]]) -> dict[str, Any] | None:
    best = max(
        (
            (_similarity(claim, item.get('content', '')), item)
            for item in contexts
            if item.get('content') and item.get('chunk_id')
        ),
        key=lambda pair: pair[0],
        default=(0.0, {}),
    )
    if best[0] < 0.45 or not best[1]:
        return None
    return {
        'evidence_chunk_id': str(best[1].get('chunk_id') or ''),
        'score': _score(best[0]),
    }


def _claim_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get('text') or value.get('claim') or value.get('statement') or '').strip()
    return str(value or '').strip()


def _has_support(value: Mapping[str, Any]) -> bool:
    return isinstance(value.get('reference_support'), Mapping) or isinstance(value.get('retrieved_support'), Mapping)


def _evidence_mapping_complete(mapping: list[dict[str, Any]], claims: list[str]) -> bool:
    if not claims:
        return True
    mapped = {str(item.get('claim') or '') for item in mapping if _has_support(item)}
    return all(claim in mapped for claim in claims)


def _sentences(text: str) -> list[str]:
    chunks = re.split(r'(?<=[。！？.!?])\s+|[\n\r]+', text)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) >= 8]


def _numeric_accuracy(answer: str, reference: str) -> float | None:
    answer_numbers = _numbers(answer)
    reference_numbers = _numbers(reference)
    if not reference_numbers:
        return None
    hits = sum(1 for number in reference_numbers if any(abs(candidate - number) <= 1e-9 for candidate in answer_numbers))
    return hits / len(reference_numbers)


def _numbers(text: str) -> list[float]:
    return [float(match.group(0)) for match in NUMBER.finditer(text)]


def _list_set_f1(case: Mapping[str, Any], answer: str, reference: str) -> float | None:
    if str(case.get('answer_type') or '').strip().lower() not in {'list', 'table'}:
        return None
    ref_items = _split_items(reference)
    answer_items = _split_items(answer)
    if not ref_items:
        return 1.0
    hit = sum(1 for ref in ref_items if any(_similarity(ref, item) >= 0.7 for item in answer_items))
    precision = hit / len(answer_items) if answer_items else 0.0
    recall = hit / len(ref_items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _split_items(text: str) -> list[str]:
    return [item.strip() for item in re.split(r'[;；,，\n\r]+', text) if item.strip()]


def _similarity(left: str, right: str) -> float:
    left_norm, right_norm = _norm(left), _norm(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _norm(value: str) -> str:
    return re.sub(r'\s+', '', str(value or '').lower())


def _ndcg(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    dcg = sum((1.0 if flag else 0.0) / math.log2(index + 2) for index, flag in enumerate(flags))
    ideal_count = sum(1 for flag in flags if flag)
    if ideal_count == 0:
        return 0.0
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal


def _list_or_default(value: Any, default: list[Any]) -> list[Any]:
    return list(value) if isinstance(value, list) and value else default


def _id_tail(value: str) -> str:
    return str(value or '').rsplit(':', 1)[-1]


def _float(value: Any, default: float | None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
