from __future__ import annotations

import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evo.operations.analysis.classify import classify_case
from evo.operations.analysis.diagnostic_sidecar import (
    build_diagnostic_plan,
    finalize_diagnostic_sidecar,
)
from evo.operations.analysis.judge import (
    COMPATIBILITY_METRIC_FIELDS,
    CORE_EXPLAINER_FIELDS,
    DIAGNOSTIC_EVIDENCE_FIELDS,
    NUMERIC_FIELDS,
    OPTIONAL_DIAGNOSTIC_EVIDENCE_FIELDS,
    OPTIONAL_SPECIALIZED_METRIC_FIELDS,
    PRIMARY_SCORE_FIELDS,
    SPECIALIZED_METRIC_FIELDS,
    judge_primary_summary,
    validate_classification_inputs,
)
from evo.operations.analysis.confirmation import (
    analysis_owned_probe_handlers,
    register_probe_handler,
    registered_probe_handlers,
    run_confirmation_probe_batch,
    run_registered_probe,
)
from evo.operations.analysis.repair_groups import FUNCTION_BLOCKS, build_repair_group_queue
from evo.operations.analysis.review import (
    build_evidence_packet as _build_evidence_packet,
    build_semantic_review_prompt,
    normalize_review_packages,
    run_semantic_review,
)
from evo.operations.analysis.summary import trace_quality
from evo.operations.analysis.trace_summary import (
    _retrieval_artifacts,
    _semantic_data,
    build_trace_summary,
)
from evo.operations.public_contracts import DatasetCase
from evo import artifacts as analysis_artifacts
from evo.artifact_runtime import OperationContext
from evo.operations import operation as operation_module
from evo.operations.operation import (
    analysis_summary_operation,
    diagnostic_plan_operation,
    diagnostic_sidecar_operation,
    evidence_packet_operation,
    evo_operations,
    probe_batch_operation,
    semantic_review_batch_operation,
    trace_clusters_operation,
)


def build_diagnostic_sidecar(
    case,
    answer,
    judge,
    trace,
    *,
    semantic_reviews=None,
    probe_observations=None,
    max_review_calls=2,
):
    plan = build_diagnostic_plan(
        case,
        answer,
        judge,
        trace,
        max_review_calls=max_review_calls,
    )
    return {
        **plan,
        **finalize_diagnostic_sidecar(
            plan,
            semantic_reviews=semantic_reviews,
            probe_observations=probe_observations,
        ),
    }


def build_evidence_packet(
    case,
    answer,
    judge,
    trace,
    *,
    review_packages=None,
    diagnostic_plan=None,
):
    plan = diagnostic_plan or build_diagnostic_plan(case, answer, judge, trace)
    return _build_evidence_packet(
        case,
        answer,
        judge,
        trace,
        review_packages=review_packages,
        diagnostic_plan=plan,
    )


def _case(**overrides):
    value = {
        'answer': 'Paris',
        'difficulty': 'easy',
        'grading_guidance': 'Must answer Paris.',
        'id': 'case-1',
        'key_points': [
            {
                'id': 'kp-1',
                'statement': 'Paris is the capital',
                'weight': 1.0,
                'required': True,
                'acceptable_variants': [],
                'evidence_chunk_ids': ['chunk-gold'],
            }
        ],
        'question': 'What is the capital of France?',
        'question_type': 'factoid',
        'reference_chunk_ids': ['chunk-gold'],
        'reference_context': ['France capital reference'],
        'reference_doc': 'doc-gold',
        'reference_doc_ids': ['doc-gold'],
        'source_preparation': {'source': 'unit-test'},
    }
    value.update(overrides)
    return value


def _answer(**overrides):
    value = {
        'case_id': 'case-1',
        'answer': 'Lyon',
        'status': 'ok',
        'trace_id': 'trace-1',
        'contexts': [
            {
                'chunk_id': 'chunk-other',
                'doc_id': 'doc-other',
                'content': 'Unrelated context.',
                'rank': 1,
            }
        ],
        'doc_ids': ['doc-other'],
        'chunk_ids': ['chunk-other'],
        'target': {'algorithm_id': 'algo-1', 'kb_id': 'kb-1'},
        'tool_errors': [],
    }
    value.update(overrides)
    return value


def _key_point(point_id='kp-1', statement='Paris is the capital', **overrides):
    value = {
        'id': point_id,
        'statement': statement,
        'weight': 1.0,
        'required': True,
        'acceptable_variants': [],
    }
    value.update(overrides)
    return value


def _eval_policy(**overrides):
    value = {
        'answer_good_threshold': 0.8,
        'answer_partial_threshold': 0.5,
        'answer_correctness_floor': 0.6,
        'groundedness_floor': 0.6,
        'answer_relevance_floor': 0.6,
        'key_point_recall_floor': 0.8,
        'contradiction_rate_ceiling': 0.0,
        'retrieval_top_k': 5,
        'top_k': 5,
        'rubric': 'Use the provided references and grading guidance.',
        'judge_model': 'evo_llm',
    }
    value.update(overrides)
    return value


def _metric_layers(*, sparse=False):
    diagnostic_fields = DIAGNOSTIC_EVIDENCE_FIELDS
    if not sparse:
        diagnostic_fields = (*diagnostic_fields, *OPTIONAL_DIAGNOSTIC_EVIDENCE_FIELDS)
    return {
        'primary_scores': list(PRIMARY_SCORE_FIELDS),
        'core_explainers': list(CORE_EXPLAINER_FIELDS),
        'diagnostic_evidence': list(diagnostic_fields),
        'specialized_metrics': list(
            SPECIALIZED_METRIC_FIELDS
            if sparse
            else (*OPTIONAL_SPECIALIZED_METRIC_FIELDS, *SPECIALIZED_METRIC_FIELDS)
        ),
        'compatibility_metrics': list(COMPATIBILITY_METRIC_FIELDS),
    }


def _score_breakdown(*, sparse=False):
    if sparse:
        return {
            'answer_quality_score': {'weights': {'answer_correctness': 0.3}, 'penalties': {}},
            'retrieval_quality_score': {'weights': {'retrieval_mrr': 0.2}},
            'overall_score': {
                'weights': {'answer_quality_score': 0.8, 'retrieval_quality_score': 0.2},
                'retrieval_not_applicable': False,
            },
        }
    return {
        'answer_quality_score': {
            'weights': {
                'answer_correctness': 0.3,
                'key_point_recall': 0.2,
                'completeness': 0.15,
                'claim_support_rate': 0.15,
                'answer_relevance': 0.1,
                'semantic_similarity': 0.05,
                'format_compliance': 0.05,
            },
            'penalties': {'contradiction_rate': 0.2},
        },
        'retrieval_quality_score': {
            'weights': {
                'retrieval_recall_at_k': 0.35,
                'retrieval_ndcg': 0.25,
                'retrieval_mrr': 0.2,
                'context_precision': 0.1,
                'context_relevance_avg': 0.1,
            },
        },
        'overall_score': {
            'weights': {'answer_quality_score': 0.8, 'retrieval_quality_score': 0.2},
            'retrieval_not_applicable': False,
        },
    }


def _zero_metrics():
    return {
        key: 0.0
        for key in (*NUMERIC_FIELDS, *OPTIONAL_SPECIALIZED_METRIC_FIELDS)
    }


def _judge(case=None, answer=None, **overrides):
    case = case or _case()
    answer = answer or _answer()
    nested_case = dict(case) | {
        'key_points': [_key_point(evidence_chunk_ids=['chunk-gold'])],
    }
    value = {
        'case_id': case['id'],
        'trace_id': answer['trace_id'],
        'case': nested_case,
        'rag_answer': dict(answer),
        'target': {'algorithm_id': 'algo-1', 'kb_id': 'kb-1'},
        'tool_errors': [],
        'eval_policy': _eval_policy(),
        'answer_correctness': 0.0,
        'answer_relevance': 0.4,
        'completeness': 0.2,
        'groundedness': 0.2,
        'format_compliance': 1.0,
        'key_point_recall': 0.0,
        'key_point_precision': 0.0,
        'semantic_similarity': 0.1,
        'numeric_accuracy': 1.0,
        'list_set_f1': 1.0,
        'claim_support_rate': 0.2,
        'unsupported_claim_rate': 0.8,
        'contradiction_rate': 0.0,
        'retrieval_hit_at_k': 0.0,
        'retrieval_recall_at_k': 0.0,
        'retrieval_precision_at_k': 0.5,
        'retrieval_mrr': 0.0,
        'retrieval_ndcg': 0.0,
        'context_relevance_avg': 0.2,
        'context_noise_rate': 0.5,
        'matched_key_points': [],
        'missing_points': [_key_point()],
        'wrong_points': [],
        'extra_points': [],
        'claims': [{'text': 'Lyon is the capital of France.', 'supported': False}],
        'unsupported_claims': [{'text': 'Lyon is the capital of France.'}],
        'contradicted_claims': [],
        'evidence_mapping': [],
        'chunk_recall': 0.0,
        'chunk_precision': 0.5,
        'doc_recall': 0.0,
        'doc_precision': 0.5,
        'context_recall': 0.0,
        'context_precision': 0.5,
        'answer_quality_score': 0.1,
        'retrieval_quality_score': 0.1,
        'overall_score': 0.1,
        'retrieval_failure_type': 'retrieval_miss',
        'quality_label': 'bad',
        'failure_type': 'wrong_answer',
        'is_correct': False,
        'reason': 'gold document was not retrieved',
        'defect': 'wrong_answer',
        'metric_layers': _metric_layers(),
        'score_breakdown': _score_breakdown(),
    }
    value.update(overrides)
    return deepcopy(value)


def _trace(**overrides):
    value = {
        'case_id': 'case-1',
        'trace_id': 'trace-1',
        'trace_source': 'lazyllm.get_single_trace',
        'route_signature': 'retrieve>llm_generate',
        'tree_text': '{retrieve{llm_generate}}',
        'stage_sequence': ['retrieve', 'llm_generate'],
        'diagnostic_stage_sequence': ['retrieve', 'llm_generate'],
        'edges': [{'source': 'retrieve-1', 'target': 'llm-1'}],
        'critical_path': ['retrieve', 'llm_generate'],
        'bottleneck_stage': 'retrieve',
        'stages': [
            {'id': 'retrieve-1', 'stage': 'retrieve', 'status': 'ok'},
            {'id': 'llm-1', 'stage': 'llm_generate', 'status': 'ok'},
        ],
        'stage_counts': {'retrieve': 1, 'llm_generate': 1},
        'latency_by_stage': {'retrieve': 12.0, 'llm_generate': 20.0},
        'error_stages': [],
        'retrieval_steps': [
            {
                'id': 'retrieve-1',
                'stage': 'retrieve',
                'doc_ids': ['doc-other'],
                'chunk_ids': ['chunk-other'],
            }
        ],
        'retrieved_doc_ids': ['doc-other'],
        'retrieved_chunk_ids': ['chunk-other'],
        'final_context_doc_ids': ['doc-other'],
        'final_context_chunk_ids': ['chunk-other'],
        'semantic_metric_keys': ['doc_ids', 'chunk_ids'],
        'features': {'node_count': 2.0, 'retrieved_doc_count': 1.0},
    }
    value.update(overrides)
    return value


def _partial_judge(case, answer):
    return _judge(
        case,
        answer,
        retrieval_failure_type='none',
        failure_type='partial_answer',
        quality_label='partial',
        reason='one required answer point is missing',
        defect='partial_answer',
        answer_quality_score=0.84,
        retrieval_quality_score=0.9,
        overall_score=0.852,
        answer_correctness=0.9,
        answer_relevance=1.0,
        completeness=0.8,
        groundedness=0.8,
        key_point_recall=0.5,
        key_point_precision=0.5,
        semantic_similarity=0.72,
        claim_support_rate=0.8,
        unsupported_claim_rate=0.2,
        retrieval_hit_at_k=1.0,
        retrieval_recall_at_k=1.0,
        retrieval_mrr=1.0,
        retrieval_ndcg=1.0,
        context_relevance_avg=0.7,
        context_noise_rate=0.2,
        chunk_recall=1.0,
        doc_recall=1.0,
        context_recall=1.0,
        matched_key_points=[
            {
                'id': 'kp-1',
                'statement': 'Paris is the capital',
                'weight': 1.0,
                'required': True,
                'acceptable_variants': [],
            }
        ],
        missing_points=[
            {
                'id': 'kp-2',
                'statement': 'Explain the answer',
                'weight': 1.0,
                'required': True,
                'acceptable_variants': [],
            }
        ],
        claims=[{'text': 'Paris is the capital of France.', 'supported': True}],
        unsupported_claims=[],
        evidence_mapping=[
            {
                'claim': 'Paris is the capital of France.',
                'evidence': 'France capital reference',
                'score': 0.82,
            }
        ],
    )


def test_classify_case_keeps_retrieval_miss_contract():
    row = classify_case(_case(), _answer(), _judge(), _trace())

    assert row['case_id'] == 'case-1'
    assert row['trace_id'] == 'trace-1'
    assert row['issue_category'] == 'retrieval'
    assert row['issue_type'] == 'reference_document_missing'
    assert row['affected_block'] == 'retrieval'
    assert row['failure_mode'] == 'reference_document_missing'
    assert row['confidence'] == 'high'
    assert row['actionable'] is True
    assert row['pending_analysis'] is False
    assert row['diagnosis_features'] == [
        'retrieval_failure_type=retrieval_miss',
        'doc_recall=0.0',
        'chunk_recall=0.0',
        'doc_precision=0.5',
        'chunk_precision=0.5',
        'retrieval_recall_at_k=0.0',
        'retrieval_mrr=0.0',
        'context_noise_rate=0.5',
        'retrieval_ndcg=0.0',
        'retrieval_precision_at_k=0.5',
        'context_relevance_avg=0.2',
    ]
    assert row['trace_evidence'][-1]['observed_value'] == {'doc_ids': [], 'chunk_ids': []}


def test_validate_classification_inputs_rejects_trace_mismatch():
    with pytest.raises(ValueError, match='trace_id must match'):
        validate_classification_inputs(
            _case(),
            _answer(trace_id='trace-answer'),
            _judge(),
            _trace(trace_id='trace-summary'),
        )


def test_contract_allows_empty_source_preparation():
    case = _case(source_preparation={})

    validate_classification_inputs(case, _answer(), _judge(case, _answer()), _trace())


def test_contract_accepts_upstream_case_without_analysis_only_metadata():
    case = _case()
    answer = _answer()

    for field in ('difficulty_rationale', 'reasoning_steps', 'source_message_id', 'type_rationale'):
        assert field not in case

    validate_classification_inputs(case, answer, _judge(case, answer), _trace())


def test_contract_accepts_pr4_legacy_case_without_key_points():
    case = _case()
    answer = _answer()
    del case['key_points']

    validate_classification_inputs(case, answer, _judge(_case(), answer), _trace())


def test_contract_accepts_pr4_nested_case_without_key_points():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    del judge['case']['key_points']

    validate_classification_inputs(case, answer, judge, _trace())


def test_public_dataset_contract_preserves_pr4_diagnostic_fields():
    key_points = [{
        'id': 'kp-1',
        'statement': 'Paris is the capital of France.',
        'evidence_chunk_ids': ['chunk-gold'],
    }]
    normalized = DatasetCase.model_validate({
        'case_id': 'case-pr4',
        'source': 'imported_csv',
        'answer': 'Paris',
        'difficulty': 'easy',
        'difficulty_rationale': '',
        'grading_guidance': 'Must answer Paris.',
        'original_id': 'source-1',
        'question': 'What is the capital of France?',
        'question_type': 'single_hop',
        'reasoning_steps': [],
        'reference_chunk_ids': ['chunk-gold'],
        'reference_context': ['Paris is the capital of France.'],
        'reference_doc': ['France'],
        'reference_doc_ids': ['doc-france'],
        'source_message_id': '',
        'source_preparation': {},
        'type_rationale': '',
        'key_points': key_points,
        'forbidden_claims': ['Paris is in Germany.'],
    }).model_dump(mode='json')

    assert normalized['key_points'] == key_points
    assert normalized['forbidden_claims'] == ['Paris is in Germany.']


def test_analysis_accepts_pr4_eval_judge_optional_fields_and_sparse_policy():
    case = _case(
        id='case-pr4',
        question='What is the supported fact?',
        answer='The supported fact is alpha.',
        question_type='single_hop',
        reference_doc_ids=[],
        reference_doc='',
        reference_chunk_ids=['chunk-a'],
        reference_context=['The supported fact is alpha.'],
    )
    answer = _answer(
        case_id='case-pr4',
        answer='The supported fact is alpha.',
        contexts=['The supported fact is alpha.'],
        chunk_ids=['chunk-a'],
        doc_ids=[],
        trace_id='trace-pr4',
    )
    judge = _judge(
        case,
        answer,
        eval_policy={'retrieval_top_k': 5},
        answer_correctness=0.9,
        answer_relevance=1.0,
        completeness=0.8,
        groundedness=0.8,
        format_compliance=1.0,
        key_point_recall=0.9,
        key_point_precision=0.9,
        semantic_similarity=0.9,
        claim_support_rate=0.9,
        unsupported_claim_rate=0.1,
        retrieval_hit_at_k=1.0,
        retrieval_recall_at_k=1.0,
        retrieval_precision_at_k=1.0,
        retrieval_mrr=1.0,
        retrieval_ndcg=1.0,
        context_relevance_avg=1.0,
        context_noise_rate=0.0,
        chunk_recall=1.0,
        chunk_precision=1.0,
        doc_recall=0.0,
        doc_precision=0.0,
        context_recall=1.0,
        context_precision=1.0,
        answer_quality_score=0.9,
        retrieval_quality_score=1.0,
        overall_score=0.92,
        matched_key_points=[],
        missing_points=[],
        wrong_points=[],
        extra_points=[],
        claims=[{'text': 'The supported fact is alpha.', 'supported': True}],
        unsupported_claims=[],
        evidence_mapping=[],
        retrieval_failure_type='none',
        quality_label='good',
        failure_type='none',
        is_correct=True,
        reason='PR4-style judge output with optional diagnostics omitted',
        defect='',
        metric_layers=_metric_layers(sparse=True),
        score_breakdown=_score_breakdown(sparse=True),
    )
    for optional in ('numeric_accuracy', 'list_set_f1', 'contradiction_rate', 'contradicted_claims'):
        judge.pop(optional, None)
    trace = _trace(
        case_id='case-pr4',
        trace_id='trace-pr4',
        retrieved_doc_ids=[],
        retrieved_chunk_ids=['chunk-a'],
        final_context_doc_ids=[],
        final_context_chunk_ids=['chunk-a'],
    )

    row = classify_case(case, answer, judge, trace)
    sidecar = build_diagnostic_sidecar(case, answer, judge, trace)

    assert row['issue_type'] == 'correct'
    assert row['judge']['eval_policy'] == {'retrieval_top_k': 5}
    assert sidecar['diagnosis_targets'] == []
    assert sidecar['checks']['target_gate_status'] == 'not_required'
    assert sidecar['checks']['judge_interface_unchanged'] is True


def test_analysis_accepts_pr4_string_claim_targets():
    case = _case()
    answer = _answer()
    judge = _judge(
        case,
        answer,
        missing_points=[],
        unsupported_claims=['unsupported statement'],
        contradicted_claims=['forbidden statement'],
        claims=[],
    )
    sidecar = build_diagnostic_sidecar(case, answer, judge, _trace())

    assert [target['target_type'] for target in sidecar['diagnosis_targets']] == [
        'unsupported_claim',
        'contradicted_claim',
    ]
    assert [target['statement'] for target in sidecar['diagnosis_targets']] == [
        'unsupported statement',
        'forbidden statement',
    ]


def test_analysis_accepts_legacy_minimal_judge_contract_as_degraded():
    case = _case()
    answer = _answer()
    enriched = _judge(case, answer)
    upstream_fields = {
        'case_id', 'trace_id', 'case', 'rag_answer', 'target', 'tool_errors',
        'eval_policy', 'answer_correctness', 'answer_relevance', 'completeness',
        'groundedness', 'format_compliance', 'chunk_recall', 'chunk_precision',
        'doc_recall', 'doc_precision', 'context_recall', 'context_precision',
        'answer_quality_score', 'retrieval_quality_score', 'overall_score',
        'retrieval_failure_type', 'quality_label', 'failure_type', 'is_correct',
        'reason', 'defect',
    }
    judge = {key: value for key, value in enriched.items() if key in upstream_fields}

    row = classify_case(case, answer, judge, _trace())
    sidecar = build_diagnostic_sidecar(case, answer, judge, _trace())

    assert row['issue_type'] == 'reference_document_missing'
    assert sidecar['judge_adapter']['status'] == 'degraded'
    assert sidecar['diagnostic_result']['repair_ready'] is False


def test_contract_rejects_target_mismatch():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer, target={'algorithm_id': 'algo-other', 'kb_id': 'kb-1'})

    with pytest.raises(ValueError, match='target algorithm_id must match'):
        validate_classification_inputs(case, answer, judge, _trace())


def test_actionable_affected_blocks_are_registered():
    allowed = {'not_applicable', 'undetermined', 'runtime_infra', 'eval_contract'}
    case = _case()
    answer = _answer(answer='Paris', doc_ids=['doc-gold'], chunk_ids=['chunk-gold'])
    format_judge = _judge(
        case,
        answer,
        retrieval_failure_type='none',
        failure_type='format_error',
        quality_label='bad',
        defect='format_error',
        answer_quality_score=0.4,
        retrieval_quality_score=1.0,
        overall_score=0.4,
        format_compliance=0.0,
        retrieval_hit_at_k=1.0,
        retrieval_recall_at_k=1.0,
        retrieval_mrr=1.0,
        retrieval_ndcg=1.0,
        context_noise_rate=0.0,
        context_recall=1.0,
        chunk_recall=1.0,
        doc_recall=1.0,
    )
    rows = [
        classify_case(_case(), _answer(), _judge(), _trace()),
        classify_case(
            case,
            answer,
            format_judge,
            _trace(
                retrieved_doc_ids=['doc-gold'],
                retrieved_chunk_ids=['chunk-gold'],
                final_context_doc_ids=['doc-gold'],
                final_context_chunk_ids=['chunk-gold'],
            ),
        ),
    ]

    for row in rows:
        block = row['affected_block']
        assert block in FUNCTION_BLOCKS or block in allowed


def test_repair_block_registry_is_valid_json_mapping():
    encoded = json.dumps(FUNCTION_BLOCKS, sort_keys=True)
    decoded = json.loads(encoded)

    assert 'retrieval' in decoded
    assert decoded['retrieval']['entrypoints']
    assert decoded['tracing_observability']['guard_metrics']


def test_judge_layers_drive_analysis_and_repair_evidence():
    case = _case()
    answer = _answer(
        answer='Paris, with an incomplete explanation.',
        doc_ids=['doc-gold'],
        chunk_ids=['chunk-gold'],
    )
    judge = _partial_judge(case, answer)
    trace = _trace(
        retrieved_doc_ids=['doc-gold'],
        retrieved_chunk_ids=['chunk-gold'],
        final_context_doc_ids=['doc-gold'],
        final_context_chunk_ids=['chunk-gold'],
    )

    row = classify_case(case, answer, judge, trace)

    assert row['issue_type'] == 'generation_incomplete_answer'
    assert row['actionable'] is True
    assert 'key_point_recall=0.5' in row['diagnosis_features']
    assert 'claim_support_rate=0.8' in row['diagnosis_features']
    assert 'missing_points_count=1' in row['diagnosis_features']
    assert [item['type'] for item in row['judge_evidence']] == [
        'judge_primary_scores',
        'judge_core_explainers',
        'judge_diagnostic_evidence',
        'judge_specialized_metrics',
        'judge_compatibility_metrics',
        'judge_score_breakdown',
        'judge_reason',
    ]

    row['cluster_id'] = 'cluster-1'
    group = build_repair_group_queue([row])[0]
    repair_evidence = group['evidence'][0]
    assert repair_evidence['primary_scores']['overall_score'] == 0.852
    assert repair_evidence['core_explainers']['key_point_recall'] == 0.5
    assert repair_evidence['diagnostic_evidence']['missing_points'][0]['id'] == 'kp-2'
    assert repair_evidence['specialized_metrics']['numeric_accuracy'] == 1.0
    assert repair_evidence['compatibility_metrics']['answer_correctness'] == 0.9
    assert repair_evidence['eval_policy']['judge_model'] == 'evo_llm'
    assert repair_evidence['score_breakdown']['overall_score']['weights']['answer_quality_score'] == 0.8


def test_judge_contract_rejects_missing_layered_field():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    del judge['key_point_recall']

    with pytest.raises(ValueError, match='fields missing: key_point_recall'):
        validate_classification_inputs(case, answer, judge, _trace())


def test_judge_contract_rejects_empty_metric_layer():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    judge['metric_layers']['core_explainers'] = []

    with pytest.raises(ValueError, match='metric layer must be a non-empty list: core_explainers'):
        validate_classification_inputs(case, answer, judge, _trace())


def test_failure_judge_uses_the_same_complete_contract():
    case = _case()
    answer = _answer(status='failed', answer='', chat_error={'type': 'timeout'})
    judge = _judge(
        case,
        answer,
        **_zero_metrics(),
        matched_key_points=[],
        missing_points=[],
        wrong_points=[],
        extra_points=[],
        claims=[],
        unsupported_claims=[],
        contradicted_claims=[],
        evidence_mapping=[],
        retrieval_failure_type='not_applicable',
        quality_label='infra_failure',
        failure_type='infra_failure',
        defect='infra_failure',
        reason='chat timeout',
    )

    row = classify_case(case, answer, judge, _trace())

    assert row['issue_category'] == 'runtime_infra'
    assert row['issue_type'] == 'rag_or_judge_infra_failure'
    assert row['judge']['metric_layers'] == judge['metric_layers']
    assert row['judge']['score_breakdown'] == judge['score_breakdown']


def test_failed_answer_without_trace_gets_synthetic_trace_summary():
    case = _case()
    answer = _answer(status='failed', answer='', trace_id='', chat_error={'type': 'chat_config_error'})
    judge = _judge(
        case,
        answer,
        **_zero_metrics(),
        trace_id='',
        rag_answer=answer,
        matched_key_points=[],
        missing_points=[],
        wrong_points=[],
        extra_points=[],
        claims=[],
        unsupported_claims=[],
        contradicted_claims=[],
        evidence_mapping=[],
        retrieval_failure_type='not_applicable',
        quality_label='infra_failure',
        failure_type='infra_failure',
        defect='infra_failure',
        reason='chat config error',
    )

    trace = build_trace_summary(case, answer)
    row = classify_case(case, answer, judge, trace)

    assert trace['trace_source'] == 'analysis.synthetic_failed_answer'
    assert trace['trace_id'] == 'missing_trace:case-1'
    assert row['issue_category'] == 'runtime_infra'
    assert row['issue_type'] == 'rag_or_judge_infra_failure'
    assert row['trace_summary']['error_stages'][0]['stage'] == 'tool_call'


def test_retrieval_not_applicable_with_references_is_contract_inconsistent():
    case = _case()
    answer = _answer(answer='Paris', doc_ids=['doc-gold'], chunk_ids=['chunk-gold'])
    judge = _judge(
        case,
        answer,
        retrieval_failure_type='not_applicable',
        failure_type='partial_answer',
        quality_label='partial',
        answer_quality_score=0.7,
        retrieval_quality_score=1.0,
        overall_score=0.7,
    )

    row = classify_case(case, answer, judge, _trace(
        retrieved_doc_ids=['doc-gold'],
        retrieved_chunk_ids=['chunk-gold'],
        final_context_doc_ids=['doc-gold'],
        final_context_chunk_ids=['chunk-gold'],
    ))

    assert row['issue_category'] == 'contract'
    assert row['issue_type'] == 'judge_contract_inconsistent'
    assert 'retrieval_failure_type=not_applicable conflicts with reference ids' in row['diagnosis_features']


def test_trace_quality_counts_incomplete_trace_summaries():
    rows = [
        {'case_id': 'case-ok', 'trace_summary': _trace()},
        {'case_id': 'case-missing', 'trace_summary': {'trace_id': 'trace-missing'}},
    ]

    quality = trace_quality(rows)

    assert quality['total'] == 2
    assert quality['complete'] == 1
    assert quality['missing'] == ['case-missing']


def test_judge_primary_summary_uses_unified_primary_scores():
    judge = _partial_judge(_case(), _answer())
    summary = judge_primary_summary([{'judge': judge}])

    assert summary['total'] == 1
    assert summary['score_averages']['overall_score'] == 0.852
    assert summary['quality_label_counts'] == {'partial': 1}
    assert summary['failure_type_counts'] == {'partial_answer': 1}
    assert summary['correct_rate'] == 0.0


def test_trace_clustering_falls_back_without_apted(monkeypatch):
    from evo.operations.analysis import cluster as cluster_module

    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    row = classify_case(case, answer, judge, _trace())

    monkeypatch.setattr(cluster_module, 'APTED', None)
    monkeypatch.setattr(cluster_module, 'Tree', None)

    clusters = cluster_module.cluster_traces((row,))

    assert clusters['total'] == 1
    assert clusters['rows'][0]['case_id'] == 'case-1'
    assert clusters['rows'][0]['cluster_id'] == 'cluster_0001'


def test_evidence_packet_groups_bounded_surface_review_inputs():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    trace = _trace(stages=[
        {'id': 'rewrite-1', 'stage': 'query_rewrite', 'status': 'ok',
         'name': 'rewrite query', 'raw_data': {'input': 'capital France', 'output': 'France capital'}},
        {'id': 'retrieve-1', 'stage': 'retrieve', 'status': 'ok',
         'name': 'search kb', 'semantic_metrics': {'doc_ids': ['doc-other'], 'chunk_ids': ['chunk-other']}},
        {'id': 'llm-1', 'stage': 'llm_generate', 'status': 'ok',
         'name': 'answer', 'raw_data': {'input': 'context', 'output': 'Lyon'}},
    ])
    packet = build_evidence_packet(
        case,
        answer,
        judge,
        trace,
        review_packages=('query_intent_review', 'answer_faithfulness_review'),
    )

    assert packet['id'] == 'analysis.evidence_packet'
    assert packet['case_id'] == 'case-1'
    assert packet['case_evidence']['question'] == 'What is the capital of France?'
    assert packet['answer_evidence']['contexts'][0]['doc_id'] == 'doc-other'
    assert packet['judge_evidence']['diagnostic_evidence_counts']['missing_points'] == 1
    assert [item['review_package'] for item in packet['surface_reviews']] == [
        'query_intent_review',
        'answer_faithfulness_review',
    ]
    assert packet['surface_reviews'][0]['stage_evidence'][0]['raw_output'] == 'France capital'
    assert packet['surface_reviews'][1]['stage_evidence'][0]['raw_output'] == 'Lyon'
    assert packet['checks']['bounded'] is True


def test_semantic_review_runs_bounded_llm_review_without_repair_control():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    trace = _trace()
    packet = build_evidence_packet(case, answer, judge, trace,
                                   review_packages=('candidate_equivalence_review',))
    captured = {}

    def fake_llm(prompt):
        captured['prompt'] = prompt
        return json.dumps({
            'review_package': 'candidate_equivalence_review',
            'findings': [],
            'rule_alignment': 'insufficient_evidence',
            'bounded_reason': 'Candidate equivalence cannot be established from this bounded packet.',
        })

    review = run_semantic_review(packet, 'candidate_equivalence_review', llm_complete=fake_llm)

    assert review['id'] == 'analysis.semantic_review'
    assert review['rule_alignment'] == 'insufficient_evidence'
    assert review['provenance']['model_invoked'] is True
    assert 'analysis_evidence_packet' in captured['prompt']
    assert 'repair_group_queue' not in captured['prompt']


def test_evidence_packet_contains_planned_surface_review_packages():
    packet = build_evidence_packet(
        _case(),
        _answer(),
        _judge(),
        _trace(),
    )

    packages = {item['review_package']: item['surface'] for item in packet['surface_reviews']}

    assert packages['query_intent_review'] == 'query_planning'
    assert packages['candidate_equivalence_review'] == 'retrieve'
    assert packages['candidate_priority_review'] == 'rerank'
    assert packages['context_completeness_review'] == 'context_selection'
    assert packages['context_expansion_review'] == 'context_expansion'
    assert packages['compact_post_func_review'] == 'compact_post_func'
    assert packages['answer_faithfulness_review'] == 'answer_generation'
    assert packages['judge_conflict_review'] == 'judge_conflict'


def test_review_contract_normalizes_package_inputs():
    assert normalize_review_packages('judge_conflict_review') == ('judge_conflict_review',)
    assert normalize_review_packages((
        'candidate_equivalence_review',
        'candidate_equivalence_review',
        '',
    )) == ('candidate_equivalence_review',)
    with pytest.raises(ValueError, match='unknown semantic review package'):
        normalize_review_packages(('freeform_review',))


def test_semantic_review_prompt_rejects_unknown_package():
    with pytest.raises(ValueError, match='unknown semantic review package'):
        build_semantic_review_prompt({'id': 'analysis.evidence_packet'}, 'freeform_review')


def test_diagnostic_sidecar_builds_non_exclusive_agenda_without_judge_change():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    trace = _trace()
    encoded_judge = json.dumps(judge, sort_keys=True)

    sidecar = build_diagnostic_sidecar(case, answer, judge, trace)
    agenda = {item['mechanism_id']: item for item in sidecar['agenda']}

    assert json.dumps(judge, sort_keys=True) == encoded_judge
    assert sidecar['id'] == 'analysis.diagnostic_sidecar'
    assert sidecar['judge_interface']['required_change'] is False
    assert sidecar['judge_adapter']['status'] == 'valid'
    assert sidecar['checks']['judge_interface_unchanged'] is True
    assert sidecar['diagnosis_targets'][0]['id'] == 'kp-1'
    assert sidecar['target_paths'][0]['final_context_status'] == 'semantic_ambiguous'
    assert sidecar['target_paths'][0]['investigation_direction'] == 'needs_review'
    assert sidecar['target_paths'][0]['next_review_package'] == 'context_completeness_review'
    assert sidecar['evidence_timeline'][0]['earliest_observable_failure']['stage'] == 'retrieve'
    assert agenda['retrieve.reference_absent']['status'] == 'needs_semantic_review'
    assert agenda['answer.available_context_ignored']['status'] == 'ruled_out'
    assert sidecar['review_plan']['review_packages'][:2] == [
        'context_completeness_review',
        'answer_faithfulness_review',
    ]
    assert 'candidate_equivalence_review' in sidecar['review_plan']['delayed_packages']
    assert {
        item.get('probe_id') for item in sidecar['confirmation_plan']['steps']
        if item.get('probe_id')
    } >= {'retrieve.rank_expand_replay', 'index.presence_probe'}
    assert sidecar['diagnostic_result']['actionability'] == 'needs_semantic_review'


def test_diagnostic_sidecar_uses_abnormal_claim_without_key_points_as_target():
    case = _case()
    del case['key_points']
    answer = _answer(
        answer='Lyon is the capital of France.',
        doc_ids=['doc-gold'],
        chunk_ids=['chunk-gold'],
    )
    judge = _judge(
        case,
        answer,
        retrieval_failure_type='none',
        failure_type='hallucination',
        quality_label='bad',
        answer_quality_score=0.2,
        retrieval_quality_score=1.0,
        overall_score=0.36,
        groundedness=0.2,
        claim_support_rate=0.0,
        unsupported_claim_rate=1.0,
        retrieval_hit_at_k=1.0,
        retrieval_recall_at_k=1.0,
        retrieval_mrr=1.0,
        retrieval_ndcg=1.0,
        context_recall=1.0,
        chunk_recall=1.0,
        doc_recall=1.0,
        matched_key_points=[],
        missing_points=[],
        wrong_points=[],
        claims=[{'text': 'Lyon is the capital of France.', 'supported': False}],
        unsupported_claims=[{'text': 'Lyon is the capital of France.'}],
        evidence_mapping=[],
    )
    trace = _trace(
        retrieved_doc_ids=['doc-gold'],
        retrieved_chunk_ids=['chunk-gold'],
        final_context_doc_ids=['doc-gold'],
        final_context_chunk_ids=['chunk-gold'],
    )

    sidecar = build_diagnostic_sidecar(case, answer, judge, trace)
    agenda = {item['mechanism_id']: item for item in sidecar['agenda']}

    assert sidecar['checks']['target_gate_status'] == 'valid'
    assert len(sidecar['diagnosis_targets']) == 1
    assert sidecar['diagnosis_targets'][0]['source'] == 'judge.unsupported_claims'
    assert sidecar['diagnosis_targets'][0]['target_type'] == 'unsupported_claim'
    assert sidecar['target_paths'][0]['investigation_direction'] == 'answer_side'
    assert sidecar['target_paths'][0]['next_review_package'] == 'answer_faithfulness_review'
    assert agenda['answer.unsupported_or_contradicted']['status'] == 'needs_semantic_review'


def test_diagnostic_sidecar_fails_closed_when_bad_judge_has_no_targets():
    case = _case()
    del case['key_points']
    answer = _answer()
    judge = _judge(
        case,
        answer,
        retrieval_failure_type='retrieval_miss',
        failure_type='wrong_answer',
        quality_label='bad',
        matched_key_points=[],
        missing_points=[],
        wrong_points=[],
        claims=[{'text': 'Lyon is the capital of France.', 'supported': True}],
        unsupported_claims=[],
        contradicted_claims=[],
        evidence_mapping=[],
    )
    trace = _trace(stages=[
        {'id': 'rewrite-1', 'stage': 'query_rewrite', 'status': 'ok'},
        {'id': 'retrieve-1', 'stage': 'retrieve', 'status': 'ok'},
        {'id': 'llm-1', 'stage': 'llm_generate', 'status': 'ok'},
    ])

    sidecar = build_diagnostic_sidecar(case, answer, judge, trace)
    agenda = {item['mechanism_id']: item for item in sidecar['agenda']}
    packet = build_evidence_packet(case, answer, judge, trace)

    assert sidecar['diagnosis_targets'] == []
    assert sidecar['judge_adapter']['status'] == 'degraded'
    assert sidecar['checks']['target_gate_status'] == 'judge_diagnosis_incomplete'
    assert sidecar['diagnostic_result']['actionability'] == 'judge_diagnosis_incomplete'
    assert agenda['query.intent_lost']['status'] == 'insufficient_evidence'
    assert agenda['retrieve.reference_absent']['status'] == 'insufficient_evidence'
    assert agenda['judge.judge_conflict']['status'] == 'needs_semantic_review'
    assert packet['judge_adapter']['status'] == 'degraded'
    assert packet['checks']['target_gate_status'] == 'judge_diagnosis_incomplete'


def test_diagnostic_sidecar_confirms_context_drop_without_exclusive_router():
    case = _case()
    answer = _answer(answer='Lyon', doc_ids=['doc-other'], chunk_ids=['chunk-other'])
    judge = _judge(
        case,
        answer,
        retrieval_failure_type='retrieval_partial',
        failure_type='wrong_answer',
        doc_recall=1.0,
        chunk_recall=1.0,
        context_recall=0.0,
        retrieval_recall_at_k=1.0,
    )
    trace = _trace(
        diagnostic_stage_sequence=['retrieve', 'rerank', 'llm_generate'],
        stage_sequence=['retrieve', 'rerank', 'llm_generate'],
        stages=[
            {'id': 'retrieve-1', 'stage': 'retrieve', 'status': 'ok'},
            {'id': 'rerank-1', 'stage': 'rerank', 'status': 'ok'},
            {'id': 'llm-1', 'stage': 'llm_generate', 'status': 'ok'},
        ],
        retrieved_doc_ids=['doc-gold'],
        retrieved_chunk_ids=['chunk-gold'],
        final_context_doc_ids=['doc-other'],
        final_context_chunk_ids=['chunk-other'],
    )
    sidecar = build_diagnostic_sidecar(case, answer, judge, trace)
    agenda = {item['mechanism_id']: item for item in sidecar['agenda']}

    assert agenda['retrieve.reference_absent']['status'] == 'ruled_out'
    assert sidecar['target_paths'][0]['final_context_status'] == 'insufficient_by_fact'
    assert sidecar['target_paths'][0]['investigation_direction'] == 'evidence_backtrack'
    assert sidecar['target_paths'][0]['breakpoint_window'] == {
        'from': 'retrieve',
        'to': 'context_assembly',
    }
    assert agenda['rerank.relevant_candidate_demoted']['status'] == 'needs_probe'
    probe_ids = [
        item.get('probe_id') for item in sidecar['confirmation_plan']['steps']
        if item.get('probe_id')
    ]
    assert probe_ids[:3] == [
        'context.selection_replay',
        'rerank.selection_replay',
        'retrieve.rank_expand_replay',
    ]
    assert agenda['context.required_evidence_dropped']['status'] == 'confirmed'
    assert sidecar['diagnostic_result']['primary_mechanism']['mechanism_id'] == (
        'context.required_evidence_dropped'
    )
    assert sidecar['diagnostic_result']['repair_ready'] is True


def test_semantic_review_uses_target_scoped_findings_schema():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    trace = _trace()
    diagnostic_plan = build_diagnostic_sidecar(case, answer, judge, trace)
    packet = build_evidence_packet(
        case,
        answer,
        judge,
        trace,
        review_packages=('candidate_equivalence_review',),
        diagnostic_plan=diagnostic_plan,
    )

    def fake_llm(prompt):
        assert 'review_package, findings, rule_alignment, bounded_reason' in prompt
        return json.dumps({
            'review_package': 'candidate_equivalence_review',
            'findings': [{
                'obligation_id': 'kp-1',
                'mechanism_id': 'retrieve.reference_absent',
                'stage': 'retrieve',
                'finding': 'not_equivalent',
                'confidence': 0.91,
                'evidence_refs': ['surface_reviews.candidate_equivalence_review'],
            }],
            'rule_alignment': 'supports_candidate',
            'bounded_reason': 'Retrieved candidate is not equivalent to the required Paris evidence.',
        })

    review = run_semantic_review(packet, 'candidate_equivalence_review', llm_complete=fake_llm)

    assert review['findings'][0]['finding'] == 'not_equivalent'
    assert review['findings'][0]['mechanism_id'] == 'retrieve.reference_absent'
    assert review['rule_alignment'] == 'supports_candidate'
    assert 'secondary_review' not in review


def test_probe_harness_only_runs_registered_readonly_handlers():
    observation = run_registered_probe(
        'rerank.selection_replay',
        {'case_id': 'case-1'},
        handlers={
            'rerank.selection_replay': lambda params: {
                'case_id': params['case_id'],
                'changed_mechanism': False,
            },
        },
    )

    assert observation['id'] == 'analysis.probe_observation'
    assert observation['probe_id'] == 'rerank.selection_replay'
    assert observation['checks']['registered_probe_only'] is True
    assert observation['observation']['case_id'] == 'case-1'
    with pytest.raises(ValueError, match='unknown analysis probe'):
        run_registered_probe('freeform.shell', {}, handlers={})


def test_semantic_review_closes_target_root_cause_loop():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer, unsupported_claims=[], claims=[])
    trace = _trace()
    review = {
        'id': 'analysis.semantic_review',
        'review_package': 'candidate_equivalence_review',
        'rule_alignment': 'supports_candidate',
        'findings': [{
            'obligation_id': 'kp-1',
            'stage': 'retrieve',
            'finding': 'not_equivalent',
            'confidence': 0.93,
            'evidence_refs': ['trace.retrieval_steps[0]'],
        }],
    }

    sidecar = build_diagnostic_sidecar(
        case,
        answer,
        judge,
        trace,
        semantic_reviews=[review],
    )

    target = sidecar['target_results'][0]
    assert target['status'] == 'confirmed'
    assert target['primary_mechanism']['mechanism_id'] == 'retrieve.reference_absent'
    assert target['primary_mechanism']['decision_source'] == 'semantic_review'
    assert target['primary_mechanism']['confidence'] == 0.93
    assert sidecar['diagnostic_result']['primary_mechanism']['target_ids'] == ['kp-1']
    assert sidecar['diagnostic_result']['fully_resolved'] is True


def test_registered_rerank_probe_changes_target_primary_root():
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    trace = _trace(
        diagnostic_stage_sequence=['retrieve', 'rerank', 'llm_generate'],
        retrieved_doc_ids=['doc-gold', 'doc-other'],
        retrieved_chunk_ids=['chunk-gold', 'chunk-other'],
        final_context_doc_ids=['doc-other'],
        final_context_chunk_ids=['chunk-other'],
    )
    probe = run_registered_probe(
        'rerank.selection_replay',
        {'target_id': 'kp-1'},
        handlers={
            'rerank.selection_replay': lambda params: {
                'target_id': params['target_id'],
                'decision': 'confirmed',
                'confidence': 0.97,
                'controlled_variables': ['query', 'candidate_set', 'context_cutoff'],
                'baseline': {'required_rank': 2, 'selected': True},
                'treatment': {'required_rank': 9, 'selected': False},
                'evidence_refs': ['replay.rank_delta'],
                'cost': {'latency_ms': 18, 'model_calls': 0},
            },
        },
    )

    sidecar = build_diagnostic_sidecar(
        case,
        answer,
        judge,
        trace,
        probe_observations=[probe],
    )

    assert probe['target_ids'] == ['kp-1']
    assert probe['decision'] == 'confirmed'
    assert probe['checks']['decision_ready'] is True
    assert probe['controlled_variables'] == ['query', 'candidate_set', 'context_cutoff']
    target = sidecar['target_results'][0]
    assert target['primary_mechanism']['mechanism_id'] == 'rerank.relevant_candidate_demoted'
    assert target['primary_mechanism']['decision_source'] == 'probe'
    assert target['evidence'][-1] == 'replay.rank_delta'
    assert sidecar['diagnostic_result']['primary_mechanism']['mechanism_id'] == (
        'rerank.relevant_candidate_demoted'
    )


def test_probe_observation_is_gated_to_its_target():
    case = _case(
        reference_doc_ids=[],
        reference_chunk_ids=[],
        key_points=[
            {
                'id': 'kp-1',
                'statement': 'Paris is the capital',
                'required': True,
                'evidence_doc_ids': ['doc-a'],
                'evidence_chunk_ids': ['chunk-a'],
            },
            {
                'id': 'kp-2',
                'statement': 'France is in Europe',
                'required': True,
                'evidence_doc_ids': ['doc-b'],
                'evidence_chunk_ids': ['chunk-b'],
            },
        ],
    )
    answer = _answer(
        contexts=[{
            'doc_id': 'doc-a',
            'chunk_id': 'chunk-a',
            'content': 'Paris is the capital.',
            'rank': 1,
        }],
        doc_ids=['doc-a'],
        chunk_ids=['chunk-a'],
    )
    judge = _judge(
        case,
        answer,
        missing_points=[
            {'id': 'kp-1', 'statement': 'Paris is the capital', 'required': True},
            {'id': 'kp-2', 'statement': 'France is in Europe', 'required': True},
        ],
        unsupported_claims=[],
        claims=[],
    )
    trace = _trace(
        diagnostic_stage_sequence=['retrieve', 'rerank', 'llm_generate'],
        retrieved_doc_ids=['doc-a', 'doc-b'],
        retrieved_chunk_ids=['chunk-a', 'chunk-b'],
        final_context_doc_ids=['doc-a'],
        final_context_chunk_ids=['chunk-a'],
    )
    probe = run_registered_probe(
        'rerank.selection_replay',
        {'target_id': 'kp-2'},
        handlers={
            'rerank.selection_replay': lambda params: {
                'target_id': params['target_id'],
                'decision': 'confirmed',
                'confidence': 0.96,
                'evidence_refs': ['replay.kp-2.rank_delta'],
            },
        },
    )

    sidecar = build_diagnostic_sidecar(
        case,
        answer,
        judge,
        trace,
        probe_observations=[probe],
    )
    targets = {item['target_id']: item for item in sidecar['target_results']}

    assert targets['kp-1']['primary_mechanism'].get('mechanism_id') != (
        'rerank.relevant_candidate_demoted'
    )
    assert targets['kp-2']['primary_mechanism']['mechanism_id'] == (
        'rerank.relevant_candidate_demoted'
    )
    assert targets['kp-1']['observation_updates'] == []
    assert targets['kp-2']['observation_updates'][0]['source_id'] == 'rerank.selection_replay'


def test_trace_summary_preserves_retriever_and_reranker_stage_boundaries():
    retriever_metrics = _semantic_data(SimpleNamespace(semantic_data={
        'query': 'capital of France',
        'filters': {'kb_id': 'kb-1'},
        'topk': 8,
        'returned_node_ids': ['chunk-gold', 'chunk-other'],
        'returned_nodes': [{'uid': 'chunk-gold', 'doc_id': 'doc-gold'}],
        'scores': [0.91, 0.62],
    }))
    reranker_metrics = _semantic_data(SimpleNamespace(semantic_data={
        'query': 'capital of France',
        'topk': 1,
        'candidate_doc_ids': ['doc-gold', 'doc-other'],
        'ranked_doc_ids': ['doc-other', 'doc-gold'],
        'candidate_nodes': [{'uid': 'chunk-gold', 'doc_id': 'doc-gold'}],
        'ranked_nodes': [{'uid': 'chunk-other', 'doc_id': 'doc-other'}],
        'rerank_model': 'reranker-current',
        'scores': [0.88, 0.31],
    }))
    artifacts = _retrieval_artifacts([
        {
            'id': 'retrieve-1',
            'stage': 'retrieve',
            'name': 'Retriever',
            'semantic_metrics': retriever_metrics,
        },
        {
            'id': 'rerank-1',
            'stage': 'rerank',
            'name': 'Reranker',
            'semantic_metrics': reranker_metrics,
        },
    ])

    retrieve_step, rerank_step = artifacts['steps']
    assert retrieve_step['returned_node_ids'] == ['chunk-gold', 'chunk-other']
    assert retrieve_step['filters'] == {'kb_id': 'kb-1'}
    assert rerank_step['candidate_doc_ids'] == ['doc-gold', 'doc-other']
    assert rerank_step['ranked_doc_ids'] == ['doc-other', 'doc-gold']
    assert rerank_step['rerank_model'] == 'reranker-current'


def test_probe_batch_reports_missing_handlers_without_confirming_a_root():
    batch = run_confirmation_probe_batch(
        {
            'steps': [
                {
                    'step_id': 'probe.rerank.selection_replay:kp-1',
                    'probe_id': 'rerank.selection_replay',
                    'target_ids': ['kp-1'],
                    'mechanism_ids': ['rerank.relevant_candidate_demoted'],
                },
                {
                    'step_id': 'probe.retrieve.rank_expand_replay:kp-1',
                    'probe_id': 'retrieve.rank_expand_replay',
                    'target_ids': ['kp-1'],
                    'mechanism_ids': ['retrieve.reference_absent'],
                },
            ]
        },
        handlers={},
        eligible_target_ids=['kp-1'],
        max_probe_calls=1,
    )

    assert batch['status'] == 'unavailable'
    assert batch['observations'] == []
    assert batch['unavailable'][0]['probe_id'] == 'rerank.selection_replay'
    assert batch['delayed'] == []
    assert len(batch['unavailable']) == 2
    assert batch['checks']['all_handlers_available'] is False


def test_probe_batch_executes_injected_handler_with_control_contract():
    captured = {}

    def rerank_handler(params):
        captured.update(params)
        return {
            'decision': 'confirmed',
            'confidence': 0.94,
            'evidence_refs': ['rerank.rank_delta'],
            'controlled_variables': params['fixed_variables'],
        }

    batch = run_confirmation_probe_batch(
        {
            'steps': [{
                'step_id': 'probe.rerank.selection_replay:kp-1',
                'probe_id': 'rerank.selection_replay',
                'target_ids': ['kp-1'],
                'mechanism_ids': ['rerank.relevant_candidate_demoted'],
                'fixed_variables': ['query', 'candidate set', 'reranker configuration'],
                'compare': ['rank_before', 'rank_after'],
            }]
        },
        handlers={'rerank.selection_replay': rerank_handler},
        context={'case': {'id': 'case-1'}},
        eligible_target_ids=['kp-1'],
    )

    assert batch['status'] == 'completed'
    assert captured['target_ids'] == ['kp-1']
    assert captured['case']['id'] == 'case-1'
    assert batch['observations'][0]['decision'] == 'confirmed'
    assert batch['observations'][0]['target_ids'] == ['kp-1']
    assert batch['observations'][0]['evidence_refs'] == ['rerank.rank_delta']


def test_analysis_owned_rerank_diff_splits_reranker_from_upstream():
    handlers = analysis_owned_probe_handlers()
    common = {
        'target_ids': ['kp-1'],
        'evidence_packet': {
            'diagnosis_targets': [{
                'id': 'kp-1',
                'reference_doc_ids': ['doc-gold'],
                'reference_chunk_ids': ['chunk-gold'],
            }]
        },
    }
    demoted = run_registered_probe(
        'rerank.selection_replay',
        {
            **common,
            'trace': {
                'retrieval_steps': [{
                    'id': 'rerank-1',
                    'stage': 'rerank',
                    'topk': 1,
                    'candidate_doc_ids': ['doc-gold', 'doc-other'],
                    'ranked_doc_ids': ['doc-other'],
                }]
            },
        },
        handlers=handlers,
    )
    upstream_missing = run_registered_probe(
        'rerank.selection_replay',
        {
            **common,
            'trace': {
                'retrieval_steps': [{
                    'id': 'rerank-2',
                    'stage': 'rerank',
                    'topk': 1,
                    'candidate_doc_ids': ['doc-other'],
                    'ranked_doc_ids': ['doc-other'],
                }]
            },
        },
        handlers=handlers,
    )

    assert demoted['decision'] == 'confirmed'
    assert demoted['cost']['model_calls'] == 0
    assert upstream_missing['decision'] == 'ruled_out'


def test_external_probe_handler_registration_is_the_runtime_extension_point(monkeypatch):
    from evo.operations.analysis import confirmation as confirmation_module

    monkeypatch.setattr(confirmation_module, '_EXTERNAL_PROBE_HANDLERS', {})

    def handler(params):
        return {'decision': 'confirmed'}

    register_probe_handler('index.presence_probe', handler)

    assert registered_probe_handlers()['index.presence_probe'] is handler


def test_analysis_flow_declares_full_diagnostic_chain_per_case():
    op_ids = [op.spec.op_id for op in evo_operations()]
    expected_chain = [
        'analysis.trace_summary',
        'analysis.classify_case',
        'analysis.diagnostic_plan',
        'analysis.evidence_packet',
        'analysis.semantic_review_batch',
        'analysis.probe_batch',
        'analysis.diagnostic_sidecar',
        'analysis.trace_clusters',
        'analysis.summary',
    ]
    assert [op_id for op_id in op_ids if op_id.startswith('analysis.')] == expected_chain

    assert {
        analysis_artifacts.ANALYSIS_DIAGNOSTIC_PLAN,
        analysis_artifacts.ANALYSIS_EVIDENCE_PACKET,
        analysis_artifacts.ANALYSIS_SEMANTIC_REVIEWS,
        analysis_artifacts.ANALYSIS_PROBE_OBSERVATIONS,
        analysis_artifacts.ANALYSIS_DIAGNOSTIC_SIDECAR,
    } <= set(analysis_artifacts.PARTITION_SET_BY_ARTIFACT)


def test_semantic_review_batch_operation_uses_lazyllm_evo_model(monkeypatch):
    captured = {}

    class FakeLazyLLMClient:
        def __init__(self, *, llm_config, model):
            captured['llm_config'] = llm_config
            captured['model'] = model

        def __call__(self, prompt, **kwargs):
            captured['prompt'] = prompt
            captured['kwargs'] = kwargs
            return 'thinking before structured answer\n' + json.dumps({
                'review_package': 'judge_conflict_review',
                'findings': [{
                    'obligation_id': 'kp-1',
                    'mechanism_id': 'judge.judge_conflict',
                    'stage': 'judge',
                    'finding': 'judge_consistent',
                    'confidence': 0.92,
                    'evidence_refs': ['judge_evidence'],
                }],
                'rule_alignment': 'supports_candidate',
                'bounded_reason': 'The supplied evidence supports the candidate.',
            })

    monkeypatch.setattr('evo.llm.LazyLLMClient', FakeLazyLLMClient)
    ctx = OperationContext('run-1', 'test-lazyllm-review', 'case-1')
    result = asyncio.run(semantic_review_batch_operation(
        ctx,
        {
            'id': 'analysis.evidence_packet',
            'case_id': 'case-1',
            'diagnosis_targets': [{'id': 'kp-1'}],
            'mechanism_candidates': [{
                'mechanism_id': 'judge.judge_conflict',
                'review_package': 'judge_conflict_review',
                'status': 'needs_semantic_review',
            }],
        },
        {
            'review_plan': {
                'max_review_calls': 1,
                'review_packages': ['judge_conflict_review'],
                'delayed_packages': [],
            }
        },
        {
            'llm_config': {
                'evo_llm': {
                    'source': 'deepseek',
                    'type': 'llm',
                    'name': 'deepseek-v4-flash',
                }
            }
        },
    )).values['semantic_reviews']

    assert captured['model'] == 'evo_llm'
    assert captured['llm_config']['evo_llm']['model'] == 'deepseek-v4-flash'
    assert captured['kwargs'] == {'stream': False}
    assert result['status'] == 'completed'
    assert result['reviews'][0]['rule_alignment'] == 'supports_candidate'


def test_analysis_operations_complete_chain_and_fail_closed_without_external_probe_handlers(
    monkeypatch,
):
    def insufficient_review_batch(packet, plan, *, llm_config, timeout_seconds):
        assert packet['id'] == 'analysis.evidence_packet'
        assert 'evo_llm' in llm_config
        assert timeout_seconds > 0
        return {
            'id': 'analysis.semantic_review_batch',
            'status': 'completed',
            'reviews': [{
                'id': 'analysis.semantic_review',
                'review_package': plan['review_packages'][0],
                'findings': [],
                'rule_alignment': 'insufficient_evidence',
            }],
            'requested_packages': plan['review_packages'],
            'completed_packages': plan['review_packages'][:1],
            'failed_packages': [],
            'delayed_packages': plan['review_packages'][1:],
            'checks': {
                'ready': True,
                'errors': [],
                'max_review_calls': plan['max_review_calls'],
            },
        }

    monkeypatch.setattr(
        operation_module,
        'run_semantic_review_batch',
        insufficient_review_batch,
    )
    ctx = OperationContext('run-1', 'test-analysis-chain', 'case-1')
    case = _case()
    answer = _answer()
    judge = _judge(case, answer)
    trace = _trace()
    classification = classify_case(case, answer, judge, trace)
    config = {
        'inputs': {'analysis_review_budget': 1, 'analysis_probe_budget': 2},
        'llm_config': {'evo_llm': {'model': 'test'}},
    }

    async def run_chain():
        plan = (await diagnostic_plan_operation(
            ctx,
            case,
            answer,
            judge,
            trace,
            config,
        )).values['diagnostic_plan']
        packet = (await evidence_packet_operation(
            ctx,
            case,
            answer,
            judge,
            trace,
            plan,
        )).values['evidence_packet']
        reviews = (await semantic_review_batch_operation(
            ctx,
            packet,
            plan,
            config,
        )).values['semantic_reviews']
        probes = (await probe_batch_operation(
            ctx,
            case,
            answer,
            judge,
            trace,
            classification,
            plan,
            packet,
            reviews,
            config,
        )).values['probe_observations']
        sidecar = (await diagnostic_sidecar_operation(
            ctx,
            plan,
            reviews,
            probes,
        )).values['diagnostic_sidecar']
        clusters = (await trace_clusters_operation(
            ctx,
            {'case-1': classification},
        )).values['clusters']
        summary = (await analysis_summary_operation(
            ctx,
            {'case-1': classification},
            clusters,
            {'case-1': sidecar},
        )).values['summary']
        return plan, reviews, probes, sidecar, summary

    plan, reviews, probes, sidecar, summary = asyncio.run(run_chain())

    assert plan['id'] == 'analysis.diagnostic_plan'
    assert reviews['checks']['max_review_calls'] == 1
    assert probes['status'] == 'unavailable'
    assert probes['observations'] == []
    assert sidecar['diagnostic_result']['fully_resolved'] is False
    assert sidecar['investigation_execution']['probe_batch']['unavailable']
    assert summary['cases'][0]['diagnosis']['probe_status'] == 'unavailable'
    assert summary['cases'][0]['diagnosis']['unavailable_probes']
