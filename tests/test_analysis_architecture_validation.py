from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_TESTS = Path(__file__).resolve().parent
for item in (_ROOT, _TESTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from evo.operations.analysis.classify import classify_case
from test_analysis_classify import (
    _answer,
    _case,
    _judge,
    _trace,
    build_diagnostic_sidecar,
    build_evidence_packet,
)


@dataclass(frozen=True)
class Scenario:
    name: str
    retrieved_docs: tuple[str, ...] = ()
    retrieved_chunks: tuple[str, ...] = ()
    final_docs: tuple[str, ...] = ()
    final_chunks: tuple[str, ...] = ()
    stages: tuple[str, ...] = ('retrieve', 'llm_generate')
    answer_text: str = 'Lyon'
    judge_overrides: dict[str, Any] = field(default_factory=dict)
    answer_overrides: dict[str, Any] = field(default_factory=dict)
    trace_overrides: dict[str, Any] = field(default_factory=dict)
    case_overrides: dict[str, Any] = field(default_factory=dict)
    expected_issue_category: str = ''
    expected_target_gate: str = ''
    expected_path_direction: str = ''
    expected_actionability: str = ''
    expected_mechanism_status: tuple[str, str] = ('', '')
    expected_step_kind: str = ''
    expect_targets: bool = True


def _synthetic_case(spec: Scenario) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    case_id = f'arch-{spec.name}'
    trace_id = f'trace-{spec.name}'
    case = _case(id=case_id, **spec.case_overrides)
    contexts = [
        {'doc_id': doc_id, 'chunk_id': chunk_id, 'content': f'context for {doc_id}/{chunk_id}', 'rank': index + 1}
        for index, (doc_id, chunk_id) in enumerate(zip(spec.final_docs, spec.final_chunks))
    ]
    if not contexts and (spec.final_docs or spec.final_chunks):
        contexts = [
            {
                'doc_id': spec.final_docs[index] if index < len(spec.final_docs) else '',
                'chunk_id': spec.final_chunks[index] if index < len(spec.final_chunks) else '',
                'content': 'partial final context',
                'rank': index + 1,
            }
            for index in range(max(len(spec.final_docs), len(spec.final_chunks)))
        ]
    answer = _answer(
        case_id=case_id,
        trace_id=trace_id,
        answer=spec.answer_text,
        doc_ids=list(spec.final_docs),
        chunk_ids=list(spec.final_chunks),
        contexts=contexts,
        **spec.answer_overrides,
    )
    judge = _judge(case, answer, **_judge_metrics_for(spec))
    trace_kwargs = {
        'case_id': case_id,
        'trace_id': trace_id,
        'route_signature': '>'.join(spec.stages),
        'stage_sequence': list(spec.stages),
        'diagnostic_stage_sequence': list(spec.stages),
        'critical_path': list(spec.stages),
        'bottleneck_stage': spec.stages[0] if spec.stages else '',
        'stages': [{'id': f'{stage}-{index + 1}', 'stage': stage, 'status': 'ok'} for index, stage in enumerate(spec.stages)],
        'stage_counts': {stage: spec.stages.count(stage) for stage in spec.stages},
        'retrieval_steps': [
            {
                'id': 'retrieve-1',
                'stage': 'retrieve',
                'doc_ids': list(spec.retrieved_docs),
                'chunk_ids': list(spec.retrieved_chunks),
            }
        ],
        'retrieved_doc_ids': list(spec.retrieved_docs),
        'retrieved_chunk_ids': list(spec.retrieved_chunks),
        'final_context_doc_ids': list(spec.final_docs),
        'final_context_chunk_ids': list(spec.final_chunks),
        'semantic_metric_keys': ['doc_ids', 'chunk_ids'],
    }
    trace_kwargs.update(spec.trace_overrides)
    trace = _trace(**trace_kwargs)
    return case, answer, judge, trace


def _judge_metrics_for(spec: Scenario) -> dict[str, Any]:
    ref_retrieved = 'doc-gold' in spec.retrieved_docs or 'chunk-gold' in spec.retrieved_chunks
    ref_final = 'doc-gold' in spec.final_docs or 'chunk-gold' in spec.final_chunks
    ref_retrieved_complete = 'doc-gold' in spec.retrieved_docs and 'chunk-gold' in spec.retrieved_chunks
    ref_final_complete = 'doc-gold' in spec.final_docs and 'chunk-gold' in spec.final_chunks
    other_retrieved = bool(set(spec.retrieved_docs + spec.retrieved_chunks) - {'doc-gold', 'chunk-gold'})
    retrieval_failure = 'none' if ref_final_complete else 'retrieval_miss'
    if ref_retrieved and not ref_retrieved_complete:
        retrieval_failure = 'retrieval_partial'
    elif ref_retrieved_complete and not ref_final_complete:
        retrieval_failure = 'retrieval_partial'
    if other_retrieved and not ref_retrieved:
        retrieval_failure = 'retrieval_miss'
    metrics: dict[str, Any] = {
        'retrieval_failure_type': retrieval_failure,
        'failure_type': 'wrong_answer',
        'quality_label': 'bad',
        'answer_quality_score': 0.2,
        'retrieval_quality_score': 0.85 if ref_retrieved else 0.15,
        'overall_score': 0.33,
        'answer_correctness': 0.2,
        'answer_relevance': 0.45,
        'completeness': 0.25,
        'groundedness': 0.3 if ref_final else 0.15,
        'key_point_recall': 0.0,
        'claim_support_rate': 0.2,
        'unsupported_claim_rate': 0.8,
        'retrieval_hit_at_k': 1.0 if ref_retrieved else 0.0,
        'retrieval_recall_at_k': 1.0 if ref_retrieved else 0.0,
        'retrieval_mrr': 0.8 if ref_retrieved else 0.0,
        'retrieval_ndcg': 0.7 if ref_retrieved else 0.0,
        'context_relevance_avg': 0.7 if ref_final else 0.25,
        'context_noise_rate': 0.2 if ref_final else 0.5,
        'chunk_recall': 1.0 if 'chunk-gold' in spec.retrieved_chunks else 0.0,
        'doc_recall': 1.0 if 'doc-gold' in spec.retrieved_docs else 0.0,
        'context_recall': 1.0 if ref_final else 0.0,
        'reason': f'synthetic architecture validation case: {spec.name}',
        'missing_points': [
            {
                'id': 'kp-1',
                'statement': 'Paris is the capital',
                'weight': 1.0,
                'required': True,
                'acceptable_variants': [],
            }
        ],
        'wrong_points': [],
        'matched_key_points': [],
        'claims': [{'text': spec.answer_text, 'supported': False}],
        'unsupported_claims': [{'text': spec.answer_text}],
        'contradicted_claims': [],
    }
    metrics.update(spec.judge_overrides)
    return metrics


SCENARIOS = (
    Scenario('retrieval-empty-miss', expected_issue_category='tracing', expected_target_gate='valid',
             expected_actionability='repair_ready', expected_mechanism_status=('retrieve.reference_absent', 'confirmed')),
    Scenario('retrieval-candidates-no-reference', retrieved_docs=('doc-other',), retrieved_chunks=('chunk-other',),
             final_docs=('doc-other',), final_chunks=('chunk-other',), expected_issue_category='retrieval',
             expected_path_direction='needs_review', expected_actionability='needs_semantic_review',
             expected_mechanism_status=('retrieve.reference_absent', 'needs_semantic_review')),
    Scenario('doc-hit-chunk-missing', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-other',),
             final_docs=('doc-gold',), final_chunks=('chunk-other',), expected_issue_category='retrieval',
             expected_path_direction='answer_side', expected_actionability='needs_semantic_review'),
    Scenario('retrieved-reference-dropped', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-other',), final_chunks=('chunk-other',), expected_issue_category='retrieval',
             expected_path_direction='evidence_backtrack', expected_actionability='repair_ready',
             expected_mechanism_status=('context.required_evidence_dropped', 'confirmed'), expected_step_kind='readonly_replay'),
    Scenario('rerank-demotion-needs-probe', retrieved_docs=('doc-gold', 'doc-noise'), retrieved_chunks=('chunk-gold', 'chunk-noise'),
             final_docs=('doc-noise',), final_chunks=('chunk-noise',), stages=('retrieve', 'rerank', 'context_assembly', 'llm_generate'),
             expected_issue_category='retrieval', expected_path_direction='evidence_backtrack',
             expected_mechanism_status=('rerank.relevant_candidate_demoted', 'needs_probe'), expected_step_kind='readonly_replay'),
    Scenario('final-context-answer-wrong', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), expected_issue_category='generation',
             expected_path_direction='answer_side', expected_actionability='needs_semantic_review',
             expected_mechanism_status=('answer.available_context_ignored', 'needs_semantic_review')),
    Scenario('partial-answer-visible-context', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='Paris',
             judge_overrides={'failure_type': 'partial_answer', 'unsupported_claims': [], 'claims': [{'text': 'Paris', 'supported': True}]},
             expected_issue_category='generation', expected_path_direction='answer_side',
             expected_actionability='needs_semantic_review'),
    Scenario('question-not-answered-visible-context', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='I do not know.',
             judge_overrides={'failure_type': 'question_not_answered'},
             expected_issue_category='generation', expected_path_direction='answer_side',
             expected_actionability='needs_semantic_review'),
    Scenario('unsupported-claim', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='Paris and Atlantis are capitals.',
             judge_overrides={'failure_type': 'hallucination', 'missing_points': [], 'unsupported_claims': [{'text': 'Atlantis is a capital.'}]},
             expected_issue_category='generation', expected_path_direction='answer_side',
             expected_mechanism_status=('answer.unsupported_or_contradicted', 'needs_semantic_review')),
    Scenario('contradicted-claim', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='Lyon is the capital.',
             judge_overrides={'failure_type': 'hallucination', 'missing_points': [], 'unsupported_claims': [],
                              'contradicted_claims': [{'text': 'Lyon is the capital.'}],
                              'claims': [{'text': 'Lyon is the capital.', 'supported': False, 'contradicted': True}]},
             expected_issue_category='generation', expected_path_direction='answer_side'),
    Scenario('query-rewrite-observed', retrieved_docs=('doc-other',), retrieved_chunks=('chunk-other',),
             final_docs=('doc-other',), final_chunks=('chunk-other',), stages=('query_rewrite', 'retrieve', 'llm_generate'),
             expected_issue_category='retrieval', expected_actionability='needs_semantic_review',
             expected_mechanism_status=('query.intent_lost', 'needs_semantic_review')),
    Scenario('query-rewrite-empty-retrieval', stages=('query_rewrite', 'retrieve', 'llm_generate'),
             expected_issue_category='tracing', expected_mechanism_status=('query.intent_lost', 'needs_semantic_review')),
    Scenario('trace-retrieval-unknown', trace_overrides={'semantic_metric_keys': [], 'retrieved_doc_ids': [], 'retrieved_chunk_ids': []},
             expected_issue_category='tracing', expected_path_direction='blocked_by_trace',
             expected_mechanism_status=('trace.metrics_missing', 'insufficient_evidence')),
    Scenario('trace-unknown-stage-count', trace_overrides={'unknown_stage_count': 2},
             expected_issue_category='tracing', expected_mechanism_status=('trace.metrics_missing', 'insufficient_evidence')),
    Scenario('judge-target-incomplete', judge_overrides={'missing_points': [], 'wrong_points': [], 'unsupported_claims': [],
                                                         'claims': [], 'contradicted_claims': []},
             expected_issue_category='tracing', expected_target_gate='judge_diagnosis_incomplete',
             expected_actionability='judge_diagnosis_incomplete', expect_targets=False),
    Scenario('retrieve-stage-error', answer_overrides={'status': 'failed'},
             judge_overrides={'failure_type': 'infra_failure', 'quality_label': 'bad'},
             trace_overrides={'error_stages': [{'stage': 'retrieve', 'message': 'timeout'}]},
             expected_issue_category='runtime_infra', expected_target_gate='terminal',
             expected_mechanism_status=('execution.stage_error', 'confirmed')),
    Scenario('rerank-stage-error', answer_overrides={'status': 'failed'},
             judge_overrides={'failure_type': 'infra_failure', 'quality_label': 'bad'},
             trace_overrides={'error_stages': [{'stage': 'rerank', 'message': 'timeout'}]},
             expected_issue_category='runtime_infra', expected_target_gate='terminal',
             expected_mechanism_status=('execution.stage_error', 'confirmed')),
    Scenario('llm-stage-error', answer_overrides={'status': 'failed'},
             judge_overrides={'failure_type': 'infra_failure', 'quality_label': 'bad'},
             trace_overrides={'error_stages': [{'stage': 'llm_generate', 'message': 'timeout'}]},
             expected_issue_category='runtime_infra', expected_target_gate='terminal',
             expected_mechanism_status=('execution.stage_error', 'confirmed')),
    Scenario('answer-status-error', answer_overrides={'status': 'failed', 'chat_error': 'upstream failed'},
             expected_issue_category='runtime_infra'),
    Scenario('judge-contract-error', judge_overrides={'failure_type': 'judge_contract_error', 'quality_label': 'bad'},
             expected_issue_category='contract', expected_target_gate='terminal'),
    Scenario('dataset-contract-error', judge_overrides={'failure_type': 'dataset_contract_error', 'quality_label': 'bad'},
             expected_issue_category='contract', expected_target_gate='terminal'),
    Scenario('format-error', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='Paris\n- bad table',
             judge_overrides={'failure_type': 'format_error', 'format_compliance': 0.0, 'retrieval_failure_type': 'none'},
             expected_issue_category='generation'),
    Scenario('retrieval-noise-rerank', retrieved_docs=('doc-gold', 'doc-noise'), retrieved_chunks=('chunk-gold', 'chunk-noise'),
             final_docs=('doc-noise',), final_chunks=('chunk-noise',), stages=('retrieve', 'rerank', 'llm_generate'),
             judge_overrides={'retrieval_failure_type': 'retrieval_noise', 'retrieval_precision_at_k': 0.1, 'context_noise_rate': 0.9},
             expected_issue_category='retrieval', expected_mechanism_status=('rerank.relevant_candidate_demoted', 'needs_probe')),
    Scenario('retrieval-noise-no-rerank', retrieved_docs=('doc-gold', 'doc-noise'), retrieved_chunks=('chunk-gold', 'chunk-noise'),
             final_docs=('doc-noise',), final_chunks=('chunk-noise',),
             judge_overrides={'retrieval_failure_type': 'retrieval_noise', 'retrieval_precision_at_k': 0.1, 'context_noise_rate': 0.9},
             expected_issue_category='retrieval'),
    Scenario('wrong-point-visible-context', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='Paris is not the capital.',
             judge_overrides={'missing_points': [], 'wrong_points': [{'id': 'kp-1', 'statement': 'Paris is the capital'}]},
             expected_issue_category='generation', expected_path_direction='answer_side'),
    Scenario('multi-target-mixed', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',),
             case_overrides={'key_points': [{'id': 'kp-1', 'statement': 'Paris is the capital', 'required': True},
                                            {'id': 'kp-2', 'statement': 'France is in Europe', 'required': True}]},
             judge_overrides={'missing_points': [{'id': 'kp-1', 'statement': 'Paris is the capital'},
                                                 {'id': 'kp-2', 'statement': 'France is in Europe'}]},
             expected_issue_category='generation', expected_path_direction='answer_side'),
    Scenario('final-context-semantic-ambiguous', retrieved_docs=('doc-other',), retrieved_chunks=('chunk-other',),
             final_docs=('doc-related',), final_chunks=('chunk-related',),
             expected_issue_category='retrieval', expected_path_direction='needs_review',
             expected_actionability='needs_semantic_review'),
    Scenario('empty-final-after-retrieve-observed', retrieved_docs=('doc-other',), retrieved_chunks=('chunk-other',),
             expected_issue_category='tracing', expected_path_direction='blocked_by_trace'),
    Scenario('no-reference-ids-hallucination',
             case_overrides={'reference_doc_ids': [], 'reference_chunk_ids': [], 'reference_context': []},
             judge_overrides={'retrieval_failure_type': 'not_applicable', 'failure_type': 'hallucination',
                              'missing_points': [], 'unsupported_claims': [{'text': 'Unsupported answer.'}]},
             expected_issue_category='generation', expected_path_direction='answer_side'),
    Scenario('correct-control', retrieved_docs=('doc-gold',), retrieved_chunks=('chunk-gold',),
             final_docs=('doc-gold',), final_chunks=('chunk-gold',), answer_text='Paris',
             judge_overrides={'failure_type': 'none', 'quality_label': 'good', 'is_correct': True,
                              'retrieval_failure_type': 'none', 'answer_quality_score': 0.95,
                              'retrieval_quality_score': 0.95, 'overall_score': 0.95,
                              'missing_points': [], 'unsupported_claims': [],
                              'claims': [{'text': 'Paris', 'supported': True}]},
             expected_issue_category='ok', expected_target_gate='not_required', expect_targets=False),
)


@pytest.mark.parametrize('spec', SCENARIOS, ids=[item.name for item in SCENARIOS])
def test_analysis_architecture_exposes_problem_root_and_evidence(spec: Scenario):
    assert len(SCENARIOS) == 30
    case, answer, judge, trace = _synthetic_case(spec)

    row = classify_case(case, answer, judge, trace)
    sidecar = build_diagnostic_sidecar(case, answer, judge, trace)
    packet = build_evidence_packet(
        case,
        answer,
        judge,
        trace,
        review_packages=sidecar['review_plan']['review_packages'],
    )

    assert row['issue_category']
    assert row['issue_type']
    assert row['affected_block']
    assert row['failure_mode']
    assert row['root_cause_reason']
    if spec.expected_issue_category:
        assert row['issue_category'] == spec.expected_issue_category

    assert sidecar['checks']['ready'] is True
    assert sidecar['checks']['judge_interface_unchanged'] is True
    assert sidecar['diagnostic_result']['actionability']
    if spec.expected_target_gate:
        assert sidecar['checks']['target_gate_status'] == spec.expected_target_gate
    if spec.expected_actionability:
        assert sidecar['diagnostic_result']['actionability'] == spec.expected_actionability
    if spec.expect_targets:
        assert sidecar['diagnosis_targets']
        assert sidecar['target_paths']
        assert sidecar['target_results']
        assert sidecar['evidence_timeline']
        assert len(sidecar['target_results']) == len(sidecar['diagnosis_targets'])
    else:
        assert sidecar['diagnosis_targets'] == []
        assert sidecar['target_results'] == []

    if spec.expected_path_direction:
        assert any(path.get('investigation_direction') == spec.expected_path_direction for path in sidecar['target_paths'])

    mechanism_id, status = spec.expected_mechanism_status
    if mechanism_id:
        assert _mechanism_status(sidecar, mechanism_id) == status

    if spec.expected_step_kind:
        assert any(step.get('kind') == spec.expected_step_kind for step in sidecar['confirmation_plan']['steps'])

    assert _has_root_or_next_step(sidecar)
    assert packet['checks']['ready'] is True
    assert packet['checks']['judge_adapter_status'] == sidecar['checks']['judge_adapter_status']
    assert packet['checks']['target_gate_status'] == sidecar['checks']['target_gate_status']


def _mechanism_status(sidecar: dict[str, Any], mechanism_id: str) -> str:
    for item in sidecar['agenda']:
        if item.get('mechanism_id') == mechanism_id:
            return str(item.get('status') or '')
    return ''


def _has_root_or_next_step(sidecar: dict[str, Any]) -> bool:
    result = sidecar['diagnostic_result']
    return bool(
        result.get('primary_mechanism')
        or result.get('alternatives')
        or result.get('missing_evidence')
        or sidecar['review_plan'].get('review_packages')
        or sidecar['confirmation_plan'].get('steps')
    )
