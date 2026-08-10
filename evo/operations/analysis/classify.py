from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evo.operations.public_contracts import algo_id, case_source_label

from . import _evidence_record as evidence
from .judge import (
    compact_features,
    diagnostic_count_features,
    judge_evidence,
    policy_number,
    validate_classification_inputs,
)
_OLD_ALIASES = {
    'coarse_category',
    'fine_category',
    'repairable',
    'recommended_action',
    'repairable_cases',
    'category_counts',
    'fine_category_counts',
    'llm_analysis_queue',
    'answer_score',
    'retrieval_score',
    'trace_missing',
    'trace_available',
}


def classify_case(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    validate_classification_inputs(case, answer, judge, trace)
    case_id = text(case.get('id') or answer.get('case_id') or judge.get('case_id'))
    decision = decide_case(case, answer, judge, trace)
    decision['actionable'] = is_actionable(decision)
    row = {
        'case_id': case_id,
        'trace_id': text(trace.get('trace_id')),
        'source': case_source_label(case),
        'algo_id': algo_id({'rag_answer': answer, 'target': judge.get('target') or {}}),
        'question_type': text(case.get('question_type')),
        **decision,
        'judge_reason': text(judge.get('reason')),
        'root_cause_reason': root_cause_reason(decision),
        'diagnosis_features': decision.pop('features'),
        'answer_evidence': decision.pop('answer_evidence'),
        'judge_evidence': decision.pop('judge_evidence'),
        'trace_evidence': decision.pop('trace_evidence'),
        'investigation_note': decision.pop('investigation_note'),
        'case': _snapshot(case, (
            'id', 'question', 'answer', 'question_type', 'difficulty', 'grading_guidance',
            'key_points', 'reference_doc_ids', 'reference_chunk_ids', 'reference_context',
            'source_preparation',
        )),
        'rag_answer': _snapshot(answer, (
            'case_id', 'trace_id', 'status', 'answer', 'doc_ids', 'chunk_ids', 'target',
            'tool_errors', 'chat_error',
        )),
        'judge': _snapshot(judge, tuple(
            key for key in judge if key not in {'case', 'rag_answer'}
        )),
        'trace_summary': _snapshot(trace, (
            'case_id', 'trace_id', 'trace_source', 'route_signature', 'tree_text',
            'stage_sequence', 'diagnostic_stage_sequence', 'unknown_stage_count',
            'critical_path', 'bottleneck_stage', 'stage_counts', 'latency_by_stage',
            'error_stages', 'retrieval_steps', 'retrieved_doc_ids', 'retrieved_chunk_ids',
            'final_context_doc_ids', 'final_context_chunk_ids', 'semantic_metric_keys',
            'features', 'trace_unavailable',
        )),
    }
    return row


def _scrub(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _scrub(raw) for key, raw in value.items() if str(key) not in _OLD_ALIASES}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    return value


def _snapshot(value: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {
        key: _scrub(value[key])
        for key in keys
        if key in value
    }


def decide_case(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    inconsistent = _judge_inconsistency(judge)
    if judge.get('failure_type') == 'judge_contract_error':
        return _row('contract', 'judge_contract_error', 'eval_contract', 'judge_contract_error', 'high',
                    False, [f'failure_type={judge["failure_type"]}'], judge)
    if inconsistent:
        return _row('contract', 'judge_contract_inconsistent', 'eval_contract', 'judge_contract_inconsistent',
                    'high', False, inconsistent, judge)
    if judge.get('failure_type') == 'dataset_contract_error':
        return _row('contract', 'dataset_contract_error', 'eval_contract', 'dataset_contract_error', 'high',
                    False, ['failure_type=dataset_contract_error'], judge)
    if answer.get('status') != 'ok' or answer.get('chat_error'):
        source = _infra_source(answer)
        return _row('runtime_infra', 'rag_or_judge_infra_failure', 'runtime_infra',
                    'rag_or_judge_infra_failure', 'high', False, [source], judge, answer=answer, trace=trace)
    if judge.get('failure_type') == 'infra_failure':
        if _answer_has_evidence(answer):
            return _row('contract', 'judge_contract_inconsistent', 'eval_contract', 'judge_contract_inconsistent',
                        'high', False, ['failure_type=infra_failure conflicts with usable rag_answer'], judge,
                        answer=answer, trace=trace)
        error = _stage_error(trace)
        if error:
            block, mode = error
            return _row('execution', 'stage_error', block, mode, 'high', False, [f'error_stage={mode}'], judge,
                        answer=answer, trace=trace)
        return _row('runtime_infra', 'rag_or_judge_infra_failure', 'runtime_infra',
                    'rag_or_judge_infra_failure', 'high', False, ['failure_type=infra_failure'], judge, trace=trace)
    if _correct(case, judge, trace):
        return _row('ok', 'correct', 'not_applicable', 'correct', 'high', False, ['quality_label=good'], judge,
                    trace=trace)
    tracing = _tracing_defect(case, answer, judge, trace)
    if tracing:
        return _row('tracing', tracing, 'tracing_observability', tracing, 'medium', True, [tracing], judge,
                    trace=trace)
    retrieval = _retrieval(case, answer, judge, trace)
    if retrieval:
        return retrieval
    generation = _generation(case, answer, judge, trace)
    return generation or _row('undetermined', 'insufficient_evidence', 'undetermined', 'insufficient_evidence',
                              'low', True, ['no deterministic rule reached threshold'], judge, answer=answer,
                              trace=trace)


def is_actionable(row: Mapping[str, Any]) -> bool:
    return (
        row['issue_category'] in {'retrieval', 'generation', 'execution'}
        and row['affected_block'] != 'undetermined'
        and row['failure_mode'] != 'insufficient_evidence'
        and row['confidence'] in {'high', 'medium'}
        and not row['pending_analysis']
    )


def root_cause_reason(row: Mapping[str, Any]) -> str:
    return f"{row['issue_category']}/{row['issue_type']} at {row['affected_block']}: " + '; '.join(row['features'][:4])


def _retrieval(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any] | None:
    if judge.get('retrieval_failure_type') in {'none', 'not_applicable'}:
        return None
    retrieved_docs = semantic_ids(trace, answer, 'retrieved_doc_ids', 'doc_ids')
    retrieved_chunks = semantic_ids(trace, answer, 'retrieved_chunk_ids', 'chunk_ids')
    final_docs = semantic_ids(trace, answer, 'final_context_doc_ids', 'doc_ids')
    final_chunks = semantic_ids(trace, answer, 'final_context_chunk_ids', 'chunk_ids')
    ref_docs, ref_chunks = ids(case.get('reference_doc_ids')), ids(case.get('reference_chunk_ids'))
    doc_hit, chunk_hit = ref_docs & retrieved_docs, ref_chunks & retrieved_chunks
    final_hit = bool(ref_docs & final_docs or ref_chunks & final_chunks)
    features = [f'retrieval_failure_type={judge["retrieval_failure_type"]}',
                f'doc_recall={judge["doc_recall"]}', f'chunk_recall={judge["chunk_recall"]}',
                f'doc_precision={judge["doc_precision"]}', f'chunk_precision={judge["chunk_precision"]}']
    features += compact_features(
        judge,
        'core_explainers',
        keys=('retrieval_recall_at_k', 'retrieval_mrr', 'context_noise_rate'),
    )
    features += compact_features(
        judge,
        'specialized_metrics',
        keys=('retrieval_ndcg', 'retrieval_precision_at_k', 'context_relevance_avg'),
    )
    if ref_docs and not doc_hit and not chunk_hit:
        return _row('retrieval', 'reference_document_missing', 'retrieval', 'reference_document_missing', 'high',
                    False, features, judge, answer=answer, trace=trace, case=case)
    if doc_hit and ref_chunks and not chunk_hit:
        return _row('retrieval', 'reference_chunk_missing', 'retrieval', 'reference_chunk_missing', 'high', False,
                    features, judge, answer=answer, trace=trace, case=case)
    if (doc_hit or chunk_hit) and ref_chunks and not (ref_chunks & final_chunks):
        return _row('retrieval', 'context_assembly_failure', 'context_assembly', 'context_reference_chunk_dropped',
                    'high', False, features + ['final_context_missing_reference'], judge,
                    answer=answer, trace=trace, case=case)
    partial_seen = (
        (ref_docs and not ref_docs <= retrieved_docs)
        or (ref_chunks and (not ref_chunks <= retrieved_chunks or not ref_chunks <= final_chunks))
    )
    if (judge.get('retrieval_failure_type') == 'retrieval_partial' or _partial_recall(judge)) and partial_seen:
        return _row('retrieval', 'partial_reference_recall', 'retrieval', 'partial_reference_recall', 'medium',
                    False, features, judge, answer=answer, trace=trace, case=case)
    extra_context = (retrieved_docs - ref_docs) or (retrieved_chunks - ref_chunks)
    if (judge.get('retrieval_failure_type') == 'retrieval_noise' or _precision_low(judge)) and extra_context:
        block = 'rerank' if 'rerank' in trace.get('diagnostic_stage_sequence', []) else 'retrieval'
        mode = 'rerank_noise_promoted' if block == 'rerank' else 'retrieval_noise'
        issue = 'rerank_failure' if block == 'rerank' else 'retrieval_noise'
        return _row('retrieval', issue, block, mode, 'medium', False, features, judge,
                    answer=answer, trace=trace, case=case)
    if final_hit:
        return None
    return _row('undetermined', 'insufficient_trace_evidence', 'undetermined', 'insufficient_trace_evidence', 'low',
                True, features + ['trace_retrieval_evidence_does_not_confirm_judge_signal'], judge,
                answer=answer, trace=trace, case=case)


def _generation(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any] | None:
    failure = text(judge.get('failure_type'))
    if failure not in {'format_error', 'question_not_answered', 'partial_answer', 'wrong_answer', 'hallucination'}:
        return None
    healthy = _retrieval_healthy(case, judge, trace, answer)
    refs_absent = judge.get('retrieval_failure_type') == 'not_applicable'
    context_present = bool(
        semantic_ids(trace, answer, 'final_context_doc_ids', 'doc_ids')
        or semantic_ids(trace, answer, 'final_context_chunk_ids', 'chunk_ids')
    )
    pending = False
    llm_completed = _stage_completed(trace, 'llm_generate')
    if not (healthy or refs_absent) and failure != 'format_error':
        return None
    if failure != 'format_error' and not llm_completed:
        return _row('undetermined', 'insufficient_evidence', 'undetermined', 'insufficient_evidence', 'low', True,
                    [f'failure_type={failure}', 'llm_generate_completion_unobserved'], judge, answer=answer,
                    trace=trace, case=case)
    mapping = {
        'format_error': ('answer_format_error', 'postprocess_serialization', 'answer_format_error'),
        'question_not_answered': ('question_not_answered', 'llm_generation', 'question_not_answered'),
        'partial_answer': ('generation_incomplete_answer', 'llm_generation', 'generation_incomplete_answer'),
        'wrong_answer': ('generation_wrong_answer', 'llm_generation', 'generation_wrong_answer'),
        'hallucination': ('generation_hallucination', 'llm_generation', 'generation_hallucination'),
    }
    issue, block, mode = mapping[failure]
    confidence = 'medium' if pending or failure in {'question_not_answered', 'partial_answer'} else 'high'
    features = [f'failure_type={failure}', f'answer_quality_score={judge.get("answer_quality_score")}']
    features += compact_features(
        judge,
        'core_explainers',
        keys=('key_point_recall', 'claim_support_rate', 'answer_relevance'),
    )
    features += compact_features(
        judge,
        'specialized_metrics',
        keys=('numeric_accuracy', 'list_set_f1', 'contradiction_rate'),
    )
    features += diagnostic_count_features(
        judge,
        keys=('missing_points', 'wrong_points', 'unsupported_claims', 'contradicted_claims'),
    )
    if refs_absent:
        features.append('retrieval_not_applicable')
    if context_present:
        features.append('trace_context_present')
    return _row('generation', issue, block, mode, confidence, pending, features, judge, answer=answer, trace=trace,
                case=case)


def _row(category: str, issue: str, block: str, mode: str, confidence: str, pending: bool, features: list[str],
         judge: Mapping[str, Any], *, answer: Mapping[str, Any] | None = None,
         trace: Mapping[str, Any] | None = None, case: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        'issue_category': category,
        'issue_type': issue,
        'affected_block': block,
        'failure_mode': mode,
        'pending_analysis': pending,
        'confidence': confidence,
        'features': unique(features),
        'answer_evidence': answer_evidence(answer or {}),
        'judge_evidence': judge_evidence(judge),
        'trace_evidence': trace_evidence(trace or {}, case or {}, answer or {}),
        'investigation_note': _note(category, issue, block, case or {}),
    }


def _judge_inconsistency(judge: Mapping[str, Any]) -> list[str]:
    issues = []
    good_threshold = _good_threshold(judge)
    if judge.get('quality_label') == 'good' and (
        judge.get('failure_type') != 'none'
        or score(judge, 'overall_score') < good_threshold
        or score(judge, 'answer_quality_score') < good_threshold
        or (judge.get('retrieval_failure_type') != 'not_applicable'
            and score(judge, 'retrieval_quality_score') < good_threshold)
    ):
        issues.append('quality_label=good conflicts with failure/score')
    if judge.get('failure_type') == 'none' and judge.get('retrieval_failure_type') not in {'none', 'not_applicable'}:
        issues.append('failure_type=none conflicts with retrieval failure')
    if judge.get('failure_type') == 'none' and score(judge, 'answer_quality_score') < good_threshold:
        issues.append('failure_type=none conflicts with answer_quality_score')
    if (
        judge.get('retrieval_failure_type') == 'not_applicable'
        and judge.get('failure_type') not in {'infra_failure', 'judge_contract_error', 'dataset_contract_error'}
    ):
        nested_case = judge.get('case') if isinstance(judge.get('case'), Mapping) else {}
        if ids(nested_case.get('reference_doc_ids')) or ids(nested_case.get('reference_chunk_ids')):
            issues.append('retrieval_failure_type=not_applicable conflicts with reference ids')
    if judge.get('is_correct') is True and (
        judge.get('quality_label') != 'good' or judge.get('failure_type') != 'none'
    ):
        issues.append('is_correct=true conflicts with quality/failure')
    if judge.get('is_correct') is False and judge.get('quality_label') == 'good':
        issues.append('is_correct=false conflicts with quality_label=good')
    return issues


def _stage_error(trace: Mapping[str, Any]) -> tuple[str, str] | None:
    mapping = {
        'tool_call': ('tool_orchestration', 'tool_error'),
        'retrieve': ('retrieval', 'retrieval_stage_error'),
        'rerank': ('rerank', 'rerank_stage_error'),
        'context_assembly': ('context_assembly', 'context_assembly_stage_error'),
        'prompt_build': ('prompt_build', 'prompt_build_stage_error'),
        'llm_generate': ('llm_generation', 'llm_generation_stage_error'),
        'postprocess': ('postprocess_serialization', 'postprocess_stage_error'),
        'stream': ('postprocess_serialization', 'stream_truncation'),
        'unknown': ('tracing_observability', 'trace_stage_unknown'),
    }
    priority = {stage: index for index, stage in enumerate(mapping)}
    errors = [
        (priority[str(item['stage'])], mapping[str(item['stage'])])
        for item in trace.get('error_stages') or []
        if isinstance(item, Mapping) and item.get('stage') in mapping
    ]
    if errors:
        return min(errors, key=lambda item: item[0])[1]
    return None


def _tracing_defect(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> str:
    unknown_value = trace.get('unknown_stage_count')
    if unknown_value is None:
        unknown_value = (trace.get('stage_counts') or {}).get('unknown') or 0
    unknown = int(unknown_value)
    if unknown and ('unknown' in trace.get('critical_path', []) or judge.get('quality_label') != 'good'):
        return 'trace_stage_unknown'
    needs_ids = (
        judge.get('retrieval_failure_type') != 'not_applicable'
        and (ids(case.get('reference_doc_ids')) or ids(case.get('reference_chunk_ids')))
    )
    refs_exist = bool(ids(case.get('reference_doc_ids')) or ids(case.get('reference_chunk_ids')))
    answer_failed = judge.get('failure_type') not in {'none', 'infra_failure'}
    retrieved_trace_ids = bool(
        trace_semantic_ids(trace, 'retrieved_doc_ids')
        or trace_semantic_ids(trace, 'retrieved_chunk_ids')
    )
    final_trace_ids = bool(
        trace_semantic_ids(trace, 'final_context_doc_ids')
        or trace_semantic_ids(trace, 'final_context_chunk_ids')
    )
    retrieved_ids = bool(
        retrieved_trace_ids
        or semantic_ids(trace, answer, 'retrieved_doc_ids', 'doc_ids')
        or semantic_ids(trace, answer, 'retrieved_chunk_ids', 'chunk_ids')
    )
    final_ids = bool(
        final_trace_ids
        or semantic_ids(trace, answer, 'final_context_doc_ids', 'doc_ids')
        or semantic_ids(trace, answer, 'final_context_chunk_ids', 'chunk_ids')
    )
    if refs_exist and judge.get('retrieval_failure_type') == 'not_applicable':
        return 'trace_metrics_missing'
    if needs_ids and not trace.get('retrieval_steps'):
        return 'trace_metrics_missing'
    if needs_ids and trace.get('semantic_metric_keys') and not retrieved_trace_ids:
        return 'trace_metrics_missing'
    if answer_failed and refs_exist and trace.get('semantic_metric_keys') and not final_trace_ids:
        return 'trace_metrics_missing'
    if needs_ids and semantic_fallback_enabled(trace) and not retrieved_ids:
        return 'trace_metrics_missing'
    if answer_failed and refs_exist and semantic_fallback_enabled(trace) and not final_ids:
        return 'trace_metrics_missing'
    return ''


def _correct(case: Mapping[str, Any], judge: Mapping[str, Any], trace: Mapping[str, Any]) -> bool:
    refs_exist = bool(ids(case.get('reference_doc_ids')) or ids(case.get('reference_chunk_ids')))
    retrieval_ok = (
        judge.get('retrieval_failure_type') == 'none'
        or (judge.get('retrieval_failure_type') == 'not_applicable' and not refs_exist)
    )
    return (
        judge.get('quality_label') == 'good'
        and judge.get('failure_type') == 'none'
        and retrieval_ok
        and judge.get('is_correct') is not False
    )


def _retrieval_healthy(
    case: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
    answer: Mapping[str, Any] | None = None,
) -> bool:
    ref_docs, ref_chunks = ids(case.get('reference_doc_ids')), ids(case.get('reference_chunk_ids'))
    answer = answer or {}
    final_docs = semantic_ids(trace, answer, 'final_context_doc_ids', 'doc_ids')
    final_chunks = semantic_ids(trace, answer, 'final_context_chunk_ids', 'chunk_ids')
    overlap_ok = not (ref_docs or ref_chunks) or bool(ref_docs & final_docs or ref_chunks & final_chunks)
    threshold = _good_threshold(judge)
    ranked_recall_ok = (
        'retrieval_recall_at_k' not in judge
        or score(judge, 'retrieval_recall_at_k') >= threshold
    )
    noise_ok = 'context_noise_rate' not in judge or score(judge, 'context_noise_rate') <= 0.40
    return (
        judge.get('retrieval_failure_type') == 'none'
        and score(judge, 'retrieval_quality_score') >= threshold
        and score(judge, 'context_recall') >= threshold
        and (not ref_docs or score(judge, 'doc_recall') >= threshold)
        and (not ref_chunks or score(judge, 'chunk_recall') >= threshold)
        and ranked_recall_ok
        and noise_ok
        and overlap_ok
    )


def _note(category: str, issue: str, block: str, case: Mapping[str, Any]) -> str:
    qtype = text(case.get('question_type'))
    suffix = f' for {qtype}' if qtype else ''
    return f'inspect {block} evidence for {category}/{issue}{suffix}'


def _infra_source(answer: Mapping[str, Any]) -> str:
    error = answer.get('chat_error')
    if isinstance(error, Mapping):
        return 'chat_error=' + text(error.get('type') or error.get('code') or 'unknown')
    return 'rag_answer.status=' + text(answer.get('status'))


def _answer_has_evidence(answer: Mapping[str, Any]) -> bool:
    return bool(
        text(answer.get('answer'))
        and (ids(answer.get('contexts')) or ids(answer.get('doc_ids')) or ids(answer.get('chunk_ids')))
    )


def _precision_low(judge: Mapping[str, Any]) -> bool:
    if 'retrieval_precision_at_k' in judge and score(judge, 'retrieval_precision_at_k') < 0.40:
        return True
    if 'context_noise_rate' in judge and score(judge, 'context_noise_rate') > 0.50:
        return True
    return score(judge, 'doc_precision') < 0.40 or score(judge, 'chunk_precision') < 0.40


def _partial_recall(judge: Mapping[str, Any]) -> bool:
    if 'retrieval_recall_at_k' in judge:
        return 0 < score(judge, 'retrieval_recall_at_k') < 1.0
    return 0 < score(judge, 'doc_recall') < 0.75 or 0 < score(judge, 'chunk_recall') < 0.75


def _stage_completed(trace: Mapping[str, Any], stage: str) -> bool:
    return any(item.get('stage') == stage and item.get('status') in {'ok', 'success', 'done', 'completed', 'finished'}
               for item in trace.get('stages') or [] if isinstance(item, Mapping))


def _good_threshold(judge: Mapping[str, Any]) -> float:
    return policy_number(judge, 'answer_good_threshold', 0.75)


def answer_evidence(answer: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = []
    if not text(answer.get('answer')):
        items.append(evidence('empty_answer', 'rag_answer.answer', ''))
    if answer.get('chat_error'):
        items.append(evidence('chat_error', 'rag_answer.chat_error', answer.get('chat_error')))
    return items


def trace_evidence(
    trace: Mapping[str, Any],
    case: Mapping[str, Any],
    answer: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ref_docs, ref_chunks = ids(case.get('reference_doc_ids')), ids(case.get('reference_chunk_ids'))
    answer = answer or {}
    retrieved_docs = semantic_ids(trace, answer, 'retrieved_doc_ids', 'doc_ids')
    retrieved_chunks = semantic_ids(trace, answer, 'retrieved_chunk_ids', 'chunk_ids')
    final_docs = semantic_ids(trace, answer, 'final_context_doc_ids', 'doc_ids')
    final_chunks = semantic_ids(trace, answer, 'final_context_chunk_ids', 'chunk_ids')
    source = semantic_id_source(trace, answer)
    return [
        evidence('route_signature', 'analysis.trace_summary.route_signature', trace.get('route_signature')),
        evidence(
            'stage_sequence',
            'analysis.trace_summary.diagnostic_stage_sequence',
            trace.get('diagnostic_stage_sequence') or [],
        ),
        evidence(
            'unknown_stage_count',
            'analysis.trace_summary.unknown_stage_count',
            trace.get('unknown_stage_count') or 0,
        ),
        evidence('error_stage', 'analysis.trace_summary.error_stages', trace.get('error_stages') or []),
        evidence('semantic_id_source', 'analysis.trace_summary.semantic_metric_keys', source),
        evidence(
            'retrieved_doc_overlap',
            'analysis.trace_summary.retrieved_doc_ids',
            sorted(ref_docs & retrieved_docs),
        ),
        evidence(
            'retrieved_chunk_overlap',
            'analysis.trace_summary.retrieved_chunk_ids',
            sorted(ref_chunks & retrieved_chunks),
        ),
        evidence(
            'final_context_reference_overlap',
            'analysis.trace_summary.final_context_ids',
            {
                'doc_ids': sorted(ref_docs & final_docs),
                'chunk_ids': sorted(ref_chunks & final_chunks),
            },
        ),
    ]


def semantic_ids(
    trace: Mapping[str, Any],
    answer: Mapping[str, Any],
    trace_key: str,
    answer_key: str,
) -> set[str]:
    trace_ids = trace_semantic_ids(trace, trace_key)
    if trace_ids or not semantic_fallback_enabled(trace):
        return trace_ids
    return ids(answer.get(answer_key))


def trace_semantic_ids(trace: Mapping[str, Any], trace_key: str) -> set[str]:
    return ids(trace.get(trace_key))


def semantic_id_source(trace: Mapping[str, Any], answer: Mapping[str, Any]) -> str:
    trace_ids = (
        trace_semantic_ids(trace, 'retrieved_doc_ids')
        or trace_semantic_ids(trace, 'retrieved_chunk_ids')
        or trace_semantic_ids(trace, 'final_context_doc_ids')
        or trace_semantic_ids(trace, 'final_context_chunk_ids')
    )
    if trace_ids:
        return 'trace'
    if semantic_fallback_enabled(trace) and (ids(answer.get('doc_ids')) or ids(answer.get('chunk_ids'))):
        return 'rag_answer_fallback'
    if trace.get('semantic_metric_keys'):
        return 'trace_missing_ids'
    return 'trace'


def semantic_fallback_enabled(trace: Mapping[str, Any]) -> bool:
    return bool(trace.get('retrieval_steps')) and not bool(trace.get('semantic_metric_keys'))


def ids(value: Any) -> set[str]:
    items = [value] if isinstance(value, str) else list(value or [])
    return {str(item).strip() for item in items if str(item or '').strip()}


def score(judge: Mapping[str, Any], key: str) -> float:
    return float(judge.get(key) or 0.0)


def text(value: Any) -> str:
    return str(value or '').strip()


def unique(items: list[str]) -> list[str]:
    return [item for item in dict.fromkeys(str(value) for value in items if str(value or '').strip())]
