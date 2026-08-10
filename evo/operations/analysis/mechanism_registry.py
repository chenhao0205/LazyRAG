from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _mechanism(
    mechanism_id: str,
    surface: str,
    stage: str,
    affected_block: str,
    failure_mode: str,
    review_package: str,
    confirmation_mode: str,
    repair_owner: str,
    validation_focus: list[str],
    order: int,
    *,
    requires_probe: bool = False,
) -> Mapping[str, Any]:
    return {
        'id': mechanism_id,
        'surface': surface,
        'stage': stage,
        'affected_block': affected_block,
        'failure_mode': failure_mode,
        'review_package': review_package,
        'confirmation_mode': confirmation_mode,
        'requires_probe': requires_probe,
        'repair_owner': repair_owner,
        'validation_focus': validation_focus,
        'order': order,
    }


MECHANISM_REGISTRY: tuple[Mapping[str, Any], ...] = (
    _mechanism('query.intent_lost', 'query_planning', 'query_rewrite', 'query_rewrite',
               'query_intent_lost', 'query_intent_review', 'semantic_review', 'query_rewrite',
               ['answer_relevance', 'retrieval_recall_at_k'], 10),
    _mechanism('retrieve.reference_absent', 'retrieve', 'retrieve', 'retrieval',
               'reference_document_missing', 'candidate_equivalence_review', 'trace_or_semantic_review', 'retrieval',
               ['doc_recall', 'chunk_recall', 'retrieval_recall_at_k'], 20),
    _mechanism('rerank.relevant_candidate_demoted', 'rerank', 'rerank', 'rerank',
               'rerank_noise_promoted', 'candidate_priority_review', 'semantic_review_or_probe', 'rerank',
               ['retrieval_ndcg', 'retrieval_mrr', 'context_precision'], 30, requires_probe=True),
    _mechanism('context.required_evidence_dropped', 'context_selection', 'context_assembly', 'context_assembly',
               'context_reference_chunk_dropped', 'context_completeness_review', 'trace_or_semantic_review',
               'context_assembly', ['context_recall', 'chunk_recall', 'groundedness'], 40),
    _mechanism('context.context_insufficient', 'context_selection', 'prompt_build', 'prompt_build',
               'prompt_context_insufficient', 'context_completeness_review', 'semantic_review', 'prompt_build',
               ['context_recall', 'answer_correctness', 'groundedness'], 50),
    _mechanism('answer.available_context_ignored', 'answer_generation', 'llm_generate', 'llm_generation',
               'generation_incomplete_answer', 'answer_faithfulness_review', 'semantic_review', 'llm_generation',
               ['answer_correctness', 'key_point_recall', 'claim_support_rate'], 60),
    _mechanism('answer.unsupported_or_contradicted', 'answer_generation', 'llm_generate', 'llm_generation',
               'generation_hallucination', 'answer_faithfulness_review', 'semantic_review', 'llm_generation',
               ['groundedness', 'claim_support_rate', 'contradiction_rate'], 70),
    _mechanism('judge.judge_conflict', 'judge_conflict', 'judge', 'eval_contract',
               'judge_conflict', 'judge_conflict_review', 'semantic_review', 'eval_contract',
               ['overall_score', 'reason'], 80),
    _mechanism('execution.stage_error', 'execution', 'runtime', 'tool_orchestration',
               'stage_error', '', 'trace', 'tool_orchestration', ['error_span_count'], 5),
    _mechanism('trace.metrics_missing', 'tracing_observability', 'trace', 'tracing_observability',
               'trace_metrics_missing', '', 'trace', 'tracing_observability', ['trace_quality'], 90),
)


def _probe(
    mechanisms: list[str],
    fixed_variables: list[str],
    compare: list[str],
    *,
    kind: str = 'readonly_replay',
) -> Mapping[str, Any]:
    return {
        'mechanisms': mechanisms,
        'kind': kind,
        'fixed_variables': fixed_variables,
        'compare': compare,
    }


PROBE_REGISTRY: Mapping[str, Mapping[str, Any]] = {
    'retrieve.rank_expand_replay': _probe(
        ['retrieve.reference_absent'],
        ['query', 'corpus/index configuration', 'filters', 'retriever configuration'],
        ['hit_at_current_k', 'hit_at_deeper_k', 'first_support_rank'],
    ),
    'query.retrieve_ab': _probe(
        ['query.intent_lost', 'retrieve.reference_absent'],
        ['corpus/index configuration', 'filters', 'retriever configuration', 'top_k'],
        ['original_query_hits', 'rewrite_query_hits', 'support_rank_delta'],
    ),
    'rerank.selection_replay': _probe(
        ['rerank.relevant_candidate_demoted'],
        ['query', 'candidate set', 'reranker configuration', 'case target'],
        ['rank_before', 'rank_after', 'score_before', 'score_after', 'selected'],
    ),
    'context.selection_replay': _probe(
        ['context.required_evidence_dropped', 'context.context_insufficient'],
        ['ranked candidates', 'selection configuration', 'case target'],
        ['selected ids', 'excluded required ids', 'token budget/cutoff'],
    ),
    'postprocess.serialization_diff': _probe(
        ['answer.available_context_ignored', 'answer.unsupported_or_contradicted'],
        ['selected nodes', 'serializer configuration', 'final request artifact'],
        ['raw_node_text', 'serialized_text', 'final_request_text'],
        kind='readonly_diff',
    ),
    'index.presence_probe': _probe(
        ['retrieve.reference_absent'],
        ['source ids', 'parser configuration', 'index/store configuration'],
        ['source_presence', 'parsed_node_presence', 'index_entry_presence'],
        kind='readonly_diff',
    ),
}


def registered_probes_for(mechanism_id: str) -> list[dict[str, Any]]:
    return [
        {'probe_id': probe_id, **dict(spec)}
        for probe_id, spec in PROBE_REGISTRY.items()
        if mechanism_id in spec.get('mechanisms', ())
    ]
