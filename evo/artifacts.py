from types import MappingProxyType


RUN_CONFIG = 'run.config'
CORPUS_SOURCE_CONFIG = 'corpus.source_config'
EVAL_TARGET_CONFIG = 'eval.target_config'
EVAL_POLICY = 'eval.policy'
REPAIR_POLICY = 'repair.policy'
ABTEST_CANDIDATE_CONFIG = 'abtest.candidate_config'

CORPUS_REPORT = 'corpus.report'
CORPUS_SNAPSHOT = 'corpus.snapshot'
DATASET_IMPORT_CASES_MANIFEST = 'dataset.import_cases_manifest'
DATASET_SELECTED_DOCS = 'dataset.selected_docs'
DATASET_BUILD_CHUNK_CANDIDATES = 'dataset.build_chunk_candidates'
DATASET_CHUNK_REQUESTS = 'dataset.chunk_requests'
DATASET_CHUNK_REQUEST = 'dataset.chunk_request'
DATASET_CHUNK = 'dataset.chunk'
DATASET_BUILD_CHUNKS_MANIFEST = 'dataset.build_chunks_manifest'
DATASET_CHUNK_ENTITY = 'dataset.chunk_entity'
DATASET_CHUNK_ENTITIES_MANIFEST = 'dataset.chunk_entities_manifest'
DATASET_ENTITY_GRAPH = 'dataset.entity_graph'
DATASET_ENTITY_CLUSTERS = 'dataset.entity_clusters'
DATASET_EMBEDDING_CLUSTER_CANDIDATES = 'dataset.embedding_cluster_candidates'
DATASET_EMBEDDING_CLUSTERS = 'dataset.embedding_clusters'
DATASET_TOPIC_MANIFEST = 'dataset.topic_manifest'
DATASET_QAPLAN_PLAN = 'dataset.qaplan_plan'
DATASET_QAPLAN_SPEC = 'dataset.qaplan_spec'
DATASET_QAPLAN_MANIFEST = 'dataset.qaplan_manifest'
DATASET_CASE_DRAFT = 'dataset.case_draft'
DATASET_CASE_ENHANCEMENT = 'dataset.case_enhancement'
DATASET_GENERATE_MANIFEST = 'dataset.generate_manifest'
DATASET_ENHANCE_MANIFEST = 'dataset.enhance_manifest'
EVAL_CASE_REQUESTS = 'eval.case_requests'
EVAL_CASE_REQUEST = 'eval.case_request'
EVAL_CASE_PREPARATION = 'eval.case_preparations'
EVAL_CASE = 'eval.cases'
EVAL_DATASET = 'eval.dataset'
EVAL_RAG_ANSWER = 'eval.rag_answers'
EVAL_JUDGE_RESULT = 'eval.judge_results'
EVAL_SUMMARY = 'eval.summary'
ANALYSIS_TRACE_SUMMARY = 'analysis.trace_summaries'
ANALYSIS_CASE_CLASSIFICATION = 'analysis.case_classifications'
ANALYSIS_TRACE_CLUSTERS = 'analysis.trace_clusters'
ANALYSIS_SUMMARY = 'analysis.summary'
REPAIR_VERIFIED_PATCH = 'repair.verified_patch'
ABTEST_CANDIDATE_SERVICE = 'abtest.candidate_service'
ABTEST_CANDIDATE_RAG_ANSWER = 'abtest.candidate_rag_answers'
ABTEST_CANDIDATE_JUDGE_RESULT = 'abtest.candidate_judge_results'
ABTEST_CANDIDATE_EVAL_SUMMARY = 'abtest.candidate_eval_summary'
ABTEST_COMPARISON = 'abtest.comparison'

APPROVAL_DATASET = 'approval.dataset'
APPROVAL_EVAL = 'approval.eval'
APPROVAL_ANALYSIS = 'approval.analysis'
APPROVAL_REPAIR = 'approval.repair'

STEPS = ('dataset', 'eval', 'analysis', 'repair', 'abtest')

SEEDS = (
    RUN_CONFIG,
    CORPUS_SOURCE_CONFIG,
    EVAL_TARGET_CONFIG,
    EVAL_POLICY,
    REPAIR_POLICY,
    ABTEST_CANDIDATE_CONFIG,
)

ROOTS = MappingProxyType({
    'dataset': EVAL_DATASET,
    'eval': EVAL_SUMMARY,
    'analysis': ANALYSIS_SUMMARY,
    'repair': REPAIR_VERIFIED_PATCH,
    'abtest': ABTEST_COMPARISON,
})

APPROVALS = MappingProxyType({
    'dataset': APPROVAL_DATASET,
    'eval': APPROVAL_EVAL,
    'analysis': APPROVAL_ANALYSIS,
    'repair': APPROVAL_REPAIR,
})

PARTITION_SET_BY_ARTIFACT = MappingProxyType({
    DATASET_CHUNK_REQUEST: DATASET_CHUNK_REQUESTS,
    DATASET_CHUNK: DATASET_CHUNK_REQUESTS,
    DATASET_CHUNK_ENTITY: DATASET_CHUNK_REQUESTS,
    EVAL_CASE_REQUEST: EVAL_CASE_REQUESTS,
    EVAL_CASE_PREPARATION: EVAL_CASE_REQUESTS,
    DATASET_QAPLAN_SPEC: EVAL_CASE_REQUESTS,
    DATASET_CASE_DRAFT: EVAL_CASE_REQUESTS,
    DATASET_CASE_ENHANCEMENT: EVAL_CASE_REQUESTS,
    EVAL_CASE: EVAL_CASE_REQUESTS,
    EVAL_RAG_ANSWER: EVAL_CASE_REQUESTS,
    EVAL_JUDGE_RESULT: EVAL_CASE_REQUESTS,
    ANALYSIS_TRACE_SUMMARY: EVAL_CASE_REQUESTS,
    ANALYSIS_CASE_CLASSIFICATION: EVAL_CASE_REQUESTS,
    ABTEST_CANDIDATE_RAG_ANSWER: EVAL_CASE_REQUESTS,
    ABTEST_CANDIDATE_JUDGE_RESULT: EVAL_CASE_REQUESTS,
})


__all__ = [name for name in globals() if name.isupper()]
