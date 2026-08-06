import json
import sys
import types
import asyncio


fake_router = types.ModuleType('evo.operations.chat_router')


class RouterChatRequest:
    pass


fake_router.RouterChatRequest = RouterChatRequest
fake_router.call_router_chat = lambda request: {}
sys.modules.setdefault('evo.operations.chat_router', fake_router)

fake_repair = types.ModuleType('json_repair')
fake_repair.repair_json = lambda raw, return_objects=False: json.loads(raw) if return_objects else raw
sys.modules.setdefault('json_repair', fake_repair)

from evo.operations.eval.judge import judge_case
from evo.operations.eval.materializers import (
    build_eval_detail_summary,
    build_eval_frontend_view,
    eval_materializers,
)
from evo.operations.eval.answer import call_chat_answer, case_kb_id, _with_case
from evo.operations.route.chat_router import RouterChatRequest, async_call_router_chat


class FakeLLM:
    def __init__(self, *, llm_config=None, model=None):
        self.llm_config = llm_config
        self.model = model

    def __call__(self, prompt, **kwargs):
        return json.dumps({
            'answer_correctness': 0.9,
            'answer_relevance': 1.0,
            'completeness': 0.8,
            'groundedness': 0.8,
            'format_compliance': 1.0,
            'failure_type': 'none',
            'reason': 'legacy judge output without diagnostic fields',
            'defect': '',
        })


class FakeContradictionLLM(FakeLLM):
    def __call__(self, prompt, **kwargs):
        return json.dumps({
            'answer_correctness': 0.8,
            'answer_relevance': 1.0,
            'completeness': 0.8,
            'groundedness': 0.7,
            'format_compliance': 1.0,
            'failure_type': 'none',
            'reason': 'judge detected a forbidden claim',
            'defect': '',
            'contradiction_rate': 1.0,
            'contradicted_claims': ['forbidden claim'],
        })


class FakeBadLLM(FakeLLM):
    def __call__(self, prompt, **kwargs):
        return json.dumps({
            'answer_correctness': 0.1,
            'answer_relevance': 0.3,
            'completeness': 0.1,
            'groundedness': 0.2,
            'format_compliance': 1.0,
            'failure_type': 'wrong_answer',
            'reason': 'answer is wrong',
            'defect': 'wrong_answer',
        })


class FakeCapturePromptLLM(FakeLLM):
    prompt = ''

    def __call__(self, prompt, **kwargs):
        type(self).prompt = prompt
        return super().__call__(prompt, **kwargs)


class FakeClaimsWithoutMappingLLM(FakeLLM):
    def __call__(self, prompt, **kwargs):
        return json.dumps({
            'answer_correctness': 0.9,
            'answer_relevance': 1.0,
            'completeness': 0.9,
            'groundedness': 0.9,
            'format_compliance': 1.0,
            'failure_type': 'none',
            'reason': 'claims returned without usable mapping',
            'defect': '',
            'claims': [{'text': 'The supported fact is alpha.', 'supported': True}],
            'evidence_mapping': [],
        })


class FakeZeroDiagnosticsLLM(FakeLLM):
    def __call__(self, prompt, **kwargs):
        return json.dumps({
            'answer_correctness': 0.9,
            'answer_relevance': 1.0,
            'completeness': 0.9,
            'groundedness': 0.9,
            'format_compliance': 1.0,
            'failure_type': 'none',
            'reason': 'diagnostic zero values are intentional',
            'defect': '',
            'key_point_recall': 0.0,
            'claim_support_rate': 0.0,
            'semantic_similarity': 0.0,
            'retrieval_recall_at_k': 0.0,
            'retrieval_mrr': 0.0,
            'retrieval_ndcg': 0.0,
            'context_relevance_avg': 0.0,
            'claims': [{'text': 'Unsupported claim.', 'supported': False}],
            'unsupported_claims': ['Unsupported claim.'],
        })


def test_eval_answer_keeps_context_chunk_alignment_for_duplicate_text():
    result = {
        'status': 'ok',
        'answer': 'final answer',
        'target': {'trace_id': 'trace-router', 'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
        'sources': [
            {'content': 'same text', 'doc_id': 'doc-a', 'chunk_id': 'chunk-a'},
            {'content': 'same text', 'doc_id': 'doc-b', 'chunk_id': 'chunk-b'},
            {'content': 'other text', 'doc_id': 'doc-c', 'chunk_id': 'chunk-c'},
        ],
    }

    answer = _with_case({'id': 'case-align', 'question': 'q'}, result)

    assert answer['contexts'] == [
        {'content': 'same text', 'doc_id': 'kb-a:doc-a', 'doc_name': 'kb-a:doc-a', 'chunk_id': 'kb-a:doc-a:chunk-a'},
        {'content': 'same text', 'doc_id': 'kb-a:doc-b', 'doc_name': 'kb-a:doc-b', 'chunk_id': 'kb-a:doc-b:chunk-b'},
        {'content': 'other text', 'doc_id': 'kb-a:doc-c', 'doc_name': 'kb-a:doc-c', 'chunk_id': 'kb-a:doc-c:chunk-c'},
    ]
    assert answer['chunk_ids'] == [
        'kb-a:doc-a:chunk-a',
        'kb-a:doc-b:chunk-b',
        'kb-a:doc-c:chunk-c',
    ]


def test_eval_answer_passes_chat_retry_policy(monkeypatch):
    captured = {}

    def fake_call_router_chat(request):
        captured['request'] = request
        return {
            'status': 'ok',
            'answer': 'final answer',
            'trace_id': request.trace_id,
            'target': {
                'trace_id': request.trace_id,
                'algorithm_id': request.algorithm_id,
                'kb_id': ';'.join(request.kb_ids),
            },
        }

    import evo.operations.eval.answer as answer_module

    monkeypatch.setattr(answer_module, 'call_router_chat', fake_call_router_chat)

    result = call_chat_answer(
        {'id': 'case-retry', 'question': 'q'},
        {
            'router_chat_url': 'http://router.local/api/chat/stream',
            'router_admin_url': 'http://router.local',
            'algorithm_id': 'algo-a',
            'session_id': '0' * 32,
            'llm_config': {'llm': {'model': 'fake'}},
            'chat_max_attempts': 3,
            'chat_retry_wait_max_seconds': 0,
        },
        'kb-a',
    )

    request = captured['request']
    assert request.max_attempts == 3
    assert request.retry_wait_max_seconds == 0
    assert result['status'] == 'ok'


def test_eval_answer_reads_dataset_pr5_kb_ids_from_source_preparation():
    case = {
        'id': 'case-pr5',
        'question': 'q',
        'source_preparation': {
            'kb_ids': ['kb-a', 'kb-b', 'kb-a'],
        },
    }

    assert case_kb_id(case, {}) == 'kb-a;kb-b'


def test_router_chat_records_retry_attempt_history(monkeypatch):
    attempts = []

    async def fake_call_once(request: RouterChatRequest):
        attempts.append(request.trace_id)
        if len(attempts) == 1:
            return {
                'status': 'failed',
                'trace_id': request.trace_id,
                'chat_error': {'type': 'chat_transport_error', 'message': 'connection reset'},
                'target': {'trace_id': request.trace_id, 'conversation_id': request.conversation_id},
            }
        return {
            'status': 'ok',
            'answer': 'final answer',
            'trace_id': request.trace_id,
            'routed_instance_host': 'worker-1',
            'target': {'trace_id': request.trace_id, 'conversation_id': request.conversation_id},
        }

    import evo.operations.route.chat_router as chat_router

    monkeypatch.setattr(chat_router, '_call_router_chat_once', fake_call_once)
    result = asyncio.run(async_call_router_chat(RouterChatRequest(
        router_chat_url='http://router.local/api/chat/stream',
        router_admin_url='http://router.local',
        algorithm_id='algo-a',
        query='q',
        kb_ids=('kb-a',),
        trace_id='0' * 32,
        max_attempts=2,
        retry_wait_max_seconds=0,
    )))

    assert result['status'] == 'ok'
    assert result['chat_attempt_count'] == 2
    assert result['chat_attempts'][0]['error_type'] == 'chat_transport_error'
    assert result['chat_attempts'][1]['status'] == 'ok'
    assert attempts[0] != attempts[1]


def test_zero_diagnostic_scores_do_not_fallback_to_positive_defaults(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeZeroDiagnosticsLLM)
    case = {
        'id': 'case_zero',
        'question': 'What fact is supported?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'alpha', 'statement': 'The supported fact is alpha.', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': {'chunk-a': 'The supported fact is alpha.'},
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_zero',
        'answer': 'Unsupported claim.',
        'contexts': [{'chunk_id': 'chunk-a', 'content': 'The supported fact is alpha.', 'rank': 1}],
        'chunk_ids': ['chunk-a'],
        'doc_ids': [],
        'trace_id': 'trace-zero',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['answer_quality_score'] == 0.555
    assert result['overall_score'] == 0.644
    assert result['quality_label'] == 'partial'
    assert result['failure_type'] == 'partial_answer'


def test_evidence_mapping_falls_back_and_separates_reference_and_retrieved_support(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeClaimsWithoutMappingLLM)
    case = {
        'id': 'case_mapping',
        'question': 'What fact is supported?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'alpha', 'statement': 'The supported fact is alpha.', 'evidence_chunk_ids': ['chunk-ref']},
        ],
        'reference_chunk_ids': ['chunk-ref'],
        'reference_context': {'chunk-ref': 'The supported fact is alpha.'},
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_mapping',
        'answer': 'The supported fact is alpha.',
        'contexts': [{'chunk_id': 'chunk-ret', 'content': 'The supported fact is alpha.', 'rank': 1}],
        'chunk_ids': ['chunk-ret'],
        'doc_ids': [],
        'trace_id': 'trace-mapping',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['evidence_mapping'] == [{
        'claim': 'The supported fact is alpha.',
        'reference_support': {'evidence_chunk_id': 'chunk-ref', 'score': 1.0},
        'retrieved_support': {'evidence_chunk_id': 'chunk-ret', 'score': 1.0},
        'derivation': 'fallback',
    }]
    assert 'evidence' not in result['evidence_mapping'][0]


def test_judge_adds_key_point_and_rank_diagnostics(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0001',
        'question': 'When is launch and what is the price?',
        'answer': 'The launch date is July 8. The price is 20 USD.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'date', 'statement': 'The launch date is July 8.', 'evidence_chunk_ids': ['chunk-a']},
            {'id': 'price', 'statement': 'The price is 20 USD.', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'reference_chunk_ids': ['chunk-a'],
        'reference_doc_ids': ['doc-a'],
        'reference_context': [
            'The launch date is July 8. The price is 20 USD.',
        ],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_0001',
        'answer': 'The launch date is July 8.',
        'contexts': [
            {'chunk_id': 'chunk-a', 'doc_id': 'doc-a', 'content': 'The launch date is July 8.', 'rank': 1},
            {'chunk_id': 'chunk-noise', 'doc_id': 'doc-noise', 'content': 'Unrelated content.', 'rank': 2},
        ],
        'chunk_ids': ['chunk-a', 'chunk-noise'],
        'doc_ids': ['doc-a', 'doc-noise'],
        'trace_id': 'trace-1',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['key_point_recall'] == 0.5
    assert result['matched_key_points'][0]['id'] == 'date'
    assert result['missing_points'][0]['id'] == 'price'
    assert result['retrieval_hit_at_k'] == 1.0
    assert result['retrieval_mrr'] == 1.0
    assert result['retrieval_precision_at_k'] == 0.5
    assert result['numeric_accuracy'] == 0.5
    assert 'list_set_f1' not in result
    assert 'contradiction_rate' not in result
    assert 'contradicted_claims' not in result
    assert result['answer_quality_score'] < 0.9
    assert result['quality_label'] == 'partial'
    assert result['metric_layers']['primary_scores'] == [
        'overall_score',
        'answer_quality_score',
        'retrieval_quality_score',
        'quality_label',
        'failure_type',
        'retrieval_failure_type',
    ]
    assert result['score_breakdown']['answer_quality_score']['weights']['key_point_recall'] == 0.20


def test_eval_judge_materializer_merges_dataset_pr5_case_enhance(monkeypatch):
    import evo.operations.eval.materializers as materializers_module

    captured = {}

    def fake_judge_case(case, answer, policy):
        captured['case'] = case
        return {
            'case_id': case['id'],
            'case': dict(case),
            'rag_answer': dict(answer),
            'trace_id': 'trace-enhance',
            'target': {},
            'tool_errors': [],
            'answer_correctness': 1.0,
            'answer_relevance': 1.0,
            'completeness': 1.0,
            'groundedness': 1.0,
            'format_compliance': 1.0,
            'failure_type': 'none',
            'reason': 'ok',
            'defect': '',
            'key_point_recall': 1.0,
            'key_point_precision': 1.0,
            'semantic_similarity': 1.0,
            'claim_support_rate': 1.0,
            'unsupported_claim_rate': 0.0,
            'retrieval_hit_at_k': 1.0,
            'retrieval_recall_at_k': 1.0,
            'retrieval_precision_at_k': 1.0,
            'retrieval_mrr': 1.0,
            'retrieval_ndcg': 1.0,
            'context_relevance_avg': 1.0,
            'context_noise_rate': 0.0,
            'answer_quality_score': 1.0,
            'retrieval_quality_score': 1.0,
            'overall_score': 1.0,
            'retrieval_failure_type': 'none',
            'quality_label': 'good',
            'is_correct': True,
            'matched_key_points': case['key_points'],
            'missing_points': [],
            'wrong_points': [],
            'extra_points': [],
            'unsupported_claims': [],
            'evidence_mapping': [],
            'claims': [],
            'metric_layers': {},
            'score_breakdown': {},
        }

    monkeypatch.setattr(materializers_module, 'judge_case', fake_judge_case)

    output = eval_materializers()['eval.judge'](None, {
        'case': {
            'id': 'case-enhance',
            'question': 'q',
            'answer': 'a',
            'key_points': [{'id': 'old', 'statement': 'old'}],
        },
        'case_enhance': {
            'key_points': [{'id': 'new', 'statement': 'new', 'evidence_chunk_ids': ['chunk-a']}],
            'forbidden_claims': ['forbidden'],
        },
        'answer': {'status': 'ok', 'answer': 'a'},
        'policy': {},
    })

    assert captured['case']['key_points'][0]['id'] == 'new'
    assert captured['case']['forbidden_claims'] == ['forbidden']
    assert output['judge']['case']['key_points'][0]['id'] == 'new'


def test_optional_forbidden_claim_metrics_output_only_when_present(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeContradictionLLM)
    case = {
        'id': 'case_0006',
        'question': 'What fact is supported?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'alpha', 'statement': 'The supported fact is alpha.', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'forbidden_claims': ['The supported fact is beta.'],
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': ['The supported fact is alpha.'],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_0006',
        'answer': 'The supported fact is beta.',
        'contexts': ['The supported fact is alpha.'],
        'chunk_ids': ['chunk-a'],
        'doc_ids': [],
        'trace_id': 'trace-6',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['contradiction_rate'] == 1.0
    assert result['contradicted_claims'] == ['forbidden claim']
    assert 'contradiction_rate' in result['metric_layers']['specialized_metrics']
    assert 'contradicted_claims' in result['metric_layers']['diagnostic_evidence']


def test_wrong_answer_is_labeled_bad(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeBadLLM)
    case = {
        'id': 'case_0008',
        'question': 'What fact is supported?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'alpha', 'statement': 'The supported fact is alpha.', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': ['The supported fact is alpha.'],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_0008',
        'answer': 'The supported fact is beta.',
        'contexts': [{'chunk_id': 'chunk-noise', 'doc_id': 'doc-noise', 'content': 'Unrelated context.'}],
        'chunk_ids': ['chunk-noise'],
        'doc_ids': ['doc-noise'],
        'trace_id': 'trace-8',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['quality_label'] == 'bad'
    assert result['failure_type'] == 'wrong_answer'
    assert result['is_correct'] is False


def test_failed_rag_answer_is_labeled_infra_failure(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0009',
        'question': 'What fact is supported?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'alpha', 'statement': 'The supported fact is alpha.', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': ['The supported fact is alpha.'],
    }
    answer = {
        'status': 'failed',
        'case_id': 'case_0009',
        'answer': '',
        'contexts': [],
        'chunk_ids': [],
        'doc_ids': [],
        'trace_id': 'trace-9',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
        'chat_error': {'type': 'chat_config_error', 'message': 'router unavailable'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['quality_label'] == 'infra_failure'
    assert result['failure_type'] == 'infra_failure'
    assert result['retrieval_failure_type'] == 'not_applicable'
    assert result['overall_score'] == 0.0


def test_eval_detail_summary_exposes_frontend_overview_fields(monkeypatch):
    import evo.llm
    import evo.operations.eval.answer_process as answer_process

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)

    def fake_panel(row, **kwargs):
        return {
            'call_chain': [
                {
                    'step': 'query_rewrite',
                    'label': 'Query Rewrite',
                    'status': 'done',
                    'duration_ms': 12.5,
                    'exclusive_duration_ms': 12.5,
                    'span_count': 1,
                    'names': ['rewrite'],
                },
                {
                    'step': 'retrieve',
                    'label': 'Retrieve',
                    'status': 'done',
                    'duration_ms': 80.0,
                    'exclusive_duration_ms': 80.0,
                    'span_count': 1,
                    'names': ['retriever'],
                },
                {
                    'step': 'rerank',
                    'label': 'Rerank',
                    'status': 'done',
                    'duration_ms': 25.0,
                    'exclusive_duration_ms': 25.0,
                    'span_count': 1,
                    'names': ['rerank'],
                },
                {
                    'step': 'generate',
                    'label': 'Generate',
                    'status': 'done',
                    'duration_ms': 210.0,
                    'exclusive_duration_ms': 210.0,
                    'span_count': 1,
                    'names': ['llm'],
                },
            ],
            'latency_expand': {
                'available': True,
                'total_duration_ms': 327.5,
                'bottleneck_stage': 'llm_generate',
                'route_signature': 'query_rewrite>retrieve>rerank>llm_generate',
                'stages': [
                    {
                        'stage': 'query_rewrite',
                        'label': 'Query Rewrite',
                        'duration_ms': 12.5,
                        'exclusive_duration_ms': 12.5,
                        'status': 'done',
                        'steps': [{'id': 's1', 'name': 'rewrite', 'latency_ms': 12.5,
                                   'exclusive_latency_ms': 12.5, 'status': 'ok', 'error': ''}],
                    },
                    {
                        'stage': 'retrieve',
                        'label': 'Retrieve',
                        'duration_ms': 80.0,
                        'exclusive_duration_ms': 80.0,
                        'status': 'done',
                        'steps': [{'id': 's2', 'name': 'retriever', 'latency_ms': 80.0,
                                   'exclusive_latency_ms': 80.0, 'status': 'ok', 'error': ''}],
                    },
                    {
                        'stage': 'rerank',
                        'label': 'Rerank',
                        'duration_ms': 25.0,
                        'exclusive_duration_ms': 25.0,
                        'status': 'done',
                        'steps': [{'id': 's3', 'name': 'rerank', 'latency_ms': 25.0,
                                   'exclusive_latency_ms': 25.0, 'status': 'ok', 'error': ''}],
                    },
                    {
                        'stage': 'llm_generate',
                        'label': 'Generate',
                        'duration_ms': 210.0,
                        'exclusive_duration_ms': 210.0,
                        'status': 'done',
                        'steps': [{'id': 's4', 'name': 'llm', 'latency_ms': 210.0,
                                   'exclusive_latency_ms': 210.0, 'status': 'ok', 'error': ''}],
                    },
                ],
            },
            'trace': {
                'trace_id': str(row.get('trace_id') or 'trace-frontend'),
                'available': True,
                'source': 'lazyllm.get_single_trace',
                'status': 'ok',
                'hint': 'use /threads/{thread_id}/results/traces/{trace_id} for raw trace detail',
                'raw_entry': 'advanced',
            },
        }

    monkeypatch.setattr(answer_process, 'build_answer_process_panel', fake_panel)
    monkeypatch.setattr('evo.operations.eval.materializers.build_answer_process_panel', fake_panel)

    case = {
        'id': 'case_frontend_1',
        'question': 'When is launch and what is the price?',
        'answer': 'The launch date is July 8. The price is 20 USD.',
        'question_type': 'single_hop',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'date', 'statement': 'The launch date is July 8.', 'evidence_chunk_ids': ['chunk-a']},
            {'id': 'price', 'statement': 'The price is 20 USD.', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'reference_chunk_ids': ['chunk-a'],
        'reference_doc_ids': ['doc-a'],
        'reference_context': ['The launch date is July 8. The price is 20 USD.'],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_frontend_1',
        'answer': 'The launch date is July 8.',
        'contexts': [
            {
                'chunk_id': 'chunk-a',
                'doc_id': 'doc-a',
                'doc_name': 'Launch Guide',
                'content': 'The launch date is July 8.',
                'rank': 1,
                'score': 0.91,
            },
            {
                'chunk_id': 'chunk-noise',
                'doc_id': 'doc-noise',
                'doc_name': 'Noise Doc',
                'content': 'Unrelated content.',
                'rank': 2,
                'score': 0.3,
            },
        ],
        'chunk_ids': ['chunk-a', 'chunk-noise'],
        'doc_ids': ['doc-a', 'doc-noise'],
        'trace_id': 'trace-frontend',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    summary = build_eval_detail_summary([judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})])

    assert summary['frontend_view_version'] == 'eval_frontend_view.v1'
    assert summary['overview']['total_cases'] == 1
    assert summary['overview']['completed_cases'] == 1
    assert summary['overview']['average_scores']['overall'] == summary['metrics']['overall_score_avg']
    assert summary['case_overviews'][0]['stage_nodes'] == [
        {'key': 'retrieval_evidence', 'label': '检索证据', 'status': 'done'},
        {'key': 'answer_generation', 'label': '生成回答', 'status': 'done'},
        {'key': 'multi_dimension_judge', 'label': '多维评测', 'status': 'done'},
        {'key': 'result_archive', 'label': '结果归档', 'status': 'done'},
    ]
    detail = summary['case_details'][0]['tabs']
    assert detail['overview']['copy_payload']['question'] == case['question']
    assert detail['overview']['key_points']
    assert detail['retrieval_evidence']['items'][0]['hit_reference'] is True
    assert detail['retrieval_evidence']['items'][0]['doc_name'] == 'Launch Guide'
    assert detail['answer_process']['trace']['trace_id'] == 'trace-frontend'
    assert detail['answer_process']['trace']['raw_entry'] == 'advanced'
    assert 'p0_note' not in detail['answer_process']['trace']
    assert detail['answer_process']['call_chain'][0] == {
        'step': 'query_rewrite',
        'label': 'Query Rewrite',
        'status': 'done',
        'duration_ms': 12.5,
        'exclusive_duration_ms': 12.5,
        'span_count': 1,
        'names': ['rewrite'],
    }
    assert detail['answer_process']['call_chain'][3]['step'] == 'generate'
    assert detail['answer_process']['call_chain'][3]['duration_ms'] == 210.0
    expand = detail['answer_process']['latency_expand']
    assert expand['available'] is True
    assert expand['total_duration_ms'] == 327.5
    assert expand['bottleneck_stage'] == 'llm_generate'
    assert [item['stage'] for item in expand['stages']] == [
        'query_rewrite', 'retrieve', 'rerank', 'llm_generate',
    ]
    assert expand['stages'][1]['steps'][0]['name'] == 'retriever'
    assert 'raw' not in summary['case_details'][0]
    guides = summary['guides']
    assert guides['badcase_rule']
    assert guides['score_metrics'][0]['key'] == 'overall'
    assert guides['retrieval_metrics'][0]['key'] == 'retrieval_hit_at_k'
    assert any(item['key'] == 'retrieval_miss' for item in guides['failure_statuses'])
    overview_tab = detail['overview']
    assert overview_tab['score_metrics'][0]['label'] == '总分'
    assert 'description' in overview_tab['score_metrics'][0]
    assert 'description' in overview_tab['failure_status']
    assert overview_tab['failure_status']['suggestion'] is not None
    judge_tab = detail['judge_evaluation']
    assert judge_tab['retrieval_metric_guide'][0]['label'] == 'Hit@K'
    assert 'description' in judge_tab['retrieval_metric_guide'][0]
    assert judge_tab['facts']['metric_guide'][0]['key'] == 'claim_support_rate'
    assert judge_tab['failure_status']['code']


def test_frontend_failure_status_guide_for_infra_and_retrieval(monkeypatch):
    import evo.llm
    import evo.operations.eval.answer_process as answer_process

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    monkeypatch.setattr(
        answer_process,
        'build_answer_process_panel',
        lambda row, **kwargs: {
            'call_chain': [],
            'latency_expand': {'available': False, 'total_duration_ms': None,
                               'bottleneck_stage': '', 'route_signature': '', 'stages': []},
            'trace': {'trace_id': '', 'available': False, 'source': 'heuristic',
                      'status': 'missing', 'hint': '', 'raw_entry': 'advanced'},
        },
    )
    monkeypatch.setattr('evo.operations.eval.materializers.build_answer_process_panel',
                        answer_process.build_answer_process_panel)

    infra = {
        'case_id': 'case_infra',
        'question': 'q',
        'quality_label': 'infra_failure',
        'failure_type': 'infra_failure',
        'retrieval_failure_type': 'not_applicable',
        'reason': 'chat failed',
        'overall_score': 0.0,
        'answer_quality_score': 0.0,
        'retrieval_quality_score': 0.0,
    }
    miss = {
        'case_id': 'case_miss',
        'question': 'q',
        'quality_label': 'bad',
        'failure_type': 'wrong_answer',
        'retrieval_failure_type': 'retrieval_miss',
        'reason': 'missed evidence',
        'overall_score': 0.2,
        'answer_quality_score': 0.2,
        'retrieval_quality_score': 0.1,
        'key_points': [],
    }
    view = build_eval_frontend_view({'rows': [infra, miss], 'total': 2, 'metrics': {},
                                     'execution_failures': [{'case_id': 'case_infra'}],
                                     'routing_failures': []})
    infra_status = view['case_details'][0]['tabs']['overview']['failure_status']
    miss_status = view['case_details'][1]['tabs']['overview']['failure_status']
    assert infra_status['code'] == 'infra_failure'
    assert infra_status['label'] == '执行异常'
    assert infra_status['is_badcase'] is False
    assert infra_status['counts_in_score_avg'] is False
    assert miss_status['code'] == 'wrong_answer'
    assert miss_status['label'] == '回答错误'
    assert miss_status['is_badcase'] is True
    assert miss_status['display_group'] == 'badcase'
    assert infra_status['display_group'] == 'exception'
    assert infra_status['is_exception'] is True
    assert view['case_overviews'][0]['status_label'] == '执行异常'
    assert view['case_overviews'][0]['boundary']['kind'] == 'execution_exception'
    assert view['case_overviews'][0]['scores_applicable'] is False
    assert view['case_overviews'][1]['status_label'] == 'Badcase'
    assert view['case_overviews'][1]['boundary']['display_group'] == 'badcase'
    assert view['overview']['status_breakdown']['exception'] == 1
    assert view['overview']['status_breakdown']['badcase'] == 1
    assert view['overview']['exception_cases'] == 1
    assert view['overview']['boundary_note']
    assert view['guides']['case_statuses'][0]['key'] == 'good'
    assert view['guides']['boundary_statuses'][2]['key'] == 'exception'


def test_compose_answer_process_panel_from_trace_summary():
    from evo.operations.eval.answer_process import compose_answer_process_panel

    summary = {
        'trace_id': 'trace-compose',
        'trace_source': 'lazyllm.get_single_trace',
        'route_signature': 'retrieve>rerank>llm_generate',
        'diagnostic_stage_sequence': ['retrieve', 'rerank', 'llm_generate'],
        'bottleneck_stage': 'llm_generate',
        'latency_by_stage': {'retrieve': 40.0, 'rerank': 10.0, 'llm_generate': 100.0},
        'features': {'trace_latency_ms': 150.0},
        'error_stages': [],
        'stages': [
            {
                'id': 'a', 'stage': 'retrieve', 'name': 'kb_retriever', 'status': 'ok',
                'latency_ms': 40.0, 'exclusive_latency_ms': 40.0, 'error': '',
            },
            {
                'id': 'b', 'stage': 'rerank', 'name': 'module_rerank', 'status': 'ok',
                'latency_ms': 10.0, 'exclusive_latency_ms': 10.0, 'error': '',
            },
            {
                'id': 'c', 'stage': 'llm_generate', 'name': 'chat', 'status': 'ok',
                'latency_ms': 100.0, 'exclusive_latency_ms': 100.0, 'error': '',
            },
        ],
    }

    panel = compose_answer_process_panel(summary, trace_id='trace-compose', has_evidence=True, has_answer=True)

    assert panel['call_chain'][0]['status'] == 'skipped'
    assert panel['call_chain'][0]['duration_ms'] is None
    assert panel['call_chain'][0]['latency_note']
    assert panel['call_chain'][1]['duration_ms'] == 40.0
    assert panel['call_chain'][2]['duration_ms'] == 10.0
    assert panel['call_chain'][3]['duration_ms'] == 100.0
    assert panel['latency_expand']['available'] is True
    assert panel['latency_expand']['incomplete'] is True
    assert 'query_rewrite' in panel['latency_expand']['missing_stages']
    assert panel['latency_expand']['total_duration_ms'] == 150.0
    assert panel['latency_expand']['bottleneck_label'] == 'Generate'
    assert panel['latency_expand']['stages'][0]['stage'] == 'retrieve'
    assert panel['trace']['raw_entry'] == 'advanced'
    assert panel['trace']['status'] == 'partial'
    assert panel['trace']['readable'] is True
    assert panel['trace']['latency_available'] is True
    assert panel['trace']['note']


def test_answer_process_prefers_reloading_heuristic_stored_without_latency(monkeypatch):
    from evo.operations.eval import answer_process as ap

    captured = {}

    def fake_summary(case, answer, *, attempts=None, retry_seconds=None):
        captured['attempts'] = attempts
        return {
            'trace_id': answer['trace_id'],
            'trace_source': 'lazyllm.get_single_trace',
            'route_signature': 'retrieve>llm_generate',
            'diagnostic_stage_sequence': ['retrieve', 'llm_generate'],
            'bottleneck_stage': 'llm_generate',
            'latency_by_stage': {'retrieve': 11.0, 'llm_generate': 22.0},
            'features': {'trace_latency_ms': 33.0},
            'error_stages': [],
            'stages': [
                {'id': '1', 'stage': 'retrieve', 'name': 'ret', 'status': 'ok',
                 'latency_ms': 11.0, 'exclusive_latency_ms': 11.0, 'error': ''},
                {'id': '2', 'stage': 'llm_generate', 'name': 'llm', 'status': 'ok',
                 'latency_ms': 22.0, 'exclusive_latency_ms': 22.0, 'error': ''},
            ],
        }

    monkeypatch.setattr(ap, 'build_trace_summary', fake_summary)
    panel = ap.build_answer_process_panel(
        {
            'case_id': 'c1',
            'trace_id': 'a' * 32,
            'rag_answer': 'hello',
            'retrieve_contexts': [{'content': 'x'}],
            'answer_process': {
                'call_chain': [{'step': 'retrieve', 'label': 'Retrieve', 'status': 'done', 'duration_ms': None}],
                'latency_expand': {'available': False, 'stages': []},
                'trace': {'trace_id': 'a' * 32, 'status': 'unavailable'},
            },
        },
        load_trace=True,
        attempts=2,
        retry_seconds=0.0,
    )
    assert captured['attempts'] == 2
    assert panel['call_chain'][1]['duration_ms'] == 11.0
    assert panel['trace']['latency_available'] is True
