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
from evo.operations.eval.materializers import build_eval_detail_summary, eval_materializers
from evo.operations.eval.answer import call_chat_answer, _with_case
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
        {'content': 'same text', 'doc_id': 'kb-a:doc-a', 'chunk_id': 'kb-a:doc-a:chunk-a'},
        {'content': 'same text', 'doc_id': 'kb-a:doc-b', 'chunk_id': 'kb-a:doc-b:chunk-b'},
        {'content': 'other text', 'doc_id': 'kb-a:doc-c', 'chunk_id': 'kb-a:doc-c:chunk-c'},
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


def test_router_chat_normalizes_retry_policy_like_go_defaults(monkeypatch):
    captured = {}

    async def fake_call_once(request: RouterChatRequest):
        captured['max_attempts'] = request.max_attempts
        captured['retry_wait_max_seconds'] = request.retry_wait_max_seconds
        return {
            'status': 'ok',
            'answer': 'final answer',
            'trace_id': request.trace_id,
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
        max_attempts=99,
        retry_wait_max_seconds=-1,
    )))

    assert result['status'] == 'ok'
    assert captured['max_attempts'] == 5
    assert captured['retry_wait_max_seconds'] == 2.0


def test_runtime_limits_eval_answer_concurrency_from_env(monkeypatch, tmp_path):
    from evo.service.runtime_port import RuntimePort

    monkeypatch.setenv('LAZYMIND_EVO_MAX_IN_FLIGHT', '6')
    monkeypatch.setenv('LAZYMIND_EVO_PARTITION_OP_LIMIT', '4')
    monkeypatch.setenv('LAZYMIND_EVO_EVAL_ANSWER_OP_LIMIT', '1')

    limits = RuntimePort(tmp_path)._concurrency_limits()

    assert limits.max_in_flight == 6
    assert limits.per_materializer['eval.answer'] == 1
    assert limits.per_materializer['eval.judge'] == 4


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


def test_judge_prompt_passes_contexts_with_chunk_ids(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeCapturePromptLLM)
    case = {
        'id': 'case_prompt',
        'question': 'What fact is supported?',
        'answer': 'The supported fact is alpha.',
        'reference_chunk_ids': ['chunk-ref'],
        'reference_context': {'chunk-ref': 'The supported fact is alpha.'},
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_prompt',
        'answer': 'The supported fact is alpha.',
        'contexts': [{'chunk_id': 'chunk-ret', 'content': 'The supported fact is alpha.', 'rank': 1}],
        'chunk_ids': ['chunk-ret'],
        'doc_ids': [],
        'trace_id': 'trace-prompt',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})
    payload = json.loads(FakeCapturePromptLLM.prompt.split('case_json: ', 1)[1])

    assert payload['reference_context'] == [{'chunk_id': 'chunk-ref', 'content': 'The supported fact is alpha.'}]
    assert payload['retrieved_contexts'] == [{'chunk_id': 'chunk-ret', 'content': 'The supported fact is alpha.'}]


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


def test_detail_summary_exposes_diagnostics(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0002',
        'question': 'What is the supported fact?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'easy',
        'key_points': ['The supported fact is alpha.'],
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': ['The supported fact is alpha.'],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_0002',
        'answer': 'The supported fact is alpha.',
        'contexts': ['The supported fact is alpha.'],
        'chunk_ids': ['chunk-a'],
        'doc_ids': [],
        'trace_id': 'trace-2',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }
    judge = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    summary = build_eval_detail_summary([judge])

    assert summary['metrics']['key_point_recall_avg'] == 1.0
    assert summary['by_question_type']['single_hop']['count'] == 1
    assert summary['by_difficulty']['easy']['correct_rate'] == 1.0
    assert summary['rows'][0]['matched_key_points']
    assert summary['rows'][0]['metric_layers']['core_explainers']
    assert summary['rows'][0]['score_breakdown']['overall_score']['weights']
    assert 'retrieval_mrr_avg' in summary['metrics']
    assert 'numeric_accuracy' not in summary['rows'][0]
    assert 'list_set_f1' not in summary['rows'][0]
    assert 'contradiction_rate' not in summary['rows'][0]
    assert 'contradicted_claims' not in summary['rows'][0]


def test_legacy_case_without_key_points_keeps_existing_label_logic(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0003',
        'question': 'What is the supported fact?',
        'answer': 'The supported fact is alpha.',
        'question_type': 'single_hop',
        'difficulty': 'easy',
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': ['The supported fact is alpha.'],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_0003',
        'answer': 'The supported fact is alpha.',
        'contexts': ['The supported fact is alpha.'],
        'chunk_ids': ['chunk-a'],
        'doc_ids': [],
        'trace_id': 'trace-3',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['key_point_recall'] == 0.9
    assert result['quality_label'] == 'good'
    assert result['failure_type'] == 'none'


def test_eval_judge_materializer_passes_diagnostic_fields_to_downstream(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0004',
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
        'case_id': 'case_0004',
        'answer': 'The launch date is July 8.',
        'contexts': [
            {'chunk_id': 'chunk-a', 'doc_id': 'doc-a', 'content': 'The launch date is July 8.', 'rank': 1},
            {'chunk_id': 'chunk-noise', 'doc_id': 'doc-noise', 'content': 'Unrelated content.', 'rank': 2},
        ],
        'chunk_ids': ['chunk-a', 'chunk-noise'],
        'doc_ids': ['doc-a', 'doc-noise'],
        'trace_id': 'trace-4',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }
    materializer = eval_materializers()['eval.judge']

    output = materializer(None, {
        'case': case,
        'answer': answer,
        'policy': {
            'judge_llm_config': {'evo_llm': {'model': 'fake'}},
            'key_point_recall_floor': 0.8,
            'unsupported_claim_rate_ceiling': 0.0,
            'retrieval_top_k': 5,
        },
    })
    judge = output['judge']

    for key in (
        'key_point_recall',
        'key_point_precision',
        'matched_key_points',
        'missing_points',
        'claim_support_rate',
        'unsupported_claims',
        'retrieval_hit_at_k',
        'retrieval_recall_at_k',
        'retrieval_precision_at_k',
        'retrieval_mrr',
        'retrieval_ndcg',
        'context_noise_rate',
        'numeric_accuracy',
        'semantic_similarity',
        'metric_layers',
        'score_breakdown',
    ):
        assert key in judge
    assert judge['missing_points'][0]['id'] == 'price'
    assert judge['eval_policy']['key_point_recall_floor'] == 0.8
    assert judge['eval_policy']['unsupported_claim_rate_ceiling'] == 0.0
    assert judge['eval_policy']['retrieval_top_k'] == 5
    assert 'retrieval_mrr' in judge['metric_layers']['core_explainers']
    assert judge['score_breakdown']['retrieval_quality_score']['weights']['retrieval_mrr'] == 0.20
    assert 'list_set_f1' not in judge
    assert 'contradiction_rate' not in judge


def test_optional_list_metric_is_output_only_for_list_answer_type(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0005',
        'question': 'Which items are supported?',
        'answer': 'alpha, beta',
        'answer_type': 'list',
        'question_type': 'table_list',
        'difficulty': 'medium',
        'key_points': [
            {'id': 'alpha', 'statement': 'alpha', 'evidence_chunk_ids': ['chunk-a']},
            {'id': 'beta', 'statement': 'beta', 'evidence_chunk_ids': ['chunk-a']},
        ],
        'reference_chunk_ids': ['chunk-a'],
        'reference_context': ['alpha and beta are supported.'],
    }
    answer = {
        'status': 'ok',
        'case_id': 'case_0005',
        'answer': 'alpha',
        'contexts': ['alpha and beta are supported.'],
        'chunk_ids': ['chunk-a'],
        'doc_ids': [],
        'trace_id': 'trace-5',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert 'list_set_f1' in result
    assert 'list_set_f1' in result['metric_layers']['specialized_metrics']


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


def test_unsupported_claim_rate_downgrades_good_answer_to_partial(monkeypatch):
    import evo.llm

    monkeypatch.setattr(evo.llm, 'LazyLLMClient', FakeLLM)
    case = {
        'id': 'case_0007',
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
        'case_id': 'case_0007',
        'answer': 'The supported fact is alpha. NASA confirmed it in 2025.',
        'contexts': ['The supported fact is alpha.'],
        'chunk_ids': ['chunk-a'],
        'doc_ids': [],
        'trace_id': 'trace-7',
        'target': {'algorithm_id': 'algo-a', 'kb_id': 'kb-a'},
    }

    result = judge_case(case, answer, {'judge_llm_config': {'evo_llm': {'model': 'fake'}}})

    assert result['unsupported_claim_rate'] > 0.0
    assert result['quality_label'] == 'partial'
    assert result['failure_type'] == 'partial_answer'


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
