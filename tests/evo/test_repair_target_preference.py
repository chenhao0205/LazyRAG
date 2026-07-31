from __future__ import annotations

from typing import Any

import pytest

from evo.operations.repair.target_contracts import PreferenceCompileRequest
from evo.operations.repair.target_preference import (
    LLMPreferenceCompiler,
    PreferenceCompilerError,
)


def _request(
    *,
    mode: str = 'legacy',
    source: str = 'user_guidance',
) -> PreferenceCompileRequest:
    return PreferenceCompileRequest.model_validate({
        'mode': mode,
        'texts': [
            {
                'evidence': {'source': source, 'index': 0},
                'text': '优先修复召回问题',
            },
        ],
        'category_options': [
            {
                'category_id': 'retrieval',
                'summary': 'Recall failures.',
                'analysis': 'The retrieval stage filters useful documents.',
            },
            {
                'category_id': 'generation',
                'summary': 'Generation failures.',
                'analysis': 'The answer omits a qualification.',
            },
        ],
        'metric_ids': ['answer_correctness', 'retrieval_recall'],
    })


def _valid_resolution(
    *,
    source: str = 'user_guidance',
    tier: str = 'prefer',
) -> dict[str, Any]:
    return {
        'category_directives': [
            {
                'category_id': 'retrieval',
                'tier': tier,
                'order': 0,
                'evidence': {'source': source, 'index': 0},
            },
        ],
        'metric_directives': [
            {
                'metric_id': 'retrieval_recall',
                'order': 0,
                'evidence': {'source': source, 'index': 0},
            },
        ],
    }


class _SequenceCompletion:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def __call__(self, prompt: str, **kwargs: Any) -> object:
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_compiler_accepts_repaired_fenced_json() -> None:
    completion = _SequenceCompletion(
        '<think>internal</think>\n```json\n'
        f'{_valid_resolution()!r}'
        '\n```',
    )
    compiler = LLMPreferenceCompiler(completion=completion)

    result = compiler(_request())

    assert result.category_directives[0].category_id == 'retrieval'
    assert result.metric_directives[0].metric_id == 'retrieval_recall'
    assert completion.kwargs == [{
        'stream': False,
        'response_format': {'type': 'json_object'},
    }]


def test_compiler_retries_after_strict_validation_failure() -> None:
    invalid = _valid_resolution()
    invalid['category_directives'][0]['order'] = '0'
    completion = _SequenceCompletion(invalid, _valid_resolution())
    compiler = LLMPreferenceCompiler(completion=completion)

    result = compiler(_request())

    assert result.category_directives[0].order == 0
    assert len(completion.prompts) == 2
    assert 'Previous validation error:' in completion.prompts[1]


def test_compiler_retries_after_unknown_identifier() -> None:
    unknown = _valid_resolution()
    unknown['category_directives'][0]['category_id'] = 'invented'
    completion = _SequenceCompletion(unknown, _valid_resolution())
    compiler = LLMPreferenceCompiler(completion=completion)

    result = compiler(_request())

    assert result.category_directives[0].category_id == 'retrieval'
    assert len(completion.prompts) == 2
    assert 'unknown category' in completion.prompts[1]


def test_compiler_failure_is_bounded_and_does_not_return_partial_output() -> None:
    completion = _SequenceCompletion(
        RuntimeError('provider unavailable'),
        {'category_directives': [], 'metric_directives': [], 'reason': 'not allowed'},
    )
    compiler = LLMPreferenceCompiler(completion=completion)

    with pytest.raises(PreferenceCompilerError, match='response='):
        compiler(_request())

    assert len(completion.prompts) == 2


def test_prompt_omits_structured_analysis_scores_and_code_spans() -> None:
    completion = _SequenceCompletion(_valid_resolution())
    compiler = LLMPreferenceCompiler(completion=completion)

    compiler(_request())
    prompt = completion.prompts[0]

    assert '"category_id":"retrieval"' in prompt
    assert '"metric_ids":' in prompt
    assert 'metric_averages' not in prompt
    assert 'all_case_average_drop' not in prompt
    assert '"code_span":' not in prompt
    assert 'Do not create scores, weights, reasons' in prompt


def test_compile_mode_rejects_evidence_from_the_wrong_phase() -> None:
    with pytest.raises(ValueError, match='initial_preference requests only accept'):
        _request(mode='initial_preference', source='user_guidance')

    with pytest.raises(ValueError, match='interrupt_guidance requests only accept'):
        _request(mode='interrupt_guidance', source='preference')


def test_initial_preference_is_soft_and_retries_hard_tiers() -> None:
    completion = _SequenceCompletion(
        _valid_resolution(source='preference', tier='must'),
        _valid_resolution(source='preference', tier='prefer'),
    )
    compiler = LLMPreferenceCompiler(completion=completion)

    result = compiler(_request(
        mode='initial_preference',
        source='preference',
    ))

    assert result.category_directives[0].tier == 'prefer'
    assert len(completion.prompts) == 2
    assert 'initial preference may only produce' in completion.prompts[1]


def test_prompt_marks_preference_and_guidance_modes() -> None:
    completion = _SequenceCompletion(
        _valid_resolution(source='preference'),
    )

    LLMPreferenceCompiler(completion=completion)(_request(
        mode='initial_preference',
        source='preference',
    ))

    prompt = completion.prompts[0]
    assert '"mode":"initial_preference"' in prompt
    assert 'initial_preference is a soft prior' in prompt
    assert 'interrupt_guidance is a current, explicit instruction' in prompt


def test_large_valid_request_is_compacted_to_prompt_budget_without_mutation() -> None:
    request = PreferenceCompileRequest.model_validate({
        'texts': [
            {
                'evidence': {'source': 'user_guidance', 'index': index},
                'text': '偏' * 2000,
            }
            for index in range(128)
        ],
        'category_options': [
            {
                'category_id': f'category_{index:03d}',
                'summary': '摘' * 1000,
                'analysis': '析' * 10000,
            }
            for index in range(200)
        ],
        'metric_ids': [
            f'metric_{index:03d}'
            for index in range(128)
        ],
    })
    completion = _SequenceCompletion({
        'category_directives': [],
        'metric_directives': [],
    })

    result = LLMPreferenceCompiler(completion=completion)(request)

    assert result.category_directives == []
    assert len(completion.prompts[0]) < 210_000
    assert len(request.texts[0].text) == 2000
    assert len(request.category_options[0].analysis) == 10000


@pytest.mark.parametrize('attempts', [True, 0, 4, 1.5])
def test_attempt_count_is_strictly_bounded(attempts: object) -> None:
    with pytest.raises(ValueError, match='attempts'):
        LLMPreferenceCompiler(
            completion=_SequenceCompletion(_valid_resolution()),
            attempts=attempts,  # type: ignore[arg-type]
        )
