from __future__ import annotations

import json

import pytest

from evo.apply.errors import ApplyError
from evo.eval_qc import run_eval_qc_batch


# ---- Fakes & factories ----------------------------------------------------

class _ScriptedLLM:
    """Pops scripted responses in order; returns '' once exhausted."""

    def __init__(self, *responses: dict) -> None:
        self.responses = [json.dumps(r, ensure_ascii=False) for r in responses]
        self.calls: list[str] = []

    def __call__(self, prompt: str, *args: object, **kwargs: object) -> str:
        self.calls.append(prompt)
        return self.responses.pop(0) if self.responses else ''


def _extract() -> dict:
    return {
        'query': {
            'unit': '在2025年北京地区，哪些材料支持这个答案？',
            'core': '2025年北京地区支持这个答案的材料',
            'qualifier': '限2025年、限北京地区',
        },
    }


def _judge(levels: dict[str, float] | None = None, *, summary: str = 'summary') -> dict:
    lv = {'query_to_gt_answer': 0.9, 'query_to_gt_text': 0.9, 'query_to_key_points': 0.9}
    lv.update(levels or {})
    return {
        'judgments': [{'logic_id': lid, 'reason': 'r', 'level': level} for lid, level in lv.items()],
        'summary_reason': summary,
    }


def _bad_judge() -> dict:
    # Only one logic_id -> fails the eval_qc_judge schema -> triggers repair.
    return {'judgments': [{'logic_id': 'query_to_gt_answer', 'reason': 'r', 'level': 0.9}],
            'summary_reason': 's'}


def _schema_invalid_high_score_judge() -> dict:
    judge = _judge()
    judge['extra'] = 'forbidden'
    for entry in judge['judgments']:
        entry['extra'] = 'forbidden'
    return judge


def _case(case_id: str = 'c1', **overrides: object) -> dict:
    case = {
        'id': case_id,
        'query': '在2025年北京地区，哪些材料支持这个答案？',
        'gt_answer': 'an answer',
        'gt_text': ['ctx'],
        'key_points': ['k'],
    }
    case.update(overrides)
    return case


# ---- A. error_format_input (stage B) --------------------------------------

@pytest.mark.parametrize(
    'overrides, expected_field',
    [
        ({'query': ''}, 'query'),
        ({'gt_answer': '  '}, 'gt_answer'),
        ({'gt_text': []}, 'gt_text'),
        ({'key_points': []}, 'key_points'),
    ],
)
def test_format_input_single_missing_field(overrides, expected_field) -> None:
    llm = _ScriptedLLM()
    [result] = run_eval_qc_batch([_case(**overrides)], judge_llm=llm)

    assert result['passed'] is False
    assert result['reject']['type'] == 'error_format_input'
    assert result['reject']['missing_fields'] == [expected_field]
    assert 'logics' not in result
    assert result['summary']
    assert llm.calls == []  # B short-circuits, LLM never invoked


def test_format_input_multiple_missing_fields() -> None:
    [result] = run_eval_qc_batch([_case(gt_text=[], key_points=[])], judge_llm=_ScriptedLLM())
    assert set(result['reject']['missing_fields']) == {'gt_text', 'key_points'}


def test_whitespace_counts_as_missing() -> None:
    [result] = run_eval_qc_batch([_case(gt_text=['   ', ''])], judge_llm=_ScriptedLLM())
    assert result['reject']['missing_fields'] == ['gt_text']


# ---- B. pass + error_unpassed (stage C, valid output) ---------------------

def test_pass_when_all_logics_pass() -> None:
    llm = _ScriptedLLM(_extract(), _judge())
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)

    assert result['passed'] is True
    assert 'reject' not in result
    assert set(result['logics']) == {'query_to_gt_answer', 'query_to_gt_text', 'query_to_key_points'}
    assert all(l['passed'] for l in result['logics'].values())
    assert result['summary'] == 'summary'


def test_unpassed_when_one_logic_below_threshold() -> None:
    llm = _ScriptedLLM(_extract(), _judge({'query_to_gt_text': 0.3}))
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)

    assert result['passed'] is False
    assert result['reject'] == {'type': 'error_unpassed'}
    assert result['logics']['query_to_gt_text']['passed'] is False
    assert result['logics']['query_to_gt_answer']['passed'] is True


def test_threshold_boundary_partial_passes_topic_fails() -> None:
    llm = _ScriptedLLM(_extract(), _judge({'query_to_gt_answer': 0.6, 'query_to_gt_text': 0.3}))
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)

    assert result['logics']['query_to_gt_answer']['passed'] is True   # 0.6 >= 0.5
    assert result['logics']['query_to_gt_text']['passed'] is False    # 0.3 <  0.5


def test_level_mapping() -> None:
    llm = _ScriptedLLM(
        _extract(),
        _judge({'query_to_gt_answer': 0.9, 'query_to_gt_text': 0.6, 'query_to_key_points': 0.3}),
    )
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)
    levels = {lid: l['level'] for lid, l in result['logics'].items()}
    assert levels == {'query_to_gt_answer': 0.9, 'query_to_gt_text': 0.6, 'query_to_key_points': 0.3}


# ---- C. LLM output anomalies (lean on lazyllm repair) ---------------------

def test_repair_recovers_to_valid_judgment() -> None:
    llm = _ScriptedLLM(_extract(), _bad_judge(), _judge())
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)

    assert result['passed'] is True
    assert len(llm.calls) == 3  # extract + bad judge + repaired judge


def test_persistent_bad_output_falls_back_to_unpassed() -> None:
    # Judge never validates (3 attempts) -> failure output -> error_unpassed, all 0.1.
    llm = _ScriptedLLM(_extract(), _bad_judge(), _bad_judge(), _bad_judge())
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)

    assert result['passed'] is False
    assert result['reject'] == {'type': 'error_unpassed'}
    assert all(l['level'] == 0.1 for l in result['logics'].values())


def test_schema_invalid_high_scores_fail_closed_after_repairs() -> None:
    # Extra fields fail JSON schema even though the weaker business validator
    # would otherwise see three high-scoring logic judgments.
    bad = _schema_invalid_high_score_judge()
    llm = _ScriptedLLM(_extract(), bad, bad, bad)
    [result] = run_eval_qc_batch([_case()], judge_llm=llm)

    assert result['passed'] is False
    assert result['reject'] == {'type': 'error_unpassed'}
    assert all(l['level'] == 0.1 for l in result['logics'].values())


def test_systemic_schema_failure_raises_apply_error() -> None:
    # Three failing cases reach EVO_MAX_SCHEMA_FAILURES (3) -> ApplyError bubbles.
    responses: list[dict] = []
    for _ in range(3):
        responses += [_extract(), _bad_judge(), _bad_judge(), _bad_judge()]
    cases = [_case(f'c{i}') for i in range(3)]
    with pytest.raises(ApplyError):
        run_eval_qc_batch(cases, judge_llm=_ScriptedLLM(*responses))


# ---- D. batch + contract --------------------------------------------------

def test_batch_echoes_ids_in_order() -> None:
    responses: list[dict] = []
    for _ in range(3):
        responses += [_extract(), _judge()]
    cases = [_case('a'), _case('b'), _case('c')]
    results = run_eval_qc_batch(cases, judge_llm=_ScriptedLLM(*responses))
    assert [r['id'] for r in results] == ['a', 'b', 'c']


def test_mixed_batch_field_shapes() -> None:
    llm = _ScriptedLLM(
        _extract(), _judge(),                                  # c1 pass
        _extract(), _judge({'query_to_gt_text': 0.3}),           # c3 unpassed
    )
    cases = [_case('c1'), _case('c2', gt_text=[]), _case('c3')]
    passed, fmt, unpassed = run_eval_qc_batch(cases, judge_llm=llm)

    assert passed['passed'] is True and 'reject' not in passed and 'logics' in passed
    assert fmt['reject']['type'] == 'error_format_input' and 'logics' not in fmt
    assert unpassed['reject']['type'] == 'error_unpassed' and 'logics' in unpassed
    # reject never carries a human 'reason'; summary is the only human field.
    for r in (passed, fmt, unpassed):
        assert 'summary' in r
        assert 'reason' not in r.get('reject', {})


def test_empty_batch() -> None:
    assert run_eval_qc_batch([], judge_llm=_ScriptedLLM()) == []


# ---- E. callbacks & control ------------------------------------------------

def test_on_progress_reports_each_case() -> None:
    responses: list[dict] = []
    for _ in range(3):
        responses += [_extract(), _judge()]
    seen: list[tuple[int, int]] = []
    run_eval_qc_batch(
        [_case('a'), _case('b'), _case('c')],
        judge_llm=_ScriptedLLM(*responses),
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen == [(1, 3), (2, 3), (3, 3)]


def test_cancel_returns_completed_partial() -> None:
    calls = {'n': 0}

    def cancel() -> bool:
        calls['n'] += 1
        return calls['n'] > 1  # allow first case, stop before second

    results = run_eval_qc_batch(
        [_case('a'), _case('b')],
        judge_llm=_ScriptedLLM(_extract(), _judge()),
        cancel=cancel,
    )
    assert [r['id'] for r in results] == ['a']
