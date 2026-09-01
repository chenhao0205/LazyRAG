
from __future__ import annotations

import os
from typing import Any, Callable, Sequence

from evo.eval_qc.constants import LOGIC_IDS
from evo.eval_qc.fields import missing_required_fields
from evo.eval_qc.judge import EvalQcRunState, run_eval_qc


def run_eval_qc_batch(
    cases: Sequence[dict[str, Any]],
    *,
    judge_llm: Any,
    config: Any | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Run eval_qc over a batch of cases.
    """
    
    threshold = _threshold(config)
    total = len(cases)
    state = EvalQcRunState()
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if cancel is not None and cancel():
            break
        results.append(_check_case(case, judge_llm=judge_llm, threshold=threshold, state=state))
        if on_progress is not None:
            on_progress(index + 1, total)
    return results


def _threshold(config: Any | None) -> float:
    eval_qc = getattr(config, 'eval_qc', None)
    threshold = getattr(eval_qc, 'threshold', None)
    if isinstance(threshold, (int, float)):
        return float(threshold)
    return float(os.getenv('EVO_EVAL_QC_THRESHOLD', '0.5'))


def _check_case(
    case: dict[str, Any], *, judge_llm: Any, threshold: float, state: EvalQcRunState
) -> dict[str, Any]:
    case_id = case.get('id')
    query = _text(case.get('query'))
    gt_answer = _text(case.get('gt_answer'))
    gt_text = _text_list(case.get('gt_text'))
    key_points = _text_list(case.get('key_points'))

    # Stage B: required-field validation (cheap, deterministic, no LLM).
    missing = missing_required_fields(query=query, gt_answer=gt_answer, gt_text=gt_text, key_points=key_points)
    if missing:
        return {
            'id': case_id,
            'passed': False,
            'reject': {'type': 'error_format_input', 'missing_fields': missing},
            'summary': f"{'、'.join(missing)} 为空",
        }

    # Stage C: three query-anchored logic judgments via the injected LLM.
    parsed = run_eval_qc(
        {'query': query, 'gt_answer': gt_answer, 'gt_text': gt_text, 'key_points': key_points},
        llm=judge_llm,
        state=state,
    )
    logics = _judge_logics(parsed.get('judgments'), threshold)
    passed = len(logics) == len(LOGIC_IDS) and all(logic['passed'] for logic in logics.values())
    result: dict[str, Any] = {
        'id': case_id,
        'passed': passed,
        'logics': logics,
        'summary': str(parsed.get('summary_reason') or ''),
    }
    if not passed:
        result['reject'] = {'type': 'error_unpassed'}
    return result


def _judge_logics(judgments: Any, threshold: float) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(judgments, list):
        for entry in judgments:
            if not isinstance(entry, dict):
                continue
            logic_id = str(entry.get('logic_id') or '').strip()
            if logic_id not in LOGIC_IDS or logic_id in by_id:
                continue
            raw_level = entry.get('level')
            level = float(raw_level) if isinstance(raw_level, (int, float)) else 0.0
            reason = str(entry.get('reason') or '').strip()
            by_id[logic_id] = {'level': level, 'passed': level >= threshold, 'reason': reason}
    for logic_id in LOGIC_IDS:
        by_id.setdefault(logic_id, {'level': 0.0, 'passed': False, 'reason': 'logic missing in model output'})
    return by_id


def _text(value: Any) -> str:
    return '' if value is None else str(value).strip()


def _text_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [text for item in value if (text := str(item).strip())]


__all__ = ['run_eval_qc_batch']
