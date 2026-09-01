from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from evo.apply.errors import ApplyError
from evo.eval_qc.constants import LOGIC_IDS
from evo.eval_qc.prompts import load as load_prompt
from evo.eval_qc.schemas import EVAL_QC_EXTRACT, EVAL_QC_JUDGE
from evo.harness.structured import parse_and_validate
from evo.runtime.config import EVO_MAX_SCHEMA_FAILURES
from evo.utils import strip_thinking

_EXTRACT_NAME = 'eval_qc_extract'
_JUDGE_NAME = 'eval_qc_judge'
_FAILURE_REASON = 'eval_qc 评估失败'
_VALID_LEVELS = frozenset({0.1, 0.3, 0.6, 0.9})

# Injected LLM: prompt str -> raw completion str (gateway-wrapped by the executor).
LLM = Callable[[str], str]

_REPAIR_SYSTEM_PROMPT = (
    'You are a strict JSON formatter. Your ONLY job is to read the user message '
    '(a previous bad output, the validation errors, and the required JSON schema) '
    'and emit ONE JSON object that satisfies the schema. Do NOT call tools, do NOT '
    'emit <think> tags, markdown fences, or any prose. Use [] for empty arrays, never '
    'null. Enum fields must use exactly one listed value. Output ONLY the JSON object.'
)
_ERROR_LIMIT = 20
_PREVIEW_CHARS = 2000


@dataclass
class EvalQcRunState:
    """Cross-case schema-failure counter; trips the ApplyError circuit-breaker."""

    schema_failures: int = 0


# ---- structured call: drive the injected LLM, validate, repair ------------

def _repair_user(original_user: str, raw: str, errors: list[str], schema: dict) -> str:
    joined = '\n'.join(f'- {e}' for e in errors[:_ERROR_LIMIT])
    preview = (raw or '')[:_PREVIEW_CHARS] or '<empty>'
    return (
        f'## 原始任务\n{original_user}\n\n'
        f'## 上次错误输出(请勿重复)\n```\n{preview}\n```\n\n'
        f'## 校验错误\n{joined}\n\n'
        f'## 必须满足的 JSON Schema\n```json\n{json.dumps(schema, ensure_ascii=False)}\n```\n\n'
        '只输出满足该 schema 的一个 JSON 对象,不要 markdown,不要解释文本。'
    )


def _structured_call(
    llm: LLM, *, system_prompt: str, user_text: str, schema: dict, state: EvalQcRunState, max_repair: int = 2
) -> dict:
    parsed: dict = {}
    errors: list[str] = []
    raw = ''
    current_user = user_text
    for attempt in range(max_repair + 1):
        sp = system_prompt if attempt == 0 else _REPAIR_SYSTEM_PROMPT
        raw = strip_thinking(llm(f'{sp}\n\n---\n\n{current_user}') or '').strip()
        if raw:
            parsed, errors, _ = parse_and_validate(raw, schema)
            if not errors:
                return parsed
        else:
            errors = ['<root>: empty response (model returned only reasoning/whitespace)']
        if attempt >= max_repair:
            break
        current_user = _repair_user(user_text, raw, errors, schema)
    state.schema_failures += 1
    if state.schema_failures >= EVO_MAX_SCHEMA_FAILURES:
        raise ApplyError(
            'SCHEMA_FOLLOW_FAILED',
            f'model failed to follow JSON schema {state.schema_failures} times; '
            'this model likely cannot drive eval_qc. '
            'Switch evo_llm to qwen3-max / deepseek-v3 / claude-sonnet.',
            {'failures': state.schema_failures, 'last_errors': errors[:5]},
            kind='permanent',
        )
    return {}


# ---- Stage 1: extract — decompose the query into core/qualifier -----------

def _normalize_query(raw_query: Any) -> dict[str, str]:
    if not isinstance(raw_query, dict):
        return {}
    core = str(raw_query.get('core') or '').strip()
    if not core:
        return {}
    return {'core': core, 'qualifier': str(raw_query.get('qualifier') or '').strip()}


def _extract_query(*, query: Any, llm: LLM, state: EvalQcRunState) -> dict[str, str]:
    user = json.dumps({'query': query}, ensure_ascii=False, indent=2)
    parsed = _structured_call(
        llm, system_prompt=load_prompt(_EXTRACT_NAME), user_text=user,
        schema=EVAL_QC_EXTRACT, state=state, max_repair=1,
    )
    return _normalize_query(parsed.get('query'))


# ---- Stage 2: judge — score the three query-anchored logics ---------------

def _judge(payload: dict[str, Any], *, query: dict[str, str], llm: LLM, state: EvalQcRunState) -> dict[str, Any]:
    user = json.dumps(
        {
            'query': query,
            'gt_answer': payload.get('gt_answer', ''),
            'gt_text': payload.get('gt_text', []),
            'key_points': payload.get('key_points', []),
        },
        ensure_ascii=False,
        indent=2,
    )
    return _structured_call(
        llm, system_prompt=load_prompt(_JUDGE_NAME), user_text=user, schema=EVAL_QC_JUDGE, state=state
    )


def _judgments_valid(parsed: dict[str, Any]) -> bool:
    judgments = parsed.get('judgments')
    if not isinstance(judgments, list):
        return False
    seen: set[str] = set()
    for entry in judgments:
        if not isinstance(entry, dict):
            return False
        logic_id = str(entry.get('logic_id') or '').strip()
        if logic_id not in LOGIC_IDS or logic_id in seen:
            return False
        if entry.get('level') not in _VALID_LEVELS:
            return False
        if not str(entry.get('reason') or '').strip():
            return False
        seen.add(logic_id)
    return seen == set(LOGIC_IDS)


def _failure_output() -> dict[str, Any]:
    return {
        'judgments': [
            {'logic_id': logic_id, 'reason': _FAILURE_REASON, 'level': 0.1} for logic_id in LOGIC_IDS
        ],
        'summary_reason': _FAILURE_REASON,
    }


# ---- Orchestration --------------------------------------------------------

def run_eval_qc(payload: dict[str, Any], *, llm: LLM, state: EvalQcRunState) -> dict[str, Any]:
    """Run extract → judge for one case; return {judgments, summary_reason}.
    """
    try:
        query = _extract_query(query=payload.get('query', ''), llm=llm, state=state)
        if not query:
            return _failure_output()
        parsed = _judge(payload, query=query, llm=llm, state=state)
        if not _judgments_valid(parsed):
            return _failure_output()
        return parsed
    except ApplyError:
        raise
    except Exception:
        return _failure_output()
