from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from evo.llm import LazyLLMClient, parse_json_object

from .schemas import TurnPlan


PROMPT = """
Translate one user message into one strict EVO TurnPlan JSON object.
Return JSON only. Do not use markdown or explanations.

The next action kinds are:
- flow: start, approve, pause, resume, rerun, retry, cancel. approve and rerun require a stage.
  retry accepts an optional stage and only retries recorded failures; rerun actively regenerates a stage.
- query: progress, run_history, stage_snapshot, case_snapshot, operation_events,
  stage_result, artifact, artifact_history.
  operation_events reads operation-internal steps, logs, errors and structured details;
  it may filter by stage, case_id, event_type and level.
- case: rerun or retry. rerun requires an explicit stage; retry only retries that case's recorded failure.
- repair_guidance: append the user's concrete Repair observation, constraint, or optimization direction to
  repair.policy.user_guidance. Preserve the user's meaning and do not invent technical conclusions.
- confirmation: respond to the pending destructive-action confirmation.
- clarify asks for missing information.
- final answers ordinary chat, explains current capabilities, or gives execution advice without changing state.

Flow retry retries only the currently recorded failures in the selected stage; successful cases stay unchanged.
Without a stage it retries failures in the current stage. Use rerun when the user wants a new successful result.
Do not invent rerun_step, rerun_case_stage, invalidate_from_step, continue, or patch_collection.
Cancelling a run requires a separate confirmation. Return the executable action first; the
application creates that confirmation. Only return confirmation when projection.has_pending_confirmation is true.
Stage approval is a flow approve action and is different from destructive-action confirmation.

Never modify, patch, replace, roll back, comment on, add, or delete an artifact, case, or configuration,
except that repair_guidance may append one user-authored observation or direction to Repair policy.
If the user requests a content or structure change, use final to explain that they must edit it in
the product UI; the Service receives the complete edited value and base version, then Flow computes
the affected recomputation. You may suggest what to edit, but must not create an executable mutation.

Use intent_catalog as the source of truth for stages and artifact ids. Configuration changes belong to the product UI.
Resolve ordinal stages from the catalog order. Pick exactly one action. Put remaining user goals
in active_agenda. Never claim that a long-running flow has completed; only describe the action.
If information is missing, return needs_input with a clarify action.
"""


class StructuredPlanError(ValueError):
    pass


def plan_next_turn(context: Mapping[str, Any], llm_config: Mapping[str, Any]) -> TurnPlan:
    schema = TurnPlan.model_json_schema()
    client = LazyLLMClient(llm_config=llm_config, model='evo_llm')
    error = ''
    raw: Any = None
    for _ in range(2):
        retry_note = f'\nPrevious validation error: {error}' if error else ''
        prompt = (
            f'{PROMPT}\n'
            f'TurnPlan JSON schema: {json.dumps(schema, ensure_ascii=False)}\n'
            f'Context: {json.dumps(context, ensure_ascii=False, sort_keys=True, default=str)}'
            f'{retry_note}'
        )
        try:
            raw = client(prompt, stream=False, response_format={'type': 'json_object'})
            return TurnPlan.model_validate(parse_json_object(raw))
        except Exception as exc:
            error = str(exc)
    raise StructuredPlanError(error or 'LLM did not return a valid turn plan')


def answer_query(context: Mapping[str, Any], result: object, llm_config: Mapping[str, Any]) -> str:
    prompt = (
        '你是 EVO 的只读查询回答器。只根据 query_result 和 flow_snapshot 回答，'
        '不编造，不发起操作，用简洁中文直接回答。\n'
        f'Context: {_json(context)}\n'
        f'Query result: {_json(result)}'
    )
    try:
        client = LazyLLMClient(llm_config=llm_config, model='evo_llm')
        return str(client(prompt, stream=False)).strip()
    except Exception:
        return '已读取当前信息，详细结果已写入本次查询记录。'


def _json(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text if len(text) <= 12000 else text[:12000]


__all__ = ['StructuredPlanError', 'answer_query', 'plan_next_turn']
