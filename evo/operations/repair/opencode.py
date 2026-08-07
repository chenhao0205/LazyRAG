from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from evo.llm import parse_json_object

from .contracts import (
    RepairAction,
    RepairAgentError,
    RepairInput,
    RepairView,
    contract_dict,
    repair_action,
)


ModelCall = Callable[..., Any]


class DecisionAgent(Protocol):
    """The sole semantic decision boundary used by one RepairSession."""

    def decide(self, view: RepairView) -> RepairAction:
        ...

    def summarize(
        self,
        objective: str,
        guidance: str,
        previous_brief: str,
        events: list[dict[str, Any]],
    ) -> str:
        ...

    def assess_finish(
        self,
        repair_input: RepairInput,
        view: RepairView,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, str]:
        ...


class OpenCodeAdapter(DecisionAgent):
    """One-shot model adapter; it never runs an inner tool or agent loop."""

    def __init__(self, model_call: ModelCall, timeout_seconds: int = 120) -> None:
        self.model_call = model_call
        self.timeout_seconds = max(1, int(timeout_seconds))

    def decide(self, view: RepairView) -> RepairAction:
        prompt = (
            'You are the only decision Agent in a RepairSession. Use the projected working memory and choose '
            'exactly one capability call. Do not describe a pipeline and do not execute tools internally. '
            'workspace can list/read/write/diff candidate source or work files; shell runs one argv command; '
            'test accepts level L0, L1, or L2; research searches or reads sources; finish requests completion. '
            'Return only JSON with exactly call_id, tool, arguments.\n'
            'Workspace arguments: {operation: list|read|write|diff, path?: string, content?: string}.\n'
            'Shell arguments: {command: string[], cwd: source|work, timeout_seconds?: int}.\n'
            'Test arguments: {level: L0|L1|L2}.\n'
            'Research arguments: {operation: search|read, query: string, urls?: string[]}.\n'
            'Finish arguments: {reason: string}.\n'
            f'RepairView: {json.dumps(contract_dict(view), ensure_ascii=False, default=str)}'
        )
        try:
            return repair_action(self._json(prompt))
        except (TypeError, ValueError) as exc:
            raise RepairAgentError('agent_action_invalid', str(exc)) from exc

    def summarize(
        self,
        objective: str,
        guidance: str,
        previous_brief: str,
        events: list[dict[str, Any]],
    ) -> str:
        prompt = (
            'Compress older Repair events into a factual working-memory brief. Preserve unresolved constraints, '
            'observed facts, failed attempts and human messages. Never replace or reinterpret the pinned objective '
            'or guidance. Return only JSON {"memory_brief": "..."}.\n'
            f'Pinned objective: {objective}\nPinned guidance: {guidance}\n'
            f'Previous brief: {previous_brief}\n'
            f'Older events: {json.dumps(events, ensure_ascii=False, default=str)}'
        )
        value = self._json(prompt)
        brief = str(value.get('memory_brief') or '').strip()
        if not brief:
            raise RepairAgentError('memory_brief_invalid')
        return brief

    def assess_finish(
        self,
        repair_input: RepairInput,
        view: RepairView,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, str]:
        prompt = (
            'Assess whether the current Repair workspace semantically satisfies the objective, guidance and '
            'case scope. Base the verdict on the current diff and evidence in RepairView, not keyword presence. '
            'Return only JSON {"satisfied": true|false, "summary": "..."}.\n'
            f'RepairInput: {json.dumps(contract_dict(repair_input), ensure_ascii=False, default=str)}\n'
            f'RepairView: {json.dumps(contract_dict(view), ensure_ascii=False, default=str)}\n'
            f'Finish request: {json.dumps(dict(arguments), ensure_ascii=False, default=str)}'
        )
        value = self._json(prompt)
        if not isinstance(value.get('satisfied'), bool):
            raise RepairAgentError('finish_assessment_invalid')
        summary = str(value.get('summary') or '').strip()
        if not summary:
            raise RepairAgentError('finish_assessment_invalid')
        return value['satisfied'], summary

    def _json(self, prompt: str) -> dict[str, Any]:
        try:
            raw = self.model_call(
                prompt,
                stream=False,
                response_format={'type': 'json_object'},
                timeout=self.timeout_seconds,
                max_retries=1,
                max_tokens=4096,
            )
            value = parse_json_object(raw)
        except Exception as exc:
            raise RepairAgentError('model_call_failed', str(exc)) from exc
        if not isinstance(value, Mapping):
            raise RepairAgentError('model_response_invalid')
        return dict(value)


__all__ = ['DecisionAgent', 'ModelCall', 'OpenCodeAdapter']
