from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .contracts import (
    RepairAction,
    RepairAgentError,
    RepairContractError,
    RepairInput,
    RepairObservation,
    RepairResult,
    RepairView,
    ResultStatus,
    contract_dict,
)
from .dispatch import CapabilityDispatcher, CapabilityFactory
from .memory import EventMemory
from .opencode import DecisionAgent
from .validation import check_completion, record_finish_evidence, validation_evidence
from .workspace import (
    DEFAULT_RUNTIME_ROOT,
    WorkspacePaths,
    changed_paths,
    initialize_workspace,
    path_in_scope,
    workspace_hash,
    write_json,
    write_patch,
)


class RepairSession:
    """One invocation over a run-scoped, working-memory-driven Agent loop."""

    def __init__(
        self,
        agent: DecisionAgent,
        capability_factory: CapabilityFactory,
        *,
        runtime_root: Path = DEFAULT_RUNTIME_ROOT,
    ) -> None:
        self.agent = agent
        self.capability_factory = capability_factory
        self.runtime_root = runtime_root

    def run(self, input: RepairInput) -> RepairResult:
        paths = initialize_workspace(input, self.runtime_root)
        memory = EventMemory(paths, input)
        dispatcher = CapabilityDispatcher(self.capability_factory(input, paths))
        turns = _positive_budget(input.budget.get('turns'), 50)
        seconds = _positive_budget(input.budget.get('seconds'), 3600)
        deadline = time.monotonic() + seconds
        used_calls: set[str] = {record.call_id for record in memory.read() if record.call_id}
        unresolved: list[str] = []
        for turn in range(1, turns + 1):
            remaining = self._remaining_budget(input.budget, turns, turn, deadline)
            if remaining['seconds'] <= 0:
                unresolved.append('time budget exhausted')
                break
            current_hash = workspace_hash(paths.source)
            action: RepairAction | None = None
            try:
                view, action = self._plan(paths, memory, current_hash, remaining, used_calls)
                observation = self._act(input, paths, dispatcher, view, action, current_hash)
            except (RepairAgentError, RepairContractError) as exc:
                call_id = action.call_id if action is not None else f'agent-{len(memory.read()) + 1:08d}'
                observation = RepairObservation(
                    call_id=call_id,
                    status='error',
                    summary=str(exc),
                    artifact_refs=[],
                    workspace_hash=workspace_hash(paths.source),
                )
            observation = self._bind_workspace_hash(observation, workspace_hash(paths.source))
            self._persist(memory, observation)
            if observation.status != 'success':
                unresolved.append(observation.summary)
            if (
                action is not None
                and action.tool == 'finish'
                and observation.status == 'success'
                and self._verify(paths, memory, observation.workspace_hash, action.call_id)
            ):
                return self._finish_result(input, paths, memory, 'success', [], 'repair completed')
        status = 'partial' if changed_paths(input.source_ref, paths.source) else 'failed'
        summary = 'repair stopped before the completion gate passed'
        return self._finish_result(input, paths, memory, status, unresolved[-10:], summary)

    def _plan(
        self,
        paths: WorkspacePaths,
        memory: EventMemory,
        current_hash: str,
        remaining: dict[str, Any],
        used_calls: set[str],
    ) -> tuple[RepairView, RepairAction]:
        view = memory.project(
            current_hash,
            validation_evidence(paths, memory.observations(), current_hash),
            remaining,
            self.agent.summarize,
        )
        action = self.agent.decide(view)
        self._validate_call_id(action, used_calls)
        used_calls.add(action.call_id)
        memory.record_action(action, current_hash)
        return view, action

    def _act(
        self,
        input: RepairInput,
        paths: WorkspacePaths,
        dispatcher: CapabilityDispatcher,
        view: RepairView,
        action: RepairAction,
        current_hash: str,
    ) -> RepairObservation:
        if action.tool != 'finish':
            return dispatcher.execute(action, current_hash)
        semantic_satisfied, assessment = self.agent.assess_finish(
            input,
            view,
            action.arguments,
        )
        changed = changed_paths(input.source_ref, paths.source)
        scope_satisfied = bool(changed) and all(path_in_scope(path, input.case_scope) for path in changed)
        return record_finish_evidence(
            paths,
            action,
            current_hash,
            semantic_satisfied,
            scope_satisfied,
            assessment,
        )

    @staticmethod
    def _persist(memory: EventMemory, observation: RepairObservation) -> None:
        memory.record_observation(observation)

    @staticmethod
    def _verify(paths: WorkspacePaths, memory: EventMemory, current_hash: str, finish_call_id: str) -> bool:
        return check_completion(paths, memory.observations(), current_hash, finish_call_id)

    @staticmethod
    def _bind_workspace_hash(observation: RepairObservation, actual_hash: str) -> RepairObservation:
        if observation.workspace_hash == actual_hash:
            return observation
        return RepairObservation(
            call_id=observation.call_id,
            status='error',
            summary=(
                f'{observation.summary}; capability reported workspace hash '
                f'{observation.workspace_hash}, actual hash is {actual_hash}'
            ),
            artifact_refs=observation.artifact_refs,
            workspace_hash=actual_hash,
        )

    @staticmethod
    def _validate_call_id(action: RepairAction, used_calls: set[str]) -> None:
        if action.call_id in used_calls:
            raise RepairContractError('call_id_reused', action.call_id)

    @staticmethod
    def _remaining_budget(
        budget: dict[str, Any],
        turns: int,
        turn: int,
        deadline: float,
    ) -> dict[str, Any]:
        remaining = dict(budget)
        remaining['turns'] = max(0, turns - turn + 1)
        remaining['seconds'] = max(0, int(deadline - time.monotonic()))
        return remaining

    @staticmethod
    def _finish_result(
        input: RepairInput,
        paths: WorkspacePaths,
        memory: EventMemory,
        status: ResultStatus,
        unresolved: list[str],
        summary: str,
    ) -> RepairResult:
        patch = write_patch(input.source_ref, paths.source, paths.control / 'result.patch')
        references = memory.artifact_refs()
        result = RepairResult(
            status=status,
            patch_ref=str(patch) if patch.stat().st_size else '',
            evidence_refs=references,
            summary=summary,
            unresolved=list(dict.fromkeys(unresolved)),
        )
        write_json(paths.result, contract_dict(result))
        memory.append('invocation.finished', contract_dict(result), workspace_hash=workspace_hash(paths.source))
        return result


def _positive_budget(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return default
    return value


__all__ = ['RepairSession']
