from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from .contracts import (
    RepairAction,
    RepairCapabilityError,
    RepairContractError,
    RepairInput,
    RepairObservation,
    RepairTool,
)
from .workspace import WorkspacePaths


EXTERNAL_TOOLS = ('workspace', 'shell', 'test', 'research')


class Capability(Protocol):
    """Every component has the same frozen input and output."""

    def __call__(self, action: RepairAction) -> RepairObservation:
        ...


class CapabilityFactory(Protocol):
    def __call__(
        self,
        repair_input: RepairInput,
        paths: WorkspacePaths,
    ) -> Mapping[RepairTool, Capability]:
        ...


class CapabilityDispatcher:
    """A routing boundary only; component implementation belongs outside the skeleton."""

    def __init__(self, capabilities: Mapping[RepairTool, Capability]) -> None:
        self.capabilities = dict(capabilities)
        missing = [tool for tool in EXTERNAL_TOOLS if tool not in self.capabilities]
        if missing:
            raise RepairContractError('capabilities_missing', ','.join(missing))

    def execute(self, action: RepairAction, current_hash: str) -> RepairObservation:
        capability = self.capabilities.get(action.tool)
        if capability is None:
            raise RepairContractError('capability_not_external', action.tool)
        try:
            observation = capability(action)
        except RepairCapabilityError as exc:
            observation = RepairObservation(
                call_id=action.call_id,
                status='error',
                summary=str(exc),
                artifact_refs=[],
                workspace_hash=current_hash,
            )
        if observation.call_id != action.call_id:
            raise RepairContractError('observation_call_id_mismatch', action.call_id)
        return observation


__all__ = ['Capability', 'CapabilityDispatcher', 'CapabilityFactory', 'EXTERNAL_TOOLS']
