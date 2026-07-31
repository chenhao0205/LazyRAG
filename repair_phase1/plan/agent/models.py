from dataclasses import dataclass
from typing import Any, Dict, Tuple

from repair_phase1.plan.opencode_session.models import WorkspacePolicy


VALID_ACTIONS = ("codesearch", "demowrite", "finish")


@dataclass(frozen=True)
class RepairContext:
    run_id: str
    category: str
    analysis: Dict[str, Any]
    user_guidance: Tuple[str, ...]
    policy: WorkspacePolicy


@dataclass(frozen=True)
class RepairAction:
    name: str
    payload: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.name not in VALID_ACTIONS:
            raise ValueError(f"Unsupported repair action: {self.name}")


@dataclass(frozen=True)
class ToolResult:
    action: RepairAction
    value: Any
