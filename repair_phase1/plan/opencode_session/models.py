from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class WorkspacePolicy:
    """Paths that an OpenCode demo-generation request may modify."""

    allowed_roots: Tuple[str, ...]
    blocked_roots: Tuple[str, ...]


@dataclass(frozen=True)
class InvestigationRequest:
    category: str
    question: str
    analysis: str
    code_spans: Tuple[Dict[str, Any], ...]
    workspace: Path


@dataclass(frozen=True)
class DemoWriteRequest:
    category: str
    method: str
    expected_observation: str
    workspace: Path
    policy: WorkspacePolicy


@dataclass(frozen=True)
class OpenCodeCallResult:
    session_id: str
    action: str
    response: Any
    changed_paths: Tuple[str, ...]


@dataclass(frozen=True)
class OpenCodeTranscript:
    session_id: str
    calls: Tuple[OpenCodeCallResult, ...]
    messages: Any
    artifact_path: Path


def normalise_relative_path(path: Path) -> str:
    value = path.as_posix()
    return value[2:] if value.startswith("./") else value
