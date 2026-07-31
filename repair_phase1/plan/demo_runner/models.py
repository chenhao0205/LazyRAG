from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


FIXED_DEMO_ENTRY = Path(".repair_demo/run_demo.py")


@dataclass(frozen=True)
class DemoRunRequest:
    workspace: Path
    artifact_dir: Path
    timeout_seconds: int = 30


@dataclass(frozen=True)
class DemoRunResult:
    status: str
    exit_code: int
    payload: Dict[str, Any]
    stdout_path: Path
    stderr_path: Path
    result_path: Path
    journal_path: Path
