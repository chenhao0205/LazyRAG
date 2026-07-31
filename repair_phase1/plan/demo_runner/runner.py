import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .models import DemoRunRequest, DemoRunResult, FIXED_DEMO_ENTRY


NO_EXIT_CODE = -1


class DemoRunner:
    def run(self, request: DemoRunRequest) -> DemoRunResult:
        artifact_dir = request.artifact_dir
        artifact_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = artifact_dir / "stdout.log"
        stderr_path = artifact_dir / "stderr.log"
        result_path = artifact_dir / "result.json"
        journal_path = artifact_dir / "journal.jsonl"
        entry = request.workspace / FIXED_DEMO_ENTRY

        if not entry.is_file():
            return self._save_result(
                "entry_missing",
                NO_EXIT_CODE,
                {},
                "",
                f"Fixed demo entry is missing: {entry}",
                stdout_path,
                stderr_path,
                result_path,
                journal_path,
            )

        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONNOUSERSITE": "1",
            "PYTHONPYCACHEPREFIX": str(artifact_dir / "pycache"),
        }
        try:
            completed = subprocess.run(
                [sys.executable, str(entry)],
                cwd=str(request.workspace),
                env=environment,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return self._save_result(
                "timed_out",
                NO_EXIT_CODE,
                {},
                as_text(exc.stdout),
                as_text(exc.stderr),
                stdout_path,
                stderr_path,
                result_path,
                journal_path,
            )

        payload = parse_payload(completed.stdout)
        status = "passed" if completed.returncode == 0 else "failed"
        return self._save_result(
            status,
            completed.returncode,
            payload,
            completed.stdout,
            completed.stderr,
            stdout_path,
            stderr_path,
            result_path,
            journal_path,
        )

    @staticmethod
    def _save_result(
        status: str,
        exit_code: int,
        payload: Dict[str, Any],
        stdout: str,
        stderr: str,
        stdout_path: Path,
        stderr_path: Path,
        result_path: Path,
        journal_path: Path,
    ) -> DemoRunResult:
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        record = {
            "status": status,
            "exit_code": exit_code,
            "payload": payload,
            "entry": FIXED_DEMO_ENTRY.as_posix(),
            "stdout": file_ref(stdout_path),
            "stderr": file_ref(stderr_path),
        }
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        journal_record = {
            "event": "demo.run",
            "time": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "exit_code": exit_code,
            "result": file_ref(result_path),
        }
        journal_path.write_text(json.dumps(journal_record, ensure_ascii=False) + "\n", encoding="utf-8")
        return DemoRunResult(status, exit_code, payload, stdout_path, stderr_path, result_path, journal_path)


def parse_payload(stdout: str) -> Dict[str, Any]:
    try:
        value = json.loads(stdout)
    except json.JSONDecodeError:
        return {}
    if not isinstance(value, dict):
        return {}
    return value


def file_ref(path: Path) -> Dict[str, str]:
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
