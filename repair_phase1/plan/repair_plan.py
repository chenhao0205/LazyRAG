import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from repair_phase1.plan.demo_runner.models import DemoRunResult


def write_json(path: Path, value: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def artifact_ref(path: Path, run_id: str, category: str) -> Dict[str, str]:
    return {
        "uri": f"phase1://{run_id}/{category}/{path.name}",
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def build_repair_plan(
    run_id: str,
    category: str,
    method_summary: str,
    code_scope: Iterable[Dict[str, Any]],
    requirements: Iterable[str],
    reason: str,
    spec_path: Path,
    demo_path: Path,
    demo_result: DemoRunResult,
) -> Dict[str, Any]:
    return {
        "id": "repair.plan",
        "target_category": category,
        "method": {
            "summary": method_summary,
            "code_scope": list(code_scope),
            "requirements": list(requirements),
        },
        "demo": {
            "reason": reason,
            "spec_ref": artifact_ref(spec_path, run_id, category),
            "demo_ref": artifact_ref(demo_path, run_id, category),
            "result_ref": artifact_ref(demo_result.result_path, run_id, category),
            "journal_ref": artifact_ref(demo_result.journal_path, run_id, category),
        },
    }
