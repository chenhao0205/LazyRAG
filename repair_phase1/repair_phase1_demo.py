import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

from repair_phase1.plan.demo_runner import DemoRunRequest, DemoRunner
from repair_phase1.plan.opencode_session.client import ManagedOpenCodeServer, OpenCodeError, OpenCodeHttpClient, choose_unused_port, wait_for_health
from repair_phase1.plan.opencode_session.models import DemoWriteRequest, InvestigationRequest, WorkspacePolicy
from repair_phase1.plan.opencode_session.tool import OpenCodeSessionTool
from repair_phase1.plan.repair_plan import build_repair_plan, write_json


DEFAULT_ANALYSIS = {
    "category": "category_01",
    "summary": "订单总价在配置 25% 税率时返回 75.0，而不是 125.0。",
    "analysis": "pricing.calculate_total 将税率当作折扣率，使用了 (1 - tax_rate)。",
    "code_span": [{"path": "src/pricing.py", "symbol": "calculate_total", "line": ["1-2"]}],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-file", help="JSON input. Defaults to the bundled tax-calculation mock.")
    parser.add_argument("--opencode-bin", default="opencode")
    parser.add_argument("--model", help="OpenCode model in provider/model form")
    parser.add_argument("--agent", help="Optional OpenCode agent name")
    parser.add_argument("--username", default=os.getenv("OPENCODE_SERVER_USERNAME"))
    parser.add_argument("--password", default=os.getenv("OPENCODE_SERVER_PASSWORD"))
    parser.add_argument("--cleanup", action="store_true")
    return parser.parse_args()


def load_analysis(path: str) -> Dict[str, Any]:
    if not path:
        return dict(DEFAULT_ANALYSIS)
    with Path(path).open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError("analysis input must be a JSON object")
    if "categories" in value:
        categories = value["categories"]
        if not isinstance(categories, dict) or not categories:
            raise ValueError("analysis.categories must be a non-empty object")
        category, details = sorted(
            categories.items(), key=lambda item: item[1].get("all_case_average_drop", 0), reverse=True
        )[0]
        return {
            "category": category,
            "summary": details["summary"],
            "analysis": details["analysis"],
            "code_span": details.get("code_span", []),
        }
    required = ("category", "summary", "analysis", "code_span")
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError(f"analysis JSON missing fields: {', '.join(missing)}")
    return value


def write_toy_project(workspace: Path) -> None:
    source_dir = workspace / "src"
    source_dir.mkdir()
    (source_dir / "pricing.py").write_text(
        "def calculate_total(prices, tax_rate):\n    return sum(prices) * (1 - tax_rate)\n",
        encoding="utf-8",
    )


def run_toolchain(args: argparse.Namespace) -> Tuple[Path, int]:
    analysis = load_analysis(args.analysis_file)
    run_id = f"repair-demo-{os.urandom(4).hex()}"
    workspace_root = Path(__file__).resolve().parent / "tmp"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=f"{run_id}-", dir=str(workspace_root)))
    artifact_dir = workspace / ".repair_artifacts"
    server = None
    try:
        write_toy_project(workspace)
        spec_path = write_json(
            artifact_dir / "spec.json",
            {
                "category": analysis["category"],
                "method": "Apply tax as an addition rather than a discount.",
                "expected_observation": "[40, 60] with 25% tax returns 125.0.",
            },
        )
        server = ManagedOpenCodeServer(args.opencode_bin, workspace, choose_unused_port(), args.username, args.password)
        server.start()
        client = OpenCodeHttpClient(server.url, args.username, args.password)
        wait_for_health(client, server)
        tool = OpenCodeSessionTool(client, f"repair-phase1-{analysis['category']}", args.model, args.agent)
        tool.investigate(
            InvestigationRequest(
                analysis["category"],
                "Which expression explains the observed incorrect total and what is the smallest valid experiment?",
                analysis["analysis"],
                tuple(analysis["code_span"]),
                workspace,
            )
        )
        tool.write_demo(
            DemoWriteRequest(
                analysis["category"],
                "Apply tax as an addition rather than a discount.",
                "[40, 60] with 25% tax returns 125.0.",
                workspace,
                WorkspacePolicy(("src",), (".git", "tests", "data", "evo")),
            )
        )
        transcript = tool.export_transcript(artifact_dir / "opencode_transcript.json")
        result = DemoRunner().run(DemoRunRequest(workspace, artifact_dir))
        plan = build_repair_plan(
            run_id,
            analysis["category"],
            "Apply tax as an addition rather than a discount.",
            analysis["code_span"],
            ("Do not hard-code individual cases.", "Do not change blocked roots."),
            "The fixed entry verified the reported tax case and emitted a structured result.",
            spec_path,
            workspace / ".repair_demo" / "run_demo.py",
            result,
        )
        write_json(artifact_dir / "repair_plan.json", plan)
        print(json.dumps({"workspace": str(workspace), "session_id": transcript.session_id, "status": result.status}, ensure_ascii=False))
        return workspace, 0 if result.status == "passed" else 1
    finally:
        if server is not None:
            server.stop()
        if args.cleanup:
            shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    args = parse_args()
    try:
        _, status = run_toolchain(args)
        return status
    except (OSError, ValueError, OpenCodeError) as exc:
        print(f"repair phase 1 failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
