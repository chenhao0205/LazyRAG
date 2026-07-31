import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple

from .models import DemoWriteRequest, InvestigationRequest, OpenCodeCallResult, OpenCodeTranscript, normalise_relative_path


class OpenCodeClient(Protocol):
    def create_session(self, title: str) -> str:
        ...

    def prompt(self, session_id: str, prompt: str, model: Optional[str] = None, agent: Optional[str] = None) -> Any:
        ...

    def messages(self, session_id: str) -> Any:
        ...


class OpenCodeScopeError(RuntimeError):
    pass


class OpenCodeSessionTool:
    def __init__(
        self,
        client: OpenCodeClient,
        title: str,
        model: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        self.client = client
        self.title = title
        self.model = model
        self.agent = agent
        self.session_id: Optional[str] = None
        self.calls: List[OpenCodeCallResult] = []

    def investigate(self, request: InvestigationRequest) -> OpenCodeCallResult:
        session_id = self._session_id()
        response = self.client.prompt(session_id, self._investigation_prompt(request), self.model, self.agent)
        result = OpenCodeCallResult(session_id, "codesearch", response, ())
        self.calls.append(result)
        return result

    def write_demo(self, request: DemoWriteRequest) -> OpenCodeCallResult:
        session_id = self._session_id()
        before = snapshot_files(request.workspace)
        response = self.client.prompt(session_id, self._demo_prompt(request), self.model, self.agent)
        after = snapshot_files(request.workspace)
        changed_paths = tuple(sorted(changed_file_paths(before, after)))
        validate_changed_paths(changed_paths, request.policy.allowed_roots, request.policy.blocked_roots)
        result = OpenCodeCallResult(session_id, "demowrite", response, changed_paths)
        self.calls.append(result)
        return result

    def export_transcript(self, artifact_path: Path) -> OpenCodeTranscript:
        session_id = self._session_id()
        messages = self.client.messages(session_id)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            json.dumps(
                {
                    "session_id": session_id,
                    "calls": [
                        {"action": call.action, "session_id": call.session_id, "changed_paths": list(call.changed_paths), "response": call.response}
                        for call in self.calls
                    ],
                    "messages": messages,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return OpenCodeTranscript(session_id, tuple(self.calls), messages, artifact_path)

    def _session_id(self) -> str:
        if self.session_id is None:
            self.session_id = self.client.create_session(self.title)
        return self.session_id

    @staticmethod
    def _investigation_prompt(request: InvestigationRequest) -> str:
        spans = json.dumps(list(request.code_spans), ensure_ascii=False)
        return f"""You are the source-investigation tool for one repair category. Work only in the current workspace.

Category: {request.category}
Upstream analysis: {request.analysis}
Relevant code spans: {spans}
Question: {request.question}

Search and read local source as needed. Reply with factual evidence: target paths, symbols, current behaviour, and a minimal experiment proposal. Do not create, edit, delete, or execute files in this call. A later call in this same session may ask you to write the demo."""

    @staticmethod
    def _demo_prompt(request: DemoWriteRequest) -> str:
        allowed = list(request.policy.allowed_roots)
        blocked = list(request.policy.blocked_roots)
        return f"""Continue in this SAME repair session and create the minimal experiment requested below.

Category: {request.category}
Method: {request.method}
Expected observation: {request.expected_observation}
Allowed source roots: {json.dumps(allowed, ensure_ascii=False)}
Blocked roots: {json.dumps(blocked, ensure_ascii=False)}

You may change only an allowed source path and files under .repair_demo/. Create exactly .repair_demo/run_demo.py. It must use the Python standard library, perform the experiment, print exactly one JSON object to stdout, and use a non-zero exit code on failure. Do not install dependencies, use the network, or run the demo yourself. Finish by listing changed files."""


def snapshot_files(workspace: Path) -> Dict[str, bytes]:
    ignored_parts = {".opencode", "__pycache__"}
    snapshot: Dict[str, bytes] = {}
    for path in workspace.rglob("*"):
        if not path.is_file() or ignored_parts.intersection(path.relative_to(workspace).parts):
            continue
        snapshot[normalise_relative_path(path.relative_to(workspace))] = path.read_bytes()
    return snapshot


def changed_file_paths(before: Dict[str, bytes], after: Dict[str, bytes]) -> Iterable[str]:
    for path in set(before).union(after):
        if before.get(path) != after.get(path):
            yield path


def validate_changed_paths(changed_paths: Iterable[str], allowed_roots: Iterable[str], blocked_roots: Iterable[str]) -> None:
    allowed = tuple(root.strip("/") for root in allowed_roots)
    blocked = tuple(root.strip("/") for root in blocked_roots)
    invalid = []
    for path in changed_paths:
        normalised = path.strip("/")
        if any(normalised == root or normalised.startswith(root + "/") for root in blocked):
            invalid.append(path)
            continue
        if normalised == ".repair_demo" or normalised.startswith(".repair_demo/"):
            continue
        if not any(normalised == root or normalised.startswith(root + "/") for root in allowed):
            invalid.append(path)
    if invalid:
        raise OpenCodeScopeError(f"OpenCode changed paths outside policy: {', '.join(sorted(invalid))}")
