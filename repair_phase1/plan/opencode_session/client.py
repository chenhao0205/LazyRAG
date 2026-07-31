import base64
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


class OpenCodeError(RuntimeError):
    pass


class OpenCodeHttpClient:
    def __init__(
        self,
        base_url: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode("utf-8")
        if self.password:
            username = self.username or "opencode"
            token = base64.b64encode((username + ":" + self.password).encode("utf-8")).decode("ascii")
            headers["Authorization"] = "Basic " + token
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeError(f"OpenCode API {method} {path} returned HTTP {exc.code}: {details}") from exc
        except urllib.error.URLError as exc:
            raise OpenCodeError(f"Unable to reach OpenCode API {self.base_url}: {exc.reason}") from exc
        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError as exc:
            raise OpenCodeError(f"OpenCode API {method} {path} did not return JSON") from exc

    def health(self) -> Dict[str, Any]:
        value = self.request("GET", "/global/health")
        if not isinstance(value, dict) or not value.get("healthy"):
            raise OpenCodeError(f"Unexpected OpenCode health response: {value!r}")
        return value

    def create_session(self, title: str) -> str:
        value = self.request("POST", "/session", {"title": title})
        try:
            return str(value["id"])
        except (TypeError, KeyError) as exc:
            raise OpenCodeError(f"OpenCode session response has no id: {value!r}") from exc

    def prompt(self, session_id: str, prompt: str, model: Optional[str] = None, agent: Optional[str] = None) -> Any:
        body: Dict[str, Any] = {"parts": [{"type": "text", "text": prompt}]}
        if model:
            provider_id, separator, model_id = model.partition("/")
            if not separator or not provider_id or not model_id:
                raise ValueError("model must use provider/model format")
            body["model"] = {"providerID": provider_id, "modelID": model_id}
        if agent:
            body["agent"] = agent
        return self.request("POST", f"/session/{session_id}/message", body)

    def messages(self, session_id: str) -> Any:
        return self.request("GET", f"/session/{session_id}/message")


class ManagedOpenCodeServer:
    def __init__(
        self,
        binary: str,
        workspace: Path,
        port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self.binary = binary
        self.workspace = workspace
        self.port = port
        self.username = username
        self.password = password
        self.process: Optional[subprocess.Popen] = None
        self.log_file = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        self.log_file = (self.workspace / "opencode-server.log").open("wb")
        environment = os.environ.copy()
        if self.password:
            environment["OPENCODE_SERVER_PASSWORD"] = self.password
            environment.setdefault("OPENCODE_SERVER_USERNAME", self.username or "opencode")
        self.process = subprocess.Popen(
            [self.binary, "serve", "--hostname", "127.0.0.1", "--port", str(self.port)],
            cwd=str(self.workspace),
            env=environment,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log_file is not None:
            self.log_file.close()


def choose_unused_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(client: OpenCodeHttpClient, server: ManagedOpenCodeServer, timeout_seconds: int = 30) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        if server.process is not None and server.process.poll() is not None:
            raise OpenCodeError(f"OpenCode server exited early; inspect {server.workspace / 'opencode-server.log'}")
        try:
            return client.health()
        except OpenCodeError as exc:
            last_error = exc
            time.sleep(0.25)
    raise OpenCodeError(f"Timed out waiting for OpenCode health: {last_error}")
