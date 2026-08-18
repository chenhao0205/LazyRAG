from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from . import demo_runtime
from .contracts import (
    RepairAction,
    RepairCapabilityError,
    RepairContractError,
    RepairInput,
    RepairObservation,
    RepairTool,
)
from .dispatch import Capability
from .opencode import OpenCodeCapability, OpenCodeEventSink, terminate_process
from .testing import RepairTestCapability, RepairTestPlan
from .workspace import (
    WorkspacePaths,
    artifact_path,
    code_changes,
    create_code_checkpoint,
    diff_summary,
    rollback_code_checkpoint,
    safe_path,
    workspace_hash,
    write_json,
)


SearchProvider = Callable[[str], list[dict[str, str]]]
ReadProvider = Callable[[str], dict[str, str]]
MAX_COMMAND_OUTPUT_CHARS = 256 * 1024


class DefaultCapabilityFactory:
    """Basic production adapters; each can be replaced without changing RepairSession."""

    def __init__(
        self,
        code_settings: Mapping[str, str],
        *,
        search: SearchProvider | None = None,
        read: ReadProvider | None = None,
        test_plan: RepairTestPlan | None = None,
        code_timeout_seconds: float = 900,
        event_sink: OpenCodeEventSink | None = None,
    ) -> None:
        self.code_settings = dict(code_settings)
        self.search = search or search_web
        self.read = read or read_web
        self.test_plan = test_plan
        self.code_timeout_seconds = code_timeout_seconds
        self.event_sink = event_sink

    def __call__(
        self,
        repair_input: RepairInput,
        paths: WorkspacePaths,
    ) -> Mapping[RepairTool, Capability]:
        return {
            'workspace': WorkspaceCapability(repair_input, paths),
            'code': OpenCodeCapability(
                repair_input,
                paths,
                self.code_settings,
                timeout_seconds=self.code_timeout_seconds,
                event_sink=self.event_sink,
            ),
            'shell': ShellCapability(paths),
            'test': (
                RepairTestCapability(self.test_plan, paths)
                if self.test_plan is not None else _test_unavailable
            ),
            'research': ResearchCapability(paths, self.search, self.read),
        }


class WorkspaceCapability:
    def __init__(self, repair_input: RepairInput, paths: WorkspacePaths) -> None:
        self.repair_input = repair_input
        self.paths = paths

    def __call__(self, action: RepairAction) -> RepairObservation:
        operation = _choice(action.arguments.get('operation'), {'list', 'read', 'write', 'diff'})
        path = str(action.arguments.get('path') or 'source')
        area, relative = _location(path)
        target = safe_path(self.paths, area, relative)
        try:
            if operation == 'list':
                summary = json.dumps(sorted(item.relative_to(target).as_posix() for item in target.rglob('*'))[:500])
            elif operation == 'read':
                summary = target.read_text(encoding='utf-8')[:50_000]
            elif operation == 'write':
                if area != 'work' or not isinstance(action.arguments.get('content'), str):
                    raise RepairContractError('workspace_write_invalid', path)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(action.arguments['content'], encoding='utf-8')
                summary = f'wrote {path}'
            else:
                summary = diff_summary(self.repair_input.source_ref, self.paths.source)
        except (OSError, UnicodeError) as exc:
            raise RepairCapabilityError('workspace_failed', str(exc)) from exc
        return _observation(action, self.paths, 'success', summary)


class ShellCapability:
    def __init__(self, paths: WorkspacePaths) -> None:
        self.paths = paths

    def __call__(self, action: RepairAction) -> RepairObservation:
        command = action.arguments.get('command')
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
            raise RepairContractError('shell_command_invalid')
        cwd = _choice(action.arguments.get('cwd', 'work'), {'work'})
        timeout = action.arguments.get('timeout_seconds', 300)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 0 < timeout <= 7200:
            raise RepairContractError('shell_timeout_invalid')
        requested = list(command)
        normalized = _demo_command(command, self.paths)
        checkpoint = create_code_checkpoint(self.paths, action.call_id)
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        timed_out = False
        stdout = ''
        stderr = ''
        stdout_truncated = False
        stderr_truncated = False
        try:
            with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
                process = subprocess.Popen(
                    _demo_sandboxed(normalized, self.paths),
                    cwd=safe_path(self.paths, cwd),
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                    env=_command_env(self.paths),
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_process(process, grace_seconds=2.0)
                stdout, stdout_truncated = _read_output(stdout_file)
                stderr, stderr_truncated = _read_output(stderr_file)
        except OSError as exc:
            rollback_code_checkpoint(checkpoint)
            raise RepairCapabilityError('shell_failed', str(exc)) from exc
        changes = code_changes(checkpoint)
        if changes:
            rollback_code_checkpoint(checkpoint)
            raise RepairCapabilityError('shell_workspace_modified', ','.join(changes[:20]))
        output: dict[str, Any] | None = None
        if not timed_out and process is not None and process.returncode == 0:
            try:
                parsed = json.loads(stdout)
                output = dict(parsed) if isinstance(parsed, Mapping) else None
            except json.JSONDecodeError:
                output = None
        status = 'success' if output is not None else 'fail'
        return_code = None if process is None else process.returncode
        result = write_json(artifact_path(self.paths.logs, action.call_id), {
            'kind': 'demo',
            'command': requested,
            'return_code': return_code,
            'timed_out': timed_out,
            'duration_seconds': round(time.monotonic() - started, 3),
            'stdout': stdout,
            'stderr': stderr,
            'stdout_truncated': stdout_truncated,
            'stderr_truncated': stderr_truncated,
            'output': output,
        })
        summary = (
            json.dumps(output, ensure_ascii=False)
            if output is not None
            else 'demo_timeout'
            if timed_out
            else stderr or stdout or f'exit code {return_code}'
        )
        return _observation(action, self.paths, status, summary[-4000:], [str(result)])


class ResearchCapability:
    def __init__(self, paths: WorkspacePaths, search: SearchProvider, read: ReadProvider) -> None:
        self.paths = paths
        self.search = search
        self.read = read

    def __call__(self, action: RepairAction) -> RepairObservation:
        operation = _choice(action.arguments.get('operation'), {'search', 'read'})
        query = _required(action.arguments.get('query'), 'research query')
        try:
            if operation == 'search':
                result: dict[str, Any] = {'query': query, 'results': self.search(query)}
            else:
                urls = action.arguments.get('urls')
                if not isinstance(urls, list) or not 0 < len(urls) <= 3:
                    raise RepairContractError('research_urls_invalid')
                result = {'query': query, 'pages': [self.read(str(url)) for url in urls]}
        except (OSError, requests.RequestException) as exc:
            raise RepairCapabilityError('research_failed', str(exc)) from exc
        log = write_json(artifact_path(self.paths.logs, action.call_id), {'kind': 'research', **result})
        return _observation(action, self.paths, 'success', json.dumps(result)[:12_000], [str(log)])


def search_web(query: str) -> list[dict[str, str]]:
    response = requests.get(
        'https://html.duckduckgo.com/html/', params={'q': query}, timeout=20,
        headers={'User-Agent': 'LazyRAG-Repair/1.0'},
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    results = []
    for item in soup.select('.result')[:5]:
        link = item.select_one('.result__a')
        if link is not None and link.get('href'):
            results.append({
                'title': link.get_text(' ', strip=True),
                'url': str(link.get('href')),
                'snippet': (item.select_one('.result__snippet') or item).get_text(' ', strip=True)[:500],
            })
    return results


def read_web(url: str) -> dict[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise RepairContractError('research_url_invalid', url)
    response = requests.get(url, timeout=20, headers={'User-Agent': 'LazyRAG-Repair/1.0'})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    return {
        'url': url,
        'title': soup.title.get_text(' ', strip=True) if soup.title else '',
        'excerpt': soup.get_text(' ', strip=True)[:8000],
    }


def _observation(
    action: RepairAction,
    paths: WorkspacePaths,
    status: str,
    summary: str,
    references: list[str] | None = None,
) -> RepairObservation:
    return RepairObservation(
        action.call_id,
        status,  # type: ignore[arg-type]
        summary or 'completed',
        references or [],
        workspace_hash(paths.source),
    )


def _location(value: str) -> tuple[str, str]:
    path = PurePosixPath(value.replace('\\', '/'))
    if path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
        raise RepairContractError('workspace_path_invalid', value)
    area, *remaining = path.parts
    if area not in {'source', 'work', 'context'}:
        raise RepairContractError('workspace_area_invalid', area)
    return area, PurePosixPath(*remaining).as_posix() if remaining else ''


def _demo_command(command: list[str], paths: WorkspacePaths) -> list[str]:
    if len(command) != 4 or Path(command[0]).name not in {'python', 'python3', Path(sys.executable).name}:
        raise RepairContractError('demo_command_invalid')
    if command[1].replace('\\', '/') != 'demo/run_demo.py' or command[2] != '--input':
        raise RepairContractError('demo_command_invalid')
    script = safe_path(paths, 'work', 'demo/run_demo.py')
    input_path = safe_path(paths, 'work', command[3])
    if not script.is_file() or not input_path.is_file() or input_path.suffix != '.json':
        raise RepairContractError('demo_command_invalid')
    return [
        sys.executable, '-I', str(Path(demo_runtime.__file__).resolve()),
        str(script), str(input_path), str(paths.source), str(MAX_COMMAND_OUTPUT_CHARS),
    ]


def _read_output(stream: Any) -> tuple[str, bool]:
    stream.seek(0)
    value = stream.read(MAX_COMMAND_OUTPUT_CHARS + 1)
    truncated = len(value) >= MAX_COMMAND_OUTPUT_CHARS
    return value[:MAX_COMMAND_OUTPUT_CHARS].decode('utf-8', errors='replace'), truncated


def _command_env(paths: WorkspacePaths) -> dict[str, str]:
    allowed = (
        'PATH', 'LANG', 'LC_ALL', 'TZ',
        'SSL_CERT_FILE', 'SSL_CERT_DIR',
    )
    environment = {key: value for key in allowed if (value := os.environ.get(key))}
    environment.update({
        'HOME': str(paths.work),
        'TMPDIR': str(paths.work),
        'PYTHONPATH': str(paths.source),
        'PYTHONDONTWRITEBYTECODE': '1',
        'REPAIR_SOURCE': str(paths.source),
        'REPAIR_WORK': str(paths.work),
    })
    return environment


def _demo_sandboxed(command: list[str], paths: WorkspacePaths) -> list[str]:
    enabled = os.getenv('LAZYRAG_REPAIR_SANDBOX_EXEC') == '1'
    binary = shutil.which('sandbox-exec') if enabled and sys.platform == 'darwin' else None
    if not binary:
        return command
    profile = (
        '(version 1) (deny default) (allow process*) (allow sysctl-read) (allow mach-lookup) '
        '(allow file-read*) '
        f'(deny file-read* (subpath "{paths.control}")) '
        f'(allow file-write* (subpath "{paths.sandbox}"))'
    )
    return [binary, '-p', profile, *command]


def _choice(value: object, choices: set[str]) -> str:
    result = str(value or '')
    if result not in choices:
        raise RepairContractError('choice_invalid', result)
    return result


def _required(value: object, name: str) -> str:
    result = str(value or '').strip()
    if not result:
        raise RepairContractError('field_required', name)
    return result


def _test_unavailable(action: RepairAction) -> RepairObservation:
    raise RepairCapabilityError('test_plan_unavailable', str(action.arguments.get('level') or ''))


__all__ = [
    'DefaultCapabilityFactory', 'ReadProvider', 'ResearchCapability',
    'SearchProvider', 'ShellCapability', 'WorkspaceCapability',
    'read_web', 'search_web',
]
