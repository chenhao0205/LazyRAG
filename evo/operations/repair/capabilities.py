from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .contracts import (
    RepairAction,
    RepairCapabilityError,
    RepairContractError,
    RepairInput,
    RepairObservation,
    RepairTool,
)
from .dispatch import Capability
from .workspace import WorkspacePaths, artifact_path, diff_summary, safe_path, workspace_hash, write_json


SearchProvider = Callable[[str], list[dict[str, str]]]
ReadProvider = Callable[[str], dict[str, str]]


class DefaultCapabilityFactory:
    """Basic production adapters; each can be replaced without changing RepairSession."""

    def __init__(
        self,
        search: SearchProvider | None = None,
        read: ReadProvider | None = None,
    ) -> None:
        self.search = search or search_web
        self.read = read or read_web

    def __call__(
        self,
        repair_input: RepairInput,
        paths: WorkspacePaths,
    ) -> Mapping[RepairTool, Capability]:
        return {
            'workspace': WorkspaceCapability(repair_input, paths),
            'shell': ShellCapability(paths),
            'test': TestCapability(repair_input, paths),
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
                if area not in {'source', 'work'} or not isinstance(action.arguments.get('content'), str):
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
        cwd = _choice(action.arguments.get('cwd', 'work'), {'source', 'work'})
        timeout = action.arguments.get('timeout_seconds', 300)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or not 0 < timeout <= 7200:
            raise RepairContractError('shell_timeout_invalid')
        try:
            completed = subprocess.run(
                _sandboxed(command, self.paths),
                cwd=safe_path(self.paths, cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env={**os.environ, 'REPAIR_SOURCE': str(self.paths.source), 'REPAIR_WORK': str(self.paths.work)},
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RepairCapabilityError('shell_failed', str(exc)) from exc
        status = 'success' if completed.returncode == 0 else 'fail'
        result = write_json(artifact_path(self.paths.logs, action.call_id), {
            'kind': 'shell', 'command': command, 'return_code': completed.returncode,
            'stdout': completed.stdout[-50_000:], 'stderr': completed.stderr[-50_000:],
        })
        summary = completed.stdout or completed.stderr or f'exit code {completed.returncode}'
        return _observation(action, self.paths, status, summary[-4000:], [str(result)])


class TestCapability:
    def __init__(self, repair_input: RepairInput, paths: WorkspacePaths) -> None:
        self.repair_input = repair_input
        self.paths = paths

    def __call__(self, action: RepairAction) -> RepairObservation:
        level = _choice(action.arguments.get('level'), {'L0', 'L1', 'L2'})
        configured = self.repair_input.constraints.get('test_commands')
        commands = configured.get(level) if isinstance(configured, Mapping) else None
        normalized = _commands(commands)
        outputs = []
        return_code = 0
        for command in normalized:
            try:
                completed = subprocess.run(
                    _sandboxed(command, self.paths), cwd=self.paths.source, capture_output=True,
                    text=True, timeout=7200, check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise RepairCapabilityError('test_failed', str(exc)) from exc
            outputs.append({'command': command, 'stdout': completed.stdout, 'stderr': completed.stderr})
            if completed.returncode:
                return_code = completed.returncode
                break
        current_hash = workspace_hash(self.paths.source)
        status = 'success' if return_code == 0 else 'fail'
        evidence = write_json(artifact_path(self.paths.evidence, action.call_id), {
            'kind': 'test', 'call_id': action.call_id, 'level': level, 'status': status,
            'workspace_hash': current_hash, 'return_code': return_code, 'outputs': outputs,
        })
        return RepairObservation(
            action.call_id, status, f'{level} {status}', [str(evidence)], current_hash,
        )


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


def _commands(value: object) -> list[list[str]]:
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return [value]
    if (
        isinstance(value, list) and value
        and all(isinstance(command, list) and command and all(isinstance(item, str) for item in command)
                for command in value)
    ):
        return value
    raise RepairContractError('test_commands_invalid')


def _sandboxed(command: list[str], paths: WorkspacePaths) -> list[str]:
    enabled = os.getenv('LAZYRAG_REPAIR_SANDBOX_EXEC') == '1'
    binary = shutil.which('sandbox-exec') if enabled and sys.platform == 'darwin' else None
    if not binary:
        return command
    profile = (
        '(version 1) (deny default) (allow process*) (allow sysctl-read) (allow mach-lookup) '
        '(allow network*) (allow file-read*) '
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


__all__ = [
    'DefaultCapabilityFactory', 'ReadProvider', 'ResearchCapability', 'SearchProvider',
    'ShellCapability', 'TestCapability', 'WorkspaceCapability', 'read_web', 'search_web',
]
