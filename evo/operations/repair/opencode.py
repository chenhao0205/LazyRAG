from __future__ import annotations

import json
import os
import select
import signal
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from evo.llm import parse_json_object

from .contracts import RepairAction, RepairCapabilityError, RepairInput, RepairObservation
from .workspace import (
    WorkspacePaths,
    artifact_path,
    code_changes,
    create_code_checkpoint,
    patch_size,
    path_in_scope,
    rollback_code_checkpoint,
    safe_path,
    workspace_hash,
    write_json,
)


OpenCodeEventSink = Callable[[str, str, str, Mapping[str, object]], None]
MAX_LOG_BYTES = 2 * 1024 * 1024
TAIL_CHARS = 4_000
_PROXY_KEYS = (
    'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY',
    'http_proxy', 'https_proxy', 'no_proxy',
)
_SAFE_ENV_KEYS = (
    'PATH', 'SHELL', 'USER', 'LANG', 'LC_ALL', 'TZ',
    'SSL_CERT_FILE', 'SSL_CERT_DIR', 'REQUESTS_CA_BUNDLE',
    *_PROXY_KEYS,
)


@dataclass(frozen=True, slots=True)
class OpenCodeRunResult:
    returncode: int
    session_id: str
    last_error: dict[str, Any] | None
    finish_reason: str
    stdout_path: Path
    events_path: Path
    stdout_truncated: bool
    events_truncated: bool


class _BoundedLog:
    def __init__(self, stream: TextIO, limit: int = MAX_LOG_BYTES) -> None:
        self.stream = stream
        self.limit = limit
        self.written = 0
        self.truncated = False

    def write(self, text: str) -> None:
        encoded = text.encode('utf-8', errors='replace')
        remaining = self.limit - self.written
        if remaining <= 0:
            self.truncated = True
            return
        retained = encoded[:remaining]
        self.stream.write(retained.decode('utf-8', errors='ignore'))
        self.stream.flush()
        self.written += len(retained)
        self.truncated = self.truncated or len(retained) < len(encoded)


class OpenCodeRunner:
    """Invocation-local adapter for the OpenCode CLI JSON event stream."""

    def __init__(
        self,
        settings: Mapping[str, str],
        runtime_root: Path,
        *,
        timeout_seconds: float = 900,
        event_sink: OpenCodeEventSink | None = None,
    ) -> None:
        self.settings = _settings(settings)
        self.runtime_root = runtime_root.resolve()
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.event_sink = event_sink
        self.session_id = ''
        self._recovered_session = False

    def run(
        self,
        *,
        workdir: Path,
        prompt: str,
        call_id: str,
        permission: Mapping[str, object],
        timeout_seconds: float | None = None,
    ) -> OpenCodeRunResult:
        call_root = self.runtime_root / 'calls' / _safe_name(call_id)
        result = self._run_once(
            workdir.resolve(), prompt, call_root, permission,
            self.session_id, timeout_seconds,
        )
        if _invalid_session(result) and self.session_id and not self._recovered_session:
            self._recovered_session = True
            result = self._run_once(
                workdir.resolve(), prompt, call_root / 'fresh-session', permission,
                '', timeout_seconds,
            )
        self.session_id = result.session_id or self.session_id
        return result

    def _run_once(
        self,
        workdir: Path,
        prompt: str,
        call_root: Path,
        permission: Mapping[str, object],
        session_id: str,
        timeout_seconds: float | None,
    ) -> OpenCodeRunResult:
        call_root.mkdir(parents=True, exist_ok=True)
        stdout_path = call_root / 'stdout.log'
        events_path = call_root / 'events.jsonl'
        timeout = min(self.timeout_seconds, timeout_seconds) if timeout_seconds is not None else self.timeout_seconds
        secrets = [self.settings.get('api_key', '')]
        secrets = [secret for secret in secrets if secret]
        environment = _process_env(
            self.settings,
            self.runtime_root / 'session',
            permission,
        )
        command = _command(self.settings, prompt, session_id)
        started = time.monotonic()
        process: subprocess.Popen[str] | None = None
        last_error: dict[str, Any] | None = None
        finish_reason = ''
        resolved_session = session_id
        tail = ''

        with stdout_path.open('w', encoding='utf-8') as stdout_stream, events_path.open(
            'w', encoding='utf-8',
        ) as events_stream:
            stdout_log = _BoundedLog(stdout_stream)
            events_log = _BoundedLog(events_stream)

            def record(raw: Mapping[str, Any]) -> dict[str, Any]:
                clean = _redact(dict(raw), secrets)
                events_log.write(json.dumps(clean, ensure_ascii=False, default=str) + '\n')
                self._emit(clean)
                return clean

            try:
                process = subprocess.Popen(
                    command,
                    cwd=workdir,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    start_new_session=True,
                )
                record({'type': 'process_start', 'status': 'running', 'pid': process.pid})
                while process.poll() is None:
                    if time.monotonic() - started > timeout:
                        last_error = record({
                            'type': 'timeout',
                            'status': 'failed',
                            'message': f'OpenCode timed out after {timeout:g}s',
                        })
                        terminate_process(process)
                        break
                    ready, _, _ = select.select([process.stdout], [], [], 0.05) if process.stdout else ([], [], [])
                    if ready:
                        resolved_session, last_error, finish_reason, tail = _consume_line(
                            ready[0].readline(), stdout_log, record,
                            resolved_session, last_error, finish_reason, tail, secrets,
                        )
                if process.stdout:
                    for line in process.stdout:
                        resolved_session, last_error, finish_reason, tail = _consume_line(
                            line, stdout_log, record,
                            resolved_session, last_error, finish_reason, tail, secrets,
                        )
                returncode = process.wait()
                record({
                    'type': 'process_exit',
                    'status': 'completed' if returncode == 0 else 'failed',
                    'returncode': returncode,
                })
                if returncode and last_error is None:
                    last_error = record({
                        'type': 'process_failed',
                        'status': 'failed',
                        'message': tail,
                    })
            except OSError as exc:
                returncode = 1
                last_error = record({
                    'type': 'process_start_failed',
                    'status': 'failed',
                    'message': str(exc),
                })
            finally:
                if process is not None and process.poll() is None:
                    terminate_process(process)

        return OpenCodeRunResult(
            returncode=returncode,
            session_id=resolved_session,
            last_error=last_error,
            finish_reason=finish_reason,
            stdout_path=stdout_path,
            events_path=events_path,
            stdout_truncated=stdout_log.truncated,
            events_truncated=events_log.truncated,
        )

    def _emit(self, event: Mapping[str, Any]) -> None:
        if self.event_sink is None:
            return
        raw_type = str(event.get('type') or 'message')
        tool = _event_tool(event)
        event_type = (
            'opencode.tool_use.search'
            if tool in {'grep', 'glob', 'list', 'read'}
            else 'opencode.tool_use.edit_file'
            if tool in {'edit', 'write', 'patch'}
            else 'opencode.error'
            if raw_type in {'error', 'timeout', 'process_failed', 'process_start_failed'}
            else 'opencode.process'
            if raw_type in {'process_start', 'process_exit'}
            else 'opencode.message'
        )
        raw_status = str(event.get('status') or '')
        status = (
            'failed'
            if event_type == 'opencode.error' or raw_status in {'error', 'failed'}
            else 'completed'
            if raw_status in {'completed', 'done', 'success', 'succeeded'}
            else 'started'
            if raw_status in {'started', 'starting'}
            else 'running'
        )
        self.event_sink(
            event_type,
            status,
            _event_message(event)[:500],
            {'tool': tool, 'event_type': raw_type},
        )


class OpenCodeCapability:
    """OpenCode adapter behind the generic Repair `code` port."""

    def __init__(
        self,
        repair_input: RepairInput,
        paths: WorkspacePaths,
        settings: Mapping[str, str],
        *,
        timeout_seconds: float,
        event_sink: OpenCodeEventSink | None,
    ) -> None:
        self.repair_input = repair_input
        self.paths = paths
        raw_patch_limit = repair_input.constraints.get('max_patch_bytes', 65536)
        self.max_patch_bytes = (
            raw_patch_limit
            if isinstance(raw_patch_limit, int) and not isinstance(raw_patch_limit, bool) and raw_patch_limit > 0
            else 65536
        )
        self.runner = OpenCodeRunner(
            settings,
            paths.logs / 'opencode',
            timeout_seconds=timeout_seconds,
            event_sink=event_sink,
        )

    def __call__(self, action: RepairAction) -> RepairObservation:
        operation = str(action.arguments.get('operation') or '')
        instruction = str(action.arguments.get('instruction') or '').strip()
        if operation not in {'inspect', 'edit_work', 'edit_source'} or not instruction:
            raise RepairCapabilityError('code_request_invalid', operation)

        checkpoint = create_code_checkpoint(self.paths, action.call_id)
        report_path = artifact_path(self.paths.work / '.opencode/reports', action.call_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.unlink(missing_ok=True)
        task_path = self.paths.context / 'code_task.json'
        if task_path.exists():
            task_path.chmod(0o600)
        task = _task(operation, instruction, report_path.relative_to(self.paths.sandbox))
        write_json(task_path, task)
        task_path.chmod(0o400)
        result = self.runner.run(
            workdir=self.paths.sandbox,
            prompt=json.dumps(task, ensure_ascii=False, indent=2),
            call_id=action.call_id,
            permission=_permission(operation, self.paths, report_path, self.repair_input.case_scope),
        )
        report_leaked_secret = _redact_report(report_path, self.runner.settings.get('api_key', ''))
        report_relative = report_path.relative_to(self.paths.sandbox).as_posix()
        changes = [path for path in code_changes(checkpoint) if path != report_relative]
        report = (
            inspect_report(report_path)
            if operation == 'inspect'
            else edit_report(report_path, operation)
        )
        if operation == 'inspect':
            invalid_changes = changes
        elif operation == 'edit_work':
            invalid_changes = [path for path in changes if not path.startswith('work/demo/')]
        else:
            invalid_changes = [
                path for path in changes
                if not path.startswith('source/')
                or not path_in_scope(path.removeprefix('source/'), self.repair_input.case_scope)
            ]
        report_mismatch = operation != 'inspect' and sorted(report.get('changed_files') or ()) != changes
        oversized = operation == 'edit_source' and patch_size(
            self.repair_input.source_ref, self.paths.source,
        ) > self.max_patch_bytes
        failure = (
            str((result.last_error or {}).get('type') or '')
            or (f'opencode_exit_{result.returncode}' if result.returncode else '')
            or ('code_report_contains_secret' if report_leaked_secret else '')
            or (str(report.get('reason') or 'code_report_invalid') if report.get('status') != 'completed' else '')
            or ('code_scope_violation' if invalid_changes else '')
            or ('code_report_diff_mismatch' if report_mismatch else '')
            or ('code_patch_too_large' if oversized else '')
        )
        references = [str(result.stdout_path), str(result.events_path)]
        if report_path.is_file():
            persisted_report = result.stdout_path.parent / 'report.json'
            shutil.copy2(report_path, persisted_report)
            references.append(str(persisted_report))
        report_path.unlink(missing_ok=True)
        if failure:
            rollback_code_checkpoint(checkpoint)
            return self._observation(action, 'error', failure, references)
        summary = json.dumps({'operation': operation, **report}, ensure_ascii=False)[:12_000]
        return self._observation(action, 'success', summary, references)

    def _observation(
        self,
        action: RepairAction,
        status: str,
        summary: str,
        references: list[str],
    ) -> RepairObservation:
        return RepairObservation(
            action.call_id,
            status,  # type: ignore[arg-type]
            summary,
            references,
            workspace_hash(self.paths.source),
        )


def inspect_report(path: Path) -> dict[str, Any]:
    try:
        value = parse_json_object(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'status': 'invalid', 'reason': type(exc).__name__, 'findings': [], 'open_questions': []}
    if not isinstance(value, Mapping) or value.get('status') != 'completed':
        return {'status': 'invalid', 'reason': 'inspect_status_invalid', 'findings': [], 'open_questions': []}
    raw_findings = value.get('findings')
    findings = [
        {
            'path': _report_path(item.get('path'), 'source'),
            'symbol': str(item.get('symbol') or '').strip(),
            'observation': str(item.get('observation') or '').strip()[:1600],
        }
        for item in (raw_findings if isinstance(raw_findings, list) else ())
        if isinstance(item, Mapping)
        and _report_path(item.get('path'), 'source')
        and str(item.get('observation') or '').strip()
    ]
    raw_questions = value.get('open_questions')
    questions = [
        str(item).strip()[:500]
        for item in (raw_questions if isinstance(raw_questions, list) else ())
        if str(item).strip()
    ]
    return {
        'status': 'completed' if findings else 'invalid',
        'reason': '' if findings else 'inspect_findings_missing',
        'findings': findings[:20],
        'open_questions': questions[:20],
    }


def edit_report(path: Path, operation: str) -> dict[str, Any]:
    try:
        value = parse_json_object(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'status': 'invalid', 'reason': type(exc).__name__, 'changed_files': []}
    if not isinstance(value, Mapping):
        return {'status': 'invalid', 'reason': 'report_not_object', 'changed_files': []}
    root = 'work/demo' if operation == 'edit_work' else 'source' if operation == 'edit_source' else ''
    if not root:
        return {'status': 'invalid', 'reason': 'edit_operation_invalid', 'changed_files': []}
    raw_changed = value.get('changed_files')
    changed = sorted({
        candidate
        for item in (raw_changed if isinstance(raw_changed, list) else ())
        if (candidate := _report_path(item, root))
    })
    entrypoint = str(value.get('entrypoint') or '').strip()
    valid = value.get('status') == 'completed' and bool(changed) and (
        operation == 'edit_source' or entrypoint == 'work/demo/run_demo.py'
    )
    result = {
        'status': 'completed' if valid else 'invalid',
        'reason': '' if valid else f'{operation}_report_invalid',
        'changed_files': changed,
        'change_intent': str(value.get('change_intent') or '').strip()[:1000],
    }
    if operation == 'edit_work':
        result['entrypoint'] = entrypoint
    return result


def _task(operation: str, instruction: str, report_path: Path) -> dict[str, Any]:
    common = {
        'mode': 'lazyrag_repair_code_capability',
        'operation': operation,
        'instruction': instruction,
        'repair_view_path': 'context/repair_view.json',
        'report_path': report_path.as_posix(),
        'constraints': [
            'Read context/repair_view.json before acting.',
            'Never use bash, web, skills, MCP tools, or subagents.',
            'Never access paths outside this workspace.',
            'Write one strict JSON object to report_path.',
        ],
    }
    if operation == 'inspect':
        common['constraints'].extend([
            'Search and read source/ without modifying it; use work/ only as supporting context.',
            'Every finding path must identify a file under source/.',
            'Report concise causal findings; do not paste large source excerpts.',
        ])
        common['report_schema'] = {
            'status': 'completed',
            'findings': [{'path': 'source/...', 'symbol': '...', 'observation': '...'}],
            'open_questions': ['...'],
        }
    elif operation == 'edit_work':
        common['constraints'].extend([
            'Create or edit files only under work/demo/.',
            'The fixed entrypoint is work/demo/run_demo.py.',
            'The entrypoint accepts --input <json-path> and prints exactly one JSON object.',
            'Do not run the Demo and do not modify source/.',
        ])
        common['report_schema'] = {
            'status': 'completed',
            'entrypoint': 'work/demo/run_demo.py',
            'changed_files': ['work/demo/run_demo.py'],
            'change_intent': '...',
        }
    else:
        common['constraints'].extend([
            'Modify only the smallest source files required by the instruction.',
            'Never modify work/, context/, control/, tests, dependencies, or generated files.',
            'Do not run tests or commands; the host owns all validation.',
        ])
        common['report_schema'] = {
            'status': 'completed',
            'changed_files': ['source/...'],
            'change_intent': 'minimal causal change',
        }
    return common


def _report_path(value: object, root: str) -> str:
    text = str(value or '').strip().replace('\\', '/')
    path = PurePosixPath(text)
    root_parts = PurePosixPath(root).parts
    if (
        not text or path.is_absolute() or path.parts[:len(root_parts)] != root_parts
        or len(path.parts) <= len(root_parts)
        or any(part in {'', '.', '..'} for part in path.parts)
    ):
        return ''
    return path.as_posix()


def _redact_report(path: Path, secret: str) -> bool:
    if not secret or not path.is_file():
        return False
    try:
        value = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return False
    if secret not in value:
        return False
    path.write_text(value.replace(secret, '<redacted>'), encoding='utf-8')
    return True


def _permission(
    operation: str,
    paths: WorkspacePaths,
    report_path: Path,
    case_scope: str = '',
) -> dict[str, object]:
    edit: dict[str, str] = {'*': 'deny'}
    if operation == 'edit_work':
        edit[f'{paths.work.as_posix().lstrip("/")}/demo/**'] = 'allow'
    elif operation == 'edit_source':
        for value in case_scope.splitlines():
            relative = value.strip()
            if not relative:
                continue
            target = safe_path(paths, 'source', relative)
            pattern = target.as_posix().lstrip('/')
            edit[f'{pattern}/**' if target.is_dir() else pattern] = 'allow'
    edit[report_path.as_posix().lstrip('/')] = 'allow'
    return {
        '*': 'deny',
        'read': {'*': 'allow', '*.env': 'deny', '*.env.*': 'deny', '*.env.example': 'allow'},
        'grep': 'allow',
        'glob': 'allow',
        'edit': edit,
        'bash': 'deny',
        'task': 'deny',
        'skill': 'deny',
        'webfetch': 'deny',
        'websearch': 'deny',
        'external_directory': 'deny',
        'doom_loop': 'deny',
    }


def _settings(raw: Mapping[str, str]) -> dict[str, str]:
    allowed = {'binary', 'model', 'provider', 'provider_model', 'npm', 'base_url', 'api_key', 'skip_auth'}
    result = {key: str(value).strip() for key, value in raw.items() if key in allowed and str(value).strip()}
    result.setdefault('binary', os.getenv('LAZYMIND_EVO_CODE_BINARY') or 'opencode')
    required = ('model', 'provider', 'provider_model', 'npm', 'base_url')
    missing = [key for key in required if not result.get(key)]
    if not result.get('api_key') and result.get('skip_auth') != 'true':
        missing.append('api_key')
    if missing:
        raise ValueError(f'opencode settings missing: {", ".join(missing)}')
    return result


def _process_env(
    settings: Mapping[str, str],
    runtime_root: Path,
    permission: Mapping[str, object],
) -> dict[str, str]:
    directories = {
        'HOME': runtime_root / 'home',
        'TMPDIR': runtime_root / 'tmp',
        'XDG_DATA_HOME': runtime_root / 'data',
        'XDG_CONFIG_HOME': runtime_root / 'config',
        'XDG_CACHE_HOME': runtime_root / 'cache',
        'XDG_STATE_HOME': runtime_root / 'state',
        'OPENCODE_CONFIG_DIR': runtime_root / 'config-dir',
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    provider = settings['provider']
    model = settings['provider_model']
    options: dict[str, str] = {'baseURL': settings['base_url']}
    if settings.get('api_key'):
        options['apiKey'] = '{env:LAZYMIND_OPENCODE_API_KEY}'
    config = {
        '$schema': 'https://opencode.ai/config.json',
        'model': settings['model'],
        'autoupdate': False,
        'share': 'disabled',
        'snapshot': False,
        'plugin': [],
        'mcp': {},
        'enabled_providers': [provider],
        'permission': dict(permission),
        'provider': {
            provider: {
                'npm': settings['npm'],
                'options': options,
                'models': {model: {'name': model}},
            },
        },
    }
    environment = {key: value for key in _SAFE_ENV_KEYS if (value := os.environ.get(key))}
    environment.update({key: str(path) for key, path in directories.items()})
    environment['OPENCODE_CONFIG_CONTENT'] = json.dumps(config, ensure_ascii=False, separators=(',', ':'))
    if settings.get('api_key'):
        environment['LAZYMIND_OPENCODE_API_KEY'] = settings['api_key']
    return environment


def _command(settings: Mapping[str, str], prompt: str, session_id: str) -> list[str]:
    command = [settings['binary'], 'run', '--format', 'json', '--pure', '--model', settings['model']]
    if session_id:
        command += ['--session', session_id]
    return [*command, prompt]


def _consume_line(
    line: str,
    stdout_log: _BoundedLog,
    record: Callable[[Mapping[str, Any]], dict[str, Any]],
    session_id: str,
    last_error: dict[str, Any] | None,
    finish_reason: str,
    tail: str,
    secrets: list[str],
) -> tuple[str, dict[str, Any] | None, str, str]:
    if not line:
        return session_id, last_error, finish_reason, tail
    clean_line = _redact(line, secrets)
    stdout_log.write(clean_line)
    tail = (tail + clean_line)[-TAIL_CHARS:]
    try:
        event = json.loads(clean_line)
    except json.JSONDecodeError:
        if clean_line.strip():
            record({'type': 'stdout', 'status': 'running', 'message': clean_line.strip()[:500]})
        return session_id, last_error, finish_reason, tail
    if not isinstance(event, Mapping):
        return session_id, last_error, finish_reason, tail
    recorded = record(event)
    part = event.get('part') if isinstance(event.get('part'), Mapping) else {}
    if event.get('type') == 'step_finish':
        finish_reason = str(part.get('reason') or event.get('reason') or '').strip()
    session_id = session_id or str(event.get('sessionID') or event.get('session_id') or '').strip()
    if event.get('type') == 'error':
        last_error = recorded
    return session_id, last_error, finish_reason, tail


def _event_tool(event: Mapping[str, Any]) -> str:
    part = event.get('part') if isinstance(event.get('part'), Mapping) else {}
    call = event.get('call') if isinstance(event.get('call'), Mapping) else {}
    return str(event.get('tool') or part.get('tool') or call.get('tool') or '')


def _event_message(event: Mapping[str, Any]) -> str:
    part = event.get('part') if isinstance(event.get('part'), Mapping) else {}
    state = part.get('state') if isinstance(part.get('state'), Mapping) else {}
    return str(
        part.get('text') or event.get('text') or event.get('message')
        or event.get('error') or state.get('error') or part.get('title') or ''
    ).strip()


def _invalid_session(result: OpenCodeRunResult) -> bool:
    if result.last_error is None:
        return False
    text = json.dumps(result.last_error, ensure_ascii=False, default=str).casefold()
    return 'session' in text and any(token in text for token in ('invalid', 'not found', 'unknown', 'expired'))


def terminate_process(process: subprocess.Popen[str], grace_seconds: float = 5.0) -> None:
    if process.poll() is not None:
        return
    for signal_value in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), signal_value)
        except ProcessLookupError:
            return
        except OSError:
            try:
                process.send_signal(signal_value)
            except ProcessLookupError:
                return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def _redact(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        result = value
        for secret in secrets:
            result = result.replace(secret, '<redacted>')
        return result
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    for secret in secrets:
        serialized = serialized.replace(secret, '<redacted>')
    return json.loads(serialized)


def _safe_name(value: str) -> str:
    return sha256(str(value).encode('utf-8')).hexdigest()[:24]


__all__ = [
    'OpenCodeCapability', 'OpenCodeEventSink', 'OpenCodeRunResult', 'OpenCodeRunner',
    'edit_report', 'inspect_report', 'terminate_process',
]
