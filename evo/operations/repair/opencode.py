from __future__ import annotations

import json
import os
import select
import signal
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from shutil import copy2
from typing import Any, NamedTuple, TextIO

from evo.artifact_runtime import record_event
from evo.llm import parse_json_object

PERMISSIONS = {
    'read': {'*': 'deny', 'source/**': 'allow'},
    'glob': {'*': 'deny', 'source/**': 'allow'},
    'grep': {'*': 'deny', 'source/**': 'allow'},
    'edit': {'*': 'deny', 'work/**': 'allow'},
    **dict.fromkeys(('bash', 'task', 'webfetch', 'websearch', 'external_directory'), 'deny'),
}
OPENCODE_FIELDS = {
    'model',
    'provider',
    'provider_model',
    'npm',
    'base_url',
    'api_key',
    'skip_auth',
}
TRACE_BY_TOOL = {
    'glob': 'opencode.tool_use.search',
    'grep': 'opencode.tool_use.search',
    'list': 'opencode.tool_use.search',
    'read': 'opencode.tool_use.read_file',
    'edit': 'opencode.tool_use.edit_file',
    'write': 'opencode.tool_use.edit_file',
    'bash': 'opencode.tool_use.run_command',
}
TRACE_BY_TYPE = {
    'setup': 'opencode.setup',
    'process_start': 'opencode.process_start',
    'process_exit': 'opencode.process_exit',
    'error': 'opencode.error',
    'timeout': 'opencode.error',
    'process_failed': 'opencode.error',
    'configuration_error': 'opencode.error',
    'prompt_write_failed': 'opencode.error',
    'process_start_failed': 'opencode.error',
}
PATH_KEYS = {'file', 'path', 'filepath', 'filePath'}


class OpenCodeRunResult(NamedTuple):
    returncode: int
    session_id: str
    last_error: dict[str, Any] | None
    finish_reason: str


class OpenCodeSession:
    """Keep all code research for one target in one resumable OpenCode session."""

    def __init__(self, *, category_id: str, workdir: Path, artifact_root: Path,
                 config: dict[str, str], timeout_s: float = 900, session_id: str = '',
                 calls: int = 0) -> None:
        self.category_id = category_id
        self.workdir = workdir
        self.artifact_root = artifact_root
        self.config = config
        self.timeout_s = timeout_s
        self.session_id = str(session_id)
        self.calls = max(0, int(calls))
        self.recovered = False

    def run(self, instruction: str, expected_result: str = '',
            timeout_s: float | None = None) -> dict[str, Any]:
        if not str(instruction).strip():
            raise ValueError('opencode_instruction_missing')
        self.calls += 1
        call_dir = self.artifact_root / 'opencode' / 'calls' / f'call-{self.calls:03d}'
        call_dirs = [call_dir]
        report_path = self.workdir / 'work' / '.opencode' / 'reports' / f'call-{self.calls:03d}.json'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        persisted_report = self.artifact_root / 'opencode' / 'reports' / report_path.name
        report_path.unlink(missing_ok=True)
        before = _workspace_snapshot(self.workdir)
        call_timeout = min(self.timeout_s, timeout_s) if timeout_s is not None else self.timeout_s
        run = run_opencode_streaming(
            workdir=str(self.workdir),
            prompt=json.dumps(_phase1_task_card(
                instruction, expected_result, self.category_id,
                report_path.relative_to(self.workdir),
            ), ensure_ascii=False, indent=2),
            artifact_dir=call_dir,
            session_id=self.session_id,
            config=self.config,
            timeout_s=max(0.1, call_timeout),
            attempt=self.calls,
        )
        if _invalid_session(run) and not self.recovered:
            self.recovered = True
            self.session_id = ''
            recovery_dir = call_dir / 'recovery-001'
            call_dirs.append(recovery_dir)
            run = run_opencode_streaming(
                workdir=str(self.workdir),
                prompt=json.dumps(_phase1_task_card(
                    instruction, expected_result, self.category_id,
                    report_path.relative_to(self.workdir),
                ), ensure_ascii=False, indent=2),
                artifact_dir=recovery_dir,
                config=self.config,
                timeout_s=max(0.1, call_timeout),
                attempt=self.calls,
            )
        self.session_id = run.session_id or self.session_id
        after = _workspace_snapshot(self.workdir)
        changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
        invalid = [path for path in changed if not path.startswith('work/')]
        report = read_opencode_report(report_path, 'phase1')
        if report_path.is_file():
            persisted_report.parent.mkdir(parents=True, exist_ok=True)
            copy2(report_path, persisted_report)
        failure = _run_failure(run)
        reported = sorted(report.get('changed_files') or ())
        mismatch = reported != changed
        reason = (
            failure or
            ('opencode_scope_violation' if invalid else '') or
            (str(report.get('reason') or 'opencode_report_invalid') if report.get('status') != 'completed' else '') or
            ('opencode_report_diff_mismatch' if mismatch else '')
        )
        return {
            'status': 'failed' if reason else 'completed',
            'reason': reason,
            'session_id': self.session_id,
            'report': report,
            'changed_files': changed,
            'invalid_changes': invalid,
            'artifacts': _call_artifacts(call_dirs, persisted_report, self.artifact_root),
        }


class _OpenCodeLogs:
    def __init__(self, stdout: TextIO, events: TextIO, secrets: list[str], attempt: int | None) -> None:
        self.stdout_stream = stdout
        self.events_stream = events
        self.secrets = secrets
        self.attempt = attempt
        self.tail = ''

    def record(self, event: dict[str, Any]) -> dict[str, Any]:
        clean = _clean(event, self.secrets)
        self.events_stream.write(json.dumps(clean, ensure_ascii=False) + '\n')
        self.events_stream.flush()
        _record_opencode_event(self.attempt, clean)
        return clean

    def write_stdout(self, line: str) -> None:
        clean = _clean(line, self.secrets)
        self.stdout_stream.write(clean)
        self.stdout_stream.flush()
        self.tail = (self.tail + clean)[-1000:]

    def failure(self, session_id: str, kind: str, message: object) -> OpenCodeRunResult:
        error = self.record({'type': kind, 'message': str(message)})
        return OpenCodeRunResult(1, session_id, error, '')


def run_opencode_streaming(*, workdir: str, prompt: str, artifact_dir: Path, session_id: str = '',
                           config: dict[str, str] | None = None, timeout_s: float = 900,
                           attempt: int | None = None
                           ) -> OpenCodeRunResult:
    started = time.monotonic()
    settings, secrets = _opencode_settings(config or {}), _secrets(config or {})
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = artifact_dir / 'opencode_prompt.json'
    stdout_path = artifact_dir / 'stdout.log'
    events_path = artifact_dir / 'events.jsonl'
    try:
        stdout_log = stdout_path.open('w', encoding='utf-8')
        events_log = events_path.open('w', encoding='utf-8')
    except Exception as exc:
        return OpenCodeRunResult(1, session_id, {'type': 'prompt_write_failed', 'message': str(exc)}, '')

    with stdout_log, events_log:
        logs = _OpenCodeLogs(stdout_log, events_log, secrets, attempt)
        if missing := _missing_config(settings):
            return logs.failure(session_id, 'configuration_error',
                                f'missing opencode config fields: {", ".join(missing)}')
        try:
            root = Path(workdir).resolve()
            prompt_path.write_text(prompt, encoding='utf-8')
        except Exception as exc:
            return logs.failure(session_id, 'prompt_write_failed', exc)

        prompt_arg = f'Follow this JSON task card exactly:\n{prompt}'
        logs.record({'type': 'setup', 'status': 'completed', 'message': f'workdir={root}'})
        logs.record({'type': 'process_start', 'status': 'running', 'message': 'starting opencode'})
        try:
            proc = subprocess.Popen(
                _cmd(prompt_arg, session_id, settings),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(root),
                env=_process_env(settings),
                start_new_session=True,
            )
        except Exception as exc:
            return logs.failure(session_id, 'process_start_failed', exc)

        session, error, finish_reason = session_id, None, ''
        while proc.poll() is None:
            now = time.monotonic()
            if now - started > timeout_s:
                error = logs.record({'type': 'timeout', 'message': f'opencode timed out after {timeout_s}s'})
                _terminate(proc)
                break
            ready, _, _ = select.select([proc.stdout], [], [], 0.05) if proc.stdout else ([], [], [])
            if not ready:
                continue
            session, error, finish_reason = _read_line(
                ready[0].readline(), logs, session, error, finish_reason,
            )
        if proc.stdout:
            drain_deadline = time.monotonic() + 1.0
            while time.monotonic() < drain_deadline:
                ready, _, _ = select.select([proc.stdout], [], [], 0.05)
                if not ready:
                    break
                line = ready[0].readline()
                if not line:
                    break
                session, error, finish_reason = _read_line(
                    line, logs, session, error, finish_reason,
                )
        returncode = proc.wait()
        logs.record({'type': 'process_exit', 'status': 'completed' if returncode == 0 else 'failed',
                     'message': f'opencode exited with code {returncode}', 'returncode': returncode})
        if returncode and not error:
            error = logs.record({'type': 'process_failed', 'message': logs.tail})
        return OpenCodeRunResult(returncode, session, error, finish_reason)


def _call_artifacts(call_dirs: list[Path], report_path: Path, artifact_root: Path) -> dict[str, dict[str, str]]:
    paths = {'report': report_path}
    for index, call_dir in enumerate(call_dirs):
        prefix = '' if index == 0 else f'recovery_{index}_'
        paths |= {
            f'{prefix}prompt': call_dir / 'opencode_prompt.json',
            f'{prefix}stdout': call_dir / 'stdout.log',
            f'{prefix}events': call_dir / 'events.jsonl',
        }
    identity = '/'.join(artifact_root.parts[-4:])
    return {
        name: {
            'uri': f'phase1://{identity}/{path.relative_to(artifact_root).as_posix()}',
            'sha256': sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in paths.items()
        if path.is_file()
    }


def _read_line(line: str, logs: _OpenCodeLogs, session: str, error: dict[str, Any] | None, finish_reason: str
               ) -> tuple[str, dict[str, Any] | None, str]:
    if not line:
        return session, error, finish_reason
    logs.write_stdout(line)
    try:
        event = _clean(json.loads(line), logs.secrets)
    except json.JSONDecodeError:
        text = _clean(line.strip(), logs.secrets)
        if text:
            logs.record({'type': 'stdout', 'status': 'running', 'message': str(text)[:300]})
        return session, error, finish_reason
    if isinstance(event, dict):
        recorded = logs.record(event)
        part = event.get('part') if isinstance(event.get('part'), dict) else {}
        if event.get('type') == 'step_finish':
            finish_reason = str(part.get('reason') or event.get('reason') or '').strip()
        return (
            session or str(event.get('sessionID') or ''),
            recorded if event.get('type') == 'error' else error,
            finish_reason,
        )
    return session, error, finish_reason


def _cmd(prompt: str, session: str, settings: dict[str, str]) -> list[str]:
    binary = os.getenv('LAZYMIND_EVO_CODE_BINARY') or 'opencode'
    args = [binary, 'run', '--format', 'json']
    if settings.get('model'):
        args += ['--model', settings['model']]
    if session:
        args += ['--session', session]
    return [*args, prompt]


def _opencode_json(settings: dict[str, str]) -> dict[str, Any]:
    provider, model = settings.get('provider', ''), settings.get('provider_model', '')
    npm = settings.get('npm', '')
    base_url, api_key = settings.get('base_url', ''), settings.get('api_key', '')
    config: dict[str, Any] = {'$schema': 'https://opencode.ai/config.json', 'permission': PERMISSIONS}
    if provider and model and npm and base_url:
        options = {'baseURL': base_url}
        if api_key:
            options['apiKey'] = api_key
        config['provider'] = {provider: {
            'npm': npm,
            'options': options,
            'models': {model: {'name': model}},
        }}
    return config


def _compact(event: dict[str, Any]) -> dict[str, Any]:
    part = event.get('part') if isinstance(event.get('part'), dict) else {}
    call = event.get('call') if isinstance(event.get('call'), dict) else {}
    state = part.get('state') if isinstance(part.get('state'), dict) else {}
    tool_input = state.get('input') if isinstance(state.get('input'), dict) else {}
    fields = list(_walk(event))
    paths = [value for key, value in fields if key in PATH_KEYS and isinstance(value, str)]
    for key in ('changed_files', 'files'):
        extra = event.get(key)
        paths += [extra] if isinstance(extra, str) else [path for path in (extra or []) if isinstance(path, str)]
    raw_type = str(event.get('type') or 'unknown')
    tool = str(event.get('tool') or part.get('tool') or call.get('tool') or '')
    message = str(
        part.get('text') or event.get('text') or event.get('message')
        or event.get('error') or state.get('error') or part.get('title') or ''
    ).strip()
    command = str(tool_input.get('command') or event.get('command') or event.get('cmd') or '')
    status = str(event.get('status') or state.get('status') or event.get('state') or '')
    return {
        'event_type': raw_type,
        'tool': tool,
        'execution_type': 'tool_use' if tool else (
            'code' if raw_type in {'text', 'stdout'} and 'diff --git' in message else
            'message' if raw_type in {'text', 'stdout'} else raw_type
        ),
        'summary': message[:500],
        'file_paths': sorted(set(paths)),
        'command': command,
        'status': 'failed' if status == 'error' else status,
        'returncode': event.get('returncode'),
    }


def _record_opencode_event(attempt: int | None, event: dict[str, Any]) -> None:
    compact = _compact(event)
    raw_type, tool = compact['event_type'], compact['tool']
    if raw_type in {'step_start', 'step_finish'}:
        return
    event_type = TRACE_BY_TOOL.get(tool) or TRACE_BY_TYPE.get(raw_type)
    if not event_type and raw_type in {'text', 'stdout'}:
        event_type = 'opencode.code' if 'diff --git' in compact['summary'] else 'opencode.message'
    event_type = event_type or 'opencode.message'
    raw_status = compact['status']
    status = (
        'failed' if event_type == 'opencode.error' or raw_status in {'error', 'failed'}
        else 'completed' if raw_status in {'completed', 'done', 'success', 'succeeded'}
        else 'started' if raw_status in {'started', 'starting'}
        else 'running'
    )
    record_event(
        event_type,
        status=status,
        source='opencode',
        attempt=attempt,
        message=compact['summary'] or compact['command'] or raw_type,
        data={
            'execution_type': compact['execution_type'],
            'tool': tool,
            'paths': compact['file_paths'],
            'command': _command_label(compact['command']),
            'returncode': compact.get('returncode'),
        },
    )


def _command_label(command: object) -> str:
    return ' '.join(str(command or '').split()[:8])[:200]


def _walk(value: Any):
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                yield str(key), child
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)


def _opencode_settings(raw: dict[str, str]) -> dict[str, str]:
    return {
        key: str(value).strip()
        for key, value in raw.items()
        if key in OPENCODE_FIELDS and str(value).strip()
    }


def _missing_config(settings: dict[str, str]) -> list[str]:
    required = ['model', 'provider', 'provider_model', 'npm', 'base_url']
    missing = [key for key in required if not settings.get(key)]
    if not settings.get('api_key') and settings.get('skip_auth') != 'true':
        missing.append('api_key')
    return missing


def _process_env(settings: dict[str, str]) -> dict[str, str]:
    env = {
        key: value for key in ('HOME', 'PATH', 'SHELL', 'USER', 'LANG', 'LC_ALL', 'TMPDIR')
        if (value := os.environ.get(key))
    }
    env['OPENCODE_CONFIG_CONTENT'] = json.dumps(_opencode_json(settings), ensure_ascii=False)
    env['OPENCODE_PERMISSION'] = json.dumps(PERMISSIONS, ensure_ascii=False)
    return env


def _terminate(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    for sig, stop in ((signal.SIGTERM, proc.terminate), (signal.SIGKILL, proc.kill)):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            stop()
        try:
            proc.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            pass


def _clean(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, '<redacted>')
        return value
    if not isinstance(value, (dict, list)):
        return value
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    for secret in secrets:
        encoded = encoded.replace(secret, '<redacted>')
    return json.loads(encoded)


def _secrets(env: dict[str, str]) -> list[str]:
    return [
        str(value)
        for key, value in env.items()
        if value and any(token in key.lower() for token in ('key', 'token', 'secret'))
    ]


def read_opencode_report(path: Path, task: str = '') -> dict[str, Any]:
    try:
        value = parse_json_object(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {'status': 'missing', 'reason': type(exc).__name__}
    if not isinstance(value, dict):
        return {'status': 'invalid', 'reason': 'report_not_object'}
    if task == 'phase1':
        summary = str(value.get('summary') or '').strip()
        changed = _report_files(value.get('changed_files'))
        commands = []
        for item in value.get('suggested_commands') or ():
            if isinstance(item, list) and item and all(isinstance(part, str) and part for part in item):
                commands.append(item[:32])
        return {
            'status': 'completed' if summary else 'invalid',
            'reason': '' if summary else 'phase1_report_summary_missing',
            'summary': summary[:4000],
            'changed_files': changed,
            'suggested_commands': commands[:8],
        }
    if task == 'formal_patch':
        changed = _report_files(value.get('files_changed'))
        locations = [dict(item) for item in value.get('confirmed_locations') or () if isinstance(item, dict)]
        valid = value.get('status') == 'edited' and bool(changed)
        return {
            'status': 'completed' if valid else 'invalid',
            'reason': '' if valid else 'formal_patch_report_invalid',
            'files_changed': changed,
            'confirmed_locations': locations[:20],
            'change_intent': str(value.get('change_intent') or '').strip(),
            'risk': str(value.get('risk') or '').strip(),
            'notes': str(value.get('notes') or '').strip()[:1000],
        }
    entrypoint = str(value.get('entrypoint') or '').strip()
    changed = _report_files(value.get('changed_files'))
    valid = entrypoint == 'demo/run_demo.py' and bool(changed)
    return {
        'status': 'completed' if valid else 'invalid',
        'reason': '' if valid else 'demo_report_invalid',
        'entrypoint': entrypoint,
        'changed_files': changed,
    }


def _phase1_task_card(instruction: str, expected_result: str, category_id: str, report_path: Path) -> dict[str, Any]:
    return {
        'mode': 'repair_phase1',
        'category_id': category_id,
        'instruction': instruction,
        'expected_result': expected_result,
        'report_path': report_path.as_posix(),
        'constraints': [
            'Use search/read tools to inspect source/. Never modify source/.',
            'Create or revise experiments only under work/.',
            'Do not execute commands or use shell/bash; the trusted runner executes suggested commands later.',
            'Suggested test scripts must be work/**/*.sh; send pytest logs to stderr and print one JSON object to stdout.',
            'Write report_path as one strict JSON object and escape every embedded quote.',
        ],
        'report_schema': {
            'summary': 'what was learned or changed this turn',
            'changed_files': ['work/...'],
            'suggested_commands': [['work/tests/run_demo.sh']],
        },
    }


def _workspace_snapshot(root: Path) -> dict[str, str]:
    result = {}
    for name in ('source', 'work'):
        for path in (root / name).rglob('*'):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if not relative.startswith('work/.opencode/'):
                    result[relative] = sha256(path.read_bytes()).hexdigest()
    return result


def _report_files(value: object) -> list[str]:
    paths = []
    for item in value if isinstance(value, list) else ():
        path = str(item or '').strip().removeprefix('./')
        if path and not Path(path).is_absolute() and '..' not in Path(path).parts:
            paths.append(Path(path).as_posix())
    return sorted(set(paths))


def _invalid_session(run: OpenCodeRunResult) -> bool:
    error = run.last_error if isinstance(run.last_error, dict) else {}
    message = str(error.get('message') or '').casefold()
    return bool(run.session_id) and 'session' in message and any(token in message for token in ('not found', 'invalid'))


def _run_failure(run: OpenCodeRunResult) -> str:
    if run.last_error:
        return str(run.last_error.get('type') or run.last_error.get('message') or 'opencode_failed')
    return f'opencode_exit_{run.returncode}' if run.returncode else ''
