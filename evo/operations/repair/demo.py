from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .memory import content_ref, write_json
from .source import source_hash


ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
SECRET = re.compile(
    r"(?i)([\"']?(?:api[_-]?key|token|secret|password|authorization)[\"']?\s*[:=]\s*)"
    r"([\"']?)([^\"'\s,;}]+)(\2)"
)
def run_command(
    work_root: Path,
    artifact_root: Path,
    command: Sequence[object],
    *,
    attempt: int,
    timeout_seconds: float,
    output_limit: int,
    expected_source_hash: str,
) -> dict[str, Any]:
    argv, relative_script = _command(command, work_root)
    if source_hash(work_root / 'source') != expected_source_hash:
        return _failed(attempt, argv, relative_script, expected_source_hash, 'source_changed')
    before = _snapshot(work_root / 'work')
    temp_dir = work_root / 'work' / '.tmp'
    temp_dir.mkdir(parents=True, exist_ok=True)
    stdout_raw = work_root / 'logs' / f'run-{attempt:03d}.stdout.raw'
    stderr_raw = work_root / 'logs' / f'run-{attempt:03d}.stderr.raw'
    env = {
        'PATH': os.environ.get('PATH', ''),
        'HOME': str(work_root),
        'LANG': os.environ.get('LANG', 'C.UTF-8'),
        'PYTHONDONTWRITEBYTECODE': '1',
        'TMPDIR': str(temp_dir),
        'PYTHONPATH': os.pathsep.join((
            str(work_root / 'source'),
            str(work_root / 'source' / 'algorithm'),
            str(work_root / 'source' / 'algorithm' / 'lazyllm'),
        )),
        'REPAIR_WORK_ROOT': str(work_root),
    }
    started = time.monotonic()
    sandboxed_command, sandboxed = _sandboxed(argv, work_root)
    with stdout_raw.open('wb') as stdout, stderr_raw.open('wb') as stderr:
        process = subprocess.Popen(
            sandboxed_command,
            cwd=str(work_root),
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            exit_code = process.wait(timeout=max(0.1, timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate(process)
            exit_code = -1
    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_bytes, stderr_bytes = stdout_raw.read_bytes(), stderr_raw.read_bytes()
    stdout_text, stdout_truncated = _sanitize(stdout_bytes, output_limit)
    stderr_text, stderr_truncated = _sanitize(stderr_bytes, output_limit)
    stdout_path = artifact_root / 'runs' / f'run-{attempt:03d}.stdout.log'
    stderr_path = artifact_root / 'runs' / f'run-{attempt:03d}.stderr.log'
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(stdout_text, encoding='utf-8')
    stderr_path.write_text(stderr_text, encoding='utf-8')
    source_changed = source_hash(work_root / 'source') != expected_source_hash
    after = _snapshot(work_root / 'work')
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    status, reason, parsed = _result_status(
        source_changed, timed_out, stdout_truncated or stderr_truncated, exit_code, stdout_text,
    )
    record = {
        'attempt': attempt,
        'status': status,
        'reason': reason,
        'script_path': relative_script,
        'command': argv,
        'source_hash': expected_source_hash,
        'sandboxed': sandboxed,
        'exit_code': exit_code,
        'duration_ms': duration_ms,
        'changed_files': changed,
        'stdout_excerpt': stdout_text[-4000:],
        'stderr_excerpt': stderr_text[-4000:],
        'output': parsed,
        'stdout_ref': content_ref(stdout_path, artifact_root),
        'stderr_ref': content_ref(stderr_path, artifact_root),
        'stdout_truncated': stdout_truncated,
        'stderr_truncated': stderr_truncated,
    }
    result_path = artifact_root / 'runs' / f'run-{attempt:03d}.json'
    write_json(result_path, record)
    stdout_raw.unlink(missing_ok=True)
    stderr_raw.unlink(missing_ok=True)
    return {**record, 'result_ref': content_ref(result_path, artifact_root)}


def request_http(
    url: str,
    method: str,
    allowed_origins: Sequence[str],
    artifact_root: Path,
    *,
    attempt: int,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    clean_url = str(url).strip()
    allowed = {_origin(item) for item in allowed_origins}
    if _origin(clean_url) not in allowed:
        raise ValueError(f'http_origin_not_allowed:{clean_url}')
    clean_method = str(method or 'GET').strip().upper()
    if clean_method not in {'GET', 'HEAD'}:
        raise ValueError('http_method_not_allowed')
    started = time.monotonic()
    status_code = None
    body = ''
    error_type = ''
    try:
        request = urllib.request.Request(clean_url, method=clean_method)
        with urllib.request.urlopen(request, timeout=max(0.1, timeout_seconds)) as response:
            status_code = int(response.status)
            body = response.read(16 * 1024).decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code)
        body = exc.read(16 * 1024).decode('utf-8', errors='replace')
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        error_type = type(exc).__name__
    record = {
        'attempt': attempt,
        'status': 'completed' if status_code is not None else 'failed',
        'url': clean_url,
        'method': clean_method,
        'status_code': status_code,
        'body_excerpt': body[:4000],
        'error_type': error_type,
        'duration_ms': int((time.monotonic() - started) * 1000),
    }
    path = artifact_root / 'runs' / f'http-{attempt:03d}.json'
    write_json(path, record)
    return {**record, 'result_ref': content_ref(path, artifact_root)}


def _command(value: Sequence[object], work_root: Path) -> tuple[list[str], str]:
    if isinstance(value, (str, bytes)) or not 2 <= len(value) <= 32:
        raise ValueError('demo_command_invalid')
    command = [str(item) for item in value]
    if any(not item or '\0' in item or '\n' in item or len(item) > 2000 for item in command):
        raise ValueError('demo_command_invalid')
    if command[0] in {'sh', '/bin/sh'}:
        if len(command) != 2:
            raise ValueError('demo_shell_command_invalid')
        resolved = _work_path(command[1], work_root, '.sh')
        return ['/bin/sh', str(resolved)], resolved.relative_to(work_root).as_posix()
    if command[0] in {'python', 'python3', sys.executable}:
        if command[1].startswith('-') or any(item in {'-c', '--command', '--eval'} for item in command[1:]):
            raise ValueError('demo_shell_or_inline_command_forbidden')
        resolved = _work_path(command[1], work_root, '.py')
        return [sys.executable, str(resolved), *command[2:]], resolved.relative_to(work_root).as_posix()
    raise ValueError('demo_executable_not_allowed')


def _work_path(value: str, work_root: Path, suffix: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.suffix != suffix:
        raise ValueError('demo_script_path_invalid')
    resolved = (work_root / path).resolve()
    work = (work_root / 'work').resolve()
    if not resolved.is_relative_to(work):
        raise ValueError('demo_script_outside_work')
    if not resolved.is_file():
        raise ValueError('demo_script_missing')
    return resolved


def _sandboxed(command: list[str], work_root: Path) -> tuple[list[str], bool]:
    sandbox = Path('/usr/bin/sandbox-exec')
    if not sandbox.is_file() or not _sandbox_available(str(sandbox)):
        return command, False
    profile = _sandbox_profile(work_root)
    profile_path = work_root / 'memory' / 'sandbox.sb'
    profile_path.write_text(profile, encoding='utf-8')
    return [str(sandbox), '-f', str(profile_path), *command], True


def _sandbox_profile(work_root: Path) -> str:
    escaped = str(work_root).replace('\\', '\\\\').replace('"', '\\"')
    return (
        '(version 1)\n'
        '(allow default)\n'
        '(deny network*)\n'
        '(deny file-write*)\n'
        '(allow file-write* (literal "/dev/null"))\n'
        f'(allow file-write* (subpath "{escaped}"))\n'
    )


@lru_cache(maxsize=1)
def _sandbox_available(binary: str) -> bool:
    try:
        result = subprocess.run(
            [binary, '-p', '(version 1)(allow default)', '/usr/bin/true'],
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def _failed(attempt: int, command: list[str], script_path: str, source_digest: str, reason: str) -> dict[str, Any]:
    return {
        'attempt': attempt,
        'status': reason,
        'reason': reason,
        'script_path': script_path,
        'command': command,
        'source_hash': source_digest,
        'sandboxed': False,
        'exit_code': -1,
        'duration_ms': 0,
        'changed_files': [],
        'stdout_excerpt': '',
        'stderr_excerpt': '',
        'output': None,
        'stdout_ref': None,
        'stderr_ref': None,
        'stdout_truncated': False,
        'stderr_truncated': False,
        'result_ref': None,
    }


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob('*')
        if path.is_file()
    }


def _result_status(
    source_changed: bool,
    timed_out: bool,
    output_too_large: bool,
    exit_code: int,
    stdout: str,
) -> tuple[str, str, dict[str, Any] | None]:
    if source_changed:
        return 'source_changed', 'source_changed', None
    if timed_out:
        return 'timeout', 'timeout', None
    if output_too_large:
        return 'output_too_large', 'output_too_large', None
    if exit_code:
        return 'process_failed', f'process_exit_{exit_code}', None
    parsed = _json_output(stdout)
    if parsed is None:
        return 'invalid_json', 'stdout_not_json_object', None
    if not isinstance(parsed.get('status'), str) or not parsed['status'].strip():
        return 'invalid_contract', 'status_missing', parsed
    if parsed['status'] != 'passed':
        return 'demo_failed', f"demo_status_{parsed['status']}", parsed
    return 'passed', '', parsed


def _json_output(value: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _terminate(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(process.pid), sig)
            process.wait(timeout=2)
            return
        except (OSError, subprocess.TimeoutExpired):
            continue
    process.kill()


def _sanitize(raw: bytes, limit: int) -> tuple[str, bool]:
    text = ANSI.sub('', raw.decode('utf-8', errors='replace'))
    text = ''.join(char for char in text if char in '\n\r\t' or ord(char) >= 32)
    text = SECRET.sub(r'\1\2<redacted>\4', text)
    encoded = text.encode('utf-8')
    if len(encoded) <= limit:
        return text, False
    half = max(1, limit // 2)
    clipped = encoded[:half] + b'\n...<truncated>...\n' + encoded[-half:]
    return clipped.decode('utf-8', errors='replace'), True


def _origin(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if (
        parsed.scheme not in {'http', 'https'}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f'http_url_invalid:{value}')
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    return f'{parsed.scheme}://{parsed.hostname}:{port}'
