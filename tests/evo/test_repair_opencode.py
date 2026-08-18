from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path

from evo.operations.repair.opencode import OpenCodeRunner, edit_report, inspect_report


def _settings(binary: Path) -> dict[str, str]:
    return {
        'binary': str(binary),
        'model': 'test-provider/test-model',
        'provider': 'test-provider',
        'provider_model': 'test-model',
        'npm': '@ai-sdk/openai-compatible',
        'base_url': 'https://example.invalid/v1',
        'api_key': 'TEST_SECRET_VALUE',
    }


def _executable(path: Path, body: str) -> Path:
    path.write_text('#!/usr/bin/env python3\n' + body, encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_runner_streams_events_with_isolated_config_and_redaction(tmp_path: Path) -> None:
    binary = _executable(tmp_path / 'fake-opencode', '''
import json
import os

config = json.loads(os.environ['OPENCODE_CONFIG_CONTENT'])
print(json.dumps({
    'type': 'text',
    'status': 'running',
    'sessionID': 'session-real',
    'message': os.environ['LAZYMIND_OPENCODE_API_KEY'],
    'config': config,
    'home': os.environ['HOME'],
}))
print(json.dumps({'type': 'step_finish', 'part': {'reason': 'stop'}}))
''')
    events = []
    runner = OpenCodeRunner(
        _settings(binary),
        tmp_path / 'runtime',
        event_sink=lambda *event: events.append(event),
    )

    result = runner.run(
        workdir=tmp_path,
        prompt='inspect',
        call_id='inspect-1',
        permission={'*': 'deny', 'read': 'allow'},
    )

    assert result.returncode == 0
    assert result.session_id == 'session-real'
    assert result.finish_reason == 'stop'
    stdout = result.stdout_path.read_text(encoding='utf-8')
    event_log = result.events_path.read_text(encoding='utf-8')
    assert 'TEST_SECRET_VALUE' not in stdout
    assert 'TEST_SECRET_VALUE' not in event_log
    assert '<redacted>' in stdout
    assert 'apiKey' in stdout
    assert '{env:LAZYMIND_OPENCODE_API_KEY}' in stdout
    assert str(tmp_path / 'runtime/session/home') in stdout
    assert events
    assert not (tmp_path / 'opencode.json').exists()


def test_runner_timeout_kills_process_group(tmp_path: Path) -> None:
    child_path = tmp_path / 'child.pid'
    binary = _executable(tmp_path / 'slow-opencode', f'''
import subprocess
import time

child = subprocess.Popen(['sleep', '30'])
open({str(child_path)!r}, 'w', encoding='utf-8').write(str(child.pid))
time.sleep(30)
''')
    runner = OpenCodeRunner(_settings(binary), tmp_path / 'runtime', timeout_seconds=1)

    result = runner.run(
        workdir=tmp_path,
        prompt='timeout',
        call_id='timeout-1',
        permission={'*': 'deny'},
    )

    assert result.returncode != 0
    assert result.last_error and result.last_error['type'] == 'timeout'
    child_pid = int(child_path.read_text(encoding='utf-8'))
    for _ in range(20):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f'OpenCode child process is still alive: {child_pid}')


def test_report_parsers_enforce_operation_contracts(tmp_path: Path) -> None:
    inspect_path = tmp_path / 'inspect.json'
    inspect_path.write_text(json.dumps({
        'status': 'completed',
        'findings': [
            {'path': 'source/pkg/code.py', 'symbol': 'repair', 'observation': 'drops chunk ids'},
            {'path': 'outside.py', 'observation': 'ignored'},
            {'path': 'source/../outside.py', 'observation': 'ignored traversal'},
        ],
        'open_questions': ['which caller consumes ids?'],
    }), encoding='utf-8')
    work_path = tmp_path / 'work.json'
    work_path.write_text(json.dumps({
        'status': 'completed',
        'entrypoint': 'work/demo/run_demo.py',
        'changed_files': ['work/demo/run_demo.py', 'work/demo/../../source/pkg/code.py', 'source/pkg/code.py'],
        'change_intent': 'reproduce id loss',
    }), encoding='utf-8')

    inspected = inspect_report(inspect_path)
    edited = edit_report(work_path, 'edit_work')

    assert inspected['status'] == 'completed'
    assert inspected['findings'] == [{
        'path': 'source/pkg/code.py',
        'symbol': 'repair',
        'observation': 'drops chunk ids',
    }]
    assert edited['status'] == 'completed'
    assert edited['changed_files'] == ['work/demo/run_demo.py']
