from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

from evo.operations.repair.capabilities import DefaultCapabilityFactory, ShellCapability
from evo.operations.repair.contracts import RepairAction, RepairInput, RepairObservation, RepairView
from evo.operations.repair.session import RepairSession
from evo.operations.repair.opencode import OpenCodeCapability
from evo.operations.repair.workspace import WorkspacePaths, initialize_workspace


def _executable(path: Path, *, mode: str) -> Path:
    path.write_text(f'''#!/usr/bin/env python3
import json
import os
import pathlib
import sys

task = json.loads(sys.argv[-1])
root = pathlib.Path.cwd()
report = root / task['report_path']
report.parent.mkdir(parents=True, exist_ok=True)
operation = task['operation']
mode = {mode!r}
if operation == 'inspect':
    if mode == 'mutate_source':
        (root / 'source/pkg/code.py').write_text('BROKEN = True\\n', encoding='utf-8')
    if mode == 'extra_report_file':
        (root / 'work/.opencode/extra.json').write_text('{{}}', encoding='utf-8')
    payload = {{
        'status': 'completed',
        'findings': [{{
            'path': 'source/pkg/code.py',
            'symbol': 'preserve_ids',
            'observation': (
                os.environ['LAZYMIND_OPENCODE_API_KEY']
                if mode == 'leak_secret'
                else 'normalization drops chunk ids before generation'
            ),
        }}],
        'open_questions': [],
    }}
elif operation == 'edit_work':
    demo = root / 'work/demo/run_demo.py'
    demo.parent.mkdir(parents=True, exist_ok=True)
    demo.write_text('import json\\nprint(json.dumps({{"observed": "ids-dropped"}}))\\n', encoding='utf-8')
    if mode == 'mutate_source':
        (root / 'source/pkg/code.py').write_text('BROKEN = True\\n', encoding='utf-8')
    payload = {{
        'status': 'completed',
        'entrypoint': 'work/demo/run_demo.py',
        'changed_files': ['work/demo/run_demo.py'],
        'change_intent': 'reproduce id loss',
    }}
else:
    changed = 'source/pkg/code.py'
    if mode != 'no_source_change':
        target = root / ('source/outside.py' if mode == 'source_outside' else changed)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            'def preserve_ids(items):\\n    return [dict(item) for item in items]\\n',
            encoding='utf-8',
        )
    payload = {{
        'status': 'completed',
        'changed_files': ['source/pkg/other.py' if mode == 'source_mismatch' else changed],
        'change_intent': 'preserve chunk ids',
    }}
report.write_text(json.dumps(payload), encoding='utf-8')
print(json.dumps({{'type': 'text', 'status': 'completed', 'sessionID': 'code-session'}}))
''', encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _input(tmp_path: Path, *, turns: int = 2, max_patch_bytes: int = 65536) -> RepairInput:
    source = tmp_path / 'original'
    (source / 'pkg').mkdir(parents=True)
    (source / 'pkg/code.py').write_text(
        'def preserve_ids(items):\n    return [{"text": item["text"]} for item in items]\n',
        encoding='utf-8',
    )
    return RepairInput(
        run_id='code-capability',
        objective='Find why retrieved chunk ids disappear.',
        guidance='Use a controlled experiment.',
        source_ref=str(source),
        case_scope='pkg',
        constraints={'max_patch_bytes': max_patch_bytes},
        budget={'turns': turns, 'seconds': 60},
    )


def _settings(binary: Path) -> dict[str, str]:
    return {
        'binary': str(binary),
        'model': 'test-provider/test-model',
        'provider': 'test-provider',
        'provider_model': 'test-model',
        'npm': '@ai-sdk/openai-compatible',
        'base_url': 'https://example.invalid/v1',
        'api_key': 'TEST_KEY',
    }


def _capability(
    tmp_path: Path,
    *,
    mode: str = 'normal',
    max_patch_bytes: int = 65536,
) -> tuple[OpenCodeCapability, WorkspacePaths, RepairInput]:
    repair_input = _input(tmp_path, max_patch_bytes=max_patch_bytes)
    paths = initialize_workspace(repair_input, tmp_path / 'runtime')
    capability = OpenCodeCapability(
        repair_input,
        paths,
        _settings(_executable(tmp_path / f'fake-opencode-{mode}', mode=mode)),
        timeout_seconds=30,
        event_sink=None,
    )
    return capability, paths, repair_input


def test_inspect_returns_findings_without_workspace_changes(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path)
    before = (paths.source / 'pkg/code.py').read_text(encoding='utf-8')

    observation = capability(RepairAction(
        'inspect-1', 'code', {'operation': 'inspect', 'instruction': 'Trace chunk ids.'},
    ))

    assert observation.status == 'success'
    assert 'preserve_ids' in observation.summary
    assert 'drops chunk ids' in observation.summary
    assert (paths.source / 'pkg/code.py').read_text(encoding='utf-8') == before
    assert len(observation.artifact_refs) == 3
    assert all(Path(reference).is_file() for reference in observation.artifact_refs)


def test_inspect_rolls_back_any_code_change(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path, mode='mutate_source')
    before = (paths.source / 'pkg/code.py').read_text(encoding='utf-8')

    observation = capability(RepairAction(
        'inspect-violation', 'code', {'operation': 'inspect', 'instruction': 'Inspect only.'},
    ))

    assert observation.status == 'error'
    assert 'code_scope_violation' in observation.summary
    assert (paths.source / 'pkg/code.py').read_text(encoding='utf-8') == before


def test_inspect_redacts_and_rejects_secret_in_report(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path, mode='leak_secret')

    observation = capability(RepairAction(
        'inspect-secret', 'code', {'operation': 'inspect', 'instruction': 'Inspect only.'},
    ))

    assert observation.status == 'error'
    assert observation.summary == 'code_report_contains_secret'
    for path in paths.root.rglob('*.json'):
        assert 'TEST_KEY' not in path.read_text(encoding='utf-8', errors='ignore')


def test_inspect_rejects_unexpected_opencode_workspace_file(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path, mode='extra_report_file')

    observation = capability(RepairAction(
        'inspect-extra', 'code', {'operation': 'inspect', 'instruction': 'Inspect only.'},
    ))

    assert observation.status == 'error'
    assert observation.summary == 'code_scope_violation'
    assert not (paths.work / '.opencode/extra.json').exists()


def test_edit_work_writes_demo_but_never_source(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path)
    before = (paths.source / 'pkg/code.py').read_text(encoding='utf-8')

    observation = capability(RepairAction(
        'demo-1', 'code', {'operation': 'edit_work', 'instruction': 'Create the controlled Demo.'},
    ))

    assert observation.status == 'success'
    assert (paths.work / 'demo/run_demo.py').is_file()
    assert (paths.source / 'pkg/code.py').read_text(encoding='utf-8') == before


def test_edit_work_rolls_back_source_scope_violation(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path, mode='mutate_source')
    before = (paths.source / 'pkg/code.py').read_text(encoding='utf-8')

    observation = capability(RepairAction(
        'demo-violation', 'code', {'operation': 'edit_work', 'instruction': 'Create the Demo only.'},
    ))

    assert observation.status == 'error'
    assert not (paths.work / 'demo/run_demo.py').exists()
    assert (paths.source / 'pkg/code.py').read_text(encoding='utf-8') == before


def test_edit_source_keeps_scoped_change(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path)

    observation = capability(RepairAction(
        'source-edit', 'code', {'operation': 'edit_source', 'instruction': 'Preserve chunk ids.'},
    ))

    assert observation.status == 'success'
    assert 'dict(item)' in (paths.source / 'pkg/code.py').read_text(encoding='utf-8')


def test_edit_source_rolls_back_scope_and_report_violations(tmp_path: Path) -> None:
    for mode, expected in (
        ('source_outside', 'code_scope_violation'),
        ('source_mismatch', 'code_report_diff_mismatch'),
        ('no_source_change', 'code_report_diff_mismatch'),
    ):
        root = tmp_path / mode
        root.mkdir()
        capability, paths, _ = _capability(root, mode=mode)
        before = (paths.source / 'pkg/code.py').read_text(encoding='utf-8')
        observation = capability(RepairAction(
            f'source-{mode}', 'code', {'operation': 'edit_source', 'instruction': 'Preserve ids.'},
        ))
        assert observation.status == 'error'
        assert observation.summary == expected
        assert (paths.source / 'pkg/code.py').read_text(encoding='utf-8') == before
        assert not (paths.source / 'outside.py').exists()


def test_edit_source_rolls_back_patch_over_limit(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path, max_patch_bytes=1)
    before = (paths.source / 'pkg/code.py').read_text(encoding='utf-8')

    observation = capability(RepairAction(
        'source-oversized', 'code', {'operation': 'edit_source', 'instruction': 'Preserve ids.'},
    ))

    assert observation.status == 'error'
    assert observation.summary == 'code_patch_too_large'
    assert (paths.source / 'pkg/code.py').read_text(encoding='utf-8') == before


def test_trusted_demo_runner_returns_json_observation(tmp_path: Path) -> None:
    capability, paths, _ = _capability(tmp_path)
    capability(RepairAction(
        'demo-code', 'code', {'operation': 'edit_work', 'instruction': 'Create the controlled Demo.'},
    ))
    (paths.work / 'inputs').mkdir()
    (paths.work / 'inputs/case.json').write_text('{"items": []}\n', encoding='utf-8')

    observation = ShellCapability(paths)(RepairAction(
        'demo-run',
        'shell',
        {'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work'},
    ))

    assert observation.status == 'success'
    assert json.loads(observation.summary) == {'observed': 'ids-dropped'}
    evidence = json.loads(Path(observation.artifact_refs[0]).read_text(encoding='utf-8'))
    assert evidence['kind'] == 'demo'
    assert evidence['output'] == {'observed': 'ids-dropped'}


def test_trusted_demo_runner_rejects_non_json_output(tmp_path: Path) -> None:
    _, paths, _ = _capability(tmp_path)
    (paths.work / 'demo').mkdir(parents=True)
    (paths.work / 'demo/run_demo.py').write_text('print("not-json")\n', encoding='utf-8')
    (paths.work / 'inputs').mkdir()
    (paths.work / 'inputs/case.json').write_text('{}\n', encoding='utf-8')

    observation = ShellCapability(paths)(RepairAction(
        'demo-invalid',
        'shell',
        {'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work'},
    ))

    assert observation.status == 'fail'
    assert 'not-json' in observation.summary


def test_trusted_demo_runner_rolls_back_workspace_mutation(tmp_path: Path) -> None:
    _, paths, _ = _capability(tmp_path)
    source_file = paths.source / 'pkg/code.py'
    before = source_file.read_text(encoding='utf-8')
    (paths.work / 'demo').mkdir(parents=True)
    (paths.work / 'demo/run_demo.py').write_text(
        'import json, os, pathlib\n'
        'pathlib.Path(os.environ["REPAIR_SOURCE"], "pkg/code.py").write_text("BROKEN = True\\n")\n'
        'print(json.dumps({"observed": true}))\n',
        encoding='utf-8',
    )
    (paths.work / 'inputs').mkdir()
    (paths.work / 'inputs/case.json').write_text('{}\n', encoding='utf-8')

    observation = ShellCapability(paths)(RepairAction(
        'demo-mutates',
        'shell',
        {'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work'},
    ))

    assert observation.status == 'fail'
    evidence = json.loads(Path(observation.artifact_refs[0]).read_text(encoding='utf-8'))
    assert 'Demo file writes are blocked' in evidence['stderr']
    assert source_file.read_text(encoding='utf-8') == before


def test_trusted_demo_runner_blocks_network_and_subprocesses(tmp_path: Path) -> None:
    _, paths, _ = _capability(tmp_path)
    (paths.work / 'demo').mkdir(parents=True)
    (paths.work / 'inputs').mkdir()
    (paths.work / 'inputs/case.json').write_text('{}\n', encoding='utf-8')

    for name, source, marker in (
        ('network', 'import socket\nsocket.socket()\n', 'socket.__new__'),
        ('process', 'import subprocess\nsubprocess.run(["true"])\n', 'subprocess.Popen'),
    ):
        (paths.work / 'demo/run_demo.py').write_text(source, encoding='utf-8')
        observation = ShellCapability(paths)(RepairAction(
            f'demo-{name}',
            'shell',
            {'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work'},
        ))
        assert observation.status == 'fail'
        evidence = json.loads(Path(observation.artifact_refs[0]).read_text(encoding='utf-8'))
        assert marker in evidence['stderr']


def test_trusted_demo_runner_bounds_output_and_kills_timeout(tmp_path: Path) -> None:
    _, paths, _ = _capability(tmp_path)
    (paths.work / 'demo').mkdir(parents=True)
    (paths.work / 'inputs').mkdir()
    (paths.work / 'inputs/case.json').write_text('{}\n', encoding='utf-8')

    (paths.work / 'demo/run_demo.py').write_text('print("x" * 300000)\n', encoding='utf-8')
    oversized = ShellCapability(paths)(RepairAction(
        'demo-oversized',
        'shell',
        {'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work'},
    ))
    oversized_evidence = json.loads(Path(oversized.artifact_refs[0]).read_text(encoding='utf-8'))
    assert oversized.status == 'fail'
    assert oversized_evidence['stdout_truncated'] is True
    assert len(oversized_evidence['stdout']) <= 256 * 1024

    (paths.work / 'demo/run_demo.py').write_text('import time\ntime.sleep(30)\n', encoding='utf-8')
    timed_out = ShellCapability(paths)(RepairAction(
        'demo-timeout',
        'shell',
        {
            'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'],
            'cwd': 'work',
            'timeout_seconds': 1,
        },
    ))
    timeout_evidence = json.loads(Path(timed_out.artifact_refs[0]).read_text(encoding='utf-8'))
    assert timed_out.status == 'fail'
    assert timeout_evidence['timed_out'] is True


def test_trusted_demo_runner_records_nonzero_exit(tmp_path: Path) -> None:
    _, paths, _ = _capability(tmp_path)
    (paths.work / 'demo').mkdir(parents=True)
    (paths.work / 'demo/run_demo.py').write_text('raise SystemExit(7)\n', encoding='utf-8')
    (paths.work / 'inputs').mkdir()
    (paths.work / 'inputs/case.json').write_text('{}\n', encoding='utf-8')

    observation = ShellCapability(paths)(RepairAction(
        'demo-nonzero',
        'shell',
        {'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work'},
    ))

    evidence = json.loads(Path(observation.artifact_refs[0]).read_text(encoding='utf-8'))
    assert observation.status == 'fail'
    assert evidence['return_code'] == 7


class _Agent:
    def __init__(self) -> None:
        self.views: list[RepairView] = []
        self.actions = [
            RepairAction('inspect-loop', 'code', {'operation': 'inspect', 'instruction': 'Trace chunk ids.'}),
            RepairAction('read-after-inspect', 'workspace', {'operation': 'read', 'path': 'source/pkg/code.py'}),
        ]

    def decide(self, view: RepairView) -> RepairAction:
        self.views.append(view)
        return self.actions.pop(0)

    def summarize(self, objective: str, guidance: str, previous_brief: str,
                  events: list[dict[str, Any]]) -> str:
        return previous_brief or f'{objective}: {guidance}: {len(events)}'

    def assess_finish(self, repair_input: RepairInput, view: RepairView,
                      arguments: dict[str, Any]) -> tuple[bool, str]:
        return False, 'not finished'


def test_inspect_observation_reaches_next_repair_view(tmp_path: Path) -> None:
    repair_input = _input(tmp_path)
    paths_holder: list[WorkspacePaths] = []
    agent = _Agent()
    binary = _executable(tmp_path / 'fake-opencode-loop', mode='normal')

    def factory(input_value: RepairInput, paths: WorkspacePaths):
        paths_holder.append(paths)
        code = OpenCodeCapability(input_value, paths, _settings(binary), timeout_seconds=30, event_sink=None)

        def other(action: RepairAction) -> RepairObservation:
            return RepairObservation(action.call_id, 'success', 'mechanical stub', [], 'placeholder')

        return {'workspace': other, 'code': code, 'shell': other, 'test': other, 'research': other}

    RepairSession(agent, factory, runtime_root=tmp_path / 'runtime-loop').run(repair_input)

    assert paths_holder
    assert len(agent.views) == 2
    recent = json.dumps(agent.views[1].recent_events, ensure_ascii=False)
    assert 'preserve_ids' in recent
    assert 'drops chunk ids' in recent


def test_repair_loop_reaches_demo_result_through_real_capabilities(tmp_path: Path) -> None:
    repair_input = _input(tmp_path, turns=5)
    binary = _executable(tmp_path / 'fake-opencode-e2e', mode='normal')

    class DemoAgent(_Agent):
        def __init__(self) -> None:
            self.views = []
            self.actions = [
                RepairAction('e2e-inspect', 'code', {
                    'operation': 'inspect', 'instruction': 'Trace chunk ids.',
                }),
                RepairAction('e2e-input', 'workspace', {
                    'operation': 'write', 'path': 'work/inputs/case.json', 'content': '{"items": []}\n',
                }),
                RepairAction('e2e-demo-code', 'code', {
                    'operation': 'edit_work', 'instruction': 'Create the controlled Demo.',
                }),
                RepairAction('e2e-demo-run', 'shell', {
                    'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'], 'cwd': 'work',
                }),
                RepairAction('e2e-observe', 'workspace', {
                    'operation': 'list', 'path': 'work/demo',
                }),
            ]

    agent = DemoAgent()
    result = RepairSession(
        agent,
        DefaultCapabilityFactory(_settings(binary)),
        runtime_root=tmp_path / 'runtime-e2e',
    ).run(repair_input)

    assert result.status == 'failed'
    assert len(agent.views) == 5
    final_memory = json.dumps(agent.views[-1].recent_events, ensure_ascii=False)
    assert 'ids-dropped' in final_memory
    root = tmp_path / 'runtime-e2e/code-capability/sandbox'
    assert (root / 'work/demo/run_demo.py').is_file()
    assert (root / 'source/pkg/code.py').read_text(encoding='utf-8').startswith('def preserve_ids')
