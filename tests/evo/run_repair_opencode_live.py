#!/usr/bin/env python3
"""Real OpenCode + real model + trusted Demo smoke for the current Repair iteration."""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evo.llm import LazyLLMClient
from evo.operations.repair.capabilities import DefaultCapabilityFactory, ShellCapability
from evo.operations.repair.contracts import RepairAction, RepairInput
from evo.operations.repair.decision import DecisionModelAdapter
from evo.operations.repair.memory import EventMemory
from evo.operations.repair.opencode import OpenCodeCapability
from evo.operations.repair.session import RepairSession
from evo.operations.repair.workspace import initialize_workspace, workspace_hash, write_json
from evo.repair_model import opencode_settings


DEFAULT_STORE = Path('/var/lib/lazymind/evo/artifact-runtime/artifact-runtime.sqlite3')
DEFAULT_RUNTIME = Path('/var/lib/lazymind/evo/repair-live')


def _model_config(store: Path, thread_id: str) -> tuple[str, dict[str, Any]]:
    connection = sqlite3.connect(f'file:{store}?mode=ro&immutable=1', uri=True)
    try:
        if thread_id:
            rows = connection.execute(
                'select run_id, payload from artifacts '
                'where run_id = ? and artifact_id = ? order by version desc',
                (thread_id, 'run.config'),
            )
        else:
            rows = connection.execute(
                'select run_id, payload from artifacts '
                'where artifact_id = ? order by rowid desc',
                ('run.config',),
            )
        for run_id, payload in rows:
            value = pickle.loads(payload)
            config = value.get('llm_config') if isinstance(value, dict) else None
            role = config.get('evo_llm') if isinstance(config, dict) else None
            if isinstance(role, dict) and role.get('api_key'):
                return str(run_id), config
    finally:
        connection.close()
    raise RuntimeError('no persisted thread contains a usable evo_llm config')


def _source(root: Path) -> Path:
    source = root / 'original'
    (source / 'pkg').mkdir(parents=True)
    (source / 'pkg/__init__.py').write_text('', encoding='utf-8')
    (source / 'pkg/rag_pipeline.py').write_text(
        'def normalize_results(items):\n'
        '    """BUG: text survives but chunk ids are dropped."""\n'
        '    return [{"text": item["text"]} for item in items]\n',
        encoding='utf-8',
    )
    return source


def _assert_secret_absent(roots: list[Path], secret: object) -> int:
    needle = str(secret or '').encode('utf-8')
    if not needle:
        raise RuntimeError('real-service smoke requires an authenticated model config')
    checked = 0
    hits = []
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            checked += 1
            try:
                if needle in path.read_bytes():
                    hits.append(str(path.relative_to(root)))
            except OSError:
                continue
    if hits:
        raise RuntimeError(f'API key leaked into Repair artifacts: {hits[:10]}')
    return checked


def _run_agent_loop(
    source: Path,
    runtime_root: Path,
    llm_config: dict[str, Any],
    role: dict[str, Any],
    progress: list[dict[str, Any]],
) -> dict[str, Any]:
    run_id = f'live-agent-{int(time.time())}'
    repair_input = RepairInput(
        run_id=run_id,
        objective=(
            'Use code evidence and a controlled Demo to determine whether normalize_results preserves '
            'retrieved chunk_id values.'
        ),
        guidance=(
            'Investigate rather than modify source. The Demo does not exist initially; do not run shell until '
            'a code.edit_work observation confirms work/demo/run_demo.py was created. A controlled input already '
            'exists at work/inputs/case.json with schema {"items": [{"text": "...", "chunk_id": "..."}]}. '
            'Use code.inspect for the causal source finding. Ask code.edit_work to call normalize_results on '
            'input["items"] and emit input_chunk_ids, output_chunk_ids, and ids_preserved using the exact '
            'chunk_id field. Then use the trusted shell command to observe the result. Base every next action '
            'on the latest observation. Do not run tests or request finish in this investigation-only iteration, '
            'and do not repeat a successful action.'
        ),
        source_ref=str(source),
        case_scope='pkg',
        constraints={
            'controlled_input': 'work/inputs/case.json',
            'must_observe': 'input chunk_id is present and output chunk_id is missing',
            'live_smoke': True,
        },
        budget={'turns': 3, 'seconds': 900},
    )
    paths = initialize_workspace(repair_input, runtime_root)
    inputs = paths.work / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    write_json(inputs / 'case.json', {
        'items': [{'text': 'refund policy', 'chunk_id': 'chunk-finance-203'}],
    })
    source_before = (paths.source / 'pkg/rag_pipeline.py').read_text(encoding='utf-8')
    client = LazyLLMClient(llm_config=llm_config, model='evo_llm')
    result = RepairSession(
        DecisionModelAdapter(client, timeout_seconds=120),
        DefaultCapabilityFactory(
            opencode_settings(role),
            code_timeout_seconds=300,
            event_sink=lambda event_type, status, message, data: progress.append({
                'event_type': event_type,
                'status': status,
                'message': message,
                'data': dict(data),
            }),
        ),
        runtime_root=runtime_root,
    ).run(repair_input)
    memory = EventMemory(paths, repair_input)
    actions = [
        {
            'call_id': event.call_id,
            'tool': event.payload.get('tool'),
            'operation': (event.payload.get('arguments') or {}).get('operation'),
        }
        for event in memory.read()
        if event.event == 'agent.action'
    ]
    observations = memory.observations()
    demo_outputs = []
    for observation in observations:
        for reference in observation.artifact_refs:
            path = Path(reference)
            if not path.is_file():
                continue
            try:
                artifact = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(artifact, dict) and artifact.get('kind') == 'demo':
                demo_outputs.append(artifact.get('output'))
    selected = {(action['tool'], action['operation']) for action in actions}
    required = {('code', 'inspect'), ('code', 'edit_work'), ('shell', None)}
    if not required.issubset(selected):
        raise RuntimeError(f'real Repair agent did not complete the investigation loop: {actions}')
    if not any(isinstance(output, dict) and output.get('ids_preserved') is False for output in demo_outputs):
        raise RuntimeError(f'real Repair agent did not observe the defect: {demo_outputs}')
    if (paths.source / 'pkg/rag_pipeline.py').read_text(encoding='utf-8') != source_before:
        raise RuntimeError('real Repair agent modified source')
    return {
        'status': 'passed',
        'repair_result_status': result.status,
        'actions': actions,
        'observation_count': len(observations),
        'demo_outputs': demo_outputs,
        'workspace_root': str(paths.root),
    }


def run(store: Path, runtime_root: Path, thread_id: str) -> dict[str, Any]:
    config_thread, llm_config = _model_config(store, thread_id)
    role = llm_config['evo_llm']
    run_id = f'live-code-{int(time.time())}'
    with TemporaryDirectory(prefix='repair-opencode-live-source-') as temporary:
        source = _source(Path(temporary))
        repair_input = RepairInput(
            run_id=run_id,
            objective=(
                'Find why normalize_results loses retrieved chunk ids and build a controlled Demo that proves it.'
            ),
            guidance=(
                'Use OpenCode for code inspection and Demo authoring. Do not modify source. '
                'The host must run the Demo and observe whether ids are preserved.'
            ),
            source_ref=str(source),
            case_scope='pkg',
            constraints={'live_smoke': True},
            budget={'turns': 8, 'seconds': 900},
        )
        paths = initialize_workspace(repair_input, runtime_root)
        memory = EventMemory(paths, repair_input)
        memory.project(
            workspace_hash(paths.source),
            [],
            {'turns': 8, 'seconds': 900},
            lambda _objective, _guidance, previous, _events: previous or 'No earlier attempts.',
        )
        progress: list[dict[str, Any]] = []
        capability = OpenCodeCapability(
            repair_input,
            paths,
            opencode_settings(role),
            timeout_seconds=300,
            event_sink=lambda event_type, status, message, data: progress.append({
                'event_type': event_type,
                'status': status,
                'message': message,
                'data': dict(data),
            }),
        )
        source_before = (paths.source / 'pkg/rag_pipeline.py').read_text(encoding='utf-8')
        inspect_observation = capability(RepairAction(
            'live-inspect',
            'code',
            {
                'operation': 'inspect',
                'instruction': (
                    'Inspect source/pkg/rag_pipeline.py. Explain exactly where chunk ids are lost and identify '
                    'the responsible function. Do not modify any code.'
                ),
            },
        ))
        if inspect_observation.status != 'success':
            raise RuntimeError(f'inspect failed: {inspect_observation.summary}')
        memory.record_action(
            RepairAction('live-inspect', 'code', {
                'operation': 'inspect', 'instruction': 'Inspect chunk id propagation.',
            }),
            workspace_hash(paths.source),
        )
        memory.record_observation(inspect_observation)
        memory.project(
            workspace_hash(paths.source),
            [],
            {'turns': 7, 'seconds': 600},
            lambda _objective, _guidance, previous, _events: previous or 'Inspection completed.',
        )

        inputs = paths.work / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        write_json(inputs / 'case.json', {
            'items': [{'text': 'refund policy', 'chunk_id': 'chunk-finance-203'}],
        })
        demo_observation = capability(RepairAction(
            'live-edit-work',
            'code',
            {
                'operation': 'edit_work',
                'instruction': (
                    'Create work/demo/run_demo.py. Parse --input JSON, import normalize_results from '
                    'pkg.rag_pipeline, call it with input["items"], and compare the exact "chunk_id" field '
                    '(not a generic "id" field) before and after the call. Print exactly one JSON object containing '
                    'input_chunk_ids, output_chunk_ids, and ids_preserved. Do not use network, subprocesses, '
                    'filesystem writes, or modify source.'
                ),
            },
        ))
        if demo_observation.status != 'success':
            raise RuntimeError(f'edit_work failed: {demo_observation.summary}')
        run_observation = ShellCapability(paths)(RepairAction(
            'live-demo-run',
            'shell',
            {
                'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'],
                'cwd': 'work',
                'timeout_seconds': 60,
            },
        ))
        if run_observation.status != 'success':
            raise RuntimeError(f'Demo failed: {run_observation.summary}')
        output = json.loads(run_observation.summary)
        if output.get('ids_preserved') is not False:
            raise RuntimeError(f'Demo did not reproduce the RAG defect: {output}')
        source_edit_observation = capability(RepairAction(
            'live-edit-source',
            'code',
            {
                'operation': 'edit_source',
                'instruction': (
                    'Modify only source/pkg/rag_pipeline.py. Fix normalize_results with the smallest change so '
                    'every input mapping, including its exact chunk_id field, is preserved. Do not modify work or '
                    'run commands.'
                ),
            },
        ))
        if source_edit_observation.status != 'success':
            raise RuntimeError(f'edit_source failed: {source_edit_observation.summary}')
        source_after = (paths.source / 'pkg/rag_pipeline.py').read_text(encoding='utf-8')
        if source_after == source_before:
            raise RuntimeError('live source edit produced no source change')
        fixed_observation = ShellCapability(paths)(RepairAction(
            'live-demo-fixed',
            'shell',
            {
                'command': ['python', 'demo/run_demo.py', '--input', 'inputs/case.json'],
                'cwd': 'work',
                'timeout_seconds': 60,
            },
        ))
        fixed_output = json.loads(fixed_observation.summary) if fixed_observation.status == 'success' else {}
        if fixed_output.get('ids_preserved') is not True:
            raise RuntimeError(f'live source edit did not fix the Demo: {fixed_observation.summary}')
        if (source / 'pkg/rag_pipeline.py').read_text(encoding='utf-8') != source_before:
            raise RuntimeError('live source edit escaped the candidate workspace')
        agent_loop = _run_agent_loop(source, runtime_root, llm_config, role, progress)
        secret_scanned_files = _assert_secret_absent(
            [paths.root, Path(agent_loop['workspace_root'])],
            role.get('api_key'),
        )
        result = {
            'status': 'passed',
            'run_id': run_id,
            'config_thread': config_thread,
            'provider': role.get('source') or role.get('provider'),
            'model': role.get('model'),
            'opencode_model': opencode_settings(role)['model'],
            'inspect_summary': json.loads(inspect_observation.summary),
            'demo_edit_summary': json.loads(demo_observation.summary),
            'demo_output': output,
            'source_edit_summary': json.loads(source_edit_observation.summary),
            'fixed_demo_output': fixed_output,
            'progress_event_count': len(progress),
            'secret_scanned_files': secret_scanned_files,
            'secret_hits': 0,
            'workspace_root': str(paths.root),
            'agent_loop': agent_loop,
        }
        write_json(paths.control / 'live-smoke-result.json', result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--runtime-root', type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument('--thread-id', default='')
    args = parser.parse_args()
    run(args.store, args.runtime_root, args.thread_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
