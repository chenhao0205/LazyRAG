#!/usr/bin/env python3
"""Real Repair source-edit + L0/L1/L2 + finish acceptance test."""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import shutil
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evo.llm import LazyLLMClient
from evo.operations.abtest.candidate import (
    async_candidate_rag_answer,
    candidate_service,
    discard_candidate,
)
from evo.operations.eval.judge import judge_case
from evo.operations.repair.capabilities import DefaultCapabilityFactory
from evo.operations.repair.contracts import RepairInput
from evo.operations.repair.decision import DecisionModelAdapter
from evo.operations.repair.memory import EventMemory
from evo.operations.repair.session import RepairSession
from evo.operations.repair.testing import build_test_plan
from evo.operations.repair.workspace import initialize_workspace
from evo.repair_model import opencode_settings


DEFAULT_STORE = Path('/var/lib/lazymind/evo/artifact-runtime/artifact-runtime.sqlite3')
DEFAULT_REPAIR_ROOT = Path('/var/lib/lazymind/evo/work/repair-live')
DEFAULT_SOURCE = Path('/app/algorithm')
BUG_LINE = '    raise RuntimeError("repair-live-injected-bug")\n'


def _artifact(store: Path, run_id: str, artifact_id: str) -> dict[str, Any]:
    connection = sqlite3.connect(f'file:{store}?mode=ro&immutable=1', uri=True)
    try:
        row = connection.execute(
            'select payload from artifacts where run_id = ? and artifact_id = ? order by version desc limit 1',
            (run_id, artifact_id),
        ).fetchone()
    finally:
        connection.close()
    value = pickle.loads(row[0]) if row else None
    if not isinstance(value, dict):
        raise RuntimeError(f'missing persisted artifact: {artifact_id}')
    return value


def _source(repair_root: Path, algorithm_source: Path, run_id: str) -> Path:
    source = repair_root / '_live_sources' / run_id
    if source.exists():
        raise RuntimeError(f'live source already exists: {source}')
    (source / 'algorithm').mkdir(parents=True)
    shutil.copytree(algorithm_source / 'lazymind', source / 'algorithm/lazymind')
    shutil.copytree(algorithm_source / 'tests/chat', source / 'algorithm/tests/chat')
    return source


def _candidate(
    run_id: str,
    source: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    service = candidate_service(
        {**config, 'thread_id': run_id, 'startup_timeout_s': 300},
        {'status': 'unvalidated', 'workspace_ref': str(source), 'diff': ''},
        SimpleNamespace(run_id=run_id),
        temporary=True,
    )
    if service.get('status') != 'ready':
        raise RuntimeError(f'candidate service failed: {service.get("healthcheck")}')
    return service


async def _answers(cases: list[dict[str, Any]], service: dict[str, Any]) -> list[dict[str, Any]]:
    return list(await asyncio.gather(*(async_candidate_rag_answer(case, service) for case in cases)))


def _discard(service: dict[str, Any]) -> None:
    result = discard_candidate(service, delete_workspace=False)
    if result.get('status') != 'completed':
        raise RuntimeError(f'candidate cleanup failed: {result}')


def _golden_cases(
    source: Path,
    candidate_config: dict[str, Any],
    kb_id: str,
    run_id: str,
) -> list[dict[str, Any]]:
    seeds = [
        {
            'id': 'repair-live-1',
            'question': '这份AI学习资料主要介绍了哪些内容？',
            'question_type': 'single_hop',
        },
        {
            'id': 'repair-live-2',
            'question': '请概括这份AI学习资料的核心主题。',
            'question_type': 'summary',
        },
    ]
    for case in seeds:
        case.update({
            'answer': '待生成',
            'reference_chunk_ids': [],
            'reference_doc_ids': [],
            'case_metadata': {'kb_id': kb_id},
        })
    service = _candidate(f'{run_id}-golden', source, candidate_config)
    try:
        answers = asyncio.run(_answers(seeds, service))
    finally:
        _discard(service)
    cases = []
    for seed, answer in zip(seeds, answers, strict=True):
        text = str(answer.get('answer') or '').strip()
        if answer.get('status') != 'ok' or not text:
            raise RuntimeError(f'golden candidate answer failed: {answer.get("chat_error")}')
        cases.append({**seed, 'answer': text})
    return cases


def _baseline_judges(
    source: Path,
    cases: list[dict[str, Any]],
    candidate_config: dict[str, Any],
    eval_policy: dict[str, Any],
    run_id: str,
) -> dict[str, dict[str, Any]]:
    service = _candidate(f'{run_id}-baseline', source, candidate_config)
    try:
        answers = asyncio.run(_answers(cases, service))
    finally:
        _discard(service)
    judges = {
        case['id']: judge_case(case, answer, eval_policy)
        for case, answer in zip(cases, answers, strict=True)
    }
    if not all(judge.get('failure_type') == 'infra_failure' for judge in judges.values()):
        raise RuntimeError(f'injected baseline did not fail as expected: {judges}')
    return judges


def _inject_bug(source: Path) -> None:
    path = source / 'algorithm/lazymind/chat/service/chat_service.py'
    text = path.read_text(encoding='utf-8')
    marker = 'async def handle_chat(request: ChatRequest) -> Union[Dict[str, Any], StreamingResponse]:\n'
    if marker not in text or BUG_LINE in text:
        raise RuntimeError('unable to inject live Repair bug')
    path.write_text(text.replace(marker, marker + BUG_LINE, 1), encoding='utf-8')


def _analysis(case_ids: list[str]) -> dict[str, Any]:
    target = 'algorithm/lazymind/chat/service/chat_service.py'
    return {
        'case_ids': case_ids,
        'rows': [
            {'case_id': case_ids[0], 'issue_category': 'execution'},
            {'case_id': case_ids[1], 'issue_category': 'ok'},
        ],
        'repair_group_queue': [{
            'group_id': 'repair-live-group',
            'issue_category': 'execution',
            'issue_type': 'runtime_error',
            'failure_mode': 'injected_unconditional_raise',
            'candidate_files': [target],
            'case_ids': [case_ids[0]],
            'evidence': [{'observation': 'handle_chat raises repair-live-injected-bug before processing'}],
        }],
    }


def _assert_secrets_absent(roots: list[Path], configs: list[dict[str, Any]]) -> int:
    secrets = []

    def collect(value: object, key: str = '') -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                collect(child, str(child_key).casefold())
        elif isinstance(value, list):
            for child in value:
                collect(child, key)
        elif _is_secret_key(key) and value:
            secrets.append(str(value).encode())

    for config in configs:
        collect(config)
    checked = 0
    hits = []
    for root in roots:
        for path in root.rglob('*'):
            if not path.is_file():
                continue
            checked += 1
            try:
                data = path.read_bytes()
            except OSError:
                continue
            if any(secret and secret in data for secret in secrets):
                hits.append(str(path))
    if hits:
        raise RuntimeError(f'secret leaked into Repair artifacts: {hits[:10]}')
    return checked


def _is_secret_key(value: str) -> bool:
    key = value.casefold().replace('-', '_')
    return key in {'api_key', 'password', 'secret', 'token'} or key.endswith(
        ('_api_key', '_password', '_secret', '_token'),
    )


def run(store: Path, repair_root: Path, algorithm_source: Path, config_run_id: str) -> dict[str, Any]:
    started = time.monotonic()
    run_id = f'repair-full-live-{int(time.time())}'
    source = _source(repair_root, algorithm_source, run_id)
    run_config = _artifact(store, config_run_id, 'run.config')
    eval_policy = _artifact(store, config_run_id, 'eval.policy')
    candidate_config = _artifact(store, config_run_id, 'abtest.candidate_config')
    repair_policy = _artifact(store, config_run_id, 'repair.policy')
    inputs = run_config.get('inputs') if isinstance(run_config.get('inputs'), dict) else {}
    kb_ids = inputs.get('kb_id') if isinstance(inputs.get('kb_id'), list) else []
    if not kb_ids:
        raise RuntimeError('persisted run has no kb_id')
    cases = _golden_cases(source, candidate_config, str(kb_ids[0]), run_id)
    _inject_bug(source)
    baseline_judges = _baseline_judges(source, cases, candidate_config, eval_policy, run_id)
    analysis = _analysis([case['id'] for case in cases])
    target = 'algorithm/lazymind/chat/service/chat_service.py'
    repair_input = RepairInput(
        run_id=run_id,
        objective=json.dumps(analysis['repair_group_queue'][0], ensure_ascii=False),
        guidance=(
            'Remove only the injected unconditional raise RuntimeError("repair-live-injected-bug") from '
            f'{target}. First inspect the source, then use code.edit_source. After the source changes, run L0, '
            'L1 and L2 on that same workspace hash, then request finish. Do not modify tests or dependencies.'
        ),
        source_ref=str(source),
        case_scope=target,
        constraints={'max_patch_bytes': 65536, 'live_acceptance': True},
        budget={'turns': 10, 'seconds': 1800},
    )
    paths = initialize_workspace(repair_input, repair_root)
    plan = build_test_plan(
        run_id,
        analysis,
        {case['id']: case for case in cases},
        baseline_judges,
        eval_policy,
        candidate_config,
    )
    role = repair_policy.get('llm_config', {}).get('evo_llm')
    if not isinstance(role, dict):
        raise RuntimeError('persisted repair policy has no evo_llm config')
    progress = []
    client = LazyLLMClient(llm_config=repair_policy['llm_config'], model='evo_llm')
    result = RepairSession(
        DecisionModelAdapter(client, timeout_seconds=120),
        DefaultCapabilityFactory(
            opencode_settings(role),
            test_plan=plan,
            code_timeout_seconds=300,
            event_sink=lambda event_type, status, message, data: progress.append({
                'event_type': event_type,
                'status': status,
                'message': message,
                'data': dict(data),
            }),
        ),
        runtime_root=repair_root,
    ).run(repair_input)
    memory = EventMemory(paths, repair_input)
    actions = [
        {
            'call_id': event.call_id,
            'tool': event.payload.get('tool'),
            'operation': (event.payload.get('arguments') or {}).get('operation'),
            'level': (event.payload.get('arguments') or {}).get('level'),
        }
        for event in memory.read()
        if event.event == 'agent.action'
    ]
    test_evidence = {}
    for observation in memory.observations():
        for reference in observation.artifact_refs:
            path = Path(reference)
            if not path.is_file() or path.parent != paths.evidence:
                continue
            value = json.loads(path.read_text(encoding='utf-8'))
            if value.get('kind') == 'test':
                test_evidence[str(value.get('level'))] = value
    if result.status != 'success' or not result.patch_ref:
        raise RuntimeError(f'full Repair did not finish: {result}')
    if set(test_evidence) != {'L0', 'L1', 'L2'} or not all(
        value.get('status') == 'success' for value in test_evidence.values()
    ):
        raise RuntimeError(f'full Repair test evidence incomplete: {test_evidence}')
    if test_evidence['L1'].get('case_ids') != [cases[0]['id']]:
        raise RuntimeError(f'L1 case set mismatch: {test_evidence["L1"].get("case_ids")}')
    if test_evidence['L2'].get('case_ids') != [case['id'] for case in cases]:
        raise RuntimeError(f'L2 case set mismatch: {test_evidence["L2"].get("case_ids")}')
    fixed = (paths.source / target).read_text(encoding='utf-8')
    if BUG_LINE in fixed:
        raise RuntimeError('full Repair left the injected bug in source')
    secret_files = _assert_secrets_absent(
        [paths.root],
        [repair_policy, eval_policy, candidate_config],
    )
    value = {
        'status': 'passed',
        'run_id': run_id,
        'model': role.get('model'),
        'opencode_model': opencode_settings(role)['model'],
        'result_status': result.status,
        'patch_bytes': Path(result.patch_ref).stat().st_size,
        'actions': actions,
        'test_evidence': {
            level: {
                'status': evidence.get('status'),
                'case_count': evidence.get('case_count'),
                'verdict': evidence.get('verdict'),
                'cleanup_status': evidence.get('cleanup_status'),
                'delta': evidence.get('delta'),
            }
            for level, evidence in sorted(test_evidence.items())
        },
        'progress_event_count': len(progress),
        'secret_scanned_files': secret_files,
        'secret_hits': 0,
        'duration_seconds': round(time.monotonic() - started, 3),
        'workspace_root': str(paths.root),
    }
    (paths.control / 'full-live-result.json').write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    shutil.rmtree(source)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--store', type=Path, default=DEFAULT_STORE)
    parser.add_argument('--repair-root', type=Path, default=DEFAULT_REPAIR_ROOT)
    parser.add_argument('--algorithm-source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--config-run-id', default='thr-d24ced48')
    args = parser.parse_args()
    run(args.store, args.repair_root, args.algorithm_source, args.config_run_id)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
