from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evo.operations.abtest.candidate import (
    async_candidate_rag_answer,
    candidate_service,
    discard_candidate,
)
from evo.operations.abtest.comparison import compare_abtest, compare_eval_detail_for_repair
from evo.operations.eval.judge import judge_case
from evo.operations.eval.summary import build_eval_detail_summary
from evo.operations.public_contracts import build_eval_summary_root

from .contracts import RepairAction, RepairContractError, RepairObservation
from .opencode import terminate_process
from .workspace import (
    WorkspacePaths,
    artifact_path,
    code_changes,
    create_code_checkpoint,
    rollback_code_checkpoint,
    workspace_hash,
    write_json,
)


MAX_TEST_OUTPUT_BYTES = 256 * 1024
L0_TIMEOUT_SECONDS = 900
CASE_CONCURRENCY = 4
_L0_COMMAND_TAILS = (
    ('-m', 'compileall', '-q', 'algorithm/lazymind/chat'),
    (
        '-m', 'pytest', '-q',
        'algorithm/tests/chat/engine/test_context_estimator.py',
        'algorithm/tests/chat/engine/test_prompt_builder.py',
        'algorithm/tests/chat/engine/prompts/test_system_prompt.py',
    ),
)


@dataclass(frozen=True, slots=True)
class RepairTestPlan:
    run_id: str
    target_category: str
    l1_case_ids: tuple[str, ...]
    l2_case_ids: tuple[str, ...]
    cases: Mapping[str, Mapping[str, Any]]
    baseline_judges: Mapping[str, Mapping[str, Any]]
    eval_policy: Mapping[str, Any]
    candidate_config: Mapping[str, Any]
    l0_commands: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        run_id = _required(self.run_id, 'test plan run_id')
        category = _required(self.target_category, 'test plan target_category')
        l1 = _case_ids(self.l1_case_ids, 'L1')
        l2 = _case_ids(self.l2_case_ids, 'L2')
        cases = _mapping_values(self.cases, 'cases')
        judges = _mapping_values(self.baseline_judges, 'baseline_judges')
        missing = sorted(set(l2) - cases.keys())
        missing_judges = sorted(set(l2) - judges.keys())
        if missing or missing_judges or not set(l1).issubset(l2):
            raise RepairContractError(
                'test_plan_case_set_invalid',
                f'missing_cases={missing}, missing_judges={missing_judges}, l1_not_l2={sorted(set(l1) - set(l2))}',
            )
        commands = tuple(tuple(str(item) for item in command) for command in self.l0_commands)
        if not commands or any(not command or any(not item for item in command) for command in commands):
            raise RepairContractError('test_plan_l0_invalid')
        object.__setattr__(self, 'run_id', run_id)
        object.__setattr__(self, 'target_category', category)
        object.__setattr__(self, 'l1_case_ids', l1)
        object.__setattr__(self, 'l2_case_ids', l2)
        object.__setattr__(self, 'cases', cases)
        object.__setattr__(self, 'baseline_judges', judges)
        object.__setattr__(self, 'eval_policy', dict(self.eval_policy))
        object.__setattr__(self, 'candidate_config', dict(self.candidate_config))
        object.__setattr__(self, 'l0_commands', commands)


def build_test_plan(
    run_id: str,
    analysis: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    baseline_judges: Mapping[str, Mapping[str, Any]],
    eval_policy: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> RepairTestPlan:
    queue = analysis.get('repair_group_queue')
    group = queue[0] if isinstance(queue, list) and queue and isinstance(queue[0], Mapping) else None
    if group is None:
        raise RepairContractError('test_plan_repair_group_missing')
    category = _required(group.get('issue_category'), 'repair group issue_category')
    rows = analysis.get('rows') if isinstance(analysis.get('rows'), list) else []
    l1 = tuple(
        str(row.get('case_id') or '').strip()
        for row in rows
        if isinstance(row, Mapping) and str(row.get('issue_category') or '').strip() == category
    )
    l2 = tuple(str(case_id).strip() for case_id in analysis.get('case_ids') or () if str(case_id).strip())
    return RepairTestPlan(
        run_id=run_id,
        target_category=category,
        l1_case_ids=l1,
        l2_case_ids=l2,
        cases=cases,
        baseline_judges=baseline_judges,
        eval_policy=eval_policy,
        candidate_config={**candidate_config, 'thread_id': run_id},
        l0_commands=tuple((sys.executable, *tail) for tail in _L0_COMMAND_TAILS),
    )


class RepairTestCapability:
    """Runtime-owned L0/L1/L2 validation behind the stable Repair test port."""

    def __init__(self, plan: RepairTestPlan, paths: WorkspacePaths) -> None:
        self.plan = plan
        self.paths = paths
        self.secrets = _secret_values(plan.eval_policy, plan.candidate_config)

    def __call__(self, action: RepairAction) -> RepairObservation:
        level = str(action.arguments.get('level') or '')
        if level not in {'L0', 'L1', 'L2'}:
            raise RepairContractError('test_level_invalid', level)
        current_hash = workspace_hash(self.paths.source)
        started = time.monotonic()
        try:
            evidence = self._run_l0(action, current_hash) if level == 'L0' else self._run_cases(
                action, level, current_hash,
            )
        except Exception as exc:
            evidence = {
                'kind': 'test',
                'call_id': action.call_id,
                'level': level,
                'status': 'error',
                'workspace_hash': workspace_hash(self.paths.source),
                'duration_seconds': round(time.monotonic() - started, 3),
                'error_type': type(exc).__name__,
                'error': _redact(str(exc), self.secrets)[:4000],
            }
        path = write_json(artifact_path(self.paths.evidence, action.call_id), evidence)
        status = 'success' if evidence['status'] == 'success' else 'error' if evidence['status'] == 'error' else 'fail'
        summary = _summary(evidence)
        return RepairObservation(action.call_id, status, summary, [str(path)], evidence['workspace_hash'])

    def _run_l0(self, action: RepairAction, current_hash: str) -> dict[str, Any]:
        checkpoint = create_code_checkpoint(self.paths, action.call_id)
        outputs = []
        changes: list[str] = []
        try:
            with tempfile.TemporaryDirectory(prefix='repair-l0-', dir=self.paths.control) as temporary:
                environment = _l0_environment(self.paths, Path(temporary))
                for command in self.plan.l0_commands:
                    output = _run_command(command, self.paths.source, environment, L0_TIMEOUT_SECONDS)
                    outputs.append(output)
                    if output['return_code'] != 0 or output['timed_out'] or output['output_truncated']:
                        break
        finally:
            changes = code_changes(checkpoint)
            if changes:
                rollback_code_checkpoint(checkpoint)
        passed = bool(outputs) and all(
            output['return_code'] == 0 and not output['timed_out'] and not output['output_truncated']
            for output in outputs
        ) and not changes
        return {
            'kind': 'test',
            'call_id': action.call_id,
            'level': 'L0',
            'status': 'success' if passed else 'fail',
            'workspace_hash': current_hash,
            'duration_seconds': round(sum(float(item['duration_seconds']) for item in outputs), 3),
            'commands': outputs,
            'workspace_changes': changes,
        }

    def _run_cases(
        self,
        action: RepairAction,
        level: str,
        current_hash: str,
    ) -> dict[str, Any]:
        case_ids = self.plan.l1_case_ids if level == 'L1' else self.plan.l2_case_ids
        checkpoint = create_code_checkpoint(self.paths, action.call_id)
        service: dict[str, Any] | None = None
        cleanup: dict[str, Any] = {'status': 'not_started'}
        started = time.monotonic()
        try:
            try:
                service = candidate_service(
                    self.plan.candidate_config,
                    {'status': 'unvalidated', 'workspace_ref': str(self.paths.source), 'diff': ''},
                    SimpleNamespace(run_id=self.plan.run_id),
                    temporary=True,
                )
                if service.get('status') != 'ready':
                    raise RuntimeError(f'candidate service failed: {service.get("healthcheck")}')
                cases = [self.plan.cases[case_id] for case_id in case_ids]
                candidate_judges = asyncio.run(_evaluate_cases(cases, service, self.plan.eval_policy))
                baseline_judges = [self.plan.baseline_judges[case_id] for case_id in case_ids]
                baseline_summary = build_eval_summary_root(self.plan.run_id, baseline_judges)
                candidate_summary = build_eval_summary_root(self.plan.run_id, candidate_judges)
                comparison = compare_abtest(self.plan.run_id, baseline_summary, candidate_summary, service)
                candidate_detail = build_eval_detail_summary(candidate_judges)
                detail = compare_eval_detail_for_repair(
                    build_eval_detail_summary(baseline_judges),
                    candidate_detail,
                )
            finally:
                if service is not None:
                    cleanup = discard_candidate(service, delete_workspace=False)
        finally:
            changes = code_changes(checkpoint)
            if changes:
                rollback_code_checkpoint(checkpoint)
        passed = comparison.get('verdict') == 'accept' and cleanup.get('status') == 'completed' and not changes
        return {
            'kind': 'test',
            'call_id': action.call_id,
            'level': level,
            'status': 'success' if passed else 'fail',
            'workspace_hash': current_hash,
            'duration_seconds': round(time.monotonic() - started, 3),
            'case_ids': list(case_ids),
            'case_count': len(case_ids),
            'candidate_algorithm_id': service.get('algorithm_id') if service else '',
            'service_health': (service.get('healthcheck') or {}).get('status') if service else 'not_started',
            'verdict': comparison.get('verdict'),
            'reasons': list(comparison.get('reasons') or ()),
            'delta': dict(comparison.get('delta') or {}),
            'case_deltas': list(detail.get('case_deltas') or ()),
            'execution_failures': list(candidate_detail.get('execution_failures') or ()),
            'cleanup_status': cleanup.get('status'),
            'workspace_changes': changes,
        }


async def _evaluate_cases(
    cases: Sequence[Mapping[str, Any]],
    service: Mapping[str, Any],
    eval_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(CASE_CONCURRENCY)

    async def evaluate(case: Mapping[str, Any]) -> dict[str, Any]:
        async with semaphore:
            answer = await async_candidate_rag_answer(case, service)
            return await asyncio.to_thread(judge_case, case, answer, eval_policy)

    return list(await asyncio.gather(*(evaluate(case) for case in cases)))


def _run_command(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    timed_out = False
    output_truncated = False
    with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
        process = subprocess.Popen(
            list(command), cwd=cwd, env=dict(environment),
            stdout=stdout_file, stderr=stderr_file, start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout_seconds:
                timed_out = True
                terminate_process(process, grace_seconds=2.0)
                break
            output_size = max(
                os.fstat(stdout_file.fileno()).st_size,
                os.fstat(stderr_file.fileno()).st_size,
            )
            if output_size > MAX_TEST_OUTPUT_BYTES:
                output_truncated = True
                terminate_process(process, grace_seconds=2.0)
                break
            time.sleep(0.05)
        process.wait()
        stdout = _read_output(stdout_file)
        stderr = _read_output(stderr_file)
    return {
        'command': list(command),
        'return_code': process.returncode,
        'timed_out': timed_out,
        'output_truncated': output_truncated or stdout[1] or stderr[1],
        'duration_seconds': round(time.monotonic() - started, 3),
        'stdout': stdout[0],
        'stderr': stderr[0],
    }


def _read_output(stream: Any) -> tuple[str, bool]:
    stream.seek(0)
    value = stream.read(MAX_TEST_OUTPUT_BYTES + 1)
    return value[:MAX_TEST_OUTPUT_BYTES].decode('utf-8', errors='replace'), len(value) > MAX_TEST_OUTPUT_BYTES


def _l0_environment(paths: WorkspacePaths, temporary: Path) -> dict[str, str]:
    allowed = ('PATH', 'LANG', 'LC_ALL', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR')
    environment = {key: value for key in allowed if (value := os.environ.get(key))}
    python_path = [str(paths.source / 'algorithm')]
    if os.environ.get('PYTHONPATH'):
        python_path.append(str(os.environ['PYTHONPATH']))
    environment.update({
        'HOME': str(temporary),
        'TMPDIR': str(temporary),
        'PYTHONPATH': os.pathsep.join(python_path),
        'PYTHONPYCACHEPREFIX': str(temporary / 'pycache'),
        'PYTHONDONTWRITEBYTECODE': '1',
        'PYTEST_ADDOPTS': '-p no:cacheprovider',
    })
    return environment


def _summary(evidence: Mapping[str, Any]) -> str:
    value = {
        'level': evidence.get('level'),
        'status': evidence.get('status'),
        'case_count': evidence.get('case_count'),
        'verdict': evidence.get('verdict'),
        'delta': evidence.get('delta'),
        'reasons': evidence.get('reasons'),
        'cleanup_status': evidence.get('cleanup_status'),
        'workspace_changes': evidence.get('workspace_changes'),
        'error': evidence.get('error'),
    }
    return json.dumps({key: item for key, item in value.items() if item not in (None, [], {}, '')}, ensure_ascii=False)


def _mapping_values(value: Mapping[str, Mapping[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping) or not all(isinstance(item, Mapping) for item in value.values()):
        raise RepairContractError('test_plan_mapping_invalid', name)
    return {str(key): dict(item) for key, item in value.items()}


def _case_ids(value: Sequence[str], level: str) -> tuple[str, ...]:
    result = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
    if not result:
        raise RepairContractError('test_plan_case_set_empty', level)
    return result


def _required(value: object, name: str) -> str:
    result = str(value or '').strip()
    if not result:
        raise RepairContractError('field_required', name)
    return result


def _secret_values(*values: Mapping[str, Any]) -> list[str]:
    secrets = []
    pending: list[tuple[str, object]] = [('', value) for value in values]
    while pending:
        key, value = pending.pop()
        if isinstance(value, Mapping):
            pending.extend((str(child_key).casefold(), child) for child_key, child in value.items())
        elif isinstance(value, (list, tuple)):
            pending.extend((key, child) for child in value)
        elif _is_secret_key(key) and value:
            secrets.append(str(value))
    return list(dict.fromkeys(secrets))


def _redact(value: str, secrets: Sequence[str]) -> str:
    result = value
    for secret in secrets:
        result = result.replace(secret, '<redacted>')
    return result


def _is_secret_key(value: str) -> bool:
    key = value.casefold().replace('-', '_')
    return key in {'api_key', 'password', 'secret', 'token'} or key.endswith(
        ('_api_key', '_password', '_secret', '_token'),
    )


__all__ = ['RepairTestCapability', 'RepairTestPlan', 'build_test_plan']
