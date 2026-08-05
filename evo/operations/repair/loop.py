from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unidiff import PatchSet

from evo.artifact_runtime import record_event
from evo.operations.public_contracts import RepairPatch, algo_id, clean_text, dump_contract
from evo.llm import LazyLLMClient

from .agent import ModelCallError, ModelCallTimeout, review_patch
from .candidate import validate_candidate_patch
from .opencode import EvoModelConfigError, build_opencode_settings, read_opencode_report, run_opencode_streaming
from .validation import pre_validate
from .workspace import (
    algorithm_source_root,
    git,
    prepare_workspace,
    reset_workspace,
    source_fingerprint,
    workspace_diff,
    workspace_fingerprint,
    workspace_path,
)


DEFAULT_SOURCE = '/app/algorithm'


def prepare_candidate_workspace(plan: Mapping[str, Any], repair_policy: Mapping[str, Any] | None = None
                                ) -> dict[str, Any]:
    policy = _runtime_policy(repair_policy)
    if plan.get('status') != 'planned':
        return {
            'status': 'failed',
            'reason': f"repair plan is not planned: {clean_text(plan.get('status')) or 'missing_status'}",
            'repair_plan_ref': _plan_ref(plan),
            'workspace_kind': 'managed_worktree',
        }
    source = algorithm_source_root(
        policy.get('candidate_source_dir') or os.getenv('LAZYMIND_EVO_CHAT_SOURCE') or DEFAULT_SOURCE
    )
    workspace = workspace_path(policy, plan)
    plan_hash = _digest({'category_id': plan.get('category_id'), 'method': plan.get('method')})
    prepare_workspace(source, workspace, plan_hash)
    return {
        'status': 'ready',
        'workspace_kind': 'managed_worktree',
        'workspace_ref': str(workspace),
        'source_dir': str(source),
        'source_hash': source_fingerprint(source)['source_hash'],
        'plan_hash': plan_hash,
        'git_head': git(workspace, 'rev-parse', '--verify', 'HEAD'),
        'repair_plan_ref': _plan_ref(plan),
    }


async def run_repair_loop(
    workspace: Mapping[str, Any],
    analysis: Mapping[str, Any],
    cases: tuple[Mapping[str, Any], ...],
    baseline_judges: tuple[Mapping[str, Any], ...],
    eval_policy: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    repair_policy: Mapping[str, Any],
    ctx: Any,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _runtime_policy(repair_policy)
    baseline_algo_id = next((value for judge in baseline_judges if (value := algo_id(judge))), '')
    ready = _ready_workspace(workspace, plan, policy)
    if ready['status'] != 'ready':
        return _result('failed', plan, workspace, [], {}, ready['reason'], baseline_algo_id)
    category = _selected_category(analysis, plan)
    if not category:
        return _result('failed', plan, workspace, [], {}, 'selected_analysis_category_missing', baseline_algo_id)
    if clean_text(analysis.get('source_hash')) != clean_text(workspace.get('source_hash')):
        return _result('failed', plan, workspace, [], {}, 'source_version_changed_since_analysis', baseline_algo_id)
    target_ids = list(category['cases'])
    case_map = {
        clean_text(case.get('id')): case
        for case in cases
        if isinstance(case, Mapping) and clean_text(case.get('id'))
    }
    baseline_map = {
        clean_text(judge.get('case_id')): judge
        for judge in baseline_judges
        if isinstance(judge, Mapping) and clean_text(judge.get('case_id'))
    }
    validation_ids = _validation_cases(analysis, plan, baseline_map, policy)
    gap = _validation_input_gap(validation_ids, case_map, baseline_map)
    if gap:
        return _result('failed', plan, workspace, [], {}, gap, baseline_algo_id)
    try:
        config = build_opencode_settings(_llm_config(policy).get('evo_llm'))
    except EvoModelConfigError as exc:
        return _result('failed', plan, workspace, [], {}, exc.reason, baseline_algo_id)
    root = Path(str(workspace['workspace_ref'])).resolve()
    attempts: list[dict[str, Any]] = []
    session_id = ''
    attempt_budget = _positive_int(policy.get('repair_attempt_budget'), 6, 1, 20)
    infra_retries = _positive_int(policy.get('infra_retry_budget'), 1, 0, 3)
    for attempt_no in range(1, attempt_budget + 1):
        reset_workspace(root)
        artifact_dir = root / '.evo_repair_logs' / 'opencode' / f'attempt_{attempt_no}'
        report_path = artifact_dir / 'worker_report.json'
        task = _task_card(plan, category, workspace, attempt_no, report_path, attempts)
        record_event('repair.attempt_started', status='started', attempt=attempt_no)
        run = run_opencode_streaming(
            workdir=str(root),
            prompt=json.dumps(task, ensure_ascii=False, indent=2),
            artifact_dir=artifact_dir,
            session_id=session_id,
            config=config,
            timeout_s=_positive_int(policy.get('opencode_timeout_s'), 900, 30, 7200),
            attempt=attempt_no,
        )
        session_id = run.session_id or session_id
        report = read_opencode_report(report_path, 'formal_patch')
        diff_info = workspace_diff(root)
        failure = _worker_failure(run, report, diff_info)
        if failure:
            pre = {'status': 'skipped', 'reason': 'worker_failed'}
            patch_review = {'status': 'skipped', 'reason': 'worker_failed'}
            candidate = {'status': 'rejected', 'accepted': False, 'reason': failure}
        else:
            pre = pre_validate(root, diff_info, plan, analysis, policy, attempt_no)
            patch_review = {'status': 'skipped', 'reason': 'pre_validation_failed'}
            candidate = {'status': 'rejected', 'accepted': False, 'reason': 'pre_validation_failed'}
            if pre['status'] == 'passed':
                try:
                    record_event('repair.patch_review_started', status='started', attempt=attempt_no)
                    reviewed = review_patch(
                        LazyLLMClient(llm_config=_llm_config(policy), model='evo_llm'),
                        plan, category, diff_info['diff'], report,
                        [_attempt_feedback(item) for item in attempts],
                        float(_positive_int(policy.get('patch_review_timeout_s'), 90, 15, 300)),
                    )
                except (ModelCallError, ModelCallTimeout) as exc:
                    patch_review = {'status': 'failed', 'reason': type(exc).__name__}
                else:
                    patch_review = {
                        'status': 'passed' if reviewed.accepted else 'failed',
                        **reviewed.model_dump(),
                    }
                record_event(
                    'repair.patch_review_completed',
                    status='completed' if patch_review['status'] == 'passed' else 'failed',
                    attempt=attempt_no,
                    data={'reason': patch_review.get('reason'), 'issues': patch_review.get('issues')},
                )
                if patch_review['status'] != 'passed':
                    candidate = {
                        'status': 'rejected', 'accepted': False, 'reason': 'patch_review_failed',
                    }
                else:
                    for retry in range(infra_retries + 1):
                        candidate = await validate_candidate_patch(
                            root, diff_info['diff'], validation_ids, target_ids, category['metric_averages'],
                            case_map, baseline_map, eval_policy, candidate_config, ctx, attempt_no,
                        )
                        if not str(candidate.get('reason') or '').startswith('candidate_eval_stopped:'):
                            break
                        if retry < infra_retries:
                            record_event('candidate.validation_retried', status='started', attempt=attempt_no,
                                         data={'retry': retry + 1, 'reason': candidate.get('reason')})
        accepted = candidate.get('accepted') is True
        attempt = {
            'attempt': attempt_no,
            'status': 'validated' if accepted else 'failed',
            'opencode': {
                'returncode': run.returncode,
                'last_error': run.last_error,
                'finish_reason': run.finish_reason,
                'session_id': session_id,
            },
            'worker_report': report,
            'pre_validation': pre,
            'patch_review': patch_review,
            'candidate_validation': candidate,
            'workspace_ref': str(root),
            'files_changed': diff_info['files'],
            'diff': diff_info['diff'],
        }
        attempts.append(attempt)
        record_event('repair.attempt_completed', status='completed' if accepted else 'failed',
                     attempt=attempt_no, data={'reason': candidate.get('reason'), 'files': diff_info['files']})
        if accepted:
            return _result('validated', plan, workspace, attempts, attempt, 'validated repair patch', baseline_algo_id)
    reset_workspace(root)
    return _result('failed', plan, workspace, attempts, {}, 'repair_attempt_budget_exhausted', baseline_algo_id)


def build_verified_patch(run_id: str, loop: Mapping[str, Any]) -> dict[str, Any]:
    if loop.get('status') != 'validated':
        raise ValueError(
            f"repair did not produce a validated patch: {clean_text(loop.get('status')) or 'missing_status'}"
        )
    diff = str(loop.get('winning_patch_diff') or '')
    files = _diff_by_file(diff)
    workspace_ref = clean_text(loop.get('workspace_ref'))
    if not files or not workspace_ref:
        raise ValueError('validated repair patch requires diff and workspace_ref')
    return dump_contract(RepairPatch, {
        'run_id': run_id,
        'algo_id': clean_text(loop.get('algo_id')),
        'candidate_algo_id': clean_text(loop.get('candidate_algo_id')),
        'status': 'verified',
        'workspace_ref': workspace_ref,
        'diff': files,
    })


def _ready_workspace(workspace: Mapping[str, Any], plan: Mapping[str, Any], policy: Mapping[str, Any]
                     ) -> dict[str, str]:
    if workspace.get('status') != 'ready' or plan.get('status') != 'planned':
        return {'status': 'failed', 'reason': 'repair_plan_or_workspace_not_ready'}
    plan_hash = _digest({'category_id': plan.get('category_id'), 'method': plan.get('method')})
    try:
        root = Path(clean_text(workspace.get('workspace_ref'))).resolve()
        expected = workspace_path(policy, plan).resolve()
        fingerprint = workspace_fingerprint(root)
        head = git(root, 'rev-parse', '--verify', 'HEAD')
    except (OSError, RuntimeError, ValueError):
        return {'status': 'failed', 'reason': 'candidate_workspace_integrity_failed'}
    valid = (
        root == expected
        and fingerprint.get('source_hash') == workspace.get('source_hash')
        and fingerprint.get('source_dir') == workspace.get('source_dir')
        and fingerprint.get('objective_hash') == plan_hash
        and head == workspace.get('git_head')
        and (root / 'algorithm' / 'lazymind' / 'chat').exists()
    )
    return {'status': 'ready', 'reason': ''} if valid else {
        'status': 'failed', 'reason': 'candidate_workspace_integrity_failed',
    }


def _selected_category(analysis: Mapping[str, Any], plan: Mapping[str, Any]) -> Mapping[str, Any]:
    categories = analysis.get('categories') if isinstance(analysis.get('categories'), Mapping) else {}
    category = categories.get(plan.get('category_id')) if isinstance(categories, Mapping) else None
    return category if isinstance(category, Mapping) else {}


def _validation_cases(analysis: Mapping[str, Any], plan: Mapping[str, Any],
                      baseline: Mapping[str, Mapping[str, Any]], policy: Mapping[str, Any]) -> list[str]:
    categories = analysis.get('categories') if isinstance(analysis.get('categories'), Mapping) else {}
    selected_id = str(plan.get('category_id') or '')
    selected = categories.get(selected_id) if isinstance(categories.get(selected_id), Mapping) else {}
    target = list((selected.get('cases') or {}).keys())
    target_budget = _positive_int(policy.get('target_case_budget'), 3, 1, 12)
    target = target[:target_budget]
    cross = []
    cross_budget = _positive_int(policy.get('cross_block_guard_budget'), 2, 0, 12)
    for category_id in sorted(categories):
        if category_id == selected_id or not isinstance(categories[category_id], Mapping):
            continue
        cross.extend((categories[category_id].get('cases') or {}).keys())
        if len(cross) >= cross_budget:
            break
    good_budget = _positive_int(policy.get('goodcase_guard_budget'), 2, 0, 12)
    used = set(target) | set(cross[:cross_budget])
    good = [case_id for case_id, judge in sorted(baseline.items())
            if case_id not in used and judge.get('quality_label') == 'good'][:good_budget]
    return list(dict.fromkeys([*target, *cross[:cross_budget], *good]))


def _validation_input_gap(case_ids: list[str], cases: Mapping[str, Any], baseline: Mapping[str, Any]) -> str:
    if not case_ids:
        return 'repair_validation_cases_missing'
    missing_cases = [case_id for case_id in case_ids if case_id not in cases]
    missing_baseline = [case_id for case_id in case_ids if case_id not in baseline]
    if missing_cases:
        return f"repair_validation_cases_missing:{','.join(missing_cases[:5])}"
    if missing_baseline:
        return f"repair_baseline_judges_missing:{','.join(missing_baseline[:5])}"
    return ''


def _task_card(plan: Mapping[str, Any], category: Mapping[str, Any], workspace: Mapping[str, Any], attempt: int,
               report_path: Path, attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        'mode': 'lazyrag_formal_repair',
        'attempt': attempt,
        'root_cause': {
            'analysis': category.get('analysis'),
            'code_span': category.get('code_span'),
        },
        'verified_method': plan.get('method'),
        'demo_validation': plan.get('demo_validation'),
        'workspace': {'path': workspace.get('workspace_ref'), 'source_dir': workspace.get('source_dir')},
        'previous_attempts': [_attempt_feedback(item) for item in attempts[-3:]],
        'constraints': [
            'Implement the verified method; do not switch to a new root-cause hypothesis or repair method.',
            'A failed root-cause metric means the verified mechanism did not activate. Use candidate case evidence '
            'to trace every earlier operation that can raise the same failure before the intended recovery path.',
            'If a KB tool succeeds but returns zero items, use its query, kb_id, and trace observation to inspect '
            'the actual index mapping and request filters; do not misclassify an empty result as an exception.',
            'Edit only files listed in root_cause.code_span.',
            'Do not copy Demo files into the formal workspace.',
            'Do not hard-code case ids, trace ids, questions, answers, document ids, or metric thresholds.',
            'Leave a non-empty git diff and do not edit tests, eval data, secrets, or vendored dependencies.',
            f'Write the required JSON report to {report_path.as_posix()}.',
        ],
        'report_schema': {
            'status': 'edited',
            'files_changed': ['algorithm/lazymind/...'],
            'confirmed_locations': [{'path': '...', 'symbol': '...', 'observation': '...'}],
            'change_intent': 'how the diff implements verified_method',
            'risk': 'low|medium|high',
            'notes': '',
        },
    }


def _attempt_feedback(attempt: Mapping[str, Any]) -> dict[str, Any]:
    candidate = attempt.get('candidate_validation') if isinstance(attempt.get('candidate_validation'), Mapping) else {}
    pre = attempt.get('pre_validation') if isinstance(attempt.get('pre_validation'), Mapping) else {}
    review = attempt.get('patch_review') if isinstance(attempt.get('patch_review'), Mapping) else {}
    return {
        'attempt': attempt.get('attempt'),
        'files_changed': list(attempt.get('files_changed') or ()),
        'pre_validation': {'status': pre.get('status'), 'reason': pre.get('reason')},
        'patch_review': {
            'status': review.get('status'), 'reason': review.get('reason'), 'issues': review.get('issues'),
        },
        'candidate_validation': {
            'status': candidate.get('status'), 'reason': candidate.get('reason'),
            'mechanism_gate': candidate.get('mechanism_gate'),
            'score_gate': candidate.get('score_gate'), 'category_metrics': candidate.get('category_metrics'),
            'case_evidence': [
                {
                    'case_id': case_id,
                    **dict((candidate.get('candidate_answer_refs') or {}).get(case_id) or {}),
                    **dict((candidate.get('candidate_judge_refs') or {}).get(case_id) or {}),
                }
                for case_id in list(candidate.get('evaluated_case_ids') or ())[:3]
            ],
        },
    }


def _worker_failure(run: Any, report: Mapping[str, Any], diff_info: Mapping[str, Any]) -> str:
    if run.last_error:
        return str(run.last_error.get('type') or run.last_error.get('message') or 'opencode_failed')
    if run.returncode:
        return f'opencode_exit_{run.returncode}'
    if report.get('status') != 'completed':
        return str(report.get('reason') or 'worker_report_invalid')
    if sorted(report.get('files_changed') or ()) != sorted(diff_info.get('files') or ()):
        return 'worker_report_diff_mismatch'
    if not str(diff_info.get('diff') or '').strip():
        return 'empty_diff'
    return ''


def _result(status: str, plan: Mapping[str, Any], workspace: Mapping[str, Any],
            attempts: list[Mapping[str, Any]], winner: Mapping[str, Any], message: str,
            algo_id_value: str) -> dict[str, Any]:
    candidate = winner.get('candidate_validation') if isinstance(winner.get('candidate_validation'), Mapping) else {}
    service = candidate.get('service') if isinstance(candidate.get('service'), Mapping) else {}
    return {
        'id': 'repair.loop_result',
        'status': status,
        'message': message,
        'algo_id': algo_id_value,
        'attempt_count': len(attempts),
        'files_changed': list(winner.get('files_changed') or ()),
        'workspace_ref': clean_text(winner.get('workspace_ref')) or clean_text(workspace.get('workspace_ref')),
        'candidate_algo_id': clean_text(service.get('algorithm_id')),
        'winning_patch_diff': str(winner.get('diff') or ''),
        'category_id': plan.get('category_id'),
        'attempts': attempts,
    }


def _diff_by_file(diff: str) -> dict[str, str]:
    result = {}
    for patched in PatchSet(diff.splitlines(True)):
        source = clean_text(patched.source_file).removeprefix('a/')
        target = clean_text(patched.target_file).removeprefix('b/')
        path = source if target == '/dev/null' else target or source
        if path:
            result[path] = str(patched)
    return result


def _plan_ref(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'status': clean_text(plan.get('status')),
        'category_id': clean_text(plan.get('category_id')),
        'plan_hash': _digest({'category_id': plan.get('category_id'), 'method': plan.get('method')}),
    }


def _runtime_policy(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _llm_config(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    value = policy.get('llm_config')
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: object, default: int, minimum: int, maximum: int) -> int:
    number = value if isinstance(value, int) and not isinstance(value, bool) else default
    return min(max(number, minimum), maximum)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]
