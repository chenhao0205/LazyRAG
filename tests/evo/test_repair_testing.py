from __future__ import annotations

import json
import sys
from pathlib import Path

from evo.operations.repair.contracts import RepairAction, RepairInput
from evo.operations.repair.testing import RepairTestCapability, RepairTestPlan, build_test_plan
from evo.operations.repair.workspace import initialize_workspace


def _source(tmp_path: Path) -> Path:
    source = tmp_path / 'source'
    (source / 'algorithm/lazymind/chat').mkdir(parents=True)
    (source / 'algorithm/lazymind/chat/module.py').write_text('VALUE = 1\n', encoding='utf-8')
    return source


def _plan(tmp_path: Path, commands: tuple[tuple[str, ...], ...]) -> tuple[RepairTestPlan, object]:
    source = _source(tmp_path)
    repair_input = RepairInput(
        run_id='testing', objective='repair', guidance='', source_ref=str(source), case_scope='algorithm/lazymind/chat',
        constraints={}, budget={'turns': 5, 'seconds': 60},
    )
    paths = initialize_workspace(repair_input, tmp_path / 'runtime')
    cases = {
        'c1': {'id': 'c1', 'question': 'q1'},
        'c2': {'id': 'c2', 'question': 'q2'},
        'c3': {'id': 'c3', 'question': 'q3'},
    }
    plan = RepairTestPlan(
        run_id='testing', target_category='retrieval', l1_case_ids=('c1', 'c2'), l2_case_ids=('c1', 'c2', 'c3'),
        cases=cases, baseline_judges={case_id: {'case_id': case_id} for case_id in cases},
        eval_policy={}, candidate_config={}, l0_commands=commands,
    )
    return plan, paths


def test_build_test_plan_freezes_category_and_full_case_sets() -> None:
    cases = {case_id: {'id': case_id} for case_id in ('c1', 'c2', 'c3')}
    judges = {case_id: {'case_id': case_id} for case_id in cases}
    plan = build_test_plan(
        'run',
        {
            'repair_group_queue': [{'issue_category': 'retrieval'}],
            'case_ids': ['c1', 'c2', 'c3'],
            'rows': [
                {'case_id': 'c1', 'issue_category': 'retrieval'},
                {'case_id': 'c2', 'issue_category': 'retrieval'},
                {'case_id': 'c3', 'issue_category': 'ok'},
            ],
        },
        cases,
        judges,
        {},
        {'router_chat_url': 'http://chat', 'router_admin_url': 'http://chat'},
    )

    assert plan.l1_case_ids == ('c1', 'c2')
    assert plan.l2_case_ids == ('c1', 'c2', 'c3')
    assert len(plan.l0_commands) == 2
    assert plan.candidate_config['thread_id'] == 'run'


def test_l0_runs_runtime_commands_and_rejects_workspace_mutation(tmp_path: Path) -> None:
    success_plan, paths = _plan(tmp_path / 'success', ((sys.executable, '-c', 'print("ok")'),))
    success = RepairTestCapability(success_plan, paths)(RepairAction('l0-ok', 'test', {'level': 'L0'}))
    assert success.status == 'success'
    evidence = json.loads(Path(success.artifact_refs[0]).read_text(encoding='utf-8'))
    assert evidence['status'] == 'success'

    mutation = 'from pathlib import Path; Path("algorithm/lazymind/chat/module.py").write_text("BROKEN")'
    mutation_plan, mutation_paths = _plan(
        tmp_path / 'mutation', ((sys.executable, '-c', mutation),),
    )
    before = (mutation_paths.source / 'algorithm/lazymind/chat/module.py').read_text(encoding='utf-8')
    failed = RepairTestCapability(mutation_plan, mutation_paths)(
        RepairAction('l0-mutation', 'test', {'level': 'L0'}),
    )
    assert failed.status == 'fail'
    assert (mutation_paths.source / 'algorithm/lazymind/chat/module.py').read_text(encoding='utf-8') == before


def test_l1_l2_use_frozen_case_sets_and_cleanup(monkeypatch, tmp_path: Path) -> None:
    plan, paths = _plan(tmp_path, ((sys.executable, '-c', 'pass'),))
    tested = []

    monkeypatch.setattr(
        'evo.operations.repair.testing.candidate_service',
        lambda *args, **kwargs: {
            'status': 'ready', 'algorithm_id': 'evo_tmp_test', 'healthcheck': {'status': 'passed'},
        },
    )
    monkeypatch.setattr(
        'evo.operations.repair.testing.discard_candidate',
        lambda *args, **kwargs: {'status': 'completed'},
    )

    async def evaluate(cases, service, policy):
        tested.append(tuple(case['id'] for case in cases))
        return [{'case_id': case['id']} for case in cases]

    monkeypatch.setattr('evo.operations.repair.testing._evaluate_cases', evaluate)
    monkeypatch.setattr('evo.operations.repair.testing.build_eval_summary_root', lambda *args, **kwargs: {})
    monkeypatch.setattr(
        'evo.operations.repair.testing.compare_abtest',
        lambda *args, **kwargs: {'verdict': 'accept', 'reasons': [], 'delta': {'overall': 0.5}},
    )
    monkeypatch.setattr('evo.operations.repair.testing.build_eval_detail_summary', lambda *args, **kwargs: {})
    monkeypatch.setattr(
        'evo.operations.repair.testing.compare_eval_detail_for_repair',
        lambda *args, **kwargs: {'case_deltas': [], 'candidate': {}},
    )
    capability = RepairTestCapability(plan, paths)

    l1 = capability(RepairAction('l1', 'test', {'level': 'L1'}))
    l2 = capability(RepairAction('l2', 'test', {'level': 'L2'}))

    assert l1.status == l2.status == 'success'
    assert tested == [('c1', 'c2'), ('c1', 'c2', 'c3')]
    assert json.loads(Path(l1.artifact_refs[0]).read_text(encoding='utf-8'))['cleanup_status'] == 'completed'
