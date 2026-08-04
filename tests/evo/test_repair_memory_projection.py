from __future__ import annotations

import copy
import json
import os

import pytest

from evo.operations.repair.memory import WorkMemory, content_ref
from evo.operations.repair.memory_projection import build_working_memory


def _target() -> dict:
    return {
        'category_id': 'category-1',
        'source_hash': 'a' * 64,
        'category': {
            'analysis': 'The retry result is discarded before response assembly.',
            'code_span': [{'path': 'algorithm/retry.py', 'symbol': 'retry'}],
            'cases': {'case-1': 'trace-1'},
            'metric_averages': {'accuracy': 0.4},
            'all_case_average_drop': 0.5,
        },
    }


def _memory(tmp_path, name='attempt-1', *, previous=(), history=None) -> WorkMemory:
    parent = tmp_path / 'run-1' / 'category-1' / 'target-1'
    artifact_root = parent / name
    work_root = tmp_path / f'{name}-work'
    artifact_root.mkdir(parents=True)
    (work_root / 'work').mkdir(parents=True)
    return WorkMemory(
        target=_target(),
        guidance=['先保留用户过滤条件', '如果冲突，以这条新意见为准'],
        scope={'allowed_roots': ['algorithm/']},
        work_root=work_root,
        artifact_root=artifact_root,
        previous_attempts=tuple(previous),
        history_attempts=None if history is None else tuple(history),
        source_digest='a' * 64,
    )


def _event_ref(number: int) -> dict[str, str]:
    return {
        'uri': f'phase1://run/category/target/attempt/events/{number}.json',
        'sha256': f'{number:064x}',
    }


def _result_ref(number: int) -> dict[str, str]:
    return {
        'uri': f'phase1://run/category/target/attempt/runs/{number}.json',
        'sha256': f'{number + 100:064x}',
    }


def _command_record(
    number: int,
    *,
    stdout_sha: str = 'b' * 64,
    status: str = 'completed',
    exit_code: int = 0,
    expected: str = 'prints the repaired result',
) -> dict:
    return {
        'event': 'command.result',
        'summary': f'{status}: python work/demo.py',
        'data': {
            'status': status,
            'command': ['python', 'work/demo.py'],
            'expected_result': expected,
            'workspace_before_sha256': 'c' * 64,
            'workspace_after_sha256': 'c' * 64,
            'exit_code': exit_code,
            'changed_files': [],
            'stdout_excerpt': 'verified output',
            'stderr_excerpt': '',
            'stdout_ref': {'uri': f'phase1://stdout/{number}', 'sha256': stdout_sha},
            'stderr_ref': {'uri': f'phase1://stderr/{number}', 'sha256': 'd' * 64},
            'result_ref': _result_ref(number),
            'duration_ms': number * 10,
        },
        '_event_ref': _event_ref(number),
    }


def _projection(records, *, section_budgets=None):
    return build_working_memory(
        target=_target(),
        guidance=['保留过滤条件', '新意见优先'],
        scope={'allowed_roots': ['algorithm/']},
        indexed_records=records,
        recent_records=records,
        workspace={'workspace_sha256': 'c' * 64},
        work_files=['work/demo.py'],
        counters={'turns': 2},
        budget={'turns': 20},
        core_refs={
            'root_cause': _result_ref(900),
            'guidance': _result_ref(901),
        },
        web_investigation={},
        section_budgets=section_budgets,
    )


def test_context_pins_root_cause_and_guidance_under_pressure(tmp_path) -> None:
    memory = _memory(tmp_path)
    for number in range(120):
        memory.record('model.failed', f'noise-{number}-' + 'x' * 1800, {'reason': 'noise'})

    first = memory.context({'turns': 120}, {'turns': 200})
    second = memory.context({'turns': 120}, {'turns': 200})

    assert first['schema_version'] == 2
    assert first['target'] == _target()
    assert first['user_guidance'] == ['先保留用户过滤条件', '如果冲突，以这条新意见为准']
    assert first['root_cause_ref']['uri'] in memory.registered_artifacts()
    assert first['guidance_history_ref']['uri'] in memory.registered_artifacts()
    assert 'recent_observations' not in first
    assert json.dumps(first, ensure_ascii=False, sort_keys=True) == json.dumps(
        second, ensure_ascii=False, sort_keys=True,
    )


def test_exact_duplicate_results_merge_without_mutating_raw_records() -> None:
    records = [_command_record(1), _command_record(2, expected='same expectation, rephrased')]
    original = copy.deepcopy(records)

    snapshot = _projection(records)

    assert len(snapshot['evidence_cards']) == 1
    assert snapshot['evidence_cards'][0]['occurrence_count'] == 2
    assert len(snapshot['investigation_ledger']) == 1
    assert snapshot['investigation_ledger'][0]['occurrence_count'] == 2
    assert snapshot['investigation_ledger'][0]['outcomes'][0]['occurrence_count'] == 2
    assert records == original


def test_same_output_on_different_workspace_results_is_not_collapsed() -> None:
    first = _command_record(1)
    second = _command_record(2)
    first['data']['workspace_after_sha256'] = '1' * 64
    second['data']['workspace_after_sha256'] = '2' * 64

    cards = _projection([first, second])['evidence_cards']

    assert len(cards) == 2
    assert {card['workspace_after_sha256'] for card in cards} == {'1' * 64, '2' * 64}


def test_conflicting_results_are_not_collapsed() -> None:
    records = [
        _command_record(1),
        _command_record(2, stdout_sha='e' * 64, status='nonzero_exit', exit_code=1),
    ]

    ledger = _projection(records)['investigation_ledger']

    assert len(ledger) == 1
    assert ledger[0]['occurrence_count'] == 2
    assert {item['status'] for item in ledger[0]['outcomes']} == {'completed', 'nonzero_exit'}


def test_noisy_investigation_keeps_bounded_outcome_summary() -> None:
    records = [
        _command_record(number, stdout_sha=f'{number:064x}')
        for number in range(1, 21)
    ]

    ledger = _projection(records)['investigation_ledger']

    assert len(ledger) == 1
    assert ledger[0]['occurrence_count'] == 20
    assert ledger[0]['distinct_outcome_count'] == 20
    assert ledger[0]['omitted_outcome_count'] == 17
    assert len(ledger[0]['outcomes']) == 3


def test_decisive_evidence_survives_tiny_budget_with_structural_fields() -> None:
    record = _command_record(1)
    record['data']['stdout_excerpt'] = 'begin-' + 'x' * 20_000 + '-tail-sentinel'
    selected_ref = record['data']['result_ref']
    records = [
        record,
        {
            'event': 'phase1.finished',
            'summary': 'supported',
            'data': {'evidence_refs': [selected_ref]},
            '_event_ref': _event_ref(2),
        },
    ]

    cards = _projection(records, section_budgets={'evidence_cards': 1})['evidence_cards']

    assert len(cards) == 1
    assert cards[0]['decisive'] is True
    assert cards[0]['status'] == 'completed'
    assert cards[0]['exit_code'] == 0
    assert cards[0]['workspace_before_sha256'] == 'c' * 64
    assert cards[0]['artifact_ref'] == selected_ref
    assert cards[0]['payload_refs']['stdout']['sha256'] == 'b' * 64
    assert cards[0]['key_observation']['truncated'] is True
    assert record['data']['stdout_excerpt'].endswith('-tail-sentinel')


def test_decisive_card_bounds_lists_and_preserves_revised_claims() -> None:
    first = _command_record(1, expected='first interpretation')
    second = _command_record(2, expected='revised interpretation')
    long_argument = 'a' * 2000
    command = ['python', *[long_argument for _ in range(31)]]
    changed_files = [f'work/{number:03d}-' + 'f' * 240 for number in range(100)]
    for record in (first, second):
        record['data']['command'] = command
        record['data']['changed_files'] = changed_files
    candidate_snapshot = _projection([first])
    assert len(candidate_snapshot['evidence_cards']) == 1
    assert candidate_snapshot['evidence_refs'] == [first['data']['result_ref']]
    assert len(
        json.dumps(candidate_snapshot['evidence_cards'][0], ensure_ascii=False).encode()
    ) <= 10_000
    records = [
        first,
        second,
        {
            'event': 'phase1.finished',
            'summary': 'supported',
            'data': {'evidence_refs': [second['data']['result_ref']]},
            '_event_ref': _event_ref(3),
        },
    ]

    snapshot = _projection(records, section_budgets={'evidence_cards': 1})
    card = snapshot['evidence_cards'][0]

    assert card['occurrence_count'] == 2
    assert card['claim'] == 'revised interpretation'
    assert card['previous_claims'] == ['first interpretation']
    assert card['command_truncated'] is True
    assert len(card['command']) == 8
    assert card['changed_files_truncated'] is True
    assert len(card['changed_files']) == 10
    assert len(json.dumps(card, ensure_ascii=False).encode()) < 10_000
    assert len(snapshot['investigation_ledger']) == 1


def test_large_opencode_request_does_not_drop_finding_or_ledger() -> None:
    report_ref = _result_ref(20)
    record = {
        'event': 'opencode.result',
        'summary': 'located the root cause',
        'data': {
            'status': 'completed',
            'reason': '',
            'instruction': 'inspect ' + 'x' * 20_000,
            'expected_result': 'locate the symbol',
            'workspace_before_sha256': 'c' * 64,
            'workspace_after_sha256': 'c' * 64,
            'report': {'summary': 'retry discards the final result'},
            'changed_files': [f'work/{number:03d}.py' for number in range(100)],
            'artifacts': {'report': report_ref},
        },
        '_event_ref': _event_ref(20),
    }

    snapshot = _projection([record])

    assert snapshot['findings'][0]['conclusion'] == 'retry discards the final result'
    assert snapshot['findings'][0]['changed_files_truncated'] is True
    assert len(snapshot['findings'][0]['subject']) <= 800
    assert len(snapshot['investigation_ledger']) == 1
    assert len(snapshot['investigation_ledger'][0]['request']['instruction']) <= 800


def test_workspace_revision_uses_content_not_mtime(tmp_path) -> None:
    memory = _memory(tmp_path)
    path = memory.work_root / 'work' / 'demo.py'
    path.write_text('print("one")\n', encoding='utf-8')
    first = memory.workspace_digest(refresh=True)

    os.utime(path, None)
    assert memory.workspace_digest(refresh=True) == first

    path.write_text('print("two")\n', encoding='utf-8')
    assert memory.workspace_digest(refresh=True) != first


def test_workspace_revision_has_unambiguous_file_boundaries(tmp_path) -> None:
    first = _memory(tmp_path / 'first')
    (first.work_root / 'work' / 'a').write_bytes(b'x\0b\0y')

    second = _memory(tmp_path / 'second')
    (second.work_root / 'work' / 'a').write_bytes(b'x')
    (second.work_root / 'work' / 'b').write_bytes(b'y')

    assert first.workspace_digest(refresh=True) != second.workspace_digest(refresh=True)


def test_old_decisive_evidence_survives_recent_event_limit(tmp_path) -> None:
    first = _memory(tmp_path, 'attempt-1')
    result_path = first.artifact_root / 'runs' / 'run-1.json'
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{"status":"completed"}\n', encoding='utf-8')
    ref = content_ref(result_path, first.artifact_root)
    workspace_sha256 = first.workspace_digest()
    first.record('command.result', 'verified result', {
        'status': 'completed',
        'command': ['python', 'work/demo.py'],
        'expected_result': 'verified',
        'workspace_before_sha256': workspace_sha256,
        'workspace_after_sha256': workspace_sha256,
        'exit_code': 0,
        'changed_files': [],
        'stdout_excerpt': 'verified',
        'stdout_ref': ref,
        'stderr_ref': ref,
        'result_ref': ref,
    })
    first.record('phase1.finished', 'supported', {'proposal': {}, 'evidence_refs': [ref]})
    for number in range(100):
        first.record('agent.decision', f'noise-{number}', {'turn': number})
    first.checkpoint('', 0)

    resumed = _memory(
        tmp_path,
        'attempt-2',
        previous=(first.artifact_root,),
        history=(first.artifact_root,),
    )
    cards = resumed.context({}, {})['evidence_cards']

    assert len(cards) == 1
    assert cards[0]['decisive'] is True
    assert cards[0]['artifact_ref'] == ref


def test_interrupted_attempt_restores_findings_but_not_session(tmp_path) -> None:
    interrupted = _memory(tmp_path, 'attempt-1')
    report_path = interrupted.artifact_root / 'opencode' / 'reports' / 'report.json'
    report_path.parent.mkdir(parents=True)
    report_path.write_text('{"summary":"retry discards the final result"}\n', encoding='utf-8')
    report_ref = content_ref(report_path, interrupted.artifact_root)
    interrupted.record('opencode.result', 'located retry result loss', {
        'status': 'completed',
        'reason': '',
        'instruction': 'locate the retry result loss',
        'expected_result': 'identify the responsible symbol',
        'workspace_before_sha256': 'c' * 64,
        'workspace_after_sha256': 'c' * 64,
        'report': {'summary': 'retry discards the final result'},
        'changed_files': [],
        'artifacts': {'report': report_ref},
    })

    resumed = _memory(
        tmp_path,
        'attempt-2',
        previous=(),
        history=(interrupted.artifact_root,),
    )

    assert resumed.restored_session == {}
    assert resumed.context({}, {})['findings'][0]['conclusion'] == (
        'retry discards the final result'
    )


def test_registered_artifact_read_is_integrity_checked_and_utf8_paged(tmp_path) -> None:
    memory = _memory(tmp_path)
    page_path = memory.artifact_root / 'web' / 'pages' / 'page.txt'
    page_path.parent.mkdir(parents=True)
    page_path.write_text('甲乙abc', encoding='utf-8')
    ref = content_ref(page_path, memory.artifact_root)
    memory.record('web.read', 'read page', {
        'status': 'completed',
        'question': 'content?',
        'pages': [{
            'status': 'readable',
            'url': 'https://example.com/page',
            'content_sha256': ref['sha256'],
            'content_ref': ref,
            'excerpt': '甲乙abc',
        }],
    })

    chunks = []
    offset = 0
    while True:
        result = memory.read_artifact(ref['uri'], offset_bytes=offset, max_bytes=4)
        chunks.append(result['content'])
        if not result['truncated']:
            break
        assert result['next_offset_bytes'] > offset
        offset = result['next_offset_bytes']
    assert ''.join(chunks) == '甲乙abc'

    with pytest.raises(ValueError, match='artifact_window_too_small'):
        memory.read_artifact(ref['uri'], max_bytes=1)
    with pytest.raises(ValueError, match='artifact_offset_not_utf8_boundary'):
        memory.read_artifact(ref['uri'], offset_bytes=1, max_bytes=4)

    guessed = memory.artifact_root / 'unregistered.txt'
    guessed.write_text('secret', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact_ref_not_registered'):
        memory.read_artifact(content_ref(guessed, memory.artifact_root)['uri'])

    page_path.write_text('tampered', encoding='utf-8')
    with pytest.raises(ValueError, match='artifact_integrity_mismatch'):
        memory.read_artifact(ref['uri'])


def test_previous_attempt_registered_artifact_can_be_read(tmp_path) -> None:
    first = _memory(tmp_path, 'attempt-1')
    result_path = first.artifact_root / 'runs' / 'result.json'
    result_path.parent.mkdir(parents=True)
    result_path.write_text('{"verified":true}\n', encoding='utf-8')
    ref = content_ref(result_path, first.artifact_root)
    first.record('http.result', 'GET endpoint -> 200', {
        'status': 'completed',
        'method': 'GET',
        'url': 'https://example.com/health',
        'status_code': 200,
        'body_excerpt': '{"verified":true}',
        'result_ref': ref,
    })

    resumed = _memory(
        tmp_path,
        'attempt-2',
        previous=(),
        history=(first.artifact_root,),
    )

    assert resumed.read_artifact(ref['uri'])['content'] == '{"verified":true}\n'
