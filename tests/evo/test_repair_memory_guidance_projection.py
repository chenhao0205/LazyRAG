from __future__ import annotations

import copy
from collections.abc import Mapping

from evo.operations.repair.memory_projection import build_working_memory, investigation_key


def _ref(number: int, kind: str = 'result') -> dict[str, str]:
    return {
        'uri': f'phase1://run/category/target/attempt/{kind}/{number}.json',
        'sha256': f'{number:064x}',
    }


def _provenance(revision_id: str, *directive_ids: str) -> dict[str, object]:
    return {
        'guidance_revision_id': revision_id,
        'active_directive_ids': list(directive_ids),
    }


def _record(
    event: str,
    number: int,
    data: Mapping[str, object],
    *,
    revision_id: str = '',
    directive_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        'event': event,
        'summary': f'{event}-{number}',
        'data': dict(data),
        '_event_ref': _ref(number, 'events'),
        **(
            {'provenance': _provenance(revision_id, *directive_ids)}
            if revision_id or directive_ids else {}
        ),
    }


def _command(number: int, revision_id: str = '', *directive_ids: str) -> dict[str, object]:
    return _record(
        'command.result',
        number,
        {
            'status': 'completed',
            'command': ['python', 'work/demo.py', str(number)],
            'expected_result': 'prints repaired output',
            'workspace_before_sha256': 'a' * 64,
            'workspace_after_sha256': 'a' * 64,
            'exit_code': 0,
            'changed_files': [],
            'stdout_excerpt': 'verified',
            'stdout_ref': _ref(number, 'stdout'),
            'stderr_ref': _ref(number, 'stderr'),
            'result_ref': _ref(number),
        },
        revision_id=revision_id,
        directive_ids=directive_ids,
    )


def _http(number: int, revision_id: str = '', *directive_ids: str) -> dict[str, object]:
    return _record(
        'http.result',
        number,
        {
            'status': 'completed',
            'method': 'GET',
            'url': f'http://127.0.0.1:{8000 + number}/health',
            'status_code': 200,
            'body_excerpt': 'ok',
            'result_ref': _ref(number),
        },
        revision_id=revision_id,
        directive_ids=directive_ids,
    )


def _opencode(number: int, revision_id: str = '', *directive_ids: str) -> dict[str, object]:
    return _record(
        'opencode.result',
        number,
        {
            'status': 'completed',
            'instruction': f'inspect symbol {number}',
            'expected_result': 'locate the repair point',
            'workspace_before_sha256': 'a' * 64,
            'workspace_after_sha256': 'a' * 64,
            'report': {'summary': 'located the repair point'},
            'changed_files': [],
            'artifacts': {'report': _ref(number, 'reports')},
        },
        revision_id=revision_id,
        directive_ids=directive_ids,
    )


def _web_read(number: int, revision_id: str = '', *directive_ids: str) -> dict[str, object]:
    return _record(
        'web.read',
        number,
        {
            'status': 'completed',
            'question': 'how does the API behave',
            'pages': [{
                'status': 'readable',
                'url': f'https://example.com/{number}',
                'canonical_url': f'https://example.com/{number}',
                'title': 'API documentation',
                'excerpt': 'documented behavior',
                'content_sha256': f'{number + 100:064x}',
                'content_ref': _ref(number, 'web'),
            }],
        },
        revision_id=revision_id,
        directive_ids=directive_ids,
    )


def _artifact_read(number: int, revision_id: str = '', *directive_ids: str) -> dict[str, object]:
    ref = _ref(number, 'payload')
    return _record(
        'artifact.read',
        number,
        {
            'status': 'completed',
            'uri': ref['uri'],
            'sha256': ref['sha256'],
            'offset_bytes': 0,
            'max_bytes': 4096,
            'returned_bytes': 8,
            'content': 'evidence',
            'artifact_ref': ref,
        },
        revision_id=revision_id,
        directive_ids=directive_ids,
    )


def _finish(number: int, revision_id: str, *evidence_refs: Mapping[str, str]) -> dict[str, object]:
    return _record(
        'phase1.finished',
        number,
        {
            'guidance_revision_id': revision_id,
            'evidence_refs': [dict(ref) for ref in evidence_refs],
        },
        revision_id=revision_id,
    )


def _guidance_state(revision_id: str = 'gr-new') -> dict[str, object]:
    return {
        'revision_id': revision_id,
        'parent_revision_id': 'gr-old',
        'effect': 'append',
        'active_content_hash': 'f' * 64,
        'active_directives': [
            {
                'directive_id': 'gd-old',
                'text': '保留原有过滤条件',
                'introduced_revision_id': 'gr-old',
            },
            {
                'directive_id': 'gd-new',
                'text': '增加空结果保护',
                'introduced_revision_id': revision_id,
            },
        ],
        'delta': {
            'added_ids': ['gd-new'],
            'superseded_ids': [],
            'withdrawn_ids': [],
        },
    }


def _projection(
    records: list[Mapping[str, object]],
    guidance_state: Mapping[str, object] | None,
) -> dict[str, object]:
    return build_working_memory(
        target={'category_id': 'category-1'},
        guidance=['legacy guidance must not leak'],
        guidance_state=guidance_state,
        scope={'allowed_roots': ['algorithm/']},
        indexed_records=records,
        recent_records=records,
        workspace={'workspace_sha256': 'a' * 64},
        work_files=[],
        counters={'turns': 1},
        budget={'turns': 20},
        core_refs={
            'root_cause': _ref(900),
            'guidance': _ref(901),
        },
        web_investigation={},
    )


def _by_kind(cards: object, kind: str) -> dict[str, object]:
    assert isinstance(cards, list)
    return next(card for card in cards if card['kind'] == kind)


def test_guidance_state_projects_active_text_once_with_lineage_metadata() -> None:
    guidance_state = _guidance_state()

    snapshot = _projection([], guidance_state)

    assert snapshot['user_guidance'] == ['保留原有过滤条件', '增加空结果保护']
    projected = snapshot['guidance_state']
    assert projected['revision_id'] == guidance_state['revision_id']
    assert projected['active_directive_count'] == 2
    assert [item['directive_id'] for item in projected['active_directives']] == [
        'gd-old', 'gd-new',
    ]
    assert all('text' not in item for item in projected['active_directives'])


def test_legacy_projection_keeps_existing_current_and_decisive_semantics() -> None:
    command = _command(1)
    records = [command, _finish(2, '', command['data']['result_ref'])]

    snapshot = _projection(records, None)

    assert snapshot['user_guidance'] == ['legacy guidance must not leak']
    assert snapshot['evidence_cards'][0]['applicability'] == 'current'
    assert snapshot['evidence_cards'][0]['decisive'] is True
    assert snapshot['evidence_refs'] == [command['data']['result_ref']]
    assert 'guidance_state' not in snapshot


def test_old_code_and_runtime_records_require_revalidation() -> None:
    records = [
        _command(1, 'gr-old', 'gd-old'),
        _http(2, 'gr-old', 'gd-old'),
        _opencode(3, 'gr-old', 'gd-old'),
    ]

    snapshot = _projection(records, _guidance_state())

    assert {
        card['kind']: card['applicability'] for card in snapshot['evidence_cards']
    } == {
        'command_result': 'needs_revalidation',
        'http_result': 'needs_revalidation',
    }
    assert _by_kind(snapshot['findings'], 'code_research')['applicability'] == 'needs_revalidation'
    assert {
        card['kind']: card['applicability'] for card in snapshot['investigation_ledger']
    } == {
        'command.result': 'needs_revalidation',
        'http.result': 'needs_revalidation',
        'opencode.result': 'needs_revalidation',
    }
    assert snapshot['evidence_refs'] == []


def test_old_web_and_artifact_observations_are_reusable() -> None:
    records = [
        _web_read(1, 'gr-old', 'gd-old'),
        _artifact_read(2, 'gr-old', 'gd-old'),
    ]

    snapshot = _projection(records, _guidance_state())

    assert _by_kind(snapshot['findings'], 'web_evidence')['applicability'] == 'reusable'
    assert _by_kind(snapshot['findings'], 'artifact_observation')['applicability'] == 'reusable'
    assert {
        card['kind']: card['applicability'] for card in snapshot['investigation_ledger']
    } == {
        'web.read': 'reusable',
        'artifact.read': 'reusable',
    }


def test_removed_directive_makes_old_records_superseded() -> None:
    state = _guidance_state()
    state['active_directives'] = [{
        'directive_id': 'gd-new',
        'text': '使用完全不同的新方向',
        'introduced_revision_id': 'gr-new',
    }]
    records = [
        _command(1, 'gr-old', 'gd-removed'),
        _web_read(2, 'gr-old', 'gd-removed'),
        _artifact_read(3, 'gr-old', 'gd-removed'),
    ]

    snapshot = _projection(records, state)

    assert snapshot['evidence_cards'][0]['applicability'] == 'superseded'
    assert {card['applicability'] for card in snapshot['findings']} == {'superseded'}
    assert {card['applicability'] for card in snapshot['investigation_ledger']} == {'superseded'}
    assert snapshot['evidence_refs'] == []


def test_only_current_revision_finish_can_make_current_evidence_decisive() -> None:
    old_command = _command(1, 'gr-old', 'gd-old')
    current_command = _command(2, 'gr-new', 'gd-old', 'gd-new')
    records = [
        old_command,
        _finish(3, 'gr-old', old_command['data']['result_ref']),
        current_command,
        _finish(4, 'gr-new', current_command['data']['result_ref']),
    ]

    snapshot = _projection(records, _guidance_state())
    cards = {card['artifact_ref']['uri']: card for card in snapshot['evidence_cards']}

    old = cards[old_command['data']['result_ref']['uri']]
    current = cards[current_command['data']['result_ref']['uri']]
    assert old['applicability'] == 'needs_revalidation'
    assert old['decisive'] is False
    assert current['applicability'] == 'current'
    assert current['decisive'] is True
    assert snapshot['evidence_refs'] == [current_command['data']['result_ref']]


def test_current_finish_cannot_promote_old_unrevalidated_evidence() -> None:
    old_command = _command(1, 'gr-old', 'gd-old')
    records = [
        old_command,
        _finish(2, 'gr-new', old_command['data']['result_ref']),
    ]

    snapshot = _projection(records, _guidance_state())

    assert snapshot['evidence_cards'][0]['applicability'] == 'needs_revalidation'
    assert snapshot['evidence_cards'][0]['decisive'] is False
    assert snapshot['evidence_refs'] == []


def test_merged_current_outcome_exposes_current_attempt_uri() -> None:
    old = _command(1, 'gr-old', 'gd-old')
    current = copy.deepcopy(old)
    current['provenance'] = _provenance('gr-new', 'gd-old', 'gd-new')
    current['_event_ref'] = _ref(2, 'events')
    current['data']['result_ref'] = _ref(2)

    snapshot = _projection([old, current], _guidance_state())

    assert len(snapshot['evidence_cards']) == 1
    assert snapshot['evidence_cards'][0]['applicability'] == 'current'
    assert snapshot['evidence_cards'][0]['artifact_ref'] == current['data']['result_ref']
    assert snapshot['evidence_refs'] == [current['data']['result_ref']]


def test_current_revision_command_becomes_stale_after_workspace_change() -> None:
    current = _command(1, 'gr-new', 'gd-old', 'gd-new')
    current['data']['workspace_after_sha256'] = 'b' * 64

    snapshot = _projection([current], _guidance_state())

    assert snapshot['evidence_cards'][0]['applicability'] == 'needs_revalidation'
    assert snapshot['evidence_refs'] == []


def test_latest_failed_revalidation_removes_old_success_from_evidence() -> None:
    succeeded = _command(1, 'gr-new', 'gd-old', 'gd-new')
    failed = copy.deepcopy(succeeded)
    failed['_event_ref'] = _ref(2, 'events')
    failed['summary'] = 'independent revalidation failed'
    failed['data']['status'] = 'failed'
    failed['data'].pop('result_ref')

    snapshot = _projection([succeeded, failed], _guidance_state())

    assert snapshot['evidence_cards'] == []
    assert snapshot['evidence_refs'] == []
    ledger = _by_kind(snapshot['investigation_ledger'], 'command.result')
    assert ledger['outcome_status_counts'] == {'completed': 1, 'failed': 1}


def test_opencode_investigation_key_is_bound_to_guidance_revision() -> None:
    common = {
        'instruction': 'inspect retry handling',
        'expected_result': 'locate the repair point',
        'workspace_before_sha256': 'a' * 64,
    }

    assert investigation_key(
        'opencode.result', {**common, 'guidance_revision_id': 'gr-old'},
    ) != investigation_key(
        'opencode.result', {**common, 'guidance_revision_id': 'gr-new'},
    )


def test_mixed_revision_ledger_prefers_current_occurrence() -> None:
    old = _web_read(1, 'gr-old', 'gd-old')
    current = _web_read(2, 'gr-new', 'gd-old', 'gd-new')
    current['data']['pages'][0]['url'] = old['data']['pages'][0]['url']
    current['data']['pages'][0]['canonical_url'] = old['data']['pages'][0]['canonical_url']
    current['data']['pages'][0]['content_sha256'] = old['data']['pages'][0]['content_sha256']
    current['data']['pages'][0]['content_ref'] = old['data']['pages'][0]['content_ref']

    snapshot = _projection([old, current], _guidance_state())
    ledger = snapshot['investigation_ledger']

    assert len(ledger) == 1
    assert ledger[0]['applicability'] == 'current'
    assert ledger[0]['outcomes'][0]['applicability'] == 'current'
    assert ledger[0]['occurrence_count'] == 2
