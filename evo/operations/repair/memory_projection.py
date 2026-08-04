from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SECTION_BUDGETS = {
    'evidence_cards': 10_000,
    'findings': 10_000,
    'investigation_ledger': 6_000,
    'recent_activity': 4_000,
}


def build_working_memory(
    *,
    target: Mapping[str, Any],
    guidance: Sequence[str],
    scope: Mapping[str, Any],
    indexed_records: Sequence[Mapping[str, Any]],
    recent_records: Sequence[Mapping[str, Any]],
    workspace: Mapping[str, Any],
    work_files: Sequence[str],
    counters: Mapping[str, int],
    budget: Mapping[str, int],
    core_refs: Mapping[str, Mapping[str, str]],
    web_investigation: Mapping[str, Any],
    guidance_state: Mapping[str, Any] | None = None,
    section_budgets: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, bounded prompt view without changing raw events.

    The root cause and user guidance are outside the dynamic section budgets. Raw
    event and payload artifacts remain append-only; this function only projects
    them into compact cards for the next model turn.
    """
    limits = {**DEFAULT_SECTION_BUDGETS, **dict(section_budgets or {})}
    current_revision_id = _guidance_revision_id(guidance_state)
    current_workspace_sha256 = str(workspace.get('workspace_sha256') or '')
    active_directive_ids, active_directives_known = _active_directive_ids(guidance_state)
    decisive_uris = _decisive_uris(indexed_records, current_revision_id)
    evidence_cards = _select_cards(
        _evidence_cards(
            indexed_records,
            decisive_uris,
            current_revision_id,
            current_workspace_sha256,
            active_directive_ids,
            active_directives_known,
        ),
        _positive_limit(limits.get('evidence_cards')),
        keep_mandatory=True,
    )
    result = {
        'schema_version': 2,
        # Compatibility fields retained for the Agent and OpenCode task card.
        'target': _copy_mapping(target),
        'user_guidance': _active_guidance_texts(guidance, guidance_state),
        'repair_scope': _copy_mapping(scope),
        'workspace': _copy_mapping(workspace),
        'evidence_refs': [
            dict(card['artifact_ref'])
            for card in evidence_cards
            if isinstance(card.get('artifact_ref'), Mapping)
            and card.get('applicability') == 'current'
        ],
        'web_investigation': _copy_mapping(web_investigation),
        'work_files': list(work_files),
        'budget': dict(budget),
        'used': dict(counters),
        # V2 structured memory sections.
        'root_cause_ref': dict(core_refs.get('root_cause') or {}),
        'guidance_history_ref': dict(core_refs.get('guidance') or {}),
        'evidence_cards': evidence_cards,
        'findings': _select_cards(
            _finding_cards(
                indexed_records,
                current_revision_id,
                current_workspace_sha256,
                active_directive_ids,
                active_directives_known,
            ),
            _positive_limit(limits.get('findings')),
        ),
        'investigation_ledger': _select_cards(
            _investigation_cards(
                indexed_records,
                current_revision_id,
                current_workspace_sha256,
                active_directive_ids,
                active_directives_known,
            ),
            _positive_limit(limits.get('investigation_ledger')),
        ),
        'recent_activity': _select_cards(
            _recent_cards(recent_records),
            _positive_limit(limits.get('recent_activity')),
        ),
    }
    if guidance_state is not None:
        result['guidance_state'] = _project_guidance_state(guidance_state)
    return result


def investigation_key(event: str, data: Mapping[str, Any]) -> str:
    """Identify one investigation request, not its outcome."""
    supplied = str(data.get('investigation_key') or '').strip()
    if supplied:
        return supplied
    return f'{event}:{_stable_hash(_investigation_identity(event, data))}'


def _investigation_identity(event: str, data: Mapping[str, Any]) -> dict[str, Any]:
    if event == 'opencode.result':
        return {
            'instruction': _normalize_text(data.get('instruction')),
            'expected_result': _normalize_text(data.get('expected_result')),
            'workspace_before_sha256': str(data.get('workspace_before_sha256') or ''),
            'guidance_revision_id': str(data.get('guidance_revision_id') or ''),
        }
    if event == 'command.result':
        # expected_result explains an observation but cannot affect execution.
        return {
            'command': list(data.get('command') or ()),
            'workspace_before_sha256': str(data.get('workspace_before_sha256') or ''),
        }
    if event == 'web.search':
        return {'query': _normalize_text(data.get('query')).casefold()}
    if event == 'web.read':
        return {
            'question': _normalize_text(data.get('question')).casefold(),
            'urls': _web_read_urls(data),
        }
    if event == 'http.result':
        return {
            'method': str(data.get('method') or 'GET').upper(),
            'url': str(data.get('url') or ''),
            'workspace_sha256': str(data.get('workspace_sha256') or ''),
        }
    if event == 'artifact.read':
        return {
            'uri': str(data.get('uri') or ''),
            'offset_bytes': _integer(data.get('offset_bytes')),
            'max_bytes': _integer(data.get('max_bytes')),
        }
    return {'event': event}


def _web_read_urls(data: Mapping[str, Any]) -> list[str]:
    supplied = data.get('requested_urls')
    if isinstance(supplied, (list, tuple)):
        return sorted({str(url) for url in supplied if str(url)})
    return sorted({
        str(page.get('requested_url') or page.get('url') or '')
        for page in data.get('pages') or ()
        if isinstance(page, Mapping)
        and str(page.get('requested_url') or page.get('url') or '')
    })


def _investigation_request(event: str, data: Mapping[str, Any]) -> dict[str, Any]:
    if event == 'opencode.result':
        return {
            'instruction': _clip_text(_normalize_text(data.get('instruction')), 600),
            'expected_result': _clip_text(_normalize_text(data.get('expected_result')), 400),
            'workspace_before_sha256': str(data.get('workspace_before_sha256') or ''),
            'guidance_revision_id': str(data.get('guidance_revision_id') or ''),
        }
    if event == 'command.result':
        command = list(data.get('command') or ())
        return {
            'command': [_clip_text(item, 120) for item in command[:6]],
            'command_truncated': len(command) > 6 or any(len(str(item)) > 120 for item in command),
            'expected_result': _clip_text(_normalize_text(data.get('expected_result')), 400),
            'workspace_before_sha256': str(data.get('workspace_before_sha256') or ''),
        }
    if event == 'web.search':
        return {'query': _clip_text(_normalize_text(data.get('query')).casefold(), 500)}
    if event == 'web.read':
        return {
            'question': _clip_text(_normalize_text(data.get('question')).casefold(), 500),
            'urls': [_clip_text(url, 350) for url in _web_read_urls(data)[:3]],
        }
    if event == 'http.result':
        return {
            'method': str(data.get('method') or 'GET').upper(),
            'url': _clip_text(data.get('url'), 1000),
            'workspace_sha256': str(data.get('workspace_sha256') or ''),
        }
    if event == 'artifact.read':
        return {
            'uri': _clip_text(data.get('uri'), 1000),
            'offset_bytes': _integer(data.get('offset_bytes')),
            'max_bytes': _integer(data.get('max_bytes')),
        }
    return {'event': event}


def _outcome_identity(event: str, data: Mapping[str, Any]) -> str:
    if event == 'command.result':
        stdout_sha256 = _ref_sha(data.get('stdout_ref'))
        value = {
            'command': list(data.get('command') or ()),
            'workspace_before_sha256': str(data.get('workspace_before_sha256') or ''),
            'workspace_after_sha256': str(data.get('workspace_after_sha256') or ''),
            'status': data.get('status'),
            'exit_code': data.get('exit_code'),
            'changed_files': sorted(str(item) for item in data.get('changed_files') or ()),
            'stdout_sha256': stdout_sha256,
            'stderr_sha256': _ref_sha(data.get('stderr_ref')),
            'output_fallback': data.get('output') if not stdout_sha256 else None,
        }
    elif event == 'http.result':
        value = {
            'method': str(data.get('method') or 'GET').upper(),
            'url': str(data.get('url') or ''),
            'workspace_sha256': str(data.get('workspace_sha256') or ''),
            'status': data.get('status'),
            'status_code': data.get('status_code'),
            'body_sha256': _stable_hash(str(data.get('body_excerpt') or '')),
            'error_type': str(data.get('error_type') or ''),
        }
    elif event == 'opencode.result':
        report = data.get('report') if isinstance(data.get('report'), Mapping) else {}
        value = {
            'instruction': _normalize_text(data.get('instruction')),
            'workspace_before_sha256': str(data.get('workspace_before_sha256') or ''),
            'workspace_after_sha256': str(data.get('workspace_after_sha256') or ''),
            'status': data.get('status'),
            'reason': data.get('reason'),
            'summary': _normalize_text(report.get('summary')),
            'changed_files': sorted(str(item) for item in data.get('changed_files') or ()),
            'report_sha256': _named_ref_sha(data.get('artifacts'), 'report'),
        }
    elif event == 'web.search':
        value = {
            'status': data.get('status'),
            'urls': [
                str(item.get('canonical_url') or item.get('url') or '')
                for item in data.get('results') or ()
                if isinstance(item, Mapping)
            ],
        }
    elif event == 'web.read':
        value = {
            'pages': [
                {
                    'status': page.get('status'),
                    'url': page.get('canonical_url') or page.get('url'),
                    'content_sha256': page.get('content_sha256') or _ref_sha(page.get('content_ref')),
                }
                for page in data.get('pages') or ()
                if isinstance(page, Mapping)
            ],
        }
    elif event == 'artifact.read':
        value = {
            'status': data.get('status'),
            'sha256': data.get('sha256'),
            'offset_bytes': data.get('offset_bytes'),
            'returned_bytes': data.get('returned_bytes'),
            'content_sha256': _stable_hash(str(data.get('content') or '')),
        }
    else:
        value = {'status': data.get('status')}
    return _stable_hash(value)


def _evidence_cards(
    records: Sequence[Mapping[str, Any]],
    decisive_uris: set[str],
    current_revision_id: str,
    current_workspace_sha256: str,
    active_directive_ids: set[str],
    active_directives_known: bool,
) -> list[dict[str, Any]]:
    latest_status: dict[tuple[str, str], str] = {}
    for record in records:
        event = str(record.get('event') or '')
        if event not in {'command.result', 'http.result'}:
            continue
        applicability = _record_applicability(
            record,
            current_revision_id,
            current_workspace_sha256,
            active_directive_ids,
            active_directives_known,
        )
        if applicability != 'superseded':
            data = _record_data(record)
            latest_status[(event, investigation_key(event, data))] = str(
                data.get('status') or ''
            )
    cards: dict[tuple[str, str], dict[str, Any]] = {}
    for order, record in enumerate(records):
        event = str(record.get('event') or '')
        if event not in {'command.result', 'http.result'}:
            continue
        data = _record_data(record)
        if data.get('status') != 'completed':
            continue
        applicability = _record_applicability(
            record,
            current_revision_id,
            current_workspace_sha256,
            active_directive_ids,
            active_directives_known,
        )
        if (
            applicability != 'superseded'
            and latest_status.get((event, investigation_key(event, data))) != 'completed'
        ):
            continue
        result_ref = _content_ref(data.get('result_ref'))
        if not result_ref:
            continue
        outcome_id = _outcome_identity(event, data)
        key = (event, outcome_id)
        event_ref = _event_ref(record)
        provenance = _project_provenance(record)
        decisive_ref = (
            result_ref
            if applicability == 'current' and result_ref.get('uri') in decisive_uris
            else {}
        )
        card = cards.get(key)
        if card is None:
            card = _new_evidence_card(
                event,
                data,
                record,
                result_ref,
                decisive_ref,
                event_ref,
                outcome_id,
                applicability,
                provenance,
                order,
            )
            cards[key] = card
            continue
        card['occurrence_count'] += 1
        card['latest_event_ref'] = event_ref
        card['_order'] = order
        _merge_claim(card, _evidence_claim(event, data, record))
        _merge_applicability(card, applicability, provenance)
        if applicability == 'current':
            card['artifact_ref'] = dict(result_ref)
        if decisive_ref:
            card['decisive'] = True
            card['artifact_ref'] = decisive_ref
            card['_mandatory'] = True
            card['_priority'] = 100
    return list(cards.values())


def _new_evidence_card(
    event: str,
    data: Mapping[str, Any],
    record: Mapping[str, Any],
    result_ref: Mapping[str, str],
    decisive_ref: Mapping[str, str],
    event_ref: Mapping[str, str],
    outcome_id: str,
    applicability: str,
    provenance: Mapping[str, Any],
    order: int,
) -> dict[str, Any]:
    claim = _evidence_claim(event, data, record)
    if event == 'command.result':
        command = list(data.get('command') or ())
        changed_files = list(data.get('changed_files') or ())
        observation = data.get('output')
        if observation is None:
            observation = '\n'.join(filter(None, (
                str(data.get('stdout_excerpt') or ''),
                str(data.get('stderr_excerpt') or ''),
            )))
        details = {
            'command': [_clip_text(item, 120) for item in command[:8]],
            'command_truncated': len(command) > 8 or any(len(str(item)) > 120 for item in command),
            'exit_code': data.get('exit_code'),
            'changed_files': [_clip_text(item, 160) for item in changed_files[:10]],
            'changed_files_truncated': len(changed_files) > 10,
            'payload_refs': _named_refs(data, ('stdout_ref', 'stderr_ref')),
        }
        kind = 'command_result'
    else:
        observation = data.get('body_excerpt')
        details = {
            'method': str(data.get('method') or ''),
            'url': _clip_text(data.get('url'), 1200),
            'status_code': data.get('status_code'),
            'payload_refs': {},
        }
        kind = 'http_result'
    decisive = bool(decisive_ref)
    return {
        'evidence_id': f'evidence-{outcome_id[:20]}',
        'kind': kind,
        'claim': _clip_text(claim, 600),
        'previous_claims': [],
        'claim_count': 1 if claim else 0,
        'omitted_claim_count': 0,
        'status': 'completed',
        'decisive': decisive,
        'applicability': applicability,
        'provenance': dict(provenance),
        'key_observation': _clip_value(observation, 800),
        **details,
        'workspace_before_sha256': str(
            data.get('workspace_before_sha256') or data.get('workspace_sha256') or ''
        ),
        'workspace_after_sha256': str(
            data.get('workspace_after_sha256') or data.get('workspace_sha256') or ''
        ),
        'artifact_ref': dict(decisive_ref or result_ref),
        'first_event_ref': dict(event_ref),
        'latest_event_ref': dict(event_ref),
        'occurrence_count': 1,
        '_priority': 100 if decisive else 70,
        '_order': order,
        '_mandatory': decisive,
        '_claim_ids': {_stable_hash(_normalize_text(claim))} if claim else set(),
        '_first_claim': _clip_text(claim, 600),
    }


def _evidence_claim(
    event: str,
    data: Mapping[str, Any],
    record: Mapping[str, Any],
) -> str:
    if event == 'command.result':
        return _normalize_text(data.get('expected_result')) or str(record.get('summary') or '')
    return str(record.get('summary') or '')


def _merge_claim(card: dict[str, Any], claim: str) -> None:
    text = _clip_text(claim, 600)
    if not text:
        return
    identity = _stable_hash(_normalize_text(claim))
    if identity in card['_claim_ids']:
        return
    card['_claim_ids'].add(identity)
    card['claim_count'] += 1
    card['claim'] = text
    first = card['_first_claim']
    card['previous_claims'] = [first] if first and first != text else []
    card['omitted_claim_count'] = card['claim_count'] - 1 - len(card['previous_claims'])


def _finding_cards(
    records: Sequence[Mapping[str, Any]],
    current_revision_id: str,
    current_workspace_sha256: str,
    active_directive_ids: set[str],
    active_directives_known: bool,
) -> list[dict[str, Any]]:
    cards: dict[str, dict[str, Any]] = {}
    for order, record in enumerate(records):
        event = str(record.get('event') or '')
        data = _record_data(record)
        findings = _record_findings(event, data, record)
        applicability = _record_applicability(
            record,
            current_revision_id,
            current_workspace_sha256,
            active_directive_ids,
            active_directives_known,
        )
        provenance = _project_provenance(record)
        for finding in findings:
            changed_files = list(finding['changed_files'])
            identity = _stable_hash({
                'kind': finding['kind'],
                'status': finding['status'],
                'conclusion': _normalize_text(finding['conclusion']),
                'changed_files': finding['changed_files'],
                'content_sha256': finding['content_sha256'],
            })
            event_ref = _event_ref(record)
            card = cards.get(identity)
            if card is None:
                cards[identity] = {
                    'finding_id': f'finding-{identity[:20]}',
                    'kind': finding['kind'],
                    'status': finding['status'],
                    'applicability': applicability,
                    'provenance': dict(provenance),
                    'subject': _clip_text(finding['subject'], 800),
                    'conclusion': _clip_text(finding['conclusion'], 2400),
                    'changed_files': [_clip_text(item, 200) for item in changed_files[:10]],
                    'changed_files_truncated': len(changed_files) > 10,
                    'artifact_refs': finding['artifact_refs'],
                    'first_event_ref': dict(event_ref),
                    'latest_event_ref': dict(event_ref),
                    'occurrence_count': 1,
                    '_priority': finding['priority'],
                    '_order': order,
                }
            else:
                _repeat(cards[identity], event_ref, order)
                _merge_applicability(cards[identity], applicability, provenance)
    return list(cards.values())


def _record_findings(
    event: str,
    data: Mapping[str, Any],
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if event == 'opencode.result':
        report = data.get('report') if isinstance(data.get('report'), Mapping) else {}
        return [{
            'kind': 'code_research',
            'status': str(data.get('status') or ''),
            'subject': _normalize_text(data.get('instruction')),
            'conclusion': str(
                report.get('summary') or data.get('reason') or record.get('summary') or ''
            ),
            'changed_files': list(data.get('changed_files') or ()),
            'artifact_refs': _artifact_refs(data.get('artifacts')),
            'content_sha256': _named_ref_sha(data.get('artifacts'), 'report'),
            'priority': 65,
        }]
    if event == 'web.read':
        return [
            {
                'kind': (
                    'web_evidence'
                    if page.get('status') == 'readable'
                    else 'web_near_duplicate'
                ),
                'status': str(page.get('status') or ''),
                'subject': str(page.get('canonical_url') or page.get('url') or ''),
                'conclusion': str(page.get('excerpt') or page.get('title') or ''),
                'changed_files': [],
                'artifact_refs': _artifact_refs(page.get('content_ref')),
                'content_sha256': str(
                    page.get('content_sha256') or _ref_sha(page.get('content_ref'))
                ),
                'priority': 50,
            }
            for page in data.get('pages') or ()
            if isinstance(page, Mapping)
            and (
                page.get('status') == 'readable'
                or (
                    page.get('status') == 'duplicate'
                    and page.get('duplicate_kind') == 'near'
                )
            )
        ]
    if event == 'artifact.read' and data.get('status') == 'completed':
        return [{
            'kind': 'artifact_observation',
            'status': 'completed',
            'subject': str(data.get('uri') or ''),
            'conclusion': str(data.get('content') or ''),
            'changed_files': [],
            'artifact_refs': _artifact_refs(data.get('artifact_ref')),
            'content_sha256': _stable_hash(str(data.get('content') or '')),
            'priority': 45,
        }]
    return []


def _investigation_cards(
    records: Sequence[Mapping[str, Any]],
    current_revision_id: str,
    current_workspace_sha256: str,
    active_directive_ids: set[str],
    active_directives_known: bool,
) -> list[dict[str, Any]]:
    supported = {
        'opencode.result', 'command.result', 'web.search', 'web.read',
        'http.result', 'artifact.read',
    }
    cards: dict[str, dict[str, Any]] = {}
    for order, record in enumerate(records):
        event = str(record.get('event') or '')
        if event not in supported:
            continue
        data = _record_data(record)
        key = investigation_key(event, data)
        outcome_id = _outcome_identity(event, data)
        event_ref = _event_ref(record)
        applicability = _record_applicability(
            record,
            current_revision_id,
            current_workspace_sha256,
            active_directive_ids,
            active_directives_known,
        )
        provenance = _project_provenance(record)
        card = cards.get(key)
        if card is None:
            card = {
                'investigation_key': key,
                'kind': event,
                'applicability': applicability,
                'provenance': dict(provenance),
                'request': _investigation_request(event, data),
                'outcomes': [],
                'occurrence_count': 0,
                '_outcomes_by_id': {},
                '_status_counts': {},
                '_priority': 40,
                '_order': order,
            }
            cards[key] = card
        outcome = card['_outcomes_by_id'].get(outcome_id)
        if outcome is None:
            outcome = {
                'outcome_id': outcome_id,
                'status': str(data.get('status') or ''),
                'applicability': applicability,
                'provenance': dict(provenance),
                'summary': _clip_text(str(record.get('summary') or ''), 300),
                'latest_event_ref': dict(event_ref),
                'occurrence_count': 1,
                '_order': order,
            }
            card['outcomes'].append(outcome)
            card['_outcomes_by_id'][outcome_id] = outcome
        else:
            outcome['occurrence_count'] += 1
            outcome['latest_event_ref'] = dict(event_ref)
            outcome['_order'] = order
            _merge_applicability(outcome, applicability, provenance)
        status = str(data.get('status') or '')
        card['_status_counts'][status] = card['_status_counts'].get(status, 0) + 1
        card['occurrence_count'] += 1
        card['_order'] = order
        _merge_applicability(card, applicability, provenance)
    return [_finalize_investigation_card(card) for card in cards.values()]


def _finalize_investigation_card(card: dict[str, Any]) -> dict[str, Any]:
    all_outcomes = sorted(
        card['_outcomes_by_id'].values(),
        key=lambda item: (
            -_applicability_priority(str(item.get('applicability') or '')),
            -int(item.get('_order') or 0),
            str(item.get('outcome_id') or ''),
        ),
    )
    selected = []
    selected_statuses = set()
    for outcome in all_outcomes:
        status = str(outcome.get('status') or '')
        if status in selected_statuses:
            continue
        selected.append(outcome)
        selected_statuses.add(status)
        if len(selected) == 3:
            break
    for outcome in all_outcomes:
        if outcome in selected:
            continue
        selected.append(outcome)
        if len(selected) == 3:
            break
    card['outcomes'] = [
        {key: value for key, value in outcome.items() if not key.startswith('_')}
        for outcome in selected
    ]
    card['distinct_outcome_count'] = len(all_outcomes)
    card['omitted_outcome_count'] = len(all_outcomes) - len(selected)
    card['outcome_status_counts'] = {
        key: card['_status_counts'][key]
        for key in sorted(card['_status_counts'])
    }
    return card


def _recent_cards(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            'event': str(record.get('event') or ''),
            'status': str(_record_data(record).get('status') or ''),
            'summary': _clip_text(str(record.get('summary') or ''), 500),
            'event_ref': _event_ref(record),
            '_priority': 10,
            '_order': order,
        }
        for order, record in enumerate(records[-80:])
    ]


def _decisive_uris(
    records: Sequence[Mapping[str, Any]],
    current_revision_id: str,
) -> set[str]:
    return {
        str(ref.get('uri') or '')
        for record in records
        if record.get('event') == 'phase1.finished'
        and (
            not current_revision_id
            or _record_revision_id(record) == current_revision_id
        )
        for ref in _record_data(record).get('evidence_refs') or ()
        if isinstance(ref, Mapping) and str(ref.get('uri') or '')
    }


def _project_guidance_state(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose lineage metadata without duplicating active directive text.

    Exact active text is already present once in ``user_guidance`` and remains
    available in ``guidance_history_ref``. Keeping a second normalized copy here
    could otherwise make a valid policy consume hundreds of thousands of tokens.
    """
    directives = value.get('active_directives')
    active = [
        {
            'directive_id': str(item.get('directive_id') or ''),
            'source_message_id': str(item.get('source_message_id') or ''),
            'introduced_revision_id': str(item.get('introduced_revision_id') or ''),
        }
        for item in (directives if isinstance(directives, (list, tuple)) else ())
        if isinstance(item, Mapping)
    ]
    delta = value.get('delta')
    return {
        'schema_version': value.get('schema_version'),
        'revision_id': str(value.get('revision_id') or ''),
        'parent_revision_id': str(value.get('parent_revision_id') or ''),
        'effect': str(value.get('effect') or ''),
        'active_content_hash': str(value.get('active_content_hash') or ''),
        'active_directive_count': len(active),
        'active_directives': active,
        'delta': _copy_mapping(delta) if isinstance(delta, Mapping) else {},
    }


def _guidance_revision_id(guidance_state: Mapping[str, Any] | None) -> str:
    if not isinstance(guidance_state, Mapping):
        return ''
    return str(guidance_state.get('revision_id') or '').strip()


def _active_guidance_texts(
    legacy_guidance: Sequence[str],
    guidance_state: Mapping[str, Any] | None,
) -> list[str]:
    if isinstance(guidance_state, Mapping) and 'active_directives' in guidance_state:
        result = []
        directives = guidance_state.get('active_directives')
        for directive in directives if isinstance(directives, (list, tuple)) else ():
            if isinstance(directive, Mapping):
                text = str(directive.get('text') or '').strip()
            else:
                text = str(directive or '').strip()
            if text:
                result.append(text)
        return result
    return [str(item).strip() for item in legacy_guidance if str(item).strip()]


def _active_directive_ids(
    guidance_state: Mapping[str, Any] | None,
) -> tuple[set[str], bool]:
    if not isinstance(guidance_state, Mapping) or 'active_directives' not in guidance_state:
        return set(), False
    directives = guidance_state.get('active_directives')
    values = directives if isinstance(directives, (list, tuple)) else ()
    result = {
        str(directive.get('directive_id') or '').strip()
        for directive in values
        if isinstance(directive, Mapping)
        and str(directive.get('directive_id') or '').strip()
    }
    return result, True


def _record_provenance(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get('provenance')
    return value if isinstance(value, Mapping) else {}


def _record_revision_id(record: Mapping[str, Any]) -> str:
    provenance = _record_provenance(record)
    revision_id = str(provenance.get('guidance_revision_id') or '').strip()
    if revision_id:
        return revision_id
    # Accept the finish-event field during the transition to top-level
    # provenance, but new producers should write provenance consistently.
    return str(_record_data(record).get('guidance_revision_id') or '').strip()


def _record_directive_ids(record: Mapping[str, Any]) -> set[str]:
    value = _record_provenance(record).get('active_directive_ids')
    values = value if isinstance(value, (list, tuple)) else ()
    return {
        str(item).strip()
        for item in values
        if str(item).strip()
    }


def _project_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    revision_id = _record_revision_id(record)
    directive_ids = sorted(_record_directive_ids(record))
    return {
        **({'guidance_revision_id': revision_id} if revision_id else {}),
        **({'active_directive_ids': directive_ids} if directive_ids else {}),
    }


def _record_applicability(
    record: Mapping[str, Any],
    current_revision_id: str,
    current_workspace_sha256: str,
    active_directive_ids: set[str],
    active_directives_known: bool,
) -> str:
    # With no Revision model, retain the exact legacy behavior: every record is
    # part of the current prompt view and old finish selections remain decisive.
    if not current_revision_id:
        return 'current'
    record_revision_id = _record_revision_id(record)
    if record_revision_id == current_revision_id:
        return _workspace_applicability(record, current_workspace_sha256)
    record_directive_ids = _record_directive_ids(record)
    if (
        active_directives_known
        and record_directive_ids
        and not record_directive_ids.issubset(active_directive_ids)
    ):
        return 'superseded'
    event = str(record.get('event') or '')
    if event in {'web.search', 'web.read', 'artifact.read'}:
        return 'reusable'
    return 'needs_revalidation'


def _workspace_applicability(
    record: Mapping[str, Any],
    current_workspace_sha256: str,
) -> str:
    if not current_workspace_sha256:
        return 'current'
    event = str(record.get('event') or '')
    data = _record_data(record)
    if event in {'command.result', 'opencode.result'}:
        return (
            'current'
            if str(data.get('workspace_after_sha256') or '') == current_workspace_sha256
            else 'needs_revalidation'
        )
    if event == 'http.result':
        return (
            'current'
            if str(data.get('workspace_sha256') or '') == current_workspace_sha256
            else 'needs_revalidation'
        )
    return 'current'


def _merge_applicability(
    card: dict[str, Any],
    applicability: str,
    provenance: Mapping[str, Any],
) -> None:
    current = str(card.get('applicability') or '')
    if _applicability_priority(applicability) < _applicability_priority(current):
        return
    card['applicability'] = applicability
    card['provenance'] = dict(provenance)


def _applicability_priority(value: str) -> int:
    return {
        'superseded': 0,
        'needs_revalidation': 1,
        'reusable': 2,
        'current': 3,
    }.get(value, -1)


def _select_cards(
    cards: Sequence[Mapping[str, Any]],
    limit: int,
    *,
    keep_mandatory: bool = False,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(card) for card in cards),
        key=lambda card: (
            -int(card.get('_priority') or 0),
            -int(card.get('_order') or 0),
            _card_id(card),
        ),
    )
    selected: list[dict[str, Any]] = []
    used = 0
    for card in ranked:
        public = {key: value for key, value in card.items() if not key.startswith('_')}
        size = len(_canonical_json(public).encode('utf-8'))
        mandatory = keep_mandatory and bool(card.get('_mandatory'))
        if not mandatory and used + size > limit:
            continue
        selected.append(public)
        used += size
    return selected


def _repeat(card: dict[str, Any], event_ref: Mapping[str, str], order: int) -> None:
    card['occurrence_count'] += 1
    card['latest_event_ref'] = dict(event_ref)
    card['_order'] = order


def _artifact_refs(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Mapping):
        items = value if isinstance(value, (list, tuple)) else (value,)
    elif value.get('uri'):
        items = (value,)
    else:
        items = tuple(value[key] for key in sorted(value))
    refs: dict[str, dict[str, str]] = {}
    for item in items:
        ref = _content_ref(item)
        if ref:
            refs.setdefault(ref['uri'], ref)
    return list(refs.values())


def _named_refs(data: Mapping[str, Any], names: Sequence[str]) -> dict[str, dict[str, str]]:
    return {
        name.removesuffix('_ref'): ref
        for name in names
        if (ref := _content_ref(data.get(name)))
    }


def _content_ref(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    uri = str(value.get('uri') or '')
    digest = str(value.get('sha256') or '')
    return {'uri': uri, **({'sha256': digest} if digest else {})} if uri else {}


def _ref_sha(value: object) -> str:
    return str(value.get('sha256') or '') if isinstance(value, Mapping) else ''


def _named_ref_sha(value: object, name: str) -> str:
    if not isinstance(value, Mapping):
        return ''
    return _ref_sha(value.get(name))


def _event_ref(record: Mapping[str, Any]) -> dict[str, str]:
    return _content_ref(record.get('_event_ref'))


def _record_data(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get('data')
    return value if isinstance(value, Mapping) else {}


def _clip_value(value: object, limit: int) -> dict[str, Any]:
    text = value if isinstance(value, str) else _canonical_json(value)
    clipped = _clip_text(text, limit)
    return {'text': clipped, 'truncated': len(text) > len(clipped)}


def _clip_text(value: object, limit: int) -> str:
    text = str(value or '')
    return text if len(text) <= limit else text[:limit - 1].rstrip() + '…'


def _normalize_text(value: object) -> str:
    return ' '.join(str(value or '').split())


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode('utf-8')).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(_canonical_json(value))


def _positive_limit(value: object) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _integer(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _card_id(card: Mapping[str, Any]) -> str:
    return str(
        card.get('evidence_id')
        or card.get('finding_id')
        or card.get('investigation_key')
        or card.get('event')
        or ''
    )


__all__ = ['DEFAULT_SECTION_BUDGETS', 'build_working_memory', 'investigation_key']
