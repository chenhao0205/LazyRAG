from __future__ import annotations

import json
from pathlib import Path

import pytest

import evo.repair_guidance as guidance_module
from evo.operations.repair.memory import WorkMemory
from evo.operations.repair.source import source_hash
from evo.repair_guidance import active_guidance, apply_guidance_transition, guidance_snapshot


def _source(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / 'candidate'
    app = root / 'algorithm' / 'lazymind' / 'chat' / 'app.py'
    app.parent.mkdir(parents=True)
    app.write_text('APPLICATION = object()\n', encoding='utf-8')
    return root, source_hash(root)


def _target(digest: str) -> dict[str, object]:
    return {
        'category_id': 'category-1',
        'source_hash': digest,
        'category': {'analysis': 'verified root cause'},
    }


def _policy(tmp_path: Path, *guidance: str) -> dict[str, object]:
    return {
        'phase1_artifact_dir': str(tmp_path / 'artifacts'),
        'user_guidance': list(guidance),
    }


def _create(
    tmp_path: Path,
    source: Path,
    digest: str,
    policy: dict[str, object],
) -> WorkMemory:
    return WorkMemory.create(
        'run-1',
        _target(digest),
        policy,
        source,
        digest,
        {'allowed_roots': ['algorithm/']},
    )


def _checkpoint_with_marker(memory: WorkMemory, value: str = 'base') -> None:
    marker = memory.work_root / 'work' / 'marker.txt'
    marker.write_text(value, encoding='utf-8')
    memory.workspace_digest(refresh=True)
    memory.checkpoint('session-1', 3)


def test_same_revision_restores_workspace_and_verified_session(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, policy)
    _checkpoint_with_marker(first)
    first.close()

    resumed = _create(tmp_path, source, digest, policy)

    assert resumed.recovery['mode'] == 'same_revision_resume'
    assert (resumed.work_root / 'work' / 'marker.txt').read_text() == 'base'
    assert resumed.restored_session['session_id'] == 'session-1'
    assert resumed.restored_session['calls'] == 3
    resumed.close()


def test_append_forks_workspace_but_resets_hidden_session(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    initial = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, initial)
    _checkpoint_with_marker(first)
    first.close()
    appended = apply_guidance_transition(
        initial,
        effect='append',
        message='同时处理空结果',
        source_message_id='message-2',
    ).policy

    resumed = _create(tmp_path, source, digest, appended)

    assert resumed.recovery['mode'] == 'fork_compatible'
    assert (resumed.work_root / 'work' / 'marker.txt').read_text() == 'base'
    assert resumed.restored_session == {}
    resumed.close()


def test_forged_append_that_drops_parent_does_not_restore_workspace(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    initial = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, initial)
    _checkpoint_with_marker(first)
    first.close()
    parent = guidance_snapshot(initial)
    message = '完全不同的新方向'
    message_id = 'message-forged'
    normalized = guidance_module.normalize_guidance_text(message)
    directive_id = guidance_module._directive_id(
        parent_revision_id=parent['revision_id'],
        source_message_id=message_id,
        effect='append',
        normalized_text=normalized,
        target_directive_ids=(),
    )
    directive = {
        'directive_id': directive_id,
        'text': message,
        'normalized_text': normalized,
        'source_message_id': message_id,
        'introduced_revision_id': '',
    }
    active_hash = guidance_module._active_content_hash([directive])
    revision_id = guidance_module._prefixed_hash('gr', {
        'schema_version': guidance_module.GUIDANCE_SCHEMA_VERSION,
        'parent_revision_id': parent['revision_id'],
        'source_message_id': message_id,
        'effect': 'append',
        'target_directive_ids': [],
        'active_content_hash': active_hash,
        'active_directive_ids': [directive_id],
    })
    directive['introduced_revision_id'] = revision_id
    forged = {
        **_policy(tmp_path, message),
        'guidance_state': {
            'schema_version': guidance_module.GUIDANCE_SCHEMA_VERSION,
            'revision_id': revision_id,
            'parent_revision_id': parent['revision_id'],
            'source_message_id': message_id,
            'effect': 'append',
            'active_content_hash': active_hash,
            'active_directives': [directive],
            'delta': {
                'added_ids': [directive_id],
                'superseded_ids': [],
                'withdrawn_ids': [],
            },
        },
    }

    assert guidance_snapshot(forged)['effect'] == 'append'
    resumed = _create(tmp_path, source, digest, forged)

    assert resumed.recovery['mode'] == 'reset_for_new_direction'
    assert not (resumed.work_root / 'work' / 'marker.txt').exists()
    resumed.close()


def test_replace_resets_workspace_and_session(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    initial = _policy(tmp_path, '修改 web.py')
    first = _create(tmp_path, source, digest, initial)
    _checkpoint_with_marker(first)
    first.close()
    replaced = apply_guidance_transition(
        initial,
        effect='replace',
        message='不要修改 web.py，只检查 memory.py',
        source_message_id='message-2',
    ).policy

    resumed = _create(tmp_path, source, digest, replaced)

    assert resumed.recovery['mode'] == 'reset_for_new_direction'
    assert not (resumed.work_root / 'work' / 'marker.txt').exists()
    assert resumed.restored_session == {}
    resumed.close()


def test_tampered_checkpoint_is_not_restored(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, policy)
    _checkpoint_with_marker(first)
    checkpoint_work = first.artifact_root / 'checkpoint' / 'work' / 'marker.txt'
    checkpoint_work.write_text('tampered', encoding='utf-8')
    first.close()

    resumed = _create(tmp_path, source, digest, policy)

    assert resumed.recovery['restore_workspace'] is False
    assert not (resumed.work_root / 'work' / 'marker.txt').exists()
    assert resumed.restored_session == {}
    resumed.close()


def test_same_revision_id_with_different_active_hash_is_rejected(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    initial = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, initial)
    _checkpoint_with_marker(first)
    first.close()
    initial_state = guidance_snapshot(initial)
    forged_state = guidance_snapshot({'user_guidance': ['完全不同的新方向']})
    forged_state['revision_id'] = initial_state['revision_id']
    forged = {
        **_policy(tmp_path, '完全不同的新方向'),
        'guidance_state': forged_state,
    }

    with pytest.raises(ValueError, match='revision_id mismatch'):
        _create(tmp_path, source, digest, forged)


def test_withdraw_restores_matching_ancestor_workspace(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    initial = _policy(tmp_path, '保留用户过滤条件')
    base = _create(tmp_path, source, digest, initial)
    _checkpoint_with_marker(base, 'ancestor')
    base.close()

    appended = apply_guidance_transition(
        initial,
        effect='append',
        message='尝试删除过滤条件',
        source_message_id='message-2',
    )
    fork = _create(tmp_path, source, digest, appended.policy)
    _checkpoint_with_marker(fork, 'new-direction')
    fork.close()
    added_id = appended.state['delta']['added_ids'][0]
    withdrawn = apply_guidance_transition(
        appended.policy,
        effect='withdraw',
        message='撤回删除过滤条件的要求',
        target_directive_ids=[added_id],
        source_message_id='message-3',
    ).policy

    restored = _create(tmp_path, source, digest, withdrawn)

    assert restored.recovery['mode'] == 'restore_ancestor'
    assert (restored.work_root / 'work' / 'marker.txt').read_text() == 'ancestor'
    assert restored.restored_session == {}
    restored.close()


def test_old_direction_is_not_reused_after_replace(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    initial = _policy(tmp_path, '联网搜索官方文档')
    old_state = guidance_snapshot(initial)
    old = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(initial),
        guidance_state=old_state,
        scope={},
        work_root=tmp_path / 'old-work',
        artifact_root=tmp_path / 'old-attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (old.work_root / 'work').mkdir(parents=True)
    old.record('web.search', 'searched old direction', {
        'status': 'completed',
        'query': 'old direction',
        'results': [{'url': 'https://example.com/old'}],
    })
    event_path = next((old.artifact_root / 'events').glob('*.json'))
    event_payload = json.loads(event_path.read_text(encoding='utf-8'))
    event_payload['provenance'] = {}
    event_path.write_text(json.dumps(event_payload), encoding='utf-8')
    replaced = apply_guidance_transition(
        initial,
        effect='replace',
        message='只检查本地代码，不要联网',
        source_message_id='message-2',
    ).policy
    new = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(replaced),
        guidance_state=guidance_snapshot(replaced),
        scope={},
        work_root=tmp_path / 'new-work',
        artifact_root=tmp_path / 'new-attempt',
        previous_attempts=(),
        history_attempts=(old.artifact_root,),
        source_digest=digest,
    )
    (new.work_root / 'work').mkdir(parents=True)

    assert new.completed_investigation(
        'web.search', {'status': 'completed', 'query': 'old direction'},
    ) is None
    assert not new.has_searched_query('old direction')
    assert new.known_urls() == set()


def test_malformed_interrupted_guidance_is_audit_only(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    old = WorkMemory(
        target=_target(digest),
        guidance=['旧方向'],
        scope={},
        work_root=tmp_path / 'old-work',
        artifact_root=tmp_path / 'old-attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (old.work_root / 'work').mkdir(parents=True)
    old.record('web.search', 'old result', {
        'status': 'completed',
        'query': 'old direction',
        'results': [{'url': 'https://example.com/old'}],
    })
    (old.artifact_root / 'memory' / 'guidance.json').write_text(
        '{"guidance_state": []}\n', encoding='utf-8',
    )

    current = WorkMemory(
        target=_target(digest),
        guidance=['当前方向'],
        scope={},
        work_root=tmp_path / 'current-work',
        artifact_root=tmp_path / 'current-attempt',
        previous_attempts=(),
        history_attempts=(old.artifact_root,),
        source_digest=digest,
    )

    assert current.completed_investigation(
        'web.search', {'status': 'completed', 'query': 'old direction'},
    ) is None
    assert current.known_urls() == set()


def test_append_can_reuse_facts_but_not_old_decisive_evidence(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    initial = _policy(tmp_path, '保留用户过滤条件')
    old_state = guidance_snapshot(initial)
    old = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(initial),
        guidance_state=old_state,
        scope={},
        work_root=tmp_path / 'old-work',
        artifact_root=tmp_path / 'old-attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (old.work_root / 'work').mkdir(parents=True)
    result_ref = old.record('command.result', 'verified old direction', {
        'status': 'completed',
        'command': ['python', 'work/demo.py'],
        'workspace_before_sha256': old.workspace_digest(),
        'workspace_after_sha256': old.workspace_digest(),
        'result_ref': {'uri': 'phase1://old/result.json', 'sha256': 'f' * 64},
    })
    old.record('phase1.finished', 'old finish', {
        'evidence_refs': [{'uri': 'phase1://old/result.json', 'sha256': 'f' * 64}],
    })
    appended = apply_guidance_transition(
        initial,
        effect='append',
        message='同时处理空结果',
        source_message_id='message-2',
    ).policy
    new = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(appended),
        guidance_state=guidance_snapshot(appended),
        scope={},
        work_root=tmp_path / 'new-work',
        artifact_root=tmp_path / 'new-attempt',
        previous_attempts=(),
        history_attempts=(old.artifact_root,),
        source_digest=digest,
    )
    (new.work_root / 'work').mkdir(parents=True)
    source_record = new.completed_investigation('command.result', {
        'status': 'completed',
        'command': ['python', 'work/demo.py'],
        'workspace_before_sha256': new.workspace_digest(),
    })

    assert source_record is not None
    assert source_record['_event_ref'] == result_ref
    assert new.evidence_refs() == []
    new.record_investigation_reuse('command.result', source_record)
    new.record_investigation_reuse('command.result', source_record)
    assert new.evidence_refs() == []
    assert len((new.artifact_root / 'journal.jsonl').read_text().splitlines()) == 1


def test_workspace_change_invalidates_current_command_evidence(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    memory = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(policy),
        guidance_state=guidance_snapshot(policy),
        scope={},
        work_root=tmp_path / 'work-root',
        artifact_root=tmp_path / 'attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (memory.work_root / 'work').mkdir(parents=True)
    verified_workspace = memory.workspace_digest()
    evidence = {'uri': 'phase1://current/result.json', 'sha256': 'f' * 64}
    memory.record('command.result', 'verified current workspace', {
        'status': 'completed',
        'command': ['python', 'work/demo.py'],
        'workspace_before_sha256': verified_workspace,
        'workspace_after_sha256': verified_workspace,
        'result_ref': evidence,
    })
    http_evidence = {'uri': 'phase1://current/http.json', 'sha256': 'e' * 64}
    memory.record('http.result', 'verified current service', {
        'status': 'completed',
        'method': 'GET',
        'url': 'http://127.0.0.1:8000/health',
        'workspace_sha256': verified_workspace,
        'result_ref': http_evidence,
    })
    assert memory.evidence_refs() == [evidence, http_evidence]

    (memory.work_root / 'work' / 'new-patch.py').write_text('changed = True\n', encoding='utf-8')
    memory.workspace_digest(refresh=True)

    assert memory.evidence_refs() == []


def test_latest_failed_revalidation_blocks_old_reuse_and_evidence(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    memory = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(policy),
        guidance_state=guidance_snapshot(policy),
        scope={},
        work_root=tmp_path / 'work-root',
        artifact_root=tmp_path / 'attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (memory.work_root / 'work').mkdir(parents=True)
    workspace = memory.workspace_digest()
    request = {
        'command': ['python', 'work/demo.py'],
        'workspace_before_sha256': workspace,
    }
    memory.record('command.result', 'initial validation passed', {
        **request,
        'status': 'completed',
        'workspace_after_sha256': workspace,
        'result_ref': {'uri': 'phase1://current/pass.json', 'sha256': 'a' * 64},
    })
    memory.record('command.result', 'forced validation failed', {
        **request,
        'status': 'failed',
        'workspace_after_sha256': workspace,
        'force_rerun': True,
        'rerun_reason': 'independent_revalidation',
    })

    assert memory.completed_investigation('command.result', request) is None
    assert memory.evidence_refs() == []


def test_state_changing_command_is_not_reused_without_replaying_workspace(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    memory = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(policy),
        guidance_state=guidance_snapshot(policy),
        scope={},
        work_root=tmp_path / 'work-root',
        artifact_root=tmp_path / 'attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (memory.work_root / 'work').mkdir(parents=True)
    before = memory.workspace_digest()
    request = {'command': ['python', 'work/create-demo.py'], 'workspace_before_sha256': before}
    memory.record('command.result', 'created a file', {
        **request,
        'status': 'completed',
        'workspace_after_sha256': 'b' * 64,
        'changed_files': ['work/generated.py'],
        'result_ref': {'uri': 'phase1://current/result.json', 'sha256': 'c' * 64},
    })

    assert memory.completed_investigation('command.result', request) is None


@pytest.mark.parametrize(
    'missing_field',
    ['workspace_sha256', 'guidance_revision_id', 'active_content_hash'],
)
def test_incomplete_checkpoint_never_restores_workspace(
    tmp_path: Path,
    missing_field: str,
) -> None:
    source, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, policy)
    _checkpoint_with_marker(first, 'trusted')
    complete_path = first.artifact_root / 'checkpoint.complete'
    complete = json.loads(complete_path.read_text())
    del complete[missing_field]
    complete_path.write_text(json.dumps(complete), encoding='utf-8')
    first.close()

    resumed = _create(tmp_path, source, digest, policy)

    assert resumed.recovery['mode'] == 'reset_for_new_direction'
    assert not (resumed.work_root / 'work' / 'marker.txt').exists()
    assert resumed.restored_session == {}
    resumed.close()


def test_tampered_checkpoint_workspace_is_not_restored(tmp_path: Path) -> None:
    source, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, policy)
    _checkpoint_with_marker(first, 'trusted')
    (first.artifact_root / 'checkpoint' / 'work' / 'marker.txt').write_text(
        'tampered', encoding='utf-8',
    )
    first.close()

    resumed = _create(tmp_path, source, digest, policy)

    assert resumed.recovery['mode'] == 'reset_for_new_direction'
    assert not (resumed.work_root / 'work' / 'marker.txt').exists()
    resumed.close()


def test_tampered_session_is_not_resumed_but_workspace_remains_recoverable(
    tmp_path: Path,
) -> None:
    source, digest = _source(tmp_path)
    policy = _policy(tmp_path, '保留用户过滤条件')
    first = _create(tmp_path, source, digest, policy)
    _checkpoint_with_marker(first, 'trusted')
    session_path = first.artifact_root / 'checkpoint' / 'session.json'
    session = json.loads(session_path.read_text())
    session['session_id'] = 'forged-session'
    session_path.write_text(json.dumps(session), encoding='utf-8')
    first.close()

    resumed = _create(tmp_path, source, digest, policy)

    assert resumed.recovery['mode'] == 'same_revision_resume'
    assert (resumed.work_root / 'work' / 'marker.txt').read_text() == 'trusted'
    assert resumed.recovery['session_resume_allowed'] is False
    assert resumed.restored_session == {}
    resumed.close()


def test_tampered_journal_event_cannot_enter_reuse_index(tmp_path: Path) -> None:
    _, digest = _source(tmp_path)
    policy = _policy(tmp_path, '联网搜索官方文档')
    old = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(policy),
        guidance_state=guidance_snapshot(policy),
        scope={},
        work_root=tmp_path / 'old-work',
        artifact_root=tmp_path / 'old-attempt',
        previous_attempts=(),
        source_digest=digest,
    )
    (old.work_root / 'work').mkdir(parents=True)
    old.record('web.search', 'trusted result', {
        'status': 'completed',
        'query': 'official docs',
        'results': [{'url': 'https://example.com/trusted'}],
    })
    event_path = next((old.artifact_root / 'events').glob('*.json'))
    event = json.loads(event_path.read_text())
    event['data']['results'] = [{'url': 'https://attacker.example/forged'}]
    event_path.write_text(json.dumps(event), encoding='utf-8')

    current = WorkMemory(
        target=_target(digest),
        guidance=active_guidance(policy),
        guidance_state=guidance_snapshot(policy),
        scope={},
        work_root=tmp_path / 'current-work',
        artifact_root=tmp_path / 'current-attempt',
        previous_attempts=(),
        history_attempts=(old.artifact_root,),
        source_digest=digest,
    )

    assert current.completed_investigation(
        'web.search', {'status': 'completed', 'query': 'official docs'},
    ) is None
    assert current.known_urls() == set()
