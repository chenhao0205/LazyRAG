from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

import evo.repair_guidance as guidance_module
from evo.repair_guidance import (
    active_guidance,
    apply_guidance_transition,
    canonicalize_guidance_policy_update,
    guidance_snapshot,
    is_append_guidance_successor,
)


def _schema_type(name: str) -> type:
    module_name = '_evo_message_intent_schemas_for_test'
    module = sys.modules.get(module_name)
    if module is None:
        path = Path(__file__).parents[2] / 'evo' / 'message_intent' / 'schemas.py'
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return getattr(module, name)


def test_repair_guidance_action_keeps_effect_and_targets_when_message_is_replaced() -> None:
    RepairGuidanceAction = _schema_type('RepairGuidanceAction')
    action = RepairGuidanceAction(
        kind='repair_guidance',
        message='planner placeholder',
        effect='amend',
        target_directive_ids=['gd_000000000000000000000000'],
    )

    preserved = action.model_copy(update={'message': '用户的原始输入'})

    assert preserved.message == '用户的原始输入'
    assert preserved.effect == 'amend'
    assert preserved.target_directive_ids == ['gd_000000000000000000000000']


def test_repair_guidance_action_requires_explicit_effect() -> None:
    RepairGuidanceAction = _schema_type('RepairGuidanceAction')

    with pytest.raises(ValueError, match='effect'):
        RepairGuidanceAction(kind='repair_guidance', message='修正上一条方向')


def test_structured_transition_requires_source_message_id() -> None:
    with pytest.raises(ValueError, match='source_message_id must be non-empty'):
        apply_guidance_transition(
            {'user_guidance': []},
            effect='append',
            message='当前方向',
        )


def test_legacy_policy_projects_stable_ids_without_mutation() -> None:
    policy = {
        'llm_config': {'model': 'test'},
        'user_guidance': [' 保留用户过滤条件 ', '保留用户过滤条件', '支持中文输入'],
    }
    before = copy.deepcopy(policy)

    first = guidance_snapshot(policy)
    second = guidance_snapshot(copy.deepcopy(policy))

    assert policy == before
    assert first == second
    assert active_guidance(policy) == ('保留用户过滤条件', '支持中文输入')
    assert first['revision_id'].startswith('gr_')
    assert all(item['directive_id'].startswith('gd_') for item in first['active_directives'])


def test_append_preserves_raw_text_and_is_deterministic() -> None:
    policy = {'user_guidance': ['保留用户过滤条件']}
    kwargs = {
        'effect': 'append',
        'message': '  同时支持中文输入  ',
        'source_message_id': 'msg-1',
    }

    first = apply_guidance_transition(policy, **kwargs)
    replay = apply_guidance_transition(copy.deepcopy(policy), **kwargs)

    assert first.changed is True
    assert first.revision_id == replay.revision_id
    assert first.state['active_directives'] == replay.state['active_directives']
    assert first.active_guidance == ('保留用户过滤条件', '同时支持中文输入')
    assert first.policy['user_guidance'] == list(first.active_guidance)
    assert policy == {'user_guidance': ['保留用户过滤条件']}


def test_normalized_duplicate_append_is_a_noop() -> None:
    policy = {'user_guidance': ['Keep   UTF-8']}
    before_revision = guidance_snapshot(policy)['revision_id']

    result = apply_guidance_transition(
        policy,
        effect='append',
        message=' keep utf-8 ',
        source_message_id='different-message',
    )

    assert result.changed is False
    assert result.revision_id == before_revision
    assert result.policy == policy


def test_amend_replaces_only_selected_directive() -> None:
    policy = {'user_guidance': ['修改 web.py', '保留中文支持']}
    state = guidance_snapshot(policy)
    old_id = state['active_directives'][0]['directive_id']

    result = apply_guidance_transition(
        policy,
        effect='amend',
        message='不要修改 web.py，改为修改 memory.py',
        target_directive_ids=[old_id],
        source_message_id='msg-amend',
    )

    assert result.active_guidance == ('不要修改 web.py，改为修改 memory.py', '保留中文支持')
    assert result.state['parent_revision_id'] == state['revision_id']
    assert result.state['delta']['superseded_ids'] == [old_id]
    assert len(result.state['delta']['added_ids']) == 1


def test_amend_to_existing_text_reuses_existing_directive() -> None:
    policy = {'user_guidance': ['方向 A', '方向 B']}
    state = guidance_snapshot(policy)
    a_id = state['active_directives'][0]['directive_id']
    b_id = state['active_directives'][1]['directive_id']

    result = apply_guidance_transition(
        policy,
        effect='amend',
        message='方向 B',
        target_directive_ids=[a_id],
        source_message_id='msg-amend-to-existing',
    )

    assert result.active_guidance == ('方向 B',)
    assert result.state['active_directives'][0]['directive_id'] == b_id
    assert result.state['delta']['added_ids'] == []
    assert result.state['delta']['superseded_ids'] == [a_id]


def test_replace_and_withdraw_have_distinct_semantics() -> None:
    initial = {'user_guidance': ['旧方向 A', '旧方向 B']}
    replaced = apply_guidance_transition(
        initial,
        effect='replace',
        message='只检查 parser',
        source_message_id='msg-replace',
    )
    assert replaced.active_guidance == ('只检查 parser',)
    assert len(replaced.state['delta']['superseded_ids']) == 2

    directive_id = replaced.state['active_directives'][0]['directive_id']
    withdrawn = apply_guidance_transition(
        replaced.policy,
        effect='withdraw',
        message='撤回上一条方向',
        target_directive_ids=[directive_id],
        source_message_id='msg-withdraw',
    )
    assert withdrawn.active_guidance == ()
    assert withdrawn.state['delta']['withdrawn_ids'] == [directive_id]
    assert '撤回上一条方向' not in withdrawn.policy['user_guidance']


@pytest.mark.parametrize('effect', ['amend', 'withdraw'])
def test_targeted_effect_requires_an_active_target(effect: str) -> None:
    with pytest.raises(ValueError, match='requires target_directive_ids'):
        apply_guidance_transition(
            {'user_guidance': ['方向 A']},
            effect=effect,
            message='修改方向',
            source_message_id='msg-invalid',
        )

    with pytest.raises(ValueError, match='target is not active'):
        apply_guidance_transition(
            {'user_guidance': ['方向 A']},
            effect=effect,
            message='修改方向',
            target_directive_ids=['gd_000000000000000000000000'],
            source_message_id='msg-invalid',
        )


def test_structured_state_must_match_compatibility_list() -> None:
    transition = apply_guidance_transition(
        {'user_guidance': []},
        effect='append',
        message='当前方向',
        source_message_id='msg-structured',
    )
    broken = {**transition.policy, 'user_guidance': ['另一个方向']}

    with pytest.raises(ValueError, match='does not match user_guidance'):
        guidance_snapshot(broken)


def test_active_guidance_has_aggregate_hot_context_limit() -> None:
    with pytest.raises(ValueError, match='active guidance exceeds'):
        guidance_snapshot({
            'user_guidance': [f'{index}' + 'x' * 3999 for index in range(9)],
        })


def test_malformed_guidance_delta_raises_value_error() -> None:
    transition = apply_guidance_transition(
        {'user_guidance': []},
        effect='append',
        message='当前方向',
        source_message_id='msg-structured',
    )
    broken = copy.deepcopy(transition.policy)
    broken['guidance_state']['delta']['added_ids'] = 1

    with pytest.raises(ValueError, match='target_directive_ids must be a list'):
        guidance_snapshot(broken)


def test_effect_label_cannot_disguise_replace_as_append() -> None:
    transition = apply_guidance_transition(
        {'user_guidance': ['旧方向']},
        effect='replace',
        message='全新方向',
        source_message_id='msg-replace',
    )
    broken = copy.deepcopy(transition.policy)
    state = broken['guidance_state']
    state['effect'] = 'append'
    state['revision_id'] = guidance_module._prefixed_hash('gr', {
        'schema_version': state['schema_version'],
        'parent_revision_id': state['parent_revision_id'],
        'source_message_id': state['source_message_id'],
        'effect': 'append',
        'target_directive_ids': [],
        'active_content_hash': state['active_content_hash'],
        'active_directive_ids': [
            item['directive_id'] for item in state['active_directives']
        ],
    })
    state['active_directives'][0]['introduced_revision_id'] = state['revision_id']

    with pytest.raises(ValueError, match='append repair guidance delta is invalid'):
        guidance_snapshot(broken)


def test_append_successor_requires_complete_parent_prefix() -> None:
    initial = {'user_guidance': ['保留旧方向']}
    appended = apply_guidance_transition(
        initial,
        effect='append',
        message='补充新方向',
        source_message_id='msg-append',
    )
    replaced = apply_guidance_transition(
        initial,
        effect='replace',
        message='补充新方向',
        source_message_id='msg-replace',
    )

    assert is_append_guidance_successor(guidance_snapshot(initial), appended.state)
    assert not is_append_guidance_successor(guidance_snapshot(initial), replaced.state)


def test_policy_update_preserves_canonical_state_when_guidance_is_unchanged() -> None:
    base = apply_guidance_transition(
        {'user_guidance': ['方向 A']},
        effect='append',
        message='方向 B',
        source_message_id='msg-append',
    ).policy

    updated = canonicalize_guidance_policy_update(
        base,
        {'user_guidance': ['方向 A', '方向 B'], 'max_calls': 12},
    )

    assert updated['guidance_state'] == base['guidance_state']
    assert updated['user_guidance'] == base['user_guidance']
    assert updated['max_calls'] == 12


def test_policy_update_allows_legacy_guidance_change_as_safe_reset() -> None:
    base = apply_guidance_transition(
        {'user_guidance': ['旧方向']},
        effect='append',
        message='补充方向',
        source_message_id='msg-append',
    ).policy

    updated = canonicalize_guidance_policy_update(
        base,
        {'user_guidance': ['完全不同的新方向'], 'max_calls': 12},
    )

    assert updated['user_guidance'] == ['完全不同的新方向']
    assert 'guidance_state' not in updated


def test_policy_update_rejects_client_supplied_structured_change() -> None:
    base = {'user_guidance': ['旧方向']}
    changed = apply_guidance_transition(
        base,
        effect='append',
        message='新方向',
        source_message_id='msg-append',
    ).policy

    with pytest.raises(ValueError, match='message-intent API'):
        canonicalize_guidance_policy_update(base, changed)


def test_structured_source_message_ids_are_bounded() -> None:
    transition = apply_guidance_transition(
        {'user_guidance': []},
        effect='append',
        message='当前方向',
        source_message_id='msg-append',
    )
    broken = copy.deepcopy(transition.policy)
    broken['guidance_state']['source_message_id'] = 'x' * 161

    with pytest.raises(ValueError, match='source_message_id exceeds'):
        guidance_snapshot(broken)
