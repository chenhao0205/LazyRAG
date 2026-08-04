from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal


GuidanceEffect = Literal['append', 'amend', 'replace', 'withdraw']
GUIDANCE_SCHEMA_VERSION = 1
MAX_ACTIVE_DIRECTIVES = 100
MAX_ACTIVE_GUIDANCE_CHARS = 32_000


@dataclass(frozen=True)
class GuidanceTransition:
    """Result of one deterministic Repair-guidance state transition."""

    policy: dict[str, Any]
    state: dict[str, Any]
    changed: bool
    previous_revision_id: str
    revision_id: str
    active_guidance: tuple[str, ...]


def normalize_guidance_text(value: object) -> str:
    """Return the stable comparison form without replacing the user's text."""
    return ' '.join(unicodedata.normalize('NFKC', str(value)).split()).casefold()


def guidance_snapshot(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Read the current guidance state, projecting legacy string lists on demand.

    The projection is pure: a legacy policy is not mutated or implicitly migrated.
    Once ``guidance_state`` exists it is authoritative and must agree with the
    compatibility ``user_guidance`` list.
    """
    if not isinstance(policy, Mapping):
        raise ValueError('repair policy must be an object')
    raw_state = policy.get('guidance_state')
    if raw_state is None:
        return _legacy_snapshot(_legacy_texts(policy))
    if not isinstance(raw_state, Mapping):
        raise ValueError('repair policy guidance_state must be an object')
    state = _validated_state(raw_state)
    compatibility = _legacy_texts(policy)
    state_texts = tuple(item['text'] for item in state['active_directives'])
    if tuple(map(normalize_guidance_text, compatibility)) != tuple(
        map(normalize_guidance_text, state_texts)
    ):
        raise ValueError('repair policy guidance_state does not match user_guidance')
    return state


def active_guidance(policy: Mapping[str, Any]) -> tuple[str, ...]:
    """Return only the currently active, user-authored guidance texts."""
    return tuple(item['text'] for item in guidance_snapshot(policy)['active_directives'])


def is_append_guidance_successor(
    parent_state: Mapping[str, Any],
    child_state: Mapping[str, Any],
) -> bool:
    """Return whether ``child_state`` is a genuine append of ``parent_state``.

    Revision ids are content hashes, not authentication tokens. Recovery must
    therefore verify the parent/child active-directive relationship instead of
    trusting the child's declared ``effect`` alone.
    """
    try:
        parent = _validated_state(parent_state)
        child = _validated_state(child_state)
    except (TypeError, ValueError):
        return False
    if (
        child['effect'] != 'append'
        or child['parent_revision_id'] != parent['revision_id']
    ):
        return False
    added_id = child['delta']['added_ids'][0]
    added = next(
        (item for item in child['active_directives'] if item['directive_id'] == added_id),
        None,
    )
    return added is not None and child['active_directives'] == [
        *parent['active_directives'],
        added,
    ]


def canonicalize_guidance_policy_update(
    base_policy: Mapping[str, Any],
    requested_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a full repair-policy update without trusting client lineage.

    Unchanged guidance inherits the server's canonical state. Legacy clients may
    still replace the text list without a structured state; that deliberately
    resets lineage. Structured changes must use the typed message-intent path,
    where the transition is derived from the current artifact head.
    """
    if not isinstance(base_policy, Mapping) or not isinstance(requested_policy, Mapping):
        raise ValueError('repair policy must be an object')
    base = guidance_snapshot(base_policy)
    requested = guidance_snapshot(requested_policy)
    value = dict(requested_policy)
    base_texts = tuple(item['normalized_text'] for item in base['active_directives'])
    requested_texts = tuple(
        item['normalized_text'] for item in requested['active_directives']
    )
    if requested_texts == base_texts:
        if requested_policy.get('guidance_state') is not None and requested != base:
            raise ValueError('unchanged repair guidance must preserve its canonical state')
        value['user_guidance'] = [item['text'] for item in base['active_directives']]
        value['guidance_state'] = base
        return value
    if requested_policy.get('guidance_state') is not None:
        raise ValueError('structured repair guidance changes require the message-intent API')
    value['user_guidance'] = [item['text'] for item in requested['active_directives']]
    value.pop('guidance_state', None)
    return value


def apply_guidance_transition(
    policy: Mapping[str, Any],
    *,
    effect: GuidanceEffect | str,
    message: str,
    target_directive_ids: Sequence[str] = (),
    source_message_id: str = '',
) -> GuidanceTransition:
    """Apply append/amend/replace/withdraw without mutating the input policy.

    ``message`` is always kept as the original user text. It becomes an active
    directive for append/amend/replace and is audit-only for withdraw. Target ids
    must name currently active directives. If the resulting normalized active text
    sequence is unchanged, no policy revision is created.
    """
    if not isinstance(policy, Mapping):
        raise ValueError('repair policy must be an object')
    if effect not in {'append', 'amend', 'replace', 'withdraw'}:
        raise ValueError(f'unsupported repair guidance effect: {effect}')
    raw_message = str(message).strip()
    if not raw_message:
        raise ValueError('repair guidance message must be non-empty')
    if len(raw_message) > 4000:
        raise ValueError('repair guidance message exceeds 4000 characters')
    message_id = str(source_message_id).strip()
    if not message_id:
        raise ValueError('repair guidance source_message_id must be non-empty')
    if len(message_id) > 160:
        raise ValueError('repair guidance source_message_id exceeds 160 characters')

    state = guidance_snapshot(policy)
    current = [dict(item) for item in state['active_directives']]
    targets = _target_ids(target_directive_ids)
    current_by_id = {item['directive_id']: item for item in current}
    unknown = [item for item in targets if item not in current_by_id]
    if unknown:
        raise ValueError(f'repair guidance target is not active: {unknown[0]}')
    if effect in {'amend', 'withdraw'} and not targets:
        raise ValueError(f'{effect} requires target_directive_ids')
    if effect in {'append', 'replace'} and targets:
        raise ValueError(f'{effect} does not accept target_directive_ids')

    normalized_message = normalize_guidance_text(raw_message)
    current_normalized = tuple(normalize_guidance_text(item['text']) for item in current)
    parent_revision_id = str(state['revision_id'])
    new_directive = {
        'directive_id': _directive_id(
            parent_revision_id=parent_revision_id,
            source_message_id=message_id,
            effect=effect,
            normalized_text=normalized_message,
            target_directive_ids=targets,
        ),
        'text': raw_message,
        'normalized_text': normalized_message,
        'source_message_id': message_id,
        'introduced_revision_id': '',
    }

    added_ids: list[str] = []
    superseded_ids: list[str] = []
    withdrawn_ids: list[str] = []
    if effect == 'append':
        if normalized_message in current_normalized:
            return _unchanged_transition(policy, state)
        proposed = [*current, new_directive]
        added_ids.append(new_directive['directive_id'])
    elif effect == 'replace':
        proposed = [new_directive]
        added_ids.append(new_directive['directive_id'])
        superseded_ids.extend(item['directive_id'] for item in current)
    elif effect == 'withdraw':
        target_set = set(targets)
        proposed = [item for item in current if item['directive_id'] not in target_set]
        withdrawn_ids.extend(targets)
    else:
        target_set = set(targets)
        first_index = min(index for index, item in enumerate(current) if item['directive_id'] in target_set)
        retained = [item for item in current if item['directive_id'] not in target_set]
        if any(normalize_guidance_text(item['text']) == normalized_message for item in retained):
            proposed = retained
        else:
            insert_at = sum(
                1 for item in current[:first_index] if item['directive_id'] not in target_set
            )
            proposed = [*retained[:insert_at], new_directive, *retained[insert_at:]]
            added_ids.append(new_directive['directive_id'])
        superseded_ids.extend(targets)

    proposed = _deduplicate_directives(proposed)
    proposed_ids = {item['directive_id'] for item in proposed}
    added_ids = [item for item in added_ids if item in proposed_ids]
    proposed_normalized = tuple(normalize_guidance_text(item['text']) for item in proposed)
    if proposed_normalized == current_normalized:
        return _unchanged_transition(policy, state)
    if len(proposed) > MAX_ACTIVE_DIRECTIVES:
        raise ValueError('repair policy user_guidance limit reached')
    _validate_total_guidance_size(item['text'] for item in proposed)

    active_content_hash = _active_content_hash(proposed)
    revision_id = _prefixed_hash('gr', {
        'schema_version': GUIDANCE_SCHEMA_VERSION,
        'parent_revision_id': parent_revision_id,
        'source_message_id': message_id,
        'effect': effect,
        'target_directive_ids': list(targets),
        'active_content_hash': active_content_hash,
        'active_directive_ids': [item['directive_id'] for item in proposed],
    })
    for item in proposed:
        if not item.get('introduced_revision_id'):
            item['introduced_revision_id'] = revision_id
    next_state = {
        'schema_version': GUIDANCE_SCHEMA_VERSION,
        'revision_id': revision_id,
        'parent_revision_id': parent_revision_id,
        'source_message_id': message_id,
        'effect': effect,
        'active_content_hash': active_content_hash,
        'active_directives': proposed,
        'delta': {
            'added_ids': added_ids,
            'superseded_ids': superseded_ids,
            'withdrawn_ids': withdrawn_ids,
        },
    }
    next_policy = {
        **dict(policy),
        'user_guidance': [item['text'] for item in proposed],
        'guidance_state': next_state,
    }
    return GuidanceTransition(
        policy=next_policy,
        state=_copy_json(next_state),
        changed=True,
        previous_revision_id=parent_revision_id,
        revision_id=revision_id,
        active_guidance=tuple(item['text'] for item in proposed),
    )


def _legacy_texts(policy: Mapping[str, Any]) -> tuple[str, ...]:
    raw = policy.get('user_guidance') or ()
    if not isinstance(raw, (list, tuple)):
        raise ValueError('repair policy user_guidance must be a list')
    result: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = str(value).strip()
        if not text:
            continue
        normalized = normalize_guidance_text(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(text)
    if len(result) > MAX_ACTIVE_DIRECTIVES:
        raise ValueError('repair policy user_guidance limit reached')
    _validate_total_guidance_size(result)
    return tuple(result)


def _legacy_snapshot(texts: Sequence[str]) -> dict[str, Any]:
    directives = [
        {
            'directive_id': _prefixed_hash('gd', {'legacy_text': normalize_guidance_text(text)}),
            'text': text,
            'normalized_text': normalize_guidance_text(text),
            'source_message_id': '',
            'introduced_revision_id': '',
        }
        for text in texts
    ]
    active_content_hash = _active_content_hash(directives)
    revision_id = _prefixed_hash('gr', {
        'schema_version': GUIDANCE_SCHEMA_VERSION,
        'effect': 'legacy',
        'active_content_hash': active_content_hash,
        'active_directive_ids': [item['directive_id'] for item in directives],
    })
    for item in directives:
        item['introduced_revision_id'] = revision_id
    return {
        'schema_version': GUIDANCE_SCHEMA_VERSION,
        'revision_id': revision_id,
        'parent_revision_id': '',
        'source_message_id': '',
        'effect': 'legacy',
        'active_content_hash': active_content_hash,
        'active_directives': directives,
        'delta': {'added_ids': [], 'superseded_ids': [], 'withdrawn_ids': []},
    }


def _validated_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    if raw.get('schema_version') != GUIDANCE_SCHEMA_VERSION:
        raise ValueError('unsupported repair guidance schema_version')
    revision_id = _identifier(raw.get('revision_id'), 'gr_', 'guidance revision_id')
    parent_revision_id = str(raw.get('parent_revision_id') or '')
    if parent_revision_id:
        _identifier(parent_revision_id, 'gr_', 'guidance parent_revision_id')
    raw_source_message_id = raw.get('source_message_id', '')
    if raw_source_message_id is None:
        raw_source_message_id = ''
    if not isinstance(raw_source_message_id, str):
        raise ValueError('repair guidance source_message_id must be a string')
    source_message_id = raw_source_message_id.strip()
    if source_message_id != raw_source_message_id:
        raise ValueError('repair guidance source_message_id is invalid')
    if len(source_message_id) > 160:
        raise ValueError('repair guidance source_message_id exceeds 160 characters')
    effect = str(raw.get('effect') or '')
    if effect not in {'legacy', 'append', 'amend', 'replace', 'withdraw'}:
        raise ValueError('repair guidance state has invalid effect')
    raw_directives = raw.get('active_directives')
    if not isinstance(raw_directives, (list, tuple)):
        raise ValueError('repair guidance active_directives must be a list')
    directives = []
    ids: set[str] = set()
    normalized: set[str] = set()
    for raw_item in raw_directives:
        if not isinstance(raw_item, Mapping):
            raise ValueError('repair guidance directive must be an object')
        directive_id = _identifier(raw_item.get('directive_id'), 'gd_', 'repair guidance directive_id')
        text = str(raw_item.get('text') or '').strip()
        normalized_text = normalize_guidance_text(text)
        if not text or normalized_text != str(raw_item.get('normalized_text') or ''):
            raise ValueError('repair guidance directive text is invalid')
        if directive_id in ids or normalized_text in normalized:
            raise ValueError('repair guidance active_directives contains duplicates')
        ids.add(directive_id)
        normalized.add(normalized_text)
        introduced_revision_id = _identifier(
            raw_item.get('introduced_revision_id'), 'gr_', 'guidance introduced_revision_id',
        )
        raw_directive_source = raw_item.get('source_message_id', '')
        if raw_directive_source is None:
            raw_directive_source = ''
        if not isinstance(raw_directive_source, str):
            raise ValueError('repair guidance directive source_message_id must be a string')
        directive_source_message_id = raw_directive_source.strip()
        if directive_source_message_id != raw_directive_source:
            raise ValueError('repair guidance directive source_message_id is invalid')
        if len(directive_source_message_id) > 160:
            raise ValueError('repair guidance directive source_message_id exceeds 160 characters')
        directives.append({
            'directive_id': directive_id,
            'text': text,
            'normalized_text': normalized_text,
            'source_message_id': directive_source_message_id,
            'introduced_revision_id': introduced_revision_id,
        })
    if len(directives) > MAX_ACTIVE_DIRECTIVES:
        raise ValueError('repair policy user_guidance limit reached')
    _validate_total_guidance_size(item['text'] for item in directives)
    active_content_hash = str(raw.get('active_content_hash') or '')
    if active_content_hash != _active_content_hash(directives):
        raise ValueError('repair guidance active_content_hash mismatch')
    raw_delta = raw.get('delta')
    if not isinstance(raw_delta, Mapping):
        raise ValueError('repair guidance delta must be an object')
    delta = {
        name: list(_target_ids(raw_delta.get(name) or ()))
        for name in ('added_ids', 'superseded_ids', 'withdrawn_ids')
    }
    active_ids = [item['directive_id'] for item in directives]
    if effect == 'legacy':
        expected_revision_id = _prefixed_hash('gr', {
            'schema_version': GUIDANCE_SCHEMA_VERSION,
            'effect': 'legacy',
            'active_content_hash': active_content_hash,
            'active_directive_ids': active_ids,
        })
    else:
        target_ids = (
            delta['superseded_ids'] if effect == 'amend'
            else delta['withdrawn_ids'] if effect == 'withdraw'
            else []
        )
        expected_revision_id = _prefixed_hash('gr', {
            'schema_version': GUIDANCE_SCHEMA_VERSION,
            'parent_revision_id': parent_revision_id,
            'source_message_id': source_message_id,
            'effect': effect,
            'target_directive_ids': target_ids,
            'active_content_hash': active_content_hash,
            'active_directive_ids': active_ids,
        })
    if revision_id != expected_revision_id:
        raise ValueError('repair guidance revision_id mismatch')
    if any(item not in active_ids for item in delta['added_ids']):
        raise ValueError('repair guidance added directive is not active')
    if any(item in active_ids for item in (*delta['superseded_ids'], *delta['withdrawn_ids'])):
        raise ValueError('inactive repair guidance delta contains an active directive')
    if any(
        item['directive_id'] in delta['added_ids']
        and item['introduced_revision_id'] != revision_id
        for item in directives
    ):
        raise ValueError('repair guidance introduced_revision_id mismatch')
    _validate_state_semantics(
        effect=effect,
        revision_id=revision_id,
        parent_revision_id=parent_revision_id,
        source_message_id=source_message_id,
        directives=directives,
        delta=delta,
    )
    return {
        'schema_version': GUIDANCE_SCHEMA_VERSION,
        'revision_id': revision_id,
        'parent_revision_id': parent_revision_id,
        'source_message_id': source_message_id,
        'effect': effect,
        'active_content_hash': active_content_hash,
        'active_directives': directives,
        'delta': delta,
    }


def _validate_state_semantics(
    *,
    effect: str,
    revision_id: str,
    parent_revision_id: str,
    source_message_id: str,
    directives: Sequence[Mapping[str, Any]],
    delta: Mapping[str, list[str]],
) -> None:
    active_ids = [str(item['directive_id']) for item in directives]
    added = list(delta['added_ids'])
    superseded = list(delta['superseded_ids'])
    withdrawn = list(delta['withdrawn_ids'])
    groups = [set(added), set(superseded), set(withdrawn)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError('repair guidance delta groups must be disjoint')
    introduced_now = [
        str(item['directive_id'])
        for item in directives
        if item['introduced_revision_id'] == revision_id
    ]

    if effect == 'legacy':
        if parent_revision_id or source_message_id or added or superseded or withdrawn:
            raise ValueError('legacy repair guidance state has invalid lineage')
        if introduced_now != active_ids:
            raise ValueError('legacy repair guidance introduced_revision_id mismatch')
        for item in directives:
            expected = _prefixed_hash('gd', {'legacy_text': item['normalized_text']})
            if item['directive_id'] != expected or item['source_message_id']:
                raise ValueError('legacy repair guidance directive provenance is invalid')
        return

    if not parent_revision_id or not source_message_id:
        raise ValueError('structured repair guidance requires parent and source message')
    if effect == 'append':
        valid_shape = len(added) == 1 and not superseded and not withdrawn
    elif effect == 'amend':
        valid_shape = len(added) <= 1 and bool(superseded) and not withdrawn
    elif effect == 'replace':
        valid_shape = (
            len(active_ids) == 1
            and added == active_ids
            and not withdrawn
        )
    else:
        valid_shape = not added and not superseded and bool(withdrawn)
    if not valid_shape:
        raise ValueError(f'{effect} repair guidance delta is invalid')
    if introduced_now != added:
        raise ValueError('repair guidance introduced directives do not match delta')

    target_ids = superseded if effect == 'amend' else withdrawn if effect == 'withdraw' else []
    for directive_id in added:
        item = next(
            (value for value in directives if value['directive_id'] == directive_id),
            None,
        )
        if item is None:
            raise ValueError('repair guidance added directive is not active')
        expected = _directive_id(
            parent_revision_id=parent_revision_id,
            source_message_id=source_message_id,
            effect=effect,
            normalized_text=str(item['normalized_text']),
            target_directive_ids=target_ids,
        )
        if item['directive_id'] != expected or item['source_message_id'] != source_message_id:
            raise ValueError('repair guidance added directive provenance is invalid')


def _target_ids(values: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError('target_directive_ids must be a list')
    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _identifier(raw, 'gd_', 'repair guidance target_directive_id')
        if value not in seen:
            seen.add(value)
            result.append(value)
    if len(result) > MAX_ACTIVE_DIRECTIVES:
        raise ValueError('too many repair guidance targets')
    return tuple(result)


def _validate_total_guidance_size(values: Iterable[object]) -> None:
    total = sum(len(str(value)) for value in values)
    if total > MAX_ACTIVE_GUIDANCE_CHARS:
        raise ValueError(
            f'repair policy active guidance exceeds {MAX_ACTIVE_GUIDANCE_CHARS} characters'
        )


def _identifier(value: object, prefix: str, label: str) -> str:
    text = str(value or '')
    suffix = text[len(prefix):] if text.startswith(prefix) else ''
    if len(suffix) != 24 or any(character not in '0123456789abcdef' for character in suffix):
        raise ValueError(f'{label} is invalid')
    return text


def _directive_id(*, parent_revision_id: str, source_message_id: str, effect: str,
                  normalized_text: str, target_directive_ids: Sequence[str]) -> str:
    return _prefixed_hash('gd', {
        'parent_revision_id': parent_revision_id,
        'source_message_id': source_message_id,
        'effect': effect,
        'normalized_text': normalized_text,
        'target_directive_ids': list(target_directive_ids),
    })


def _deduplicate_directives(directives: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in directives:
        normalized = normalize_guidance_text(item.get('text'))
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(dict(item))
    return result


def _active_content_hash(directives: Sequence[Mapping[str, Any]]) -> str:
    return _stable_hash([normalize_guidance_text(item.get('text')) for item in directives])


def _unchanged_transition(policy: Mapping[str, Any], state: Mapping[str, Any]) -> GuidanceTransition:
    revision_id = str(state['revision_id'])
    texts = tuple(str(item['text']) for item in state['active_directives'])
    return GuidanceTransition(
        policy=dict(policy),
        state=_copy_json(state),
        changed=False,
        previous_revision_id=revision_id,
        revision_id=revision_id,
        active_guidance=texts,
    )


def _prefixed_hash(prefix: str, value: object) -> str:
    return f'{prefix}_{_stable_hash(value)[:24]}'


def _stable_hash(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _copy_json(value: object) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


__all__ = [
    'GUIDANCE_SCHEMA_VERSION', 'GuidanceEffect', 'GuidanceTransition',
    'active_guidance', 'apply_guidance_transition',
    'canonicalize_guidance_policy_update', 'guidance_snapshot',
    'is_append_guidance_successor', 'normalize_guidance_text',
]
