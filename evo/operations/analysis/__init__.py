import json
from collections.abc import Mapping
from hashlib import sha1, sha256
from typing import Any

from evo.operations.public_contracts import clean_text


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if value in (None, ''):
        return []
    return [value]


def _ids(value: Any) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Mapping):
        values = [value.get(key) for key in ('id', 'doc_id', 'chunk_id') if value.get(key)]
    else:
        values = list(value or [])
    return [str(item).strip() for item in values if str(item or '').strip()]


def _stable_id(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return sha1(payload.encode('utf-8')).hexdigest()


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=str,
    )
    return sha256(payload.encode('utf-8')).hexdigest()


def _clip_text(value: Any, limit: int) -> str:
    text = clean_text(value)
    return text if len(text) <= limit else text[:limit - 3] + '...'


def _evidence_record(kind: str, field: str, value: Any) -> dict[str, Any]:
    return {'type': kind, 'source_field': field, 'observed_value': value}


def _unique_text_values(*values: Any) -> list[str]:
    items: list[str] = []
    for value in values:
        for item in _as_list(value):
            text = clean_text(item)
            if text and text not in items:
                items.append(text)
    return items
