from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_source_config(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError('source_config must be a mapping')

    raw_kb_ids = value.get('kb_ids', value.get('kb_id', []))
    if not isinstance(raw_kb_ids, list):
        raise ValueError('source_config.kb_id must be a list')
    kb_ids = _unique_text(raw_kb_ids, 'source_config.kb_id')

    csv_sources: list[dict[str, str]] = []
    normalized_sources = 'csv_sources' in value
    raw_csv_data = value.get('csv_sources', value.get('csv_data', []))
    if not isinstance(raw_csv_data, list):
        raise ValueError('source_config.csv_data must be a list')
    for index, raw in enumerate(raw_csv_data):
        if not isinstance(raw, Mapping):
            raise ValueError(f'source_config.csv_data[{index}] must be a mapping')
        if normalized_sources:
            if set(raw) != {'kb_id', 'path'}:
                raise ValueError(
                    f'source_config.csv_sources[{index}] must contain kb_id and path'
                )
            kb_id, path = raw['kb_id'], raw['path']
        else:
            if len(raw) != 1:
                raise ValueError(
                    f'source_config.csv_data[{index}] must contain one kb_id and path'
                )
            kb_id, path = next(iter(raw.items()))
        kb_id = _text(kb_id, f'source_config.csv_data[{index}].kb_id')
        path = _text(path, f'source_config.csv_data[{index}].path')
        csv_sources.append({'kb_id': kb_id, 'path': path})
        if kb_id not in kb_ids:
            kb_ids.append(kb_id)

    csv_path = str(value.get('csv_path') or '').strip()
    if csv_path:
        if not kb_ids:
            raise ValueError('source_config.csv_path requires at least one kb_id')
        csv_sources.append({'kb_id': kb_ids[0], 'path': csv_path})

    if not kb_ids:
        raise ValueError('source_config requires at least one knowledge base')

    target = value.get('target_case_count')
    if isinstance(target, bool) or not isinstance(target, int) or target < 1:
        raise ValueError('source_config.target_case_count must be a positive integer')

    deduplicated: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for source in csv_sources:
        key = (source['kb_id'], source['path'])
        if key not in seen:
            seen.add(key)
            deduplicated.append(source)

    return {
        'kb_ids': kb_ids,
        'csv_sources': deduplicated,
        'target_case_count': target,
    }


def _unique_text(values: list[object], name: str) -> list[str]:
    output = [_text(value, name) for value in values]
    if len(set(output)) != len(output):
        raise ValueError(f'{name} must contain unique values')
    return output


def _text(value: object, name: str) -> str:
    text = str(value or '').strip()
    if not text:
        raise ValueError(f'{name} must be non-empty')
    return text


__all__ = ['normalize_source_config']
