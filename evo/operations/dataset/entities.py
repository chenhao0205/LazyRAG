from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .llm_json import call_json

DEFAULT_MAX_ENTITIES_PER_CHUNK = 10
MAX_ENTITIES_PER_CHUNK_LIMIT = 100
ENTITY_PROMPT = """Extract the named entities from the given text, limiting the output to the top entities.
Ensure the number of entities does not exceed the specified maximum.

Return only JSON in this format:
{{"entities":["..."]}}

max_num: {max_num}
text:
{text}"""


@dataclass(frozen=True)
class ChunkEntitiesExtractParams:
    max_entities_per_chunk: int = DEFAULT_MAX_ENTITIES_PER_CHUNK

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'ChunkEntitiesExtractParams':
        return cls(max_entities_per_chunk=_positive_int(
            data.get('max_entities_per_chunk'),
            DEFAULT_MAX_ENTITIES_PER_CHUNK,
            MAX_ENTITIES_PER_CHUNK_LIMIT,
            'max_entities_per_chunk',
        ))


@dataclass(frozen=True)
class ChunkEntitiesExtractManifestParams:
    max_entities_per_chunk: int = DEFAULT_MAX_ENTITIES_PER_CHUNK

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'ChunkEntitiesExtractManifestParams':
        return cls(max_entities_per_chunk=_positive_int(
            data.get('max_entities_per_chunk'),
            DEFAULT_MAX_ENTITIES_PER_CHUNK,
            MAX_ENTITIES_PER_CHUNK_LIMIT,
            'max_entities_per_chunk',
        ))

    def to_dict(self) -> dict[str, int]:
        return {'max_entities_per_chunk': self.max_entities_per_chunk}


def chunk_entities_extract(
    ctx: Any,
    inputs: Mapping[str, object],
    llm_complete: Callable[[str], Any] | None = None,
) -> Mapping[str, object]:
    chunk = _mapping(inputs.get('chunk'), 'chunk')
    params = ChunkEntitiesExtractParams.from_dict(
        _mapping(inputs.get('chunk_entities_extract_params'), 'chunk_entities_extract_params')
    )
    available = chunk.get('available') is not False
    entities = [] if not available else _extract_entities(
        str(chunk.get('text') or ''), params, _llm_complete(llm_complete)
    )
    payload = {
        'available': available,
        'chunk_id': _required_str(chunk, 'chunk_id'),
        'doc_id': _required_str(chunk, 'doc_id'),
        'group': _required_str(chunk, 'group'),
        'entities': entities,
    }
    if kb_id := str(chunk.get('kb_id') or ''):
        payload['kb_id'] = kb_id
    return {'chunk_entity': payload}


def chunk_entities_extract_manifest(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
    built = _mapping(inputs.get('build_chunks_manifest'), 'build_chunks_manifest')
    params = ChunkEntitiesExtractManifestParams.from_dict(
        _mapping(inputs.get('chunk_entities_extract_manifest_params'), 'chunk_entities_extract_manifest_params')
    )
    chunks = _chunks(built)
    entities = _entity_tuple(inputs.get('chunk_entities'), params)
    by_chunk_id = _entities_by_chunk_id(entities)
    missing = [chunk['chunk_id'] for chunk in chunks if chunk['chunk_id'] not in by_chunk_id]
    if missing:
        raise ValueError(f'missing ChunkEntity for chunk ids: {missing}')

    output_chunks = []
    for chunk in chunks:
        value = {
            'available': chunk['available'],
            'chunk_id': chunk['chunk_id'],
            'doc_id': chunk['doc_id'],
            'group': chunk['group'],
            'partition': chunk['partition'],
            'entities': list(by_chunk_id[chunk['chunk_id']]['entities']),
        }
        if chunk['kb_id']:
            value['kb_id'] = chunk['kb_id']
        output_chunks.append(value)
    return {'chunk_entities_extract_manifest': {
        'chunks': output_chunks,
        'stats': _chunk_entities_stats(output_chunks),
        'params': params.to_dict(),
    }}


def _extract_entities(text: str, params: ChunkEntitiesExtractParams, complete: Callable[[str], Any]) -> list[str]:
    if not text.strip():
        raise ValueError('chunk.text must be a non-empty string')
    prompt = ENTITY_PROMPT.format(max_num=params.max_entities_per_chunk, text=text)
    return call_json(
        complete,
        prompt,
        lambda value: _parse_entities_response(value, params.max_entities_per_chunk),
    )


def _chunks(built: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = built.get('chunks')
    if not isinstance(value, list) or not value:
        raise ValueError('built_chunks.chunks must be a non-empty list')
    chunks = []
    seen = set()
    for item in value:
        chunk = _mapping(item, 'built_chunks.chunks[]')
        chunk_id = _required_str(chunk, 'chunk_id')
        if chunk_id in seen:
            raise ValueError(f'duplicate built chunk_id: {chunk_id}')
        seen.add(chunk_id)
        chunks.append({
            'available': chunk.get('available') is not False,
            'kb_id': str(chunk.get('kb_id') or ''),
            'chunk_id': chunk_id,
            'doc_id': _required_str(chunk, 'doc_id'),
            'group': _required_str(chunk, 'group'),
            'partition': str(chunk.get('partition') or ''),
        })
    return chunks


def _entity_tuple(value: object, params: ChunkEntitiesExtractManifestParams) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, tuple):
        raise ValueError('chunk_entities input must be a partitioned tuple')
    output = []
    for item in value:
        entity = _mapping(item, 'chunk_entities[]')
        _validate_entities(entity.get('entities'), params.max_entities_per_chunk)
        output.append(entity)
    return tuple(output)


def _entities_by_chunk_id(entities: tuple[Mapping[str, Any], ...]) -> dict[str, Mapping[str, Any]]:
    by_chunk_id: dict[str, Mapping[str, Any]] = {}
    for entity in entities:
        chunk_id = _required_str(entity, 'chunk_id')
        if chunk_id in by_chunk_id:
            raise ValueError(f'duplicate ChunkEntity.chunk_id: {chunk_id}')
        _required_str(entity, 'doc_id')
        _required_str(entity, 'group')
        by_chunk_id[chunk_id] = entity
    return by_chunk_id


def _parse_entities_response(value: Any, max_entities: int) -> list[str]:
    data = value if isinstance(value, dict) else json.loads(str(value).strip())
    if not isinstance(data, Mapping):
        raise ValueError('expected JSON object')
    return _validate_entities(data.get('entities'), max_entities)


def _validate_entities(value: Any, max_entities: int) -> list[str]:
    if not isinstance(value, list):
        raise ValueError('entities must be list[string]')
    if len(value) > max_entities:
        raise ValueError(f'entities must contain at most {max_entities} items')
    entities = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError('entities must contain only non-empty strings')
        entities.append(item)
    return entities


def _chunk_entities_stats(chunks: list[Mapping[str, Any]]) -> dict[str, Any]:
    available_chunks = [chunk for chunk in chunks if chunk.get('available') is not False]
    entity_lists = [list(chunk.get('entities') or []) for chunk in available_chunks]
    return {
        'slot_count': len(chunks),
        'available_count': len(available_chunks),
        'placeholder_count': len(chunks) - len(available_chunks),
        'entity_count': sum(len(items) for items in entity_lists),
        'empty_entity_count': sum(1 for items in entity_lists if not items),
        'doc_count': len({
            str(chunk.get('doc_id') or '') for chunk in available_chunks if str(chunk.get('doc_id') or '')
        }),
        'group_counts': dict(Counter(str(chunk.get('group') or '') for chunk in available_chunks)),
    }


def _llm_complete(complete: Callable[[str], Any] | None) -> Callable[[str], Any]:
    if complete is not None:
        return complete
    from evo.llm import LazyLLMClient

    return LazyLLMClient(model='evo_llm')


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _required_str(value: Mapping[str, Any], name: str) -> str:
    text = str(value.get(name) or '').strip()
    if not text:
        raise ValueError(f'{name} must be a non-empty string')
    return text


def _positive_int(value: Any, default: int, maximum: int, name: str) -> int:
    if value is None:
        return default
    try:
        output = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a positive integer') from exc
    if output < 1:
        raise ValueError(f'{name} must be a positive integer')
    if output > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return output
