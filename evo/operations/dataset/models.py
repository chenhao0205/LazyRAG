from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any


@dataclass(slots=True)
class ChunkSource:
    """Source metadata for a dataset chunk."""

    kb_id: str
    doc_id: str
    filename: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """Standard chunk work unit for evo dataset flows."""

    chunk_id: str
    text: str
    embedding: dict[str, Any]
    entities: list[str]
    group: str
    type: str
    source: ChunkSource


def chunk_from_docnode(
    node: Any,
    *,
    kb_id: str,
    doc_id: str,
    group: str,
    doc: dict[str, Any] | None = None,
) -> Chunk:
    node_metadata = dict(getattr(node, 'metadata', {}) or {})
    node_global_metadata = dict(getattr(node, 'global_metadata', {}) or {})
    chunk_id = str(getattr(node, 'uid', '') or '').strip()
    text = str(getattr(node, 'text', '') or '')
    resolved_group = str(getattr(node, 'group', '') or group or '').strip()
    resolved_kb_id = str(kb_id or '').strip()
    resolved_doc_id = str(doc_id or '').strip()

    _require(chunk_id, 'chunk_id')
    _require(text.strip(), 'text')
    _require(resolved_kb_id, 'kb_id')
    _require(resolved_doc_id, 'doc_id')
    _require(resolved_group, 'group')

    embedding = normalize_embedding(getattr(node, 'embedding', {}) or {})
    doc_data = dict(doc or {})
    filename = (
        str(doc_data.get('filename') or '')
        or str(doc_data.get('display_name') or '')
        or str(node_global_metadata.get('filename') or '')
        or str(node_global_metadata.get('file_name') or '')
    )

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        embedding=embedding,
        entities=[],
        group=resolved_group,
        type=str(node_metadata.get('type') or node_metadata.get('node_type') or ''),
        source=ChunkSource(
            kb_id=resolved_kb_id,
            doc_id=resolved_doc_id,
            filename=filename,
            metadata={
                'doc': doc_data,
                'node_metadata': node_metadata,
                'node_global_metadata': node_global_metadata,
            },
        ),
    )


def chunks_from_docnodes(
    nodes: Iterable[Any],
    *,
    kb_id: str,
    doc_id: str,
    group: str,
    doc: dict[str, Any] | None = None,
) -> list[Chunk]:
    return [chunk_from_docnode(node, kb_id=kb_id, doc_id=doc_id, group=group, doc=doc) for node in nodes]


def _require(value: str, name: str) -> None:
    if not value:
        raise ValueError(f'{name} is required')


def normalize_embedding(value: Any) -> dict[str, Any]:
    """Normalize external embedding values to the Chunk contract.

    Embeddings can come from Milvus/client adapters as lists, numpy arrays, or
    numeric strings. The internal representation is always JSON-compatible
    ``dict[str, list[float]]``.
    """
    if not isinstance(value, Mapping):
        raise ValueError('embedding must be a mapping')

    if len(value) != 1:
        raise ValueError('embedding must contain exactly one model')
    model, raw_vector = next(iter(value.items()))
    try:
        vector = list(raw_vector)
    except TypeError as exc:
        raise ValueError('embedding vector must be iterable') from exc
    if not vector:
        raise ValueError('embedding vector must be non-empty')
    converted = []
    for item in vector:
        if isinstance(item, bool):
            raise ValueError('embedding vector must contain numbers')
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ValueError('embedding vector must contain numbers') from exc
        if not math.isfinite(number):
            raise ValueError('embedding vector must contain finite numbers')
        converted.append(number)
    norm = math.sqrt(sum(item * item for item in converted))
    if norm == 0:
        raise ValueError('embedding vector norm must be positive')
    return {'model': str(model), 'vector': [item / norm for item in converted]}
