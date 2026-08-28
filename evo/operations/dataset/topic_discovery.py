from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from numbers import Integral, Real
from typing import Any

from .llm_json import call_json

DEFAULT_ENTITY_SIMILARITY_THRESHOLD = 0.9
DEFAULT_EDGE_SCORE_THRESHOLD = 0.01
DEFAULT_NOISY_ENTITY_TOP_PERCENT = 0.05
DEFAULT_TOPIC_MERGE_SIMILARITY_THRESHOLD = 0.95
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_N_COMPONENTS = 10
DEFAULT_MIN_CLUSTER_SIZE = 2
DEFAULT_MIN_SAMPLES = 2
DEFAULT_MAX_TOPICS_PER_CLUSTER = 3
DEFAULT_MAX_CHARS_PER_CHUNK_FOR_LABEL = 2048
DEFAULT_MAX_LABEL_SOURCE_CHUNKS = 8

EMBEDDING_LABEL_PROMPT = """Generate concise topic labels for this cluster of chunks.
Return only JSON in this format:
{{"topics":["..."]}}

max_topics: {max_topics}
chunks:
{chunks}"""


@dataclass(frozen=True)
class EntityBuildGraphParams:
    entity_similarity_threshold: float = DEFAULT_ENTITY_SIMILARITY_THRESHOLD
    edge_score_threshold: float = DEFAULT_EDGE_SCORE_THRESHOLD
    noisy_entity_top_percent: float = DEFAULT_NOISY_ENTITY_TOP_PERCENT

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EntityBuildGraphParams':
        return cls(
            entity_similarity_threshold=_bounded_float(
                data.get('entity_similarity_threshold'),
                DEFAULT_ENTITY_SIMILARITY_THRESHOLD,
                'entity_similarity_threshold',
            ),
            edge_score_threshold=_bounded_float(
                data.get('edge_score_threshold'),
                DEFAULT_EDGE_SCORE_THRESHOLD,
                'edge_score_threshold',
            ),
            noisy_entity_top_percent=_bounded_float(
                data.get('noisy_entity_top_percent'),
                DEFAULT_NOISY_ENTITY_TOP_PERCENT,
                'noisy_entity_top_percent',
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            'entity_similarity_threshold': self.entity_similarity_threshold,
            'edge_score_threshold': self.edge_score_threshold,
            'noisy_entity_top_percent': self.noisy_entity_top_percent,
        }


@dataclass(frozen=True)
class EntityClusterParams:
    topic_merge_similarity_threshold: float = DEFAULT_TOPIC_MERGE_SIMILARITY_THRESHOLD

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EntityClusterParams':
        return cls(topic_merge_similarity_threshold=_bounded_float(
            data.get('topic_merge_similarity_threshold'),
            DEFAULT_TOPIC_MERGE_SIMILARITY_THRESHOLD,
            'topic_merge_similarity_threshold',
        ))

    def to_dict(self) -> dict[str, float]:
        return {'topic_merge_similarity_threshold': self.topic_merge_similarity_threshold}


@dataclass(frozen=True)
class EmbeddingClusterParams:
    umap_n_neighbors: int = DEFAULT_UMAP_N_NEIGHBORS
    umap_n_components: int = DEFAULT_UMAP_N_COMPONENTS
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE
    min_samples: int = DEFAULT_MIN_SAMPLES

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EmbeddingClusterParams':
        return cls(
            umap_n_neighbors=_minimum_int(data.get('umap_n_neighbors'), DEFAULT_UMAP_N_NEIGHBORS,
                                          'umap_n_neighbors', 2),
            umap_n_components=_positive_int(data.get('umap_n_components'), DEFAULT_UMAP_N_COMPONENTS,
                                            'umap_n_components'),
            min_cluster_size=_minimum_int(data.get('min_cluster_size'), DEFAULT_MIN_CLUSTER_SIZE,
                                          'min_cluster_size', 2),
            min_samples=_positive_int(data.get('min_samples'), DEFAULT_MIN_SAMPLES, 'min_samples'),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            'umap_n_neighbors': self.umap_n_neighbors,
            'umap_n_components': self.umap_n_components,
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
        }


@dataclass(frozen=True)
class EmbeddingLabelParams:
    max_topics_per_cluster: int = DEFAULT_MAX_TOPICS_PER_CLUSTER
    max_chars_per_chunk_for_label: int = DEFAULT_MAX_CHARS_PER_CHUNK_FOR_LABEL
    max_label_source_chunks: int = DEFAULT_MAX_LABEL_SOURCE_CHUNKS

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'EmbeddingLabelParams':
        return cls(
            max_topics_per_cluster=_positive_int(
                data.get('max_topics_per_cluster'), DEFAULT_MAX_TOPICS_PER_CLUSTER,
                'max_topics_per_cluster', maximum=20,
            ),
            max_chars_per_chunk_for_label=_positive_int(
                data.get('max_chars_per_chunk_for_label'), DEFAULT_MAX_CHARS_PER_CHUNK_FOR_LABEL,
                'max_chars_per_chunk_for_label', maximum=20000,
            ),
            max_label_source_chunks=_positive_int(
                data.get('max_label_source_chunks'), DEFAULT_MAX_LABEL_SOURCE_CHUNKS,
                'max_label_source_chunks',
            ),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            'max_topics_per_cluster': self.max_topics_per_cluster,
            'max_chars_per_chunk_for_label': self.max_chars_per_chunk_for_label,
            'max_label_source_chunks': self.max_label_source_chunks,
        }


def topic_discovery_entity_build_graph(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
    params = EntityBuildGraphParams.from_dict(_mapping(
        inputs.get('topic_discovery_entity_build_graph_params'),
        'topic_discovery_entity_build_graph_params',
    ))
    source_chunks = _chunk_entity_tuple(inputs.get('chunk_entity'))
    nodes, skipped = _entity_nodes(source_chunks)
    if not nodes:
        if source_chunks and all(chunk.get('available') is False for chunk in source_chunks):
            return {'entity_graph': _empty_entity_graph(source_chunks, skipped, params)}
        raise ValueError('topic_discovery_entity_build_graph requires at least one valid node')

    noisy_entities = _noisy_entities(nodes, params)
    edges = _entity_edges(nodes, noisy_entities, params)
    return {'entity_graph': {
        'nodes': nodes,
        'edges': edges,
        'skipped_chunks': skipped,
        'stats': {
            'source_chunk_count': len(source_chunks),
            'node_count': len(nodes),
            'edge_count': len(edges),
            'skipped_chunk_count': len(skipped),
            'noisy_entity_count': len(noisy_entities),
        },
        'params': params.to_dict(),
    }}


def topic_discovery_entity_cluster(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
    params = EntityClusterParams.from_dict(_mapping(
        inputs.get('topic_discovery_entity_cluster_params'),
        'topic_discovery_entity_cluster_params',
    ))
    graph = _mapping(inputs.get('entity_graph'), 'entity_graph')
    if graph.get('nodes') == [] and graph.get('edges') == []:
        return {'entity_clusters': {
            'clusters': [],
            'stats': {'source_node_count': 0, 'source_edge_count': 0, 'edge_cluster_count': 0,
                      'singleton_cluster_count': 0, 'cluster_count': 0, 'topic_merge_count': 0},
            'params': params.to_dict(),
        }}
    nodes = _graph_nodes(graph.get('nodes'))
    edges = _graph_edges(graph.get('edges'), {node['chunk_id'] for node in nodes})

    edge_clusters = _edge_entity_clusters(nodes, edges)
    merged_clusters, merge_count = _merge_entity_clusters(edge_clusters, params)
    covered = {chunk_id for cluster in merged_clusters for chunk_id in cluster['chunk_ids']}
    singleton_clusters = _singleton_entity_clusters(nodes, covered)
    clusters = [*merged_clusters, *singleton_clusters]
    for index, cluster in enumerate(clusters, 1):
        cluster['cluster_id'] = f'entity_cluster_{index:06d}'

    return {'entity_clusters': {
        'clusters': clusters,
        'stats': {
            'source_node_count': len(nodes),
            'source_edge_count': len(edges),
            'edge_cluster_count': len(merged_clusters),
            'singleton_cluster_count': len(singleton_clusters),
            'cluster_count': len(clusters),
            'topic_merge_count': merge_count,
        },
        'params': params.to_dict(),
    }}


def topic_discovery_embedding_cluster(
    ctx: Any,
    inputs: Mapping[str, object],
    reducer: Callable[[list[list[float]], EmbeddingClusterParams], list[list[float]]] | None = None,
    clusterer: Callable[[list[list[float]], EmbeddingClusterParams], list[int]] | None = None,
) -> Mapping[str, object]:
    params = EmbeddingClusterParams.from_dict(_mapping(
        inputs.get('topic_discovery_embedding_cluster_params'),
        'topic_discovery_embedding_cluster_params',
    ))
    chunks = _chunk_tuple(inputs.get('chunk'))
    embedding_chunks, skipped = _embedding_chunks(chunks)
    required_count = _required_embedding_chunk_count(params)
    if len(embedding_chunks) < required_count:
        capacity_skipped = [
            {
                'chunk_id': chunk['chunk_id'],
                'reason': 'insufficient_embedding_capacity',
                'detail': f'{len(embedding_chunks)} eligible embedding chunks; {required_count} required',
            }
            for chunk in embedding_chunks
        ]
        candidates = {
            'clusters': [], 'skipped_chunks': [*skipped, *capacity_skipped],
            'stats': {
                'source_chunk_count': len(chunks),
                'eligible_embedding_chunk_count': len(embedding_chunks),
                'embedding_chunk_count': 0,
                'required_embedding_chunk_count': required_count,
                'skipped_chunk_count': len(skipped) + len(capacity_skipped),
                'candidate_count': 0,
                'noise_candidate_count': 0,
            },
            'params': params.to_dict(),
        }
        return {
            'embedding_cluster_candidates': candidates,
            'embedding_label_requests': (),
            'embedding_label_request': {},
        }

    matrix = [chunk['vector'] for chunk in embedding_chunks]
    reduced = reducer(matrix, params) if reducer is not None else _umap_reduce(matrix, params)
    labels = clusterer(reduced, params) if clusterer is not None else _hdbscan_cluster(reduced, params)
    if len(labels) != len(embedding_chunks):
        raise ValueError('embedding cluster labels must match embedding chunk count')
    clusters, noise_count = _embedding_candidates(embedding_chunks, labels)

    candidates = {
        'clusters': clusters,
        'skipped_chunks': skipped,
        'stats': {
            'source_chunk_count': len(chunks),
            'eligible_embedding_chunk_count': len(embedding_chunks),
            'embedding_chunk_count': len(embedding_chunks),
            'required_embedding_chunk_count': required_count,
            'skipped_chunk_count': len(skipped),
            'candidate_count': len(clusters),
            'noise_candidate_count': noise_count,
        },
        'params': params.to_dict(),
    }
    chunks_by_id = {_required_str(chunk, 'chunk_id'): chunk for chunk in chunks}
    requests = {
        cluster['candidate_id']: _embedding_label_request(cluster, chunks_by_id)
        for cluster in clusters
    }
    return {
        'embedding_cluster_candidates': candidates,
        'embedding_label_requests': tuple(requests),
        'embedding_label_request': requests,
    }


def topic_discovery_embedding_label_cluster(
    ctx: Any,
    inputs: Mapping[str, object],
    llm_complete: Callable[[str], Any] | None = None,
) -> Mapping[str, object]:
    """Label one independently scheduled embedding candidate."""
    del ctx
    params = EmbeddingLabelParams.from_dict(_mapping(
        inputs.get('topic_discovery_embedding_label_params'),
        'topic_discovery_embedding_label_params',
    ))
    request = _mapping(inputs.get('request'), 'request')
    candidate = _candidate_clusters([request])[0]
    source_chunks = _label_request_chunks(request, candidate['chunk_ids'])
    complete = llm_complete or _default_llm_complete()
    prompt = _embedding_label_prompt(source_chunks[:params.max_label_source_chunks], params)
    topics = call_json(
        complete,
        prompt,
        lambda value: _parse_topics(value, params.max_topics_per_cluster),
    )
    return {'embedding_cluster': {
        'cluster_id': candidate['candidate_id'],
        'cluster_type': 'embedding',
        'topics': topics,
        'chunk_ids': list(candidate['chunk_ids']),
        'chunk_count': candidate['chunk_count'],
        'scores': dict(candidate['scores']),
        'metadata': dict(candidate['metadata']),
    }}


def topic_discovery_embedding_label_manifest(
    ctx: Any,
    inputs: Mapping[str, object],
) -> Mapping[str, object]:
    """Publish embedding clusters only after every candidate partition succeeded."""
    del ctx
    params = EmbeddingLabelParams.from_dict(_mapping(
        inputs.get('topic_discovery_embedding_label_params'),
        'topic_discovery_embedding_label_params',
    ))
    candidates = _mapping(inputs.get('embedding_cluster_candidates'), 'embedding_cluster_candidates')
    expected = _candidate_clusters(candidates.get('clusters'))
    partition_ids = _partition_ids(inputs.get('embedding_label_requests'))
    expected_ids = tuple(cluster['candidate_id'] for cluster in expected)
    if partition_ids != expected_ids:
        raise ValueError('embedding label request partitions must match candidate order')

    labeled = _labeled_embedding_clusters(inputs.get('embedding_cluster'))
    by_cluster_id = {cluster['cluster_id']: cluster for cluster in labeled}
    if len(by_cluster_id) != len(labeled) or set(by_cluster_id) != set(expected_ids):
        raise ValueError('missing or duplicate labeled embedding cluster')
    ordered = [by_cluster_id[candidate_id] for candidate_id in expected_ids]
    return {'embedding_clusters': {
        'clusters': ordered,
        'skipped_chunks': list(candidates.get('skipped_chunks') or []),
        'stats': {
            'candidate_count': len(expected),
            'cluster_count': len(ordered),
            'labeled_cluster_count': len(ordered),
        },
        'params': params.to_dict(),
    }}


def topic_discovery_embedding_label(
    ctx: Any,
    inputs: Mapping[str, object],
    llm_complete: Callable[[str], Any] | None = None,
) -> Mapping[str, object]:
    params = EmbeddingLabelParams.from_dict(_mapping(
        inputs.get('topic_discovery_embedding_label_params'),
        'topic_discovery_embedding_label_params',
    ))
    candidates = _mapping(inputs.get('embedding_cluster_candidates'), 'embedding_cluster_candidates')
    clusters = _candidate_clusters(candidates.get('clusters'))
    if not clusters:
        return {'embedding_clusters': {
            'clusters': [], 'skipped_chunks': list(candidates.get('skipped_chunks') or []),
            'stats': {'candidate_count': 0, 'cluster_count': 0, 'labeled_cluster_count': 0},
            'params': params.to_dict(),
        }}
    chunks_by_id = {chunk['chunk_id']: chunk for chunk in _chunk_tuple(inputs.get('chunk'))}
    complete = llm_complete or _default_llm_complete()

    output_clusters = []
    for index, cluster in enumerate(clusters, 1):
        source_chunks = [_chunk_for_label(chunks_by_id, chunk_id) for chunk_id in cluster['chunk_ids']]
        prompt = _embedding_label_prompt(source_chunks[:params.max_label_source_chunks], params)
        topics = call_json(
            complete,
            prompt,
            lambda value: _parse_topics(value, params.max_topics_per_cluster),
        )
        output_clusters.append({
            'cluster_id': f'embedding_cluster_{index:06d}',
            'cluster_type': 'embedding',
            'topics': topics,
            'chunk_ids': list(cluster['chunk_ids']),
            'chunk_count': cluster['chunk_count'],
            'scores': dict(cluster['scores']),
            'metadata': dict(cluster['metadata']),
        })

    return {'embedding_clusters': {
        'clusters': output_clusters,
        'skipped_chunks': list(candidates.get('skipped_chunks') or []),
        'stats': {
            'candidate_count': len(clusters),
            'cluster_count': len(output_clusters),
            'labeled_cluster_count': len(output_clusters),
        },
        'params': params.to_dict(),
    }}


def topic_discovery_manifest(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
    entity = _mapping(inputs.get('entity_clusters'), 'entity_clusters')
    embedding = _mapping(inputs.get('embedding_clusters'), 'embedding_clusters')
    entity_clusters = _final_clusters(entity.get('clusters'), 'entity')
    embedding_clusters = _final_clusters(embedding.get('clusters'), 'embedding')

    topics = []
    for cluster_type, source_clusters in (('entity', entity_clusters), ('embedding', embedding_clusters)):
        question_type = 'precision' if cluster_type == 'entity' else 'reasoning'
        for cluster in source_clusters:
            for name in cluster['topics']:
                topics.append({
                    'topic_id': f'topic_{len(topics) + 1:06d}',
                    'name': name,
                    'question_type': question_type,
                    'chunk_ids': list(cluster['chunk_ids']),
                    'chunk_count': cluster['chunk_count'],
                })
    return {'topic_discovery_manifest': {
        'topics': topics,
        'stats': {
            'total_topic_count': len(topics),
            'question_types': {
                'precision': {'count': sum(topic['question_type'] == 'precision' for topic in topics)},
                'reasoning': {'count': sum(topic['question_type'] == 'reasoning' for topic in topics)},
            },
        },
    }}


def _empty_entity_graph(
    source_chunks: tuple[Mapping[str, Any], ...],
    skipped: list[dict[str, Any]],
    params: EntityBuildGraphParams,
) -> dict[str, Any]:
    return {
        'nodes': [], 'edges': [], 'skipped_chunks': skipped,
        'stats': {'source_chunk_count': len(source_chunks), 'node_count': 0, 'edge_count': 0,
                  'skipped_chunk_count': len(skipped), 'noisy_entity_count': 0},
        'params': params.to_dict(),
    }


def _chunk_entity_tuple(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, tuple):
        raise ValueError('chunk_entity input must be a partitioned tuple')
    return tuple(_mapping(item, 'chunk_entity[]') for item in value)


def _chunk_tuple(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, tuple):
        raise ValueError('chunk input must be a partitioned tuple')
    return tuple(_mapping(item, 'chunk[]') for item in value)


def _entity_nodes(chunks: tuple[Mapping[str, Any], ...]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    nodes = []
    skipped = []
    for chunk in chunks:
        chunk_id = _required_str(chunk, 'chunk_id')
        entities = _string_list(chunk.get('entities'), 'entities', allow_empty=True)
        if chunk.get('available') is False:
            skipped.append({'chunk_id': chunk_id, 'reason': 'unavailable_chunk', 'detail': 'chunk is unavailable'})
            continue
        if not entities:
            skipped.append({'chunk_id': chunk_id, 'reason': 'empty_entities', 'detail': 'chunk entities is empty'})
            continue
        nodes.append({
            'chunk_id': chunk_id,
            'doc_id': _required_str(chunk, 'doc_id'),
            'group': _required_str(chunk, 'group'),
            'entities': entities,
        })
    return nodes, skipped


def _noisy_entities(nodes: list[Mapping[str, Any]], params: EntityBuildGraphParams) -> set[str]:
    counts = Counter(_entity_key(entity) for node in nodes for entity in node['entities'])
    noisy_count = int(len(counts) * params.noisy_entity_top_percent)
    return {entity for entity, _ in counts.most_common(noisy_count)}


def _entity_edges(
    nodes: list[dict[str, Any]],
    noisy_entities: set[str],
    params: EntityBuildGraphParams,
) -> list[dict[str, Any]]:
    edges = []
    for left_index, left in enumerate(nodes):
        for right in nodes[left_index + 1:]:
            edge = _entity_edge(left, right, noisy_entities, params)
            if edge is not None:
                edges.append(edge)
    return edges


def _entity_edge(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    noisy_entities: set[str],
    params: EntityBuildGraphParams,
) -> dict[str, Any] | None:
    comparisons = 0
    overlapped_items: list[str] = []
    seen: set[str] = set()
    for source in left['entities']:
        source_key = _entity_key(source)
        if source_key in noisy_entities:
            continue
        for target in right['entities']:
            target_key = _entity_key(target)
            if target_key in noisy_entities:
                continue
            comparisons += 1
            if _jaro_winkler(source_key, target_key) < params.entity_similarity_threshold:
                continue
            topic = target if len(target.strip()) > len(source.strip()) else source
            topic_key = _entity_key(topic)
            if topic_key not in seen:
                seen.add(topic_key)
                overlapped_items.append(topic.strip())
    if comparisons == 0:
        return None
    score = len(overlapped_items) / comparisons
    if score < params.edge_score_threshold:
        return None
    return {
        'source_chunk_id': left['chunk_id'],
        'target_chunk_id': right['chunk_id'],
        'score': score,
        'overlapped_items': overlapped_items,
    }


def _graph_nodes(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError('entity_graph.nodes must be a non-empty list')
    nodes = []
    seen = set()
    for item in value:
        node = _mapping(item, 'entity_graph.nodes[]')
        chunk_id = _required_str(node, 'chunk_id')
        if chunk_id in seen:
            raise ValueError(f'duplicate node chunk_id: {chunk_id}')
        seen.add(chunk_id)
        nodes.append({
            'chunk_id': chunk_id,
            'doc_id': _required_str(node, 'doc_id'),
            'group': _required_str(node, 'group'),
            'entities': _string_list(node.get('entities'), 'entities'),
        })
    return nodes


def _graph_edges(value: Any, chunk_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError('entity_graph.edges must be a list')
    edges = []
    for item in value:
        edge = _mapping(item, 'entity_graph.edges[]')
        source = _required_str(edge, 'source_chunk_id')
        target = _required_str(edge, 'target_chunk_id')
        if source not in chunk_ids or target not in chunk_ids:
            raise ValueError('edge endpoint must belong to entity_graph.nodes')
        edges.append({
            'source_chunk_id': source,
            'target_chunk_id': target,
            'score': _number(edge.get('score'), 'score'),
            'overlapped_items': _string_list(edge.get('overlapped_items'), 'overlapped_items'),
        })
    return edges


def _edge_entity_clusters(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {node['chunk_id']: index for index, node in enumerate(nodes)}
    clusters = []
    for topic in _topics_in_order(edges):
        topic_edges = [edge for edge in edges if topic in edge['overlapped_items']]
        for component in _connected_components(topic_edges):
            chunk_ids = sorted(component, key=lambda chunk_id: order[chunk_id])
            clusters.append(_cluster_payload('entity', [topic], chunk_ids))
    return clusters


def _topics_in_order(edges: list[dict[str, Any]]) -> list[str]:
    topics = []
    seen = set()
    for edge in edges:
        for topic in edge['overlapped_items']:
            key = _entity_key(topic)
            if key in seen:
                continue
            seen.add(key)
            topics.append(topic)
    return topics


def _connected_components(edges: list[dict[str, Any]]) -> list[set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        source = edge['source_chunk_id']
        target = edge['target_chunk_id']
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)

    components = []
    seen = set()
    for start in adjacency:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        component = set()
        while stack:
            current = stack.pop()
            component.add(current)
            for next_item in adjacency[current]:
                if next_item in seen:
                    continue
                seen.add(next_item)
                stack.append(next_item)
        components.append(component)
    return components


def _merge_entity_clusters(
    clusters: list[dict[str, Any]],
    params: EntityClusterParams,
) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    merge_count = 0
    for cluster in clusters:
        target = next((
            item for item in output
            if _topic_similarity(item['topics'], cluster['topics']) >= params.topic_merge_similarity_threshold
        ), None)
        if target is None:
            output.append(cluster)
            continue
        merge_count += 1
        target['topics'] = _unique([*target['topics'], *cluster['topics']])
        target['chunk_ids'] = _unique([*target['chunk_ids'], *cluster['chunk_ids']])
        target['chunk_count'] = len(target['chunk_ids'])
    return output, merge_count


def _topic_similarity(left: list[str], right: list[str]) -> float:
    return max(_jaro_winkler(_entity_key(a), _entity_key(b)) for a in left for b in right)


def _singleton_entity_clusters(nodes: list[dict[str, Any]], covered: set[str]) -> list[dict[str, Any]]:
    return [
        _cluster_payload('entity', node['entities'], [node['chunk_id']])
        for node in nodes
        if node['chunk_id'] not in covered
    ]


def _embedding_chunks(chunks: tuple[Mapping[str, Any], ...]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    output = []
    skipped = []
    dimension: int | None = None
    for chunk in chunks:
        chunk_id = _required_str(chunk, 'chunk_id')
        if chunk.get('available') is False:
            skipped.append({'chunk_id': chunk_id, 'reason': 'unavailable_chunk', 'detail': 'chunk is unavailable'})
            continue
        try:
            vector = _embedding_vector(chunk.get('embedding'))
        except ValueError as exc:
            skipped.append({'chunk_id': chunk_id, 'reason': 'invalid_embedding', 'detail': str(exc)})
            continue
        if dimension is None:
            dimension = len(vector)
        elif len(vector) != dimension:
            skipped.append({'chunk_id': chunk_id, 'reason': 'dimension_mismatch',
                            'detail': f'embedding dimension {len(vector)} does not match expected {dimension}'})
            continue
        output.append({'chunk_id': chunk_id, 'vector': vector})
    return output, skipped


def _required_embedding_chunk_count(params: EmbeddingClusterParams) -> int:
    return max(
        params.umap_n_neighbors + 1,
        params.umap_n_components + 2,
        params.min_cluster_size,
        params.min_samples,
    )


def _embedding_vector(value: Any) -> list[float]:
    if not isinstance(value, Mapping):
        raise ValueError('embedding must be a mapping')
    if not isinstance(value.get('model'), str) or not value['model']:
        raise ValueError('embedding model is required')
    raw = value.get('vector')
    if not isinstance(raw, list) or not raw:
        raise ValueError('embedding vector must be a non-empty list')
    vector = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError('embedding vector must be list[number]')
        vector.append(float(item))
    if _vector_norm(vector) == 0:
        raise ValueError('embedding vector norm must be positive')
    return vector


def _umap_reduce(matrix: list[list[float]], params: EmbeddingClusterParams) -> list[list[float]]:
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ValueError('UMAP dependency is required') from exc
    reducer = umap.UMAP(n_neighbors=params.umap_n_neighbors, n_components=params.umap_n_components, random_state=42)
    return [[float(item) for item in row] for row in reducer.fit_transform(matrix).tolist()]


def _hdbscan_cluster(matrix: list[list[float]], params: EmbeddingClusterParams) -> list[int]:
    try:
        import hdbscan  # type: ignore
    except ImportError as exc:
        raise ValueError('HDBSCAN dependency is required') from exc
    labels = hdbscan.HDBSCAN(
        min_cluster_size=params.min_cluster_size,
        min_samples=params.min_samples,
    ).fit_predict(matrix)
    return [int(label) for label in labels]


def _embedding_candidates(chunks: list[dict[str, Any]], labels: list[int]) -> tuple[list[dict[str, Any]], int]:
    by_label: dict[int, list[str]] = {}
    noise_count = 0
    for chunk, label in zip(chunks, labels, strict=True):
        if isinstance(label, bool) or not isinstance(label, Integral):
            raise ValueError('embedding cluster labels must be integers')
        label = int(label)
        if label == -1:
            noise_count += 1
            label = -1_000_000 - noise_count
        by_label.setdefault(label, []).append(chunk['chunk_id'])

    order = {chunk['chunk_id']: index for index, chunk in enumerate(chunks)}
    clusters = []
    for index, (_, chunk_ids) in enumerate(
        sorted(by_label.items(), key=lambda item: min(order[chunk_id] for chunk_id in item[1])),
        1,
    ):
        ordered_chunk_ids = sorted(chunk_ids, key=lambda chunk_id: order[chunk_id])
        clusters.append({
            'candidate_id': f'embedding_candidate_{index:06d}',
            'cluster_type': 'embedding',
            'topics': [],
            'chunk_ids': ordered_chunk_ids,
            'chunk_count': len(ordered_chunk_ids),
            'scores': {},
            'metadata': {},
        })
    return clusters, noise_count


def _candidate_clusters(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError('embedding_cluster_candidates.clusters must be a list')
    clusters = []
    for item in value:
        cluster = _mapping(item, 'embedding_cluster_candidates.clusters[]')
        chunk_ids = _string_list(cluster.get('chunk_ids'), 'chunk_ids')
        chunk_count = _positive_int(cluster.get('chunk_count'), len(chunk_ids), 'chunk_count')
        if chunk_count != len(chunk_ids):
            raise ValueError('chunk_count must match chunk_ids length')
        clusters.append({
            'candidate_id': _required_str(cluster, 'candidate_id'),
            'cluster_type': _required_str(cluster, 'cluster_type'),
            'chunk_ids': chunk_ids,
            'chunk_count': chunk_count,
            'scores': _optional_mapping(cluster.get('scores')),
            'metadata': _optional_mapping(cluster.get('metadata')),
        })
    return clusters


def _embedding_label_request(
    candidate: Mapping[str, Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    chunk_ids = list(candidate['chunk_ids'])
    source_chunks = []
    for chunk_id in chunk_ids:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError(f'missing chunk for embedding candidate: {chunk_id}')
        source_chunks.append(chunk)
    return {
        'candidate_id': candidate['candidate_id'],
        'cluster_type': candidate['cluster_type'],
        'topics': [],
        'chunk_ids': chunk_ids,
        'chunk_count': candidate['chunk_count'],
        'scores': dict(candidate['scores']),
        'metadata': dict(candidate['metadata']),
        'chunks': [{
            'chunk_id': chunk_id,
            'kb_id': str(chunk.get('kb_id') or ''),
            'doc_id': str(chunk.get('doc_id') or ''),
            'text': str(chunk.get('text') or ''),
        } for chunk_id, chunk in zip(chunk_ids, source_chunks, strict=True)],
    }


def _label_request_chunks(request: Mapping[str, Any], expected_ids: list[str]) -> list[Mapping[str, Any]]:
    raw = request.get('chunks')
    if not isinstance(raw, list):
        raise ValueError('request.chunks must be a list')
    chunks = tuple(_mapping(item, 'request.chunks[]') for item in raw)
    actual_ids = [_required_str(chunk, 'chunk_id') for chunk in chunks]
    if actual_ids != expected_ids:
        raise ValueError('request.chunks must match chunk_ids order')
    if len(set(actual_ids)) != len(actual_ids):
        raise ValueError('request.chunks must not contain duplicate chunk_id')
    return [_chunk_for_label({chunk_id: chunk for chunk_id, chunk in zip(actual_ids, chunks, strict=True)}, chunk_id)
            for chunk_id in expected_ids]


def _partition_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not all(isinstance(item, str) and item for item in value):
        raise ValueError('embedding_label_requests must be a tuple of partition ids')
    if len(set(value)) != len(value):
        raise ValueError('embedding_label_requests must not contain duplicate partition ids')
    return value


def _labeled_embedding_clusters(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, tuple):
        raise ValueError('embedding_cluster must be a partitioned tuple')
    output = []
    for item in value:
        cluster = _mapping(item, 'embedding_cluster[]')
        chunk_ids = _string_list(cluster.get('chunk_ids'), 'chunk_ids')
        chunk_count = _positive_int(cluster.get('chunk_count'), len(chunk_ids), 'chunk_count')
        if chunk_count != len(chunk_ids):
            raise ValueError('chunk_count must match chunk_ids length')
        if _required_str(cluster, 'cluster_type') != 'embedding':
            raise ValueError('cluster_type must be embedding')
        output.append({
            'cluster_id': _required_str(cluster, 'cluster_id'),
            'cluster_type': 'embedding',
            'topics': _string_list(cluster.get('topics'), 'topics'),
            'chunk_ids': chunk_ids,
            'chunk_count': chunk_count,
            'scores': _optional_mapping(cluster.get('scores')),
            'metadata': _optional_mapping(cluster.get('metadata')),
        })
    return output


def _chunk_for_label(chunks_by_id: Mapping[str, Mapping[str, Any]], chunk_id: str) -> Mapping[str, Any]:
    chunk = chunks_by_id.get(chunk_id)
    if chunk is None:
        raise ValueError(f'missing chunk for label source: {chunk_id}')
    if chunk.get('available') is False:
        raise ValueError(f'chunk is unavailable for label source: {chunk_id}')
    text = str(chunk.get('text') or '')
    if not text.strip():
        raise ValueError(f'missing chunk text for label source: {chunk_id}')
    return chunk


def _embedding_label_prompt(chunks: list[Mapping[str, Any]], params: EmbeddingLabelParams) -> str:
    text = '\n'.join(
        f'- {chunk["chunk_id"]}: {str(chunk.get("text") or "")[:params.max_chars_per_chunk_for_label]}'
        for chunk in chunks
    )
    return EMBEDDING_LABEL_PROMPT.format(max_topics=params.max_topics_per_cluster, chunks=text)


def _parse_topics(value: Any, maximum: int) -> list[str]:
    data = value if isinstance(value, Mapping) else json.loads(str(value).strip())
    topics = _string_list(data.get('topics') if isinstance(data, Mapping) else None, 'topics')
    if len(topics) > maximum:
        raise ValueError(f'topics must contain at most {maximum} items')
    return topics


def _final_clusters(value: Any, cluster_type: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f'{cluster_type}_clusters.clusters must be a list')
    clusters = []
    for item in value:
        cluster = _mapping(item, f'{cluster_type}_clusters.clusters[]')
        if _required_str(cluster, 'cluster_type') != cluster_type:
            raise ValueError(f'cluster_type must be {cluster_type}')
        topics = _string_list(cluster.get('topics'), 'topics')
        chunk_ids = _string_list(cluster.get('chunk_ids'), 'chunk_ids')
        _optional_mapping(cluster.get('scores'))
        _optional_mapping(cluster.get('metadata'))
        chunk_count = _positive_int(cluster.get('chunk_count'), len(chunk_ids), 'chunk_count')
        if chunk_count != len(chunk_ids):
            raise ValueError('chunk_count must match chunk_ids length')
        clusters.append({
            'source_cluster_id': _required_str(cluster, 'cluster_id'),
            'topics': topics,
            'chunk_ids': chunk_ids,
            'chunk_count': chunk_count,
        })
    return clusters


def _cluster_payload(cluster_type: str, topics: list[str], chunk_ids: list[str]) -> dict[str, Any]:
    return {
        'cluster_id': '',
        'cluster_type': cluster_type,
        'topics': _unique(topics),
        'chunk_ids': list(chunk_ids),
        'chunk_count': len(chunk_ids),
        'scores': {},
        'metadata': {},
    }


def _default_llm_complete() -> Callable[[str], Any]:
    from evo.llm import LazyLLMClient

    return LazyLLMClient(model='evo_llm')


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _optional_mapping(value: object) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, 'mapping')


def _required_str(value: Mapping[str, Any], name: str) -> str:
    text = str(value.get(name) or '').strip()
    if not text:
        raise ValueError(f'{name} must be a non-empty string')
    return text


def _string_list(value: Any, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be list[string]')
    output = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f'{name} must contain only non-empty strings')
        output.append(item.strip())
    if not output and not allow_empty:
        raise ValueError(f'{name} must be non-empty')
    return output


def _bounded_float(value: Any, default: float, name: str) -> float:
    if value is None:
        return default
    output = _number(value, name)
    if output < 0 or output > 1:
        raise ValueError(f'{name} must be between 0 and 1')
    return output


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a number')
    return float(value)


def _positive_int(value: Any, default: int, name: str, *, maximum: int | None = None) -> int:
    if value is None:
        output = default
    else:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f'{name} must be a positive integer')
        output = int(value)
    if output < 1:
        raise ValueError(f'{name} must be a positive integer')
    if maximum is not None and output > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return output


def _minimum_int(value: Any, default: int, name: str, minimum: int) -> int:
    output = _positive_int(value, default, name)
    if output < minimum:
        raise ValueError(f'{name} must be an integer greater than 1')
    return output


def _entity_key(value: str) -> str:
    return value.strip().lower()


def _unique(values: list[str]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        key = _entity_key(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = _vector_norm(vector)
    return [item / norm for item in vector]


def _vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(item * item for item in vector))


def _jaro_winkler(left: str, right: str, prefix_scale: float = 0.1) -> float:
    if left == right:
        return 1.0
    jaro = _jaro_similarity(left, right)
    prefix = 0
    for left_char, right_char in zip(left[:4], right[:4]):
        if left_char != right_char:
            break
        prefix += 1
    return jaro + prefix * prefix_scale * (1 - jaro)


def _jaro_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    match_distance = max(len(left), len(right)) // 2 - 1
    left_matches = [False] * len(left)
    right_matches = [False] * len(right)
    matches = 0
    for left_index, left_char in enumerate(left):
        start = max(0, left_index - match_distance)
        end = min(left_index + match_distance + 1, len(right))
        for right_index in range(start, end):
            if right_matches[right_index] or left_char != right[right_index]:
                continue
            left_matches[left_index] = True
            right_matches[right_index] = True
            matches += 1
            break
    if matches == 0:
        return 0.0
    transpositions = 0
    right_index = 0
    for left_index, left_char in enumerate(left):
        if not left_matches[left_index]:
            continue
        while not right_matches[right_index]:
            right_index += 1
        if left_char != right[right_index]:
            transpositions += 1
        right_index += 1
    return (
        matches / len(left)
        + matches / len(right)
        + (matches - transpositions / 2) / matches
    ) / 3
