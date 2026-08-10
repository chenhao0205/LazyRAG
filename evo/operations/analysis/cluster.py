from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from rapidfuzz.distance import Levenshtein
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from evo.operations.public_contracts import (
    clean_text as _text,
    mapping_or_empty as _mapping,
    number_or_default as _number,
)

try:
    from apted import APTED
    from apted.helpers import Tree
except ModuleNotFoundError:
    APTED = None
    Tree = None

TRACE_FEATURES = (
    'node_count', 'edge_count', 'max_depth', 'branching_factor_avg', 'error_span_count', 'trace_latency_ms',
    'exclusive_latency_ms', 'retrieved_doc_count', 'retrieved_chunk_count',
)
STAGES = ('query_rewrite', 'retrieve', 'rerank', 'context_assembly', 'prompt_build', 'tool_call', 'llm_generate',
          'postprocess', 'stream')


@dataclass(frozen=True)
class ClusterConfig:
    categorical_weight: float = 0.35
    route_weight: float = 0.25
    tree_weight: float = 0.25
    numeric_weight: float = 0.15
    distance_threshold: float = 0.45


DEFAULT_CLUSTER_CONFIG = ClusterConfig()
MAX_EXACT_CLUSTER_CASES = 500


def cluster_traces(classifications: tuple[Mapping[str, Any], ...]) -> dict[str, Any]:
    rows = sorted(
        (dict(row) for row in classifications if isinstance(row, Mapping)),
        key=lambda row: _text(row.get('case_id')),
    )
    if len(rows) != len(classifications):
        raise ValueError('analysis.trace_clusters classifications must all be mappings')
    if not rows:
        return _result(rows, [], [])
    scalable = len(rows) > MAX_EXACT_CLUSTER_CASES
    if len(rows) < 5 or scalable:
        labels = _small_labels(rows)
        matrix = np.zeros((len(rows), 1))
    else:
        matrix = _feature_matrix(rows)
        distances = _distances(rows, matrix, DEFAULT_CLUSTER_CONFIG)
        labels = _cluster_labels(distances, DEFAULT_CLUSTER_CONFIG)
    _assign_stable_ids(rows, labels)
    if scalable:
        _assign_bucket_outliers(rows, labels)
    else:
        _assign_outliers(rows, distances if len(rows) >= 20 else None)
    groups = _groups(rows)
    for members in groups.values():
        for row in members:
            row['cluster_size'] = len(members)
            row['outlier'] = float(row.get('outlier_score') or 0.0) >= 0.8
    clusters = [
        _cluster_summary(cluster_id, members, matrix, rows)
        for cluster_id, members in sorted(groups.items())
    ]
    outliers = [
        {
            'case_id': row['case_id'],
            'cluster_id': row['cluster_id'],
            'outlier_score': row['outlier_score'],
        }
        for row in sorted(rows, key=lambda r: _text(r.get('case_id')))
        if row.get('outlier')
    ]
    return _result(rows, sorted(clusters, key=lambda c: c['cluster_id']), outliers)


def _result(
    rows: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    outliers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        'id': 'analysis.trace_clusters',
        'total': len(rows),
        'strategy': (
            'fingerprint_bucket'
            if len(rows) > MAX_EXACT_CLUSTER_CASES
            else 'agglomerative_distance'
        ),
        'clusters': clusters,
        'outliers': outliers,
        'rows': [
            {
                'case_id': row.get('case_id', ''),
                'cluster_id': row.get('cluster_id', ''),
                'cluster_size': row.get('cluster_size', 0),
                'outlier': bool(row.get('outlier')),
                'outlier_score': row.get('outlier_score', 0.0),
                'issue_type': row.get('issue_type', ''),
                'affected_block': row.get('affected_block', ''),
                'failure_mode': row.get('failure_mode', ''),
                'route_signature': _trace(row).get('route_signature', ''),
            }
            for row in rows
        ],
    }


def _small_labels(rows: list[dict[str, Any]]) -> list[int]:
    labels: list[int] = []
    seen: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        key = (_text(row.get('issue_type')), _text(row.get('affected_block')), _text(row.get('failure_mode')),
               _text(_trace(row).get('route_signature')))
        seen.setdefault(key, len(seen))
        labels.append(seen[key])
    return labels


def _feature_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    raw = DictVectorizer(sparse=False).fit_transform([_feature_row(row) for row in rows])
    if raw.size == 0:
        return np.zeros((len(rows), 1))
    return StandardScaler().fit_transform(raw)


def _feature_row(row: Mapping[str, Any]) -> dict[str, float | str]:
    trace, judge = _trace(row), _mapping(row.get('judge'))
    features: dict[str, float | str] = {
        'question_type': _text(row.get('question_type') or _mapping(row.get('case')).get('question_type')),
        'issue_category': _text(row.get('issue_category')),
        'issue_type': _text(row.get('issue_type')),
        'affected_block': _text(row.get('affected_block')),
        'failure_mode': _text(row.get('failure_mode')),
        'confidence': _text(row.get('confidence')),
        'bottleneck_stage': _text(trace.get('bottleneck_stage')),
        'pending_analysis': float(bool(row.get('pending_analysis'))),
        'actionable': float(bool(row.get('actionable'))),
    }
    for key in ('answer_quality_score', 'retrieval_quality_score', 'overall_score', 'context_recall',
                'context_precision', 'chunk_recall', 'chunk_precision', 'doc_recall', 'doc_precision'):
        features[key] = _number(judge.get(key))
    trace_features = trace.get('features') if isinstance(trace.get('features'), Mapping) else {}
    for key in TRACE_FEATURES:
        features[f'trace.{key}'] = _number(trace_features.get(key))
    for stage in STAGES:
        features[f'trace.stage_count.{stage}'] = _number(trace_features.get(f'stage_count.{stage}'))
        features[f'trace.latency.{stage}'] = _number(trace_features.get(f'latency.{stage}'))
    return features


def _distances(rows: list[dict[str, Any]], matrix: np.ndarray,
               config: ClusterConfig = DEFAULT_CLUSTER_CONFIG) -> np.ndarray:
    return np.nan_to_num(
        config.categorical_weight * _categorical_distances(rows)
        + config.route_weight * _route_distances(rows)
        + config.tree_weight * _tree_distances(rows)
        + config.numeric_weight * pairwise_distances(matrix, metric='cosine')
    )


def _cluster_labels(distances: np.ndarray, config: ClusterConfig = DEFAULT_CLUSTER_CONFIG) -> np.ndarray:
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=config.distance_threshold,
        metric='precomputed',
        linkage='average',
    )
    return model.fit_predict(distances)


def _categorical_distances(rows: list[dict[str, Any]]) -> np.ndarray:
    fields = ('question_type', 'issue_category', 'issue_type', 'affected_block', 'failure_mode', 'confidence',
              'bottleneck_stage')
    values = [
        tuple(_text(row.get(field) or _trace(row).get(field)) for field in fields)
        for row in rows
    ]
    distances = np.zeros((len(rows), len(rows)))
    for i, left in enumerate(values):
        for j in range(i + 1, len(values)):
            value = sum(a != b for a, b in zip(left, values[j], strict=True)) / len(fields)
            distances[i, j] = distances[j, i] = value
    return distances


def _route_distances(rows: list[dict[str, Any]]) -> np.ndarray:
    distances = np.zeros((len(rows), len(rows)))
    for i, left in enumerate(rows):
        for j in range(i + 1, len(rows)):
            value = Levenshtein.normalized_distance(
                _text(_trace(left).get('route_signature')),
                _text(_trace(rows[j]).get('route_signature')),
            )
            distances[i, j] = distances[j, i] = float(value)
    return distances


def _tree_distances(rows: list[dict[str, Any]]) -> np.ndarray:
    distances = np.zeros((len(rows), len(rows)))
    tree_texts = [_text(_trace(row).get('tree_text')) or '{unknown}' for row in rows]
    if APTED is None or Tree is None:
        for i, left in enumerate(tree_texts):
            for j in range(i + 1, len(tree_texts)):
                distances[i, j] = distances[j, i] = float(
                    Levenshtein.normalized_distance(left, tree_texts[j])
                )
        return distances

    trees = [_tree(text) for text in tree_texts]
    sizes = [max(1, text.count('{')) for text in tree_texts]
    for i, left in enumerate(trees):
        for j in range(i + 1, len(trees)):
            distances[i, j] = distances[j, i] = APTED(left, trees[j]).compute_edit_distance() / max(sizes[i], sizes[j])
    return distances


def _assign_stable_ids(rows: list[dict[str, Any]], labels: list[int] | np.ndarray) -> None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=True):
        grouped[int(label)].append(row)
    ordered = sorted(grouped.values(), key=lambda members: (_fingerprint(members), _text(members[0].get('case_id'))))
    for index, members in enumerate(ordered, 1):
        for row in members:
            row['cluster_id'] = f'cluster_{index:04d}'


def _assign_outliers(rows: list[dict[str, Any]], distances: np.ndarray | None) -> None:
    if distances is None or len(rows) < 20:
        for row in rows:
            row['outlier_score'] = 0.0
        return
    lof = LocalOutlierFactor(n_neighbors=min(20, len(rows) - 1), metric='precomputed', contamination='auto')
    lof.fit_predict(distances)
    raw = -lof.negative_outlier_factor_
    low, high = float(np.min(raw)), float(np.max(raw))
    scores = [0.0 if high <= low else float((value - low) / (high - low)) for value in raw]
    for row, score in zip(rows, scores, strict=True):
        row['outlier_score'] = round(score, 4)


def _assign_bucket_outliers(
    rows: list[dict[str, Any]],
    labels: list[int] | np.ndarray,
) -> None:
    counts = Counter(int(label) for label in labels)
    largest = max(counts.values(), default=1)
    for row, label in zip(rows, labels, strict=True):
        size = counts[int(label)]
        row['outlier_score'] = round(1.0 - min(1.0, size / max(2, largest)), 4)


def _cluster_summary(cluster_id: str, members: list[dict[str, Any]], matrix: np.ndarray,
                     rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(members) == 1 or (matrix.shape[1] == 1 and not np.any(matrix)):
        rep = members[0]
    else:
        positions = {id(row): index for index, row in enumerate(rows)}
        indices = [positions[id(row)] for row in members]
        rep = rows[indices[int(np.argmin(
            pairwise_distances(
                matrix[indices],
                np.mean(matrix[indices], axis=0).reshape(1, -1),
            ).ravel()
        ))]]
    issues = Counter(_text(row.get('issue_type')) for row in members)
    blocks = Counter(_text(row.get('affected_block')) for row in members)
    modes = Counter(_text(row.get('failure_mode')) for row in members)
    routes = Counter(_text(_trace(row).get('route_signature')) for row in members)
    return {
        'cluster_id': cluster_id,
        'size': len(members),
        'case_ids': [_text(row.get('case_id')) for row in members],
        'representative_case_id': _text(rep.get('case_id')),
        'dominant_issue_type': issues.most_common(1)[0][0],
        'dominant_affected_block': blocks.most_common(1)[0][0],
        'dominant_failure_mode': modes.most_common(1)[0][0],
        'common_route_signature': routes.most_common(1)[0][0] if routes else '',
        'issue_type_counts': dict(issues),
        'affected_block_counts': dict(blocks),
        'failure_mode_counts': dict(modes),
        'avg_overall_score': _avg(_mapping(row.get('judge')).get('overall_score') for row in members),
        'avg_retrieval_quality_score': _avg(
            _mapping(row.get('judge')).get('retrieval_quality_score')
            for row in members
        ),
        'avg_answer_quality_score': _avg(_mapping(row.get('judge')).get('answer_quality_score') for row in members),
    }


def _groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row.get('cluster_id'))].append(row)
    return dict(grouped)


def _fingerprint(rows: list[Mapping[str, Any]]) -> tuple[str, str, str, str]:
    first = rows[0]
    return (_text(first.get('issue_type')), _text(first.get('affected_block')), _text(first.get('failure_mode')),
            _text(_trace(first).get('route_signature')))


def _tree(value: str) -> Any:
    try:
        return Tree.from_text(value)
    except Exception:
        return Tree.from_text('{unknown}')


def _trace(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(row.get('trace_summary'))


def _avg(values: Any) -> float:
    rows = [_number(value) for value in values]
    return round(float(np.mean(rows)), 4) if rows else 0.0
