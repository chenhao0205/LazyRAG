from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from evo.operations.dataset.qaplan import LANE_NAMES, LANES, _lane_counts, _topics


def auto_case_count(import_manifest: object) -> int:
    if not isinstance(import_manifest, Mapping):
        return 0
    stats = import_manifest.get('stats')
    if not isinstance(stats, Mapping):
        return 0
    allocation = stats.get('case_allocation')
    if not isinstance(allocation, Mapping):
        return 0
    auto = allocation.get('auto_case_count')
    if isinstance(auto, bool) or not isinstance(auto, int) or auto < 0:
        return 0
    return auto


def eligible_lane_counts(topic_manifest: object) -> dict[str, int]:
    topics = _topics(topic_manifest if isinstance(topic_manifest, Mapping) else {'topics': []})
    counts = dict.fromkeys(LANE_NAMES, 0)
    for lane, question_type, difficulty in LANES:
        required = {'easy': 1, 'medium': 2, 'hard': 3}[difficulty]
        counts[lane] = sum(
            1 for topic in topics
            if topic['question_type'] == question_type and topic['chunk_count'] >= required
        )
    return counts


def default_lane_distribution_exceeds_capacity(
    import_manifest: object,
    topic_manifest: object,
    plan_params: object,
) -> bool:
    auto = auto_case_count(import_manifest)
    if auto == 0:
        return False
    lane_counts = _lane_counts(plan_params if plan_params is not None else {}, auto)
    eligible = eligible_lane_counts(topic_manifest)
    return any(lane_counts[lane] > eligible[lane] for lane in LANE_NAMES)


def question_type_difficulties(lane_counts: Mapping[str, int]) -> dict[str, dict[str, int]]:
    result = {
        question_type: {'easy': 0, 'medium': 0, 'hard': 0}
        for question_type in ('precision', 'reasoning')
    }
    for lane, question_type, difficulty in LANES:
        result[question_type][difficulty] = int(lane_counts[lane])
    return result


def question_type_capacities(eligible: Mapping[str, int]) -> dict[str, dict[str, int]]:
    return question_type_difficulties(eligible)


def _automatic_plan_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    stats = manifest.get('stats')
    if not isinstance(stats, Mapping):
        raise ValueError('cases overview manifest.stats is invalid')
    total = stats.get('auto_case_count')
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise ValueError('cases overview manifest.auto_case_count is invalid')
    summaries = manifest.get('lane_summaries')
    if not isinstance(summaries, list):
        raise ValueError('cases overview manifest.lane_summaries is invalid')
    result = {
        question_type: {'total': 0, 'difficulties': {'easy': 0, 'medium': 0, 'hard': 0}}
        for question_type in ('precision', 'reasoning')
    }
    for item in summaries:
        if not isinstance(item, Mapping):
            raise ValueError('cases overview manifest lane summary is invalid')
        question_type = item.get('question_type')
        difficulty = item.get('difficulty')
        if question_type not in result or difficulty not in result[question_type]['difficulties']:
            raise ValueError('cases overview manifest lane summary is invalid')
        allocated = item.get('allocated_case_count')
        if isinstance(allocated, bool) or not isinstance(allocated, int) or allocated < 0:
            raise ValueError('cases overview lane allocated_case_count is invalid')
        result[question_type]['total'] += allocated
        result[question_type]['difficulties'][difficulty] += allocated
    if sum(item['total'] for item in result.values()) != total:
        raise ValueError('cases overview manifest automatic totals are inconsistent')
    return {'total': total, 'question_types': result}


def project_automatic_plan(
    *,
    manifest: Mapping[str, Any] | None,
    params: Mapping[str, Any],
    topic_manifest: Mapping[str, Any] | None,
    import_manifest: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    auto = auto_case_count(import_manifest)
    if auto == 0:
        return None
    eligible = eligible_lane_counts(topic_manifest or {'topics': []})
    capacities_by_type = question_type_capacities(eligible)
    if manifest is not None:
        plan = _automatic_plan_from_manifest(manifest)
        for question_type in ('precision', 'reasoning'):
            plan['question_types'][question_type]['capacities'] = capacities_by_type[question_type]
        return plan

    lane_counts = _lane_counts(params, auto)
    difficulties_by_type = question_type_difficulties(lane_counts)
    return {
        'total': auto,
        'question_types': {
            question_type: {
                'total': sum(difficulties_by_type[question_type].values()),
                'difficulties': difficulties_by_type[question_type],
                'capacities': capacities_by_type[question_type],
            }
            for question_type in ('precision', 'reasoning')
        },
    }
