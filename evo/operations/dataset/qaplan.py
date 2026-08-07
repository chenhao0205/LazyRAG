from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR
from math import isfinite
from typing import Any, Mapping


LANES = (
    ('entity_precision_easy', 'entity', 'precision', 'easy'),
    ('entity_precision_medium', 'entity', 'precision', 'medium'),
    ('entity_precision_hard', 'entity', 'precision', 'hard'),
    ('embedding_reasoning_easy', 'embedding', 'reasoning', 'easy'),
    ('embedding_reasoning_medium', 'embedding', 'reasoning', 'medium'),
    ('embedding_reasoning_hard', 'embedding', 'reasoning', 'hard'),
)
LANE_NAMES = tuple(lane[0] for lane in LANES)
REFERENCE_COUNTS = {'easy': 1, 'medium': 2, 'hard': 3}
PRECISION_INSTRUCTION = '''- 必须围绕给定 topic 组织问题，并在 question 中显式出现该 topic 名称。
- question 只围绕一个问题目标。多个 references 可以共同补足唯一答案，但不能被拼接成多个并列子问。
- answer 的结论必须完全由 references 中可直接找到的事实组成。允许抽取、并列、去重和格式化整合直接事实。
- 不得要求或使用计算、比较、资格判断、时间先后判断、因果推断或其他新关系建立；'''
REASONING_INSTRUCTION = '''- 围绕给定 topic 选择问题；topic 是选题引导，references 是唯一事实依据。
- question 必须指向一个唯一、可判定的最终结论，而不能是多个并列问题。
- answer 不能只是任一 reference 中一句话的直接复述；必须将 references 中明确给出的事实、条件或关系进行闭合的归纳或推导后得出。
- 不得依赖外部常识、主观判断、开放式总结或资料未建立的关系。'''


def qaplan_plan(
    ctx: Any,
    inputs: Mapping[str, object],
    *,
    case_ids: tuple[str, ...] | None = None,
) -> dict[str, object]:
    case_ids = _case_ids(case_ids or (), 'qaplan_plan')
    source_config = _mapping(inputs.get('source_config'), 'source_config')
    kb_ids = _string_list(source_config.get('kb_ids'), 'kb_ids')
    imported = _mapping(inputs.get('import_cases_manifest'), 'import_cases_manifest')
    allocation = _mapping(
        _mapping(imported.get('stats'), 'import_cases_manifest.stats').get('case_allocation'),
        'case_allocation',
    )
    target_case_count = _positive_int(allocation.get('target_case_count'), 'target_case_count')
    auto_case_count = _non_negative_int(allocation.get('auto_case_count'), 'auto_case_count')
    import_case_count = _non_negative_int(allocation.get('import_case_count'), 'import_case_count')
    assignments = _mapping(allocation.get('assignments'), 'assignments')
    if target_case_count != len(case_ids) or set(assignments) != set(case_ids):
        raise ValueError('case assignments must match runtime case partitions')
    generated_ids = [
        case_id
        for case_id in case_ids
        if _mapping(assignments[case_id], 'assignment').get('mode') == 'generated'
    ]
    if len(generated_ids) != auto_case_count:
        raise ValueError('auto_case_count must match generated assignments')
    if auto_case_count == 0:
        return {'qaplan_plan': {'source': {'kb_ids': kb_ids}, 'items': [], 'stats': {
            'target_case_count': target_case_count, 'import_case_count': import_case_count,
            'auto_case_count': 0, 'planned_case_count': 0, 'lane_summaries': []},
            'params': {'lane_ratios': {}, 'resolved_lane_quotas': {}, 'lane_order': list(LANE_NAMES)}}}

    ratios = _lane_ratios(inputs.get('qaplan_plan_params'))
    quotas = _allocate_quotas(auto_case_count, ratios)
    chunks = _chunk_map(inputs.get('chunk'))
    clusters = _clusters(inputs.get('topic_discovery_manifest'), chunks)

    items: list[dict[str, object]] = []
    lane_summaries: list[dict[str, object]] = []
    for lane, cluster_type, question_type, difficulty in LANES:
        quota = quotas[lane]
        if quota == 0:
            lane_summaries.append({
                'lane': lane,
                'allocated_case_count': 0,
                'candidate_cluster_count': 0,
                'topic_capacity': 0,
                'selected_cluster_count': 0,
            })
            continue

        reference_count = REFERENCE_COUNTS[difficulty]
        candidates = [
            cluster
            for cluster in clusters
            if cluster['cluster_type'] == cluster_type and cluster['chunk_count'] >= reference_count
        ]
        capacity = sum(len(cluster['topics']) for cluster in candidates)
        if capacity < quota:
            raise ValueError(f'{lane} quota {quota} exceeds topic capacity {capacity}')

        selected = _select_topics(candidates, quota)
        selected_clusters = {cluster['cluster_id'] for cluster, _, _ in selected}
        lane_summaries.append({
            'lane': lane,
            'allocated_case_count': quota,
            'candidate_cluster_count': len(candidates),
            'topic_capacity': capacity,
            'selected_cluster_count': len(selected_clusters),
        })

        for cluster, topic, selection_round in selected:
            references = _references(cluster, reference_count, chunks)
            items.append({
                'case_id': generated_ids[len(items)],
                'plan_item_id': f'qaplan_item_{len(items) + 1:06d}',
                'lane': lane,
                'question_type': question_type,
                'difficulty': difficulty,
                'cluster_id': cluster['cluster_id'],
                'cluster_type': cluster_type,
                'topic': topic,
                'references': references,
                'selection_round': selection_round,
            })

    payload = {
        'source': {'kb_ids': kb_ids},
        'items': items,
        'stats': {
            'target_case_count': target_case_count,
            'import_case_count': import_case_count,
            'auto_case_count': auto_case_count,
            'planned_case_count': len(items),
            'lane_summaries': lane_summaries,
        },
        'params': {
            'lane_ratios': ratios,
            'resolved_lane_quotas': quotas,
            'lane_order': list(LANE_NAMES),
        },
    }
    return {'qaplan_plan': payload}


def qaplan_spec(
    ctx: Any,
    inputs: Mapping[str, object],
    *,
    case_ids: tuple[str, ...] | None = None,
    partition_key: str | None = None,
) -> dict[str, object]:
    case_ids = _case_ids(case_ids or (), 'qaplan_spec')
    case_id = _text(
        partition_key or getattr(ctx, 'partition_key', None),
        'qaplan_spec output partition',
    )
    if case_id not in case_ids:
        raise ValueError('preparation output partition must belong to runtime case partitions')

    imported = _mapping(inputs.get('import_cases_manifest'), 'import_cases_manifest')
    allocation = _mapping(
        _mapping(imported.get('stats'), 'import_cases_manifest.stats').get('case_allocation'),
        'case_allocation',
    )
    assignment = _mapping(_mapping(allocation.get('assignments'), 'assignments').get(case_id), 'assignment')
    mode = _choice(assignment.get('mode'), ('imported', 'generated'), 'assignment.mode')
    if mode == 'imported':
        row = _positive_int(assignment.get('source_row_number'), 'assignment.source_row_number')
        details = imported.get('details')
        if not isinstance(details, list):
            raise ValueError('import_cases_manifest.details must be a list')
        detail = next((
            item
            for item in details
            if isinstance(item, Mapping) and item.get('source_row_number') == row
        ), None)
        case = _mapping(_mapping(detail, 'loaded detail').get('case'), 'loaded detail.case')
        if _text(case.get('id'), 'loaded detail.case.id') != case_id:
            raise ValueError('loaded detail case id mismatch')
        return {'qaplan_spec': {'id': case_id, 'mode': 'imported', 'imported_case': dict(case)}}

    qaplan = _mapping(inputs.get('qaplan_plan'), 'qaplan_plan')
    items = qaplan.get('items')
    if not isinstance(items, list):
        raise ValueError('qaplan.items must be a list')
    stats = _mapping(qaplan.get('stats'), 'qaplan.stats')
    planned_case_count = _non_negative_int(stats.get('planned_case_count'), 'qaplan.stats.planned_case_count')
    if planned_case_count != len(items):
        raise ValueError('planned_case_count must match qaplan.items')
    item = next((
        item
        for item in items
        if isinstance(item, Mapping) and item.get('case_id') == case_id
    ), None)
    item = _mapping(item, 'qaplan item for case')
    question_type = _choice(item.get('question_type'), ('precision', 'reasoning'), 'question_type')
    difficulty = _choice(item.get('difficulty'), ('easy', 'medium', 'hard'), 'difficulty')
    topic = _text(item.get('topic'), 'topic')
    references = _build_references(item.get('references'), difficulty)
    instruction = _instruction(question_type, topic, len(references))

    preparation = {
        'id': case_id, 'mode': 'generated',
        'question_type': question_type,
        'difficulty': difficulty,
        'instruction': instruction,
        'topic': topic,
        'source': {'kb_ids': list(dict.fromkeys(
            reference.get('kb_id', '')
            for reference in references
            if reference.get('kb_id')
        ))},
        'qaplan': {
            'plan_item_id': _text(item.get('plan_item_id'), 'plan_item_id'),
            'lane': _text(item.get('lane'), 'lane'),
            'cluster_id': _text(item.get('cluster_id'), 'cluster_id'),
            'cluster_type': _choice(item.get('cluster_type'), ('entity', 'embedding'), 'cluster_type'),
            'selection_round': _positive_int(item.get('selection_round'), 'selection_round'),
        },
        'references': references,
    }
    return {'qaplan_spec': preparation}


def qaplan_manifest(ctx: Any, inputs: Mapping[str, object]) -> dict[str, object]:
    plan = _mapping(inputs.get('qaplan_plan'), 'qaplan_plan')
    stats = _mapping(plan.get('stats'), 'qaplan_plan.stats')
    planned_case_count = _non_negative_int(stats.get('planned_case_count'), 'qaplan_plan.stats.planned_case_count')
    imported = _mapping(inputs.get('import_cases_manifest'), 'import_cases_manifest')
    allocation = _mapping(
        _mapping(imported.get('stats'), 'import_cases_manifest.stats').get('case_allocation'),
        'case_allocation',
    )
    target_count = _positive_int(allocation.get('target_case_count'), 'target_case_count')
    assignments = _mapping(allocation.get('assignments'), 'assignments')
    specs = inputs.get('qaplan_specs')
    if not isinstance(specs, tuple) or len(specs) != target_count:
        raise ValueError('qaplan_specs count must match target case partition count')

    spec_ids = [_text(_mapping(spec, 'qaplan_specs[]').get('id'), 'qaplan_specs[].id') for spec in specs]
    if len(set(spec_ids)) != len(spec_ids):
        raise ValueError('qaplan_specs ids must be unique')
    if set(spec_ids) != set(assignments):
        raise ValueError('qaplan_specs ids must match assignments')
    generated_ids = set()
    for spec in specs:
        value = _mapping(spec, 'qaplan_specs[]')
        case_id = _text(value.get('id'), 'qaplan_specs[].id')
        mode = _choice(value.get('mode'), ('imported', 'generated'), 'qaplan_specs[].mode')
        expected = _choice(
            _mapping(assignments[case_id], 'assignment').get('mode'),
            ('imported', 'generated'),
            'assignment.mode',
        )
        if mode != expected:
            raise ValueError('qaplan spec mode must match assignment')
        if mode == 'generated':
            generated_ids.add(case_id)
    if len(generated_ids) != planned_case_count:
        raise ValueError('generated qaplan spec count must match planned_case_count')
    return {'qaplan_manifest': {'case_count': len(spec_ids)}}


def _case_ids(values: object, operation: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f'{operation} requires runtime case_ids')
    if any(not isinstance(case_id, str) or not case_id.strip() for case_id in values):
        raise ValueError('runtime case_ids must contain non-empty strings')
    if len(set(values)) != len(values):
        raise ValueError('runtime case_ids must be unique')
    return values


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must be a non-empty list')
    return [_text(item, name) for item in value]


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value


def _lane_ratios(value: object) -> dict[str, object]:
    params = _mapping(value if value is not None else {}, 'qaplan_plan_params')
    raw = params.get('lane_ratios', {})
    raw = _mapping(raw, 'lane_ratios')
    ratios: dict[str, object] = {}
    total = Decimal('0')
    for lane in LANE_NAMES:
        current = raw.get(lane, 1)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            raise ValueError('lane_ratios values must be numbers')
        if isinstance(current, float) and not isfinite(current):
            raise ValueError('lane_ratios values must be finite')
        ratio = Decimal(str(current))
        if ratio < 0:
            raise ValueError('lane_ratios values must be non-negative')
        ratios[lane] = current
        total += ratio
    if total <= 0:
        raise ValueError('lane_ratios must contain a positive value')
    return ratios


def _allocate_quotas(target_case_count: int, ratios: Mapping[str, object]) -> dict[str, int]:
    values = {lane: Decimal(str(ratios[lane])) for lane in LANE_NAMES}
    total = sum(values.values(), Decimal('0'))
    raw = {lane: Decimal(target_case_count) * values[lane] / total for lane in LANE_NAMES}
    quotas = {lane: int(raw[lane].to_integral_value(rounding=ROUND_FLOOR)) for lane in LANE_NAMES}
    remainder = target_case_count - sum(quotas.values())
    ordered = sorted(
        range(len(LANE_NAMES)),
        key=lambda index: (
            -(raw[LANE_NAMES[index]] - quotas[LANE_NAMES[index]]),
            index,
        ),
    )
    for index in ordered[:remainder]:
        quotas[LANE_NAMES[index]] += 1
    return quotas


def _chunk_map(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, tuple):
        raise ValueError('chunk input must be a partitioned tuple')
    chunks: dict[str, dict[str, str]] = {}
    for index, raw in enumerate(value):
        item = _mapping(raw, f'chunk[{index}]')
        if not isinstance(item.get('available'), bool):
            raise ValueError('chunk.available must be boolean')
        chunk_id = _text(item.get('chunk_id'), 'chunk_id')
        if not item['available']:
            continue
        if chunk_id in chunks:
            raise ValueError('available chunk_id values must be unique')
        chunks[chunk_id] = {
            'chunk_id': chunk_id,
            'kb_id': _text(item.get('kb_id'), 'chunk.kb_id') if item.get('kb_id') else '',
            'doc_id': _text(item.get('doc_id'), 'doc_id'),
            'text': item.get('text') if isinstance(item.get('text'), str) else '',
        }
    return chunks


def _clusters(value: object, chunks: Mapping[str, Mapping[str, str]]) -> list[dict[str, object]]:
    manifest = _mapping(value, 'topic_discovery_manifest')
    raw_clusters = manifest.get('clusters')
    if not isinstance(raw_clusters, list):
        raise ValueError('topic_discovery_manifest.clusters must be a list')
    output: list[dict[str, object]] = []
    cluster_ids: set[str] = set()
    for index, raw in enumerate(raw_clusters):
        item = _mapping(raw, f'clusters[{index}]')
        cluster_id = _text(item.get('cluster_id'), 'cluster_id')
        if cluster_id in cluster_ids:
            raise ValueError('cluster_id values must be unique')
        cluster_ids.add(cluster_id)
        cluster_type = _choice(item.get('cluster_type'), ('entity', 'embedding'), 'cluster_type')
        topics = _text_list(item.get('topics'), 'topics')
        chunk_ids = _text_list(item.get('chunk_ids'), 'chunk_ids')
        chunk_count = _positive_int(item.get('chunk_count'), 'chunk_count')
        if chunk_count != len(chunk_ids):
            raise ValueError('chunk_count must match chunk_ids length')
        if any(chunk_id not in chunks for chunk_id in chunk_ids):
            raise ValueError('cluster references a missing or unavailable chunk')
        output.append({
            'cluster_id': cluster_id,
            'cluster_type': cluster_type,
            'topics': topics,
            'chunk_ids': chunk_ids,
            'chunk_count': chunk_count,
        })
    return output


def _select_topics(candidates: list[dict[str, object]], quota: int) -> list[tuple[dict[str, object], str, int]]:
    selected: list[tuple[dict[str, object], str, int]] = []
    topic_index = 0
    while len(selected) < quota:
        for cluster in candidates:
            topics = cluster['topics']
            if topic_index < len(topics):
                selected.append((cluster, topics[topic_index], topic_index + 1))
                if len(selected) == quota:
                    return selected
        topic_index += 1
    return selected


def _references(
    cluster: Mapping[str, object],
    count: int,
    chunks: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    for chunk_id in cluster['chunk_ids'][:count]:
        chunk = chunks[chunk_id]
        if not chunk['text'].strip():
            raise ValueError('referenced chunk text must be non-empty')
        references.append(dict(chunk))
    return references


def _build_references(value: object, difficulty: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError('references must be a list')
    expected = REFERENCE_COUNTS[difficulty]
    if len(value) != expected:
        raise ValueError(f'references count must match {difficulty}')
    output: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f'references[{index}]')
        output.append({
            'kb_id': _text(item.get('kb_id'), 'reference kb_id') if item.get('kb_id') else '',
            'chunk_id': _text(item.get('chunk_id'), 'reference chunk_id'),
            'doc_id': _text(item.get('doc_id'), 'reference doc_id'),
            'text': _text(item.get('text'), 'reference text'),
        })
    return output


def _instruction(question_type: str, topic: str, reference_count: int) -> str:
    if question_type == 'precision':
        return PRECISION_INSTRUCTION
    return REASONING_INSTRUCTION


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _text_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must be a non-empty list')
    return [_text(item, name) for item in value]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _choice(value: object, choices: tuple[str, ...], name: str) -> str:
    item = _text(value, name)
    if item not in choices:
        raise ValueError(f'{name} is invalid')
    return item
