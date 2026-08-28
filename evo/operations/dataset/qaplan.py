from __future__ import annotations

from typing import Any, Mapping


LANES = (
    ('precision_easy', 'precision', 'easy'), ('precision_medium', 'precision', 'medium'),
    ('precision_hard', 'precision', 'hard'), ('reasoning_easy', 'reasoning', 'easy'),
    ('reasoning_medium', 'reasoning', 'medium'), ('reasoning_hard', 'reasoning', 'hard'),
)
LANE_NAMES = tuple(item[0] for item in LANES)
REFERENCE_COUNTS = {'easy': 1, 'medium': 2, 'hard': 3}
PRECISION_INSTRUCTION = '''- 必须围绕给定 topic 组织问题，并在 question 中显式出现该 topic 名称。
- question 只围绕一个问题目标。多个 references 可以共同补足唯一答案，但不能被拼接成多个并列子问。
- answer 的结论必须完全由 references 中可直接找到的事实组成。允许抽取、并列、去重和格式化整合直接事实。
- 不得要求或使用计算、比较、资格判断、时间先后判断、因果推断或其他新关系建立；'''
REASONING_INSTRUCTION = '''- 围绕给定 topic 选择问题；topic 是选题引导，references 是唯一事实依据。
- question 必须指向一个唯一、可判定的最终结论，而不能是多个并列问题。
- answer 不能只是任一 reference 中一句话的直接复述；必须将 references 中明确给出的事实、条件或关系进行闭合的归纳或推导后得出。
- 不得依赖外部常识、主观判断、开放式总结或资料未建立的关系。'''


def qaplan_plan(ctx: Any, inputs: Mapping[str, object]) -> dict[str, object]:
    case_ids = _case_ids(ctx, 'qaplan_plan')
    allocation = _allocation(inputs)
    target, imported_count, auto_count, assignments = allocation
    if target != len(case_ids) or set(assignments) != set(case_ids):
        raise ValueError('case assignments must match runtime case partitions')
    generated_ids = [case_id for case_id in case_ids if _mode(assignments[case_id]) == 'generated']
    if len(generated_ids) != auto_count:
        raise ValueError('auto_case_count must match generated assignments')
    quotas = _lane_counts(inputs.get('qaplan_plan_params'), auto_count)
    if auto_count == 0:
        quotas = dict.fromkeys(LANE_NAMES, 0)
        topics = []
    else:
        topics = _topics(inputs.get('topic_discovery_manifest'))

    items, summaries = [], []
    for lane, question_type, difficulty in LANES:
        candidates = [topic for topic in topics if topic['question_type'] == question_type and topic['chunk_count'] >= REFERENCE_COUNTS[difficulty]]
        quota = quotas[lane]
        summaries.append({'lane': lane, 'allocated_case_count': quota, 'eligible_topic_count': len(candidates)})
        if quota > len(candidates):
            raise ValueError(f'{lane} quota {quota} exceeds eligible topics {len(candidates)}')
        for topic in candidates[:quota]:
            index = len(items)
            items.append({'case_id': generated_ids[index], 'plan_item_id': f'qaplan_item_{index + 1:06d}',
                          'lane': lane, 'question_type': question_type, 'difficulty': difficulty,
                          'topic_id': topic['topic_id']})
    return {'qaplan_plan': {'items': items, 'stats': {
        'target_case_count': target, 'import_case_count': imported_count, 'auto_case_count': auto_count,
        'planned_case_count': len(items), 'lane_summaries': summaries,
    }, 'params': {'lane_case_counts': quotas, 'lane_order': list(LANE_NAMES)}}}


def qaplan_spec(ctx: Any, inputs: Mapping[str, object]) -> dict[str, object]:
    case_ids = _case_ids(ctx, 'qaplan_spec')
    output = getattr(ctx, 'output_key_by_name', {}).get('qaplan_spec')
    case_id = _text(getattr(output, 'partition', None), 'qaplan_spec output partition')
    if case_id not in case_ids:
        raise ValueError('preparation output partition must belong to runtime case partitions')
    return {'qaplan_spec': build_qaplan_spec(
        case_id,
        inputs.get('import_cases_manifest'),
        inputs.get('qaplan_plan'),
        inputs.get('topic_discovery_manifest'),
        inputs.get('chunk'),
    )}


def build_qaplan_spec(case_id: str, import_cases_manifest: object, qaplan_plan_value: object,
                      topic_manifest: object, chunks_value: object) -> dict[str, object]:
    """Build one Case specification from the same facts used by the qaplan operation."""
    imported = _mapping(import_cases_manifest, 'import_cases_manifest')
    _, _, _, assignments = _allocation({'import_cases_manifest': import_cases_manifest})
    assignment = _mapping(assignments.get(case_id), 'assignment')
    if _mode(assignment) == 'imported':
        row = _positive_int(assignment.get('source_row_number'), 'assignment.source_row_number')
        details = imported.get('details')
        if not isinstance(details, list): raise ValueError('import_cases_manifest.details must be a list')
        detail = next((item for item in details if isinstance(item, Mapping) and item.get('source_row_number') == row), None)
        case = _mapping(_mapping(detail, 'loaded detail').get('case'), 'loaded detail.case')
        if _text(case.get('id'), 'loaded detail.case.id') != case_id: raise ValueError('loaded detail case id mismatch')
        return {'id': case_id, 'mode': 'imported', 'imported_case': dict(case)}

    plan = _mapping(qaplan_plan_value, 'qaplan_plan')
    raw_items = plan.get('items')
    if not isinstance(raw_items, list): raise ValueError('qaplan.items must be a list')
    matches = [item for item in raw_items if isinstance(item, Mapping) and item.get('case_id') == case_id]
    if len(matches) != 1: raise ValueError('qaplan item for case must be unique')
    item = _mapping(matches[0], 'qaplan item for case')
    question_type = _choice(item.get('question_type'), ('precision', 'reasoning'), 'question_type')
    difficulty = _choice(item.get('difficulty'), ('easy', 'medium', 'hard'), 'difficulty')
    topic_id = _text(item.get('topic_id'), 'topic_id')
    topics = _topics(topic_manifest)
    topic_matches = [topic for topic in topics if topic['topic_id'] == topic_id]
    if len(topic_matches) != 1: raise ValueError('topic_id must resolve to exactly one current topic')
    topic = topic_matches[0]
    if topic['question_type'] != question_type or topic['chunk_count'] < REFERENCE_COUNTS[difficulty]:
        raise ValueError('topic_id does not satisfy question_type or chunk_count')
    chunks = _chunk_map(chunks_value)
    references = []
    for chunk_id in topic['chunk_ids'][:REFERENCE_COUNTS[difficulty]]:
        if chunk_id not in chunks: raise ValueError('topic_id references an unavailable current chunk')
        references.append(chunks[chunk_id])
    return {'id': case_id, 'mode': 'generated', 'question_type': question_type,
        'difficulty': difficulty, 'topic': {'topic_id': topic_id, 'name': topic['name']},
        'instruction': _instruction(question_type),
        'qaplan': {'plan_item_id': _text(item.get('plan_item_id'), 'plan_item_id'), 'lane': _text(item.get('lane'), 'lane')},
        'references': references}


def qaplan_manifest(ctx: Any, inputs: Mapping[str, object]) -> dict[str, object]:
    plan = _mapping(inputs.get('qaplan_plan'), 'qaplan_plan')
    stats = _mapping(plan.get('stats'), 'qaplan_plan.stats')
    target, imported_count, auto_count, assignments = _allocation(inputs)
    if tuple(_non_negative_int(stats.get(key), f'qaplan_plan.stats.{key}') for key in ('target_case_count', 'import_case_count', 'auto_case_count')) != (target, imported_count, auto_count):
        raise ValueError('qaplan_plan stats must match case allocation')
    planned = _non_negative_int(stats.get('planned_case_count'), 'planned_case_count')
    if planned != auto_count: raise ValueError('planned_case_count must match auto_case_count')
    raw = stats.get('lane_summaries')
    if not isinstance(raw, list) or len(raw) != len(LANES): raise ValueError('qaplan_plan.stats.lane_summaries must contain the six lanes')
    summaries=[]
    for (lane, question_type, difficulty), value in zip(LANES, raw, strict=True):
        value=_mapping(value, 'lane summary')
        if value.get('lane') != lane: raise ValueError('qaplan lane summaries must follow the six-lane order')
        allocated=_non_negative_int(value.get('allocated_case_count'), 'allocated_case_count'); eligible=_non_negative_int(value.get('eligible_topic_count'), 'eligible_topic_count')
        if allocated > eligible: raise ValueError('allocated_case_count cannot exceed eligible_topic_count')
        summaries.append({'lane':lane,'question_type':question_type,'difficulty':difficulty,'allocated_case_count':allocated,'eligible_topic_count':eligible})
    if sum(item['allocated_case_count'] for item in summaries) != planned: raise ValueError('lane allocated case counts must match planned_case_count')
    specs=inputs.get('qaplan_specs')
    if not isinstance(specs, tuple) or len(specs)!=target: raise ValueError('qaplan_specs count must match target case partition count')
    ids=[_text(_mapping(spec,'qaplan_specs[]').get('id'),'qaplan_specs[].id') for spec in specs]
    if len(set(ids)) != len(ids) or set(ids) != set(assignments): raise ValueError('qaplan_specs ids must be unique and match assignments')
    for spec in specs:
        value=_mapping(spec,'qaplan_specs[]'); case_id=_text(value.get('id'),'qaplan_specs[].id')
        if _choice(value.get('mode'),('imported','generated'),'qaplan_specs[].mode') != _mode(assignments[case_id]): raise ValueError('qaplan spec mode must match assignment')
    return {'qaplan_manifest': {'stats': {'target_case_count':target,'import_case_count':imported_count,'auto_case_count':auto_count,'planned_case_count':planned}, 'lane_summaries':summaries}}


def _allocation(inputs: Mapping[str, object]) -> tuple[int, int, int, Mapping[str, object]]:
    manifest=_mapping(inputs.get('import_cases_manifest'),'import_cases_manifest'); stats=_mapping(manifest.get('stats'),'import_cases_manifest.stats'); value=_mapping(stats.get('case_allocation'),'case_allocation')
    target=_positive_int(value.get('target_case_count'),'target_case_count'); imported=_non_negative_int(value.get('import_case_count'),'import_case_count'); auto=_non_negative_int(value.get('auto_case_count'),'auto_case_count'); assignments=_mapping(value.get('assignments'),'assignments')
    if target != imported + auto: raise ValueError('target_case_count must equal imported and automatic case counts')
    return target, imported, auto, assignments

def _lane_counts(value: object, auto: int) -> dict[str,int]:
    params=_mapping(value if value is not None else {},'qaplan_plan_params')
    if 'lane_ratios' in params: raise ValueError('lane_ratios is not supported')
    raw=params.get('lane_case_counts')
    if raw is None:
        quotient,remainder=divmod(auto,len(LANE_NAMES)); return {lane: quotient + int(index < remainder) for index,lane in enumerate(LANE_NAMES)}
    raw=_mapping(raw,'lane_case_counts')
    if set(raw)!=set(LANE_NAMES): raise ValueError('lane_case_counts must contain exactly the six lanes')
    counts={lane:_non_negative_int(raw[lane],'lane_case_counts') for lane in LANE_NAMES}
    if sum(counts.values()) != auto: raise ValueError('lane_case_counts must sum to auto_case_count')
    return counts

def _topics(value: object) -> list[dict[str, object]]:
    manifest=_mapping(value,'topic_discovery_manifest'); raw=manifest.get('topics')
    if not isinstance(raw,list): raise ValueError('topic_discovery_manifest.topics must be a list')
    output=[]; ids=set()
    for item in raw:
        value=_mapping(item,'topics[]'); topic_id=_text(value.get('topic_id'),'topic_id')
        if topic_id in ids: raise ValueError('topic_id values must be unique')
        ids.add(topic_id); chunk_ids=_string_list(value.get('chunk_ids'),'chunk_ids'); count=_positive_int(value.get('chunk_count'),'chunk_count')
        if count != len(chunk_ids): raise ValueError('chunk_count must match chunk_ids length')
        output.append({'topic_id':topic_id,'name':_text(value.get('name'),'topic.name'),'question_type':_choice(value.get('question_type'),('precision','reasoning'),'topic.question_type'),'chunk_ids':chunk_ids,'chunk_count':count})
    return output

def _chunk_map(value: object) -> dict[str,dict[str,str]]:
    if not isinstance(value,tuple): raise ValueError('chunk input must be a partitioned tuple')
    output={}
    for raw in value:
        item=_mapping(raw,'chunk[]'); chunk_id=_text(item.get('chunk_id'),'chunk_id')
        if not item.get('available'): continue
        if chunk_id in output: raise ValueError('available chunk_id values must be unique')
        output[chunk_id]={'kb_id':_text(item.get('kb_id'),'chunk.kb_id'),'doc_id':_text(item.get('doc_id'),'chunk.doc_id'),'chunk_id':chunk_id,'text':_text(item.get('text'),'chunk.text')}
    return output

def _instruction(question_type: str) -> str: return PRECISION_INSTRUCTION if question_type == 'precision' else REASONING_INSTRUCTION
def _mode(value: object) -> str: return _choice(_mapping(value,'assignment').get('mode'),('imported','generated'),'assignment.mode')
def _case_ids(ctx: Any, operation: str) -> tuple[str,...]:
    values=getattr(ctx,'case_ids',())
    if not isinstance(values,tuple) or not values or len(set(values))!=len(values): raise ValueError(f'{operation} requires unique runtime case_ids')
    return tuple(_text(value,'case_id') for value in values)
def _mapping(value: object,name: str)->Mapping[str,object]:
    if not isinstance(value,Mapping): raise ValueError(f'{name} must be a mapping')
    return value
def _text(value:object,name:str)->str:
    if not isinstance(value,str) or not value.strip(): raise ValueError(f'{name} must be a non-empty string')
    return value
def _string_list(value:object,name:str)->list[str]:
    if not isinstance(value,list) or not value: raise ValueError(f'{name} must be a non-empty list')
    return [_text(item,name) for item in value]
def _positive_int(value:object,name:str)->int:
    if isinstance(value,bool) or not isinstance(value,int) or value<1: raise ValueError(f'{name} must be a positive integer')
    return value
def _non_negative_int(value:object,name:str)->int:
    if isinstance(value,bool) or not isinstance(value,int) or value<0: raise ValueError(f'{name} must be a non-negative integer')
    return value
def _choice(value:object,choices:tuple[str,...],name:str)->str:
    result=_text(value,name)
    if result not in choices: raise ValueError(f'{name} is invalid')
    return result
