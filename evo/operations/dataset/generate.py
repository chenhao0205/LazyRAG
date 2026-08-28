from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .llm_json import call_json


REFERENCE_COUNTS = {'easy': 1, 'medium': 2, 'hard': 3}
SHARED_PROMPT = '''你将基于给定的 topic 和 reference materials，生成一条可用于 RAG 评测的 QA。

只能依据提供的 reference materials 生成内容；topic 仅用于选题引导，不能作为额外事实来源。
question 和 answer 都必须实质性使用全部提供的 references，不得只基于其中的子集生成内容。
question 必须指代明确、范围不超过 references，且应存在唯一、可判定的答案；不得生成开放式、主观性或并列拼接的多个问题。
question、answer 与 grading_guidance 使用 references 的主要语言，并保留必要的专有名词和缩写。
answer 必须是简短、完整、直接给出准确结论的陈述句；不要展示推理过程、附加背景信息或使用 Markdown。

只返回一个 JSON object，不要包含 Markdown 或其他字段：
{
  "question": "...",
  "answer": "...",
  "grading_guidance": "..."
}

grading_guidance 简要说明本题考察意图，辅助后续判罚；不要罗列答案要点、禁止项或复述答案。'''


def generate(
    ctx: Any,
    inputs: Mapping[str, object],
    llm_complete: Callable[[str], str] | None = None,
) -> dict[str, object]:
    output_key = getattr(ctx, 'output_key_by_name', {}).get('case')
    case_id = _text(getattr(output_key, 'partition', None), 'case output partition')
    preparation = _mapping(inputs.get('qaplan_spec'), 'qaplan_spec')
    if _text(preparation.get('id'), 'qaplan_spec.id') != case_id:
        raise ValueError('qaplan_spec.id must match case output partition')
    mode = _choice(preparation.get('mode'), ('imported', 'generated'), 'qaplan_spec.mode')
    if mode == 'imported':
        case = _mapping(preparation.get('imported_case'), 'qaplan_spec.imported_case')
        if _text(case.get('id'), 'imported_case.id') != case_id:
            raise ValueError('imported_case.id must match case output partition')
        _choice(case.get('question_type'), ('precision', 'reasoning'), 'imported_case.question_type')
        _optional_choice(case.get('difficulty'), ('easy', 'medium', 'hard'), 'imported_case.difficulty')
        _text(case.get('question'), 'imported_case.question')
        _text(case.get('answer'), 'imported_case.answer')
        _text(case.get('grading_guidance'), 'imported_case.grading_guidance')
        _optional_string_list(case.get('reference_chunk_ids'), 'imported_case.reference_chunk_ids')
        return {'case': dict(case)}

    question_type = _choice(preparation.get('question_type'), ('precision', 'reasoning'), 'question_type')
    difficulty = _choice(preparation.get('difficulty'), ('easy', 'medium', 'hard'), 'difficulty')
    instruction = _text(preparation.get('instruction'), 'instruction')
    topic_value = _mapping(preparation.get('topic'), 'topic')
    _text(topic_value.get('topic_id'), 'topic.topic_id')
    topic = _text(topic_value.get('name'), 'topic.name')
    references = _references(preparation.get('references'), difficulty)
    _mapping(preparation.get('qaplan'), 'qaplan_spec.qaplan')

    run_config = _mapping(inputs.get('run_config'), 'run_config')
    llm_config = _mapping(run_config.get('llm_config'), 'run_config.llm_config')
    complete = llm_complete or _llm_complete(llm_config)
    generated = call_json(
        complete,
        _prompt(instruction, topic, references),
        _generated_fields,
        repair_instruction=_repair_instruction,
    )

    return {'case': {
        'id': case_id,
        'question_type': question_type,
        'difficulty': difficulty,
        'question': generated['question'],
        'answer': generated['answer'],
        'grading_guidance': generated['grading_guidance'],
        'references': references,
        'reference_context': [{'chunk_id': item['chunk_id'], 'text': item['text']} for item in references],
        'reference_chunk_ids': [item['chunk_id'] for item in references],
        'reference_doc_ids': list(dict.fromkeys(item['doc_id'] for item in references)),
        'source_preparation': {'kb_ids': list(dict.fromkeys(item['kb_id'] for item in references))},
    }}


def generate_manifest(ctx: Any, inputs: Mapping[str, object]) -> dict[str, object]:
    values = inputs.get('cases')
    if not isinstance(values, tuple) or not values:
        raise ValueError('cases must be a non-empty partitioned tuple')

    cases = []
    for index, raw in enumerate(values, 1):
        case = _mapping(raw, f'cases[{index}]')
        reference_chunk_ids = _string_list(case.get('reference_chunk_ids'), 'reference_chunk_ids')
        raw_references = case.get('references')
        if raw_references is not None:
            if not isinstance(raw_references, list) or [
                _text(_mapping(value, 'references[]').get('chunk_id'), 'references[].chunk_id')
                for value in raw_references
            ] != reference_chunk_ids:
                raise ValueError('references must match reference_chunk_ids')
        cases.append({
            'id': _text(case.get('id'), 'id'),
            'question_type': _choice(case.get('question_type'), ('precision', 'reasoning'), 'question_type'),
            'difficulty': _choice(case.get('difficulty'), ('easy', 'medium', 'hard'), 'difficulty'),
            'reference_count': len(reference_chunk_ids),
        })
    if len({item['id'] for item in cases}) != len(cases):
        raise ValueError('id values must be unique')
    imported = _mapping(inputs.get('import_cases_manifest'), 'import_cases_manifest')
    import_stats = _mapping(imported.get('stats'), 'import_cases_manifest.stats')
    allocation = _mapping(import_stats.get('case_allocation'), 'import_cases_manifest.stats.case_allocation')
    import_count = _non_negative_int(allocation.get('import_case_count'), 'import_case_count')
    auto_count = _non_negative_int(allocation.get('auto_case_count'), 'auto_case_count')
    if import_count + auto_count != len(cases):
        raise ValueError('import and auto case counts must match generated cases')

    return {'generate_manifest': {
        'cases': cases,
        'stats': {
            'case_count': len(cases),
            'question_type_counts': {
                name: sum(1 for item in cases if item['question_type'] == name)
                for name in ('precision', 'reasoning')
            },
            'difficulty_counts': {
                name: sum(1 for item in cases if item['difficulty'] == name)
                for name in ('easy', 'medium', 'hard')
            },
            'import_case_count': import_count,
            'generated_case_count': auto_count,
        },
    }}


def _references(value: object, difficulty: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError('references must be a list')
    if len(value) != REFERENCE_COUNTS[difficulty]:
        raise ValueError(f'references count must match {difficulty}')
    references = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f'references[{index}]')
        references.append({
            'kb_id': _text(item.get('kb_id'), 'reference kb_id'),
            'chunk_id': _text(item.get('chunk_id'), 'reference chunk_id'),
            'doc_id': _text(item.get('doc_id'), 'reference doc_id'),
            'text': _text(item.get('text'), 'reference text'),
        })
    return references


def _prompt(instruction: str, topic: str, references: list[dict[str, str]]) -> str:
    materials = '\n\n'.join(
        f'<reference id="ref_{index}">\n{item["text"]}\n</reference>'
        for index, item in enumerate(references, 1)
    )
    return f'{SHARED_PROMPT}\n\n{instruction}\n\nTopic: {topic}\n\nReference materials:\n{materials}'


def _repair_instruction(error: Exception) -> str:
    return f'上一份 JSON 未通过校验，请重新生成完整 JSON，不要解释或复述失败内容。\n校验错误：{error}'


def _generated_fields(value: Mapping[str, object]) -> dict[str, str]:
    return {
        'question': _text(value.get('question'), 'generated question'),
        'answer': _text(value.get('answer'), 'generated answer'),
        'grading_guidance': _text(value.get('grading_guidance'), 'generated grading_guidance'),
    }


def _llm_complete(llm_config: Mapping[str, object]) -> Callable[[str], str]:
    from evo.llm import LazyLLMClient

    return LazyLLMClient(llm_config=llm_config)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{name} must contain non-empty strings')
    return [_text(item, name) for item in value]


def _optional_string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a list')
    return [_text(item, name) for item in value]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value


def _choice(value: object, choices: tuple[str, ...], name: str) -> str:
    item = _text(value, name)
    if item not in choices:
        raise ValueError(f'{name} is invalid')
    return item


def _optional_choice(value: object, choices: tuple[str, ...], name: str) -> str:
    if value in (None, ''):
        return ''
    return _choice(value, choices, name)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer')
    return value
