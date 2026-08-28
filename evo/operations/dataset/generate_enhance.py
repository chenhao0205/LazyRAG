from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .llm_json import call_json


KEY_POINTS_PROMPT = '''基于给定的 question、answer 和 reference materials，提取 answer 必须命中的原子事实。

只能依据给定的 answer 和 reference materials；不得引入 answer 未包含的新事实。
每条 key point 只能表达一项完整、可单独判断的事实，并由一个或多个 reference 支持。

只返回一个 JSON object，不要包含 Markdown 或其他字段：
{
  "key_points": [{"statement": "...", "evidence_reference_ids": ["ref_1"]}]
}

key_points 必须有 1 至 5 条。evidence_reference_ids 只能使用提供的短别名。'''

FORBIDDEN_CLAIMS_PROMPT = '''基于给定的 question、answer 和 reference materials，提取本题容易误答、且被 reference materials 明确否定的具体结论。

只能依据给定的 reference materials；不得生成通用禁止语、无证据推测或与本题无关的结论。

只返回一个 JSON object，不要包含 Markdown 或其他字段：
{
  "forbidden_claims": ["..."]
}

forbidden_claims 可以为空，最多 3 条。'''


def generate_enhance(
    ctx: Any,
    inputs: Mapping[str, object],
    llm_complete: Callable[[str], str] | None = None,
) -> dict[str, object]:
    case = _mapping(inputs.get('case'), 'case')
    preparation = case.get('source_preparation')
    if isinstance(preparation, Mapping) and preparation.get('dataset_mode') == 'imported':
        key_points = case.get('key_points', [])
        forbidden_claims = case.get('forbidden_claims', [])
        if not isinstance(key_points, list):
            raise ValueError('case.key_points must be a list')
        if not isinstance(forbidden_claims, list):
            raise ValueError('case.forbidden_claims must be a list')
        return {'case_enhance': {
            'key_points': [dict(item) if isinstance(item, Mapping) else item for item in key_points],
            'forbidden_claims': list(forbidden_claims),
        }}
    question = _text(case.get('question'), 'case.question')
    answer = _text(case.get('answer'), 'case.answer')
    references = _references(case)
    run_config = _mapping(inputs.get('run_config'), 'run_config')
    llm_config = _mapping(run_config.get('llm_config'), 'run_config.llm_config')
    complete = llm_complete or _llm_complete(llm_config)
    aliases = {f'ref_{index}': item['chunk_id'] for index, item in enumerate(references, 1)}
    failed_key_points: dict[str, object] = {}

    def accept_key_points(value: Mapping[str, object]) -> list[dict[str, object]]:
        failed_key_points.clear()
        failed_key_points.update(value)
        return _key_points(value.get('key_points'), aliases)

    key_points = call_json(
        complete,
        _prompt(KEY_POINTS_PROMPT, question, answer, references),
        accept_key_points,
        repair_instruction=lambda error: _key_points_repair(error, aliases, failed_key_points),
    )
    forbidden_claims = call_json(
        complete,
        _prompt(FORBIDDEN_CLAIMS_PROMPT, question, answer, references),
        lambda value: _string_list(value.get('forbidden_claims'), 'forbidden_claims', minimum=0, maximum=3),
        repair_instruction=_repair_instruction,
    )

    return {'case_enhance': {
        'key_points': [
            {
                'id': f'key_point_{index}',
                'statement': item['statement'],
                'evidence_chunk_ids': [aliases[reference_id] for reference_id in item['evidence_reference_ids']],
            }
            for index, item in enumerate(key_points, 1)
        ],
        'forbidden_claims': forbidden_claims,
    }}


def generate_enhance_manifest(ctx: Any, inputs: Mapping[str, object]) -> dict[str, object]:
    enhancements = inputs.get('case_enhances')
    if not isinstance(enhancements, tuple) or not enhancements:
        raise ValueError('case_enhances must be a non-empty partitioned tuple')
    for enhancement in enhancements:
        value = _mapping(enhancement, 'case_enhances[]')
        if not isinstance(value.get('key_points'), list):
            raise ValueError('case_enhances[].key_points must be a list')
        if not isinstance(value.get('forbidden_claims'), list):
            raise ValueError('case_enhances[].forbidden_claims must be a list')
    return {'generate_enhance_manifest': {'case_count': len(enhancements)}}


def _references(case: Mapping[str, object]) -> list[dict[str, str]]:
    raw_context = case.get('reference_context')
    if not isinstance(raw_context, list):
        raise ValueError('case.reference_context must be a list')
    context: dict[str, str] = {}
    for index, raw in enumerate(raw_context):
        item = _mapping(raw, f'case.reference_context[{index}]')
        chunk_id = _text(item.get('chunk_id'), f'case.reference_context[{index}].chunk_id')
        if chunk_id in context:
            raise ValueError('case.reference_context chunk ids must be unique')
        context[chunk_id] = _text(item.get('text'), f'case.reference_context[{index}].text')
    chunk_ids = _string_list(case.get('reference_chunk_ids'), 'case.reference_chunk_ids')
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError('case.reference_chunk_ids must be unique')
    raw_references = case.get('references')
    if raw_references is not None:
        if not isinstance(raw_references, list):
            raise ValueError('case.references must be a list')
        for expected_id, expected_text, raw in zip(chunk_ids, (context[item] for item in chunk_ids), raw_references, strict=True):
            item = _mapping(raw, 'case.references[]')
            if _text(item.get('chunk_id'), 'case.references[].chunk_id') != expected_id or _text(item.get('text'), 'case.references[].text') != expected_text:
                raise ValueError('case.references must match reference_context and reference_chunk_ids')
            _text(item.get('kb_id'), 'case.references[].kb_id')
            _text(item.get('doc_id'), 'case.references[].doc_id')

    references = []
    for chunk_id in chunk_ids:
        references.append({
            'chunk_id': chunk_id,
            'text': _text(context.get(chunk_id), f'case.reference_context[{chunk_id}]'),
        })
    return references


def _prompt(instruction: str, question: str, answer: str, references: list[dict[str, str]]) -> str:
    materials = '\n\n'.join(
        f'<reference id="ref_{index}">\n{item["text"]}\n</reference>'
        for index, item in enumerate(references, 1)
    )
    return f'{instruction}\n\nQuestion: {question}\n\nAnswer: {answer}\n\nReference materials:\n{materials}'


def _key_points(value: object, aliases: Mapping[str, str]) -> list[dict[str, object]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 5:
        raise ValueError('key_points must contain 1 to 5 items')

    key_points = []
    for raw in value:
        item = _mapping(raw, 'key_points[]')
        evidence_reference_ids = _string_list(
            item.get('evidence_reference_ids'), 'key_points[].evidence_reference_ids'
        )
        if not set(evidence_reference_ids).issubset(aliases):
            raise ValueError(
                'key_points evidence_reference_ids must reference provided references: '
                f'got={evidence_reference_ids!r} allowed={sorted(aliases)!r}'
            )
        key_points.append({
            'statement': _text(item.get('statement'), 'key_points[].statement'),
            'evidence_reference_ids': evidence_reference_ids,
        })
    return key_points


def _key_points_repair(
    error: Exception,
    aliases: Mapping[str, str],
    payload: Mapping[str, object],
) -> str:
    previous = json.dumps({'key_points': payload.get('key_points')}, ensure_ascii=False)
    if len(previous) > 2000:
        previous = f'{previous[:2000]}…'
    allowed = ', '.join(sorted(aliases))
    return (
        '上一份 JSON 未通过校验，请重新生成完整 JSON，不要解释或复述失败内容。\n'
        f'校验错误：{error}\n'
        f'evidence_reference_ids 只能使用这些短别名：{allowed}。'
        '不要把法条编号、chunk id 或其他数字当成引用 id。\n'
        f'上一份 JSON：{previous}'
    )


def _repair_instruction(error: Exception) -> str:
    return f'上一份 JSON 未通过校验，请重新生成完整 JSON，不要解释或复述失败内容。\n校验错误：{error}'


def _llm_complete(llm_config: Mapping[str, object]) -> Callable[[str], str]:
    from evo.llm import LazyLLMClient

    return LazyLLMClient(llm_config=llm_config)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _string_list(value: object, name: str, *, minimum: int = 1, maximum: int | None = None) -> list[str]:
    if not isinstance(value, list) or len(value) < minimum or (maximum is not None and len(value) > maximum):
        raise ValueError(f'{name} must contain {minimum} to {maximum or "more"} non-empty strings')
    return [_text(item, name) for item in value]


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value
