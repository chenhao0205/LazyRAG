from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any


_FIELD_WEIGHTS = (
    (5, ('candidate_files',)),
    (4, ('function_block_id', 'affected_block')),
    (3, ('issue_category',)),
    (2, ('issue_type', 'failure_mode')),
    (1, ('adjacent_blocks',)),
)

_ALIASES = (
    ('retrieval', 'retrieve', 'retriever', '检索', '召回'),
    ('rerank', 're-rank', 'ranker', '重排'),
    ('generation', 'generate', 'generator', 'llm_generate', '生成', '回答'),
    ('execution', 'execute', 'tool_call', '执行', '工具调用'),
    ('tracing', 'trace', '链路', '追踪', '日志', '观测'),
    ('query_rewrite', 'rewrite', '查询改写', '问题改写'),
    ('context_assembly', 'context', '上下文'),
    ('prompt_build', 'prompt', '提示词'),
)

_NEGATION = re.compile(
    r'(?:不要|别|禁止|排除|忽略|无需|不用|不改|avoid|exclude|ignore|do\s+not|don\'t)'
    r'[^,，。;；]{0,16}$',
)
_ENGLISH_TERM = re.compile(r'^[a-z0-9_./-]+$')
_TOKEN = re.compile(r'[a-z0-9]+(?:[._/-][a-z0-9]+)*|[\u4e00-\u9fff]{2,}')


def rerank_repair_groups(
    queue: Sequence[Mapping[str, Any]],
    user_guidance: str | Sequence[str],
) -> list[dict[str, Any]]:
    """Apply optional user guidance without replacing Analysis' base ordering.

    Analysis owns the initial order.  Guidance only moves explicitly matching
    groups; ties retain their original position so the result is deterministic.
    The input mappings are copied and never mutated.
    """
    groups = []
    for item in queue:
        if not isinstance(item, Mapping):
            raise TypeError('repair group queue entries must be mappings')
        groups.append(dict(item))

    guidance = _guidance_text(user_guidance)
    if not guidance:
        return groups

    ranked = [(_guidance_score(group, guidance), index, group) for index, group in enumerate(groups)]
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [group for _, _, group in ranked]


def _guidance_score(group: Mapping[str, Any], guidance: str) -> int:
    score = 0
    for weight, fields in _FIELD_WEIGHTS:
        terms = set()
        for field in fields:
            terms.update(_terms(group.get(field)))
        score += weight * _field_polarity(guidance, terms)
    return score


def _field_polarity(guidance: str, terms: set[str]) -> int:
    positive = False
    negative = False
    for term in terms:
        for start in _occurrences(guidance, term):
            prefix = guidance[max(0, start - 24):start]
            if _NEGATION.search(prefix):
                negative = True
            else:
                positive = True
    return int(positive) - int(negative)


def _terms(value: object) -> set[str]:
    values = value if isinstance(value, (list, tuple, set)) else (value,)
    result: set[str] = set()
    for item in values:
        text = _normalize(item)
        if not text:
            continue
        result.add(text)
        name = PurePosixPath(text).name
        if name:
            result.add(name)
        result.update(token for token in _TOKEN.findall(text) if len(token) > 1)

    expanded = set(result)
    for aliases in _ALIASES:
        if result.intersection(aliases):
            expanded.update(aliases)
    return expanded


def _occurrences(text: str, term: str) -> list[int]:
    if not term:
        return []
    if _ENGLISH_TERM.fullmatch(term) and '/' not in term and '.' not in term:
        pattern = re.compile(rf'(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])')
        return [match.start() for match in pattern.finditer(text)]
    starts = []
    offset = 0
    while (index := text.find(term, offset)) >= 0:
        starts.append(index)
        offset = index + len(term)
    return starts


def _guidance_text(value: str | Sequence[str]) -> str:
    values = (value,) if isinstance(value, str) else value
    return ' '.join(filter(None, (_normalize(item) for item in values)))


def _normalize(value: object) -> str:
    return ' '.join(str(value or '').strip().lower().split())


__all__ = ['rerank_repair_groups']
