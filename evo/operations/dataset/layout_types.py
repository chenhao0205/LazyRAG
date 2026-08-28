from __future__ import annotations


CANONICAL_LAYOUT_TYPE_IDS = (
    'text', 'heading', 'paragraph', 'table', 'formula', 'figure', 'code', 'list', 'unknown',
)
LAYOUT_TYPE_NAMES = {
    'text': '文本', 'heading': '标题', 'paragraph': '段落', 'table': '表格',
    'formula': '公式', 'figure': '图片', 'code': '代码', 'list': '列表', 'unknown': '其他',
}
# Dataset owns this profile intentionally: the parser framework currently exposes active node groups,
# but not an authoritative declaration of standard layout types produced by each algorithm.
PARSER_LAYOUT_TYPES_BY_ALGORITHM = {
    'general_algo': CANONICAL_LAYOUT_TYPE_IDS,
    '__default__': CANONICAL_LAYOUT_TYPE_IDS,
}
_LAYOUT_TYPE_ALIASES = {
    'content': 'text',
    'header': 'heading',
    'title': 'heading',
    'equation': 'formula',
    'image': 'figure',
    'code_block': 'code',
}


def canonical_layout_type(value: object) -> str:
    raw = str(value or '').strip().lower()
    normalized = _LAYOUT_TYPE_ALIASES.get(raw, raw)
    return normalized if normalized in CANONICAL_LAYOUT_TYPE_IDS else 'unknown'


def validate_layout_types(values: list[str]) -> list[str]:
    invalid = [value for value in values if value not in CANONICAL_LAYOUT_TYPE_IDS]
    if invalid:
        raise ValueError('allowed_types contains an unsupported standard layout type')
    return values
