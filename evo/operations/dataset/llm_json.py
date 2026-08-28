from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from json_repair import repair_json

DEFAULT_LLM_JSON_TIMEOUT_SECONDS = 180
T = TypeVar('T')
logger = logging.getLogger(__name__)


def call_json(
    llm: Callable[..., Any],
    prompt: str,
    validate: Callable[[Mapping[str, Any]], T],
    *,
    repair_instruction: Callable[[Exception], str] | None = None,
) -> T:
    """Call an LLM in JSON mode and retry one content failure with repair guidance."""
    last_error: Exception | None = None
    for attempt in range(2):
        current_prompt = prompt
        if attempt and repair_instruction is not None:
            assert last_error is not None
            current_prompt = f'{prompt}\n\n{repair_instruction(last_error)}'
        raw = _invoke(llm, current_prompt)
        parsed: object = raw
        try:
            parsed = _json_object(raw)
            return validate(parsed)
        except Exception as exc:
            logger.warning(
                'LLM JSON validation failed attempt=%s/2 error=%s payload=%s',
                attempt + 1,
                exc,
                _preview(parsed),
            )
            if attempt:
                raise ValueError(f'LLM JSON call failed after 2 attempts: {exc}') from exc
            last_error = exc
    raise AssertionError('unreachable')


def _invoke(llm: Callable[..., Any], prompt: str) -> Any:
    kwargs = {
        'stream': False,
        'response_format': {'type': 'json_object'},
        'timeout': DEFAULT_LLM_JSON_TIMEOUT_SECONDS,
    }
    try:
        parameters = inspect.signature(llm).parameters.values()
    except (TypeError, ValueError):
        return llm(prompt, **kwargs)
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return llm(prompt, **kwargs)
    accepted = {parameter.name for parameter in parameters}
    return llm(prompt, **{name: value for name, value in kwargs.items() if name in accepted})


def _preview(value: object, *, limit: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        text = str(value)
    if len(text) > limit:
        return f'{text[:limit]}…'
    return text


def _json_object(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    text = re.sub(r'<think>.*?</think>', '', str(raw), flags=re.DOTALL).strip()
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            text = text[start:end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = repair_json(text, return_objects=True)
    if not isinstance(value, Mapping):
        raise ValueError(f'LLM response JSON must be an object, got {type(value).__name__}')
    return value
