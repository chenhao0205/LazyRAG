from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


MODEL_NOT_CONFIGURED = 2001300
MODEL_NOT_ALLOWED = 2001301

OPENCODE_PROVIDER = 'opencode'
OPENCODE_PROVIDER_LABEL = 'OpenCode Zen'


OPENCODE_MODELS = frozenset({
    'big-pickle',
    'claude-fable-5',
    'claude-haiku-4-5',
    'claude-opus-4-1',
    'claude-opus-4-5',
    'claude-opus-4-6',
    'claude-opus-4-7',
    'claude-opus-4-8',
    'claude-sonnet-4',
    'claude-sonnet-4-5',
    'claude-sonnet-4-6',
    'claude-sonnet-5',
    'deepseek-v4-flash',
    'deepseek-v4-flash-free',
    'deepseek-v4-pro',
    'gemini-3-flash',
    'gemini-3.1-pro',
    'gemini-3.5-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.6-flash',
    'glm-5',
    'glm-5.1',
    'glm-5.2',
    'gpt-5',
    'gpt-5-codex',
    'gpt-5-nano',
    'gpt-5.1',
    'gpt-5.1-codex',
    'gpt-5.1-codex-max',
    'gpt-5.1-codex-mini',
    'gpt-5.2',
    'gpt-5.2-codex',
    'gpt-5.3-codex',
    'gpt-5.3-codex-spark',
    'gpt-5.4',
    'gpt-5.4-mini',
    'gpt-5.4-nano',
    'gpt-5.4-pro',
    'gpt-5.5',
    'gpt-5.5-pro',
    'gpt-5.6-luna',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'grok-4.5',
    'grok-build-0.1',
    'kimi-k2.5',
    'kimi-k2.6',
    'kimi-k2.7-code',
    'laguna-s-2.1-free',
    'mimo-v2.5-free',
    'minimax-m2.5',
    'minimax-m2.7',
    'minimax-m3',
    'nemotron-3-ultra-free',
    'north-mini-code-free',
    'qwen3.5-plus',
    'qwen3.6-plus',
})


class EvoModelConfigError(ValueError):
    def __init__(self, code: int, reason: str, *, provider: str = '', model: str = '',
                 missing_fields: tuple[str, ...] = ()) -> None:
        super().__init__(reason)
        self.code = code
        self.reason = reason
        self.provider = provider
        self.model = model
        self.missing_fields = missing_fields

    def detail(self) -> dict[str, Any]:
        data: dict[str, Any] = {'reason': self.reason, 'model_role': 'evo_llm'}
        if self.provider:
            data['provider'] = self.provider
        if self.model:
            data['model'] = self.model
        if self.missing_fields:
            data['missing_fields'] = list(self.missing_fields)
        return {
            'code': self.code,
            'message': (
                '请先完成 evo_llm 模型配置'
                if self.code == MODEL_NOT_CONFIGURED
                else '当前配置的自进化模型不支持自进化'
            ),
            'data': data,
        }


def filter_opencode_models(models: object) -> list[str]:
    if isinstance(models, (str, bytes, Mapping)) or not isinstance(models, Iterable):
        return []

    allowed: list[str] = []
    seen: set[str] = set()
    for value in models:
        model = str(value or '').strip().casefold()
        if model in OPENCODE_MODELS and model not in seen:
            allowed.append(model)
            seen.add(model)
    return allowed


def build_opencode_settings(model: object, *, api_key: object = '') -> dict[str, str]:
    model_id = str(model or '').strip().casefold()
    key = str(api_key or '').strip()
    if not model_id:
        raise EvoModelConfigError(
            MODEL_NOT_CONFIGURED,
            'model_config_incomplete',
            missing_fields=('model',),
        )
    if model_id not in OPENCODE_MODELS:
        raise EvoModelConfigError(
            MODEL_NOT_ALLOWED,
            'evo_llm_not_allowed',
            provider=OPENCODE_PROVIDER,
            model=model_id,
        )
    if not key:
        raise EvoModelConfigError(
            MODEL_NOT_CONFIGURED,
            'model_config_incomplete',
            provider=OPENCODE_PROVIDER,
            model=model_id,
            missing_fields=('api_key',),
        )
    return {
        'model': f'{OPENCODE_PROVIDER}/{model_id}',
        'provider': OPENCODE_PROVIDER,
        'provider_model': model_id,
        'provider_label': OPENCODE_PROVIDER_LABEL,
        'api_key': key,
    }
