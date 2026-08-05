from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from evo.llm import parse_json_object

from .contracts import (
    RepairAction,
    RepairAgentError,
    RepairInput,
    RepairView,
    contract_dict,
    repair_action,
)


ModelCall = Callable[..., Any]

MODEL_NOT_CONFIGURED = 2001300
MODEL_NOT_ALLOWED = 2001301


def _models(*names: str) -> dict[str, str]:
    return {name.casefold(): name for name in names}


EVO_MODEL_ALLOWLIST = _models(
    'claude-haiku-4-5', 'claude-opus-4-7', 'claude-sonnet-4-6',
    'deepseek-v4-flash', 'deepseek-v4-pro',
    'GLM-5', 'GLM-5.1',
    'kimi-k2.5', 'kimi-k2.6',
    'MiniMax-M2.5', 'MiniMax-M2.7',
    'gpt-5', 'gpt-5-nano', 'gpt-5.1', 'gpt-5.2', 'gpt-5.4', 'gpt-5.4-mini',
    'gpt-5.4-nano', 'gpt-5.4-pro', 'gpt-5.5', 'gpt-5.5-pro',
    'qwen3.5-plus', 'qwen3.6-plus',
)
PROVIDER_ALIASES = {
    'alibaba': 'qwen',
    'alibabacn': 'qwen',
    'anthropic': 'claude',
    'claude': 'claude',
    'dashscope': 'qwen',
    'deepseek': 'deepseek',
    'glm': 'glm',
    'kimi': 'kimi',
    'minimax': 'minimax',
    'moonshot': 'kimi',
    'moonshotai': 'kimi',
    'openai': 'openai',
    'qwen': 'qwen',
    'siliconflow': 'siliconflow',
    'zhipu': 'glm',
    'zhipuai': 'glm',
}
OPENCODE_PROVIDERS = {
    'claude': ('anthropic', '@ai-sdk/anthropic', {}),
    'deepseek': ('deepseek', '@ai-sdk/openai-compatible', {
        'https://api.deepseek.com/v1': 'https://api.deepseek.com',
    }),
    'glm': ('zhipuai', '@ai-sdk/openai-compatible', {}),
    'kimi': ('moonshotai-cn', '@ai-sdk/openai-compatible', {
        'https://api.moonshot.cn': 'https://api.moonshot.cn/v1',
    }),
    'minimax': ('minimax-cn', '@ai-sdk/anthropic', {
        'https://api.minimaxi.com/v1': 'https://api.minimaxi.com/anthropic/v1',
    }),
    'openai': ('openai', '@ai-sdk/openai', {
        'https://api.openai.com': 'https://api.openai.com/v1',
    }),
    'qwen': ('alibaba-cn', '@ai-sdk/openai-compatible', {
        'https://dashscope.aliyuncs.com': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    }),
    'siliconflow': ('siliconflow-cn', '@ai-sdk/openai-compatible', {}),
}
GENERIC_OPENCODE_PROVIDER = ('lazyrag-openai-compatible', '@ai-sdk/openai-compatible', {})


class EvoModelConfigError(ValueError):
    def __init__(
        self,
        code: int,
        reason: str,
        provider: str = '',
        model: str = '',
        missing_fields: tuple[str, ...] = (),
    ) -> None:
        super().__init__(code, reason, provider, model, missing_fields)
        self.code = code
        self.reason = reason
        self.provider = provider
        self.model = model
        self.missing_fields = missing_fields

    def __str__(self) -> str:
        return self.reason

    def detail(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            'reason': self.reason,
            'model_role': 'evo_llm',
        }
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


def resolve_evo_model(role: object) -> tuple[str, str]:
    if not isinstance(role, Mapping):
        raise EvoModelConfigError(MODEL_NOT_CONFIGURED, 'model_not_configured')

    raw_provider = _text(role.get('provider')) or _text(role.get('source'))
    raw_model = _text(role.get('model'))
    base_url = _text(role.get('base_url'))
    api_key = _text(role.get('api_key'))
    missing = tuple(
        field
        for field, value in (
            ('source', raw_provider),
            ('model', raw_model),
            ('base_url', base_url),
            ('api_key', api_key or ('skip_auth' if role.get('skip_auth') is True else '')),
        )
        if not value
    )
    if missing:
        raise EvoModelConfigError(
            MODEL_NOT_CONFIGURED,
            'model_config_incomplete',
            provider=raw_provider,
            model=raw_model,
            missing_fields=missing,
        )

    if not EVO_MODEL_ALLOWLIST.get(_compatibility_model_key(raw_model)):
        raise EvoModelConfigError(
            MODEL_NOT_ALLOWED,
            'evo_llm_not_allowed',
            provider=raw_provider,
            model=raw_model,
        )
    provider = PROVIDER_ALIASES.get(_provider_key(raw_provider), '')
    return provider, raw_model


def build_opencode_settings(llm_config: object) -> dict[str, str]:
    provider, model = resolve_evo_model(llm_config)
    opencode_provider, npm, rewrites = OPENCODE_PROVIDERS.get(provider, GENERIC_OPENCODE_PROVIDER)
    raw = llm_config if isinstance(llm_config, Mapping) else {}
    base_url = _text(raw.get('base_url')).rstrip('/')
    return {
        'model': f'{opencode_provider}/{model}',
        'provider': opencode_provider,
        'provider_model': model,
        'npm': npm,
        'base_url': rewrites.get(base_url, base_url),
        'api_key': _text(raw.get('api_key')),
        'skip_auth': 'true' if raw.get('skip_auth') is True else '',
    }


def _provider_key(value: str) -> str:
    return re.sub(r'[\s_.-]+', '', value.casefold())


def _compatibility_model_key(value: str) -> str:
    return _text(value).rsplit('/', 1)[-1].casefold()


def _text(value: object) -> str:
    return str(value or '').strip()


class DecisionAgent(Protocol):
    """The sole semantic decision boundary used by one RepairSession."""

    def decide(self, view: RepairView) -> RepairAction:
        ...

    def summarize(
        self,
        objective: str,
        guidance: str,
        previous_brief: str,
        events: list[dict[str, Any]],
    ) -> str:
        ...

    def assess_finish(
        self,
        repair_input: RepairInput,
        view: RepairView,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, str]:
        ...


class OpenCodeAdapter(DecisionAgent):
    """One-shot model adapter; it never runs an inner tool or agent loop."""

    def __init__(
        self,
        model_call: ModelCall,
        timeout_seconds: int = 120,
        settings: Mapping[str, str] | None = None,
    ) -> None:
        self.model_call = model_call
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.settings = dict(settings or {})

    @classmethod
    def from_llm_config(
        cls,
        model_call: ModelCall,
        llm_config: object,
        timeout_seconds: int = 120,
    ) -> OpenCodeAdapter:
        return cls(
            model_call,
            timeout_seconds,
            build_opencode_settings(llm_config),
        )

    def decide(self, view: RepairView) -> RepairAction:
        prompt = (
            'You are the only decision Agent in a RepairSession. Use the projected working memory and choose '
            'exactly one capability call. Do not describe a pipeline and do not execute tools internally. '
            'workspace can list/read/write/diff candidate source or work files; shell runs one argv command; '
            'test accepts level L0, L1, or L2; research searches or reads sources; finish requests completion. '
            'Return only JSON with exactly call_id, tool, arguments.\n'
            'Workspace arguments: {operation: list|read|write|diff, path?: string, content?: string}.\n'
            'Shell arguments: {command: string[], cwd: source|work, timeout_seconds?: int}.\n'
            'Test arguments: {level: L0|L1|L2}.\n'
            'Research arguments: {operation: search|read, query: string, urls?: string[]}.\n'
            'Finish arguments: {reason: string}.\n'
            f'RepairView: {json.dumps(contract_dict(view), ensure_ascii=False, default=str)}'
        )
        try:
            return repair_action(self._json(prompt))
        except (TypeError, ValueError) as exc:
            raise RepairAgentError('agent_action_invalid', str(exc)) from exc

    def summarize(
        self,
        objective: str,
        guidance: str,
        previous_brief: str,
        events: list[dict[str, Any]],
    ) -> str:
        prompt = (
            'Compress older Repair events into a factual working-memory brief. Preserve unresolved constraints, '
            'observed facts, failed attempts and human messages. Never replace or reinterpret the pinned objective '
            'or guidance. Return only JSON {"memory_brief": "..."}.\n'
            f'Pinned objective: {objective}\nPinned guidance: {guidance}\n'
            f'Previous brief: {previous_brief}\n'
            f'Older events: {json.dumps(events, ensure_ascii=False, default=str)}'
        )
        value = self._json(prompt)
        brief = str(value.get('memory_brief') or '').strip()
        if not brief:
            raise RepairAgentError('memory_brief_invalid')
        return brief

    def assess_finish(
        self,
        repair_input: RepairInput,
        view: RepairView,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, str]:
        prompt = (
            'Assess whether the current Repair workspace semantically satisfies the objective, guidance and '
            'case scope. Base the verdict on the current diff and evidence in RepairView, not keyword presence. '
            'Return only JSON {"satisfied": true|false, "summary": "..."}.\n'
            f'RepairInput: {json.dumps(contract_dict(repair_input), ensure_ascii=False, default=str)}\n'
            f'RepairView: {json.dumps(contract_dict(view), ensure_ascii=False, default=str)}\n'
            f'Finish request: {json.dumps(dict(arguments), ensure_ascii=False, default=str)}'
        )
        value = self._json(prompt)
        if not isinstance(value.get('satisfied'), bool):
            raise RepairAgentError('finish_assessment_invalid')
        summary = str(value.get('summary') or '').strip()
        if not summary:
            raise RepairAgentError('finish_assessment_invalid')
        return value['satisfied'], summary

    def _json(self, prompt: str) -> dict[str, Any]:
        try:
            options: dict[str, Any] = {
                'stream': False,
                'response_format': {'type': 'json_object'},
                'timeout': self.timeout_seconds,
                'max_retries': 1,
                'max_tokens': 4096,
            }
            if self.settings.get('provider') == 'deepseek':
                options['thinking'] = {'type': 'disabled'}
            raw = self.model_call(
                prompt,
                **options,
            )
            value = parse_json_object(raw)
        except Exception as exc:
            raise RepairAgentError('model_call_failed', str(exc)) from exc
        if not isinstance(value, Mapping):
            raise RepairAgentError('model_response_invalid')
        return dict(value)


__all__ = [
    'DecisionAgent',
    'EVO_MODEL_ALLOWLIST',
    'MODEL_NOT_ALLOWED',
    'MODEL_NOT_CONFIGURED',
    'ModelCall',
    'EvoModelConfigError',
    'OpenCodeAdapter',
    'build_opencode_settings',
    'resolve_evo_model',
]
