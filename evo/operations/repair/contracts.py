from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AgentAction(BaseModel):
    """One tool choice. Tool-specific details stay inside the free request object."""

    model_config = ConfigDict(extra='forbid')

    action: Literal[
        'opencode', 'run_command', 'search_web', 'read_web', 'http_request',
        'read_artifact', 'finish', 'stop',
    ]
    reason: str = Field(min_length=1, max_length=2000)
    request: dict[str, Any] = Field(default_factory=dict)


class PatchReview(BaseModel):
    model_config = ConfigDict(extra='forbid')

    matches_verified_method: bool
    preserves_contracts_and_data_scope: bool
    minimal: bool
    issues: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=6,
    )
    reason: str = Field(min_length=1, max_length=1200)

    @property
    def accepted(self) -> bool:
        return self.matches_verified_method and self.preserves_contracts_and_data_scope and self.minimal

    @model_validator(mode='after')
    def validate_verdict(self) -> Self:
        if self.accepted and self.issues:
            raise ValueError('an accepted patch review must not contain issues')
        if not self.accepted and not self.issues:
            raise ValueError('a rejected patch review requires concrete issues')
        return self


def validate_analysis(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        'source_hash', 'all_case_metric_averages', 'categories',
    }:
        raise ValueError('phase1_input_fields_invalid')
    digest = str(value.get('source_hash') or '')
    if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
        raise ValueError('source_hash_invalid')
    all_metrics = _metric_map(value.get('all_case_metric_averages'))
    raw_categories = value.get('categories')
    if not isinstance(raw_categories, Mapping) or not raw_categories:
        raise ValueError('categories_empty')
    categories = {}
    seen_cases = set()
    for raw_id, raw_category in raw_categories.items():
        category_id = str(raw_id or '').strip()
        if not category_id or category_id != raw_id or not isinstance(raw_category, Mapping):
            raise ValueError('category_invalid')
        expected = {'metric_averages', 'all_case_average_drop', 'code_span', 'analysis', 'cases'}
        if set(raw_category) != expected:
            raise ValueError('category_fields_invalid')
        metrics = _metric_map(raw_category.get('metric_averages'))
        if set(metrics) != set(all_metrics):
            raise ValueError('category_metric_keys_mismatch')
        drop = raw_category.get('all_case_average_drop')
        if (
            isinstance(drop, bool)
            or not isinstance(drop, (int, float))
            or not math.isfinite(drop)
            or not 0 <= drop <= 1
        ):
            raise ValueError('category_drop_invalid')
        spans = [_code_span(item) for item in raw_category.get('code_span') or ()]
        analysis = str(raw_category.get('analysis') or '').strip()
        cases = raw_category.get('cases')
        if not spans or not analysis or not isinstance(cases, Mapping) or not cases:
            raise ValueError('category_root_cause_incomplete')
        normalized_cases = {}
        for raw_case_id, raw_trace_id in cases.items():
            case_id, trace_id = str(raw_case_id or '').strip(), str(raw_trace_id or '').strip()
            if not case_id or not trace_id or case_id in seen_cases:
                raise ValueError('category_cases_invalid')
            seen_cases.add(case_id)
            normalized_cases[case_id] = trace_id
        categories[category_id] = {
            'metric_averages': metrics,
            'all_case_average_drop': float(drop),
            'code_span': spans,
            'analysis': analysis,
            'cases': normalized_cases,
        }
    return {
        'source_hash': digest,
        'all_case_metric_averages': all_metrics,
        'categories': categories,
    }


def select_category(categories: Mapping[str, Mapping[str, Any]]) -> tuple[str, Mapping[str, Any]]:
    category_id = min(
        categories,
        key=lambda item: (-float(categories[item]['all_case_average_drop']), item),
    )
    return category_id, categories[category_id]


def build_supported_plan(
    category_id: str,
    category: Mapping[str, Any],
    phase1: Mapping[str, Any],
    guidance: list[str],
) -> dict[str, Any]:
    proposal = phase1.get('proposal')
    validation = phase1.get('validation')
    if not isinstance(proposal, Mapping):
        raise ValueError('proposal_invalid')
    target = str(proposal.get('target') or '').strip()
    change = str(proposal.get('change') or '').strip()
    expected = str(proposal.get('expected_result') or '').strip()
    if not target or not change or not expected:
        raise ValueError('proposal_incomplete')
    if not isinstance(validation, Mapping) or validation.get('verdict') != 'supports':
        raise ValueError('validation_invalid')
    evidence = list(validation.get('evidence_refs') or ())
    for ref in evidence:
        _validate_content_ref(ref)
    if not evidence:
        raise ValueError('validation_evidence_missing')
    _validate_content_ref(validation.get('workspace_ref'))
    _validate_content_ref(validation.get('journal_ref'))
    requirements = [str(item).strip() for item in guidance if str(item).strip()]
    return {
        'id': 'repair.plan',
        'status': 'planned',
        'category_id': category_id,
        'method': {
            'target': target,
            'change': change,
            'expected_result': expected,
            'constraints': requirements,
        },
        'demo_validation': dict(validation),
    }


def _metric_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError('metrics_invalid')
    result = {}
    for raw_name, raw_score in value.items():
        name = str(raw_name or '').strip()
        if (
            not name
            or name != raw_name
            or isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(raw_score)
            or not 0 <= raw_score <= 1
        ):
            raise ValueError('metric_invalid')
        result[name] = float(raw_score)
    return result


def _code_span(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {'path', 'symbol'}:
        raise ValueError('code_span_invalid')
    path, symbol = str(value.get('path') or '').strip(), str(value.get('symbol') or '').strip()
    parts = PurePosixPath(path).parts
    if (
        not path
        or not symbol
        or path.startswith('/')
        or '\\' in path
        or any(part in {'', '.', '..'} for part in parts)
    ):
        raise ValueError('code_span_invalid')
    return {'path': path, 'symbol': symbol}


def _validate_content_ref(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {'uri', 'sha256'}:
        raise ValueError('content_ref_invalid')
    uri, digest = str(value.get('uri') or '').strip(), str(value.get('sha256') or '')
    if not uri or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
        raise ValueError('content_ref_invalid')
