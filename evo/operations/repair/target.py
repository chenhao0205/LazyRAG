from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import fsum
from typing import Any, Literal, Protocol

from pydantic import ValidationError

from .target_contracts import (
    AnalysisCategory,
    AnalysisSummaryInput,
    CategoryDirective,
    CompiledPreference,
    CompiledRankingIntent,
    InterruptibleRankedCategory,
    InterruptibleRankingSignals,
    MetricDirective,
    PreferenceCategoryOption,
    PreferenceCompileMode,
    PreferenceCompileRequest,
    PreferenceEvidenceRef,
    PreferenceResolution,
    PreferenceText,
    RankedCategory,
    RankingSignals,
    RepairTargetContext,
    RepairTargetPreparation,
    TargetCommitRequest,
    TargetGuidanceInterrupt,
    TargetRankingState,
    TargetWarning,
    validate_preference_resolution,
)

_TIER_ORDER = {
    'must': 0,
    'prefer': 1,
    'normal': 2,
    'defer': 3,
}
_MISSING_USER_ORDER = 1_000_000
_PREFERENCE_SUMMARY_LIMIT = 500
_PREFERENCE_ANALYSIS_LIMIT = 1000
_MAX_TARGET_WARNINGS = 400
_LOGGER = logging.getLogger(__name__)


class PreferenceCompiler(Protocol):
    def __call__(
        self,
        request: PreferenceCompileRequest,
    ) -> PreferenceResolution | Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class _CategoryCard:
    category_id: str
    category: AnalysisCategory


@dataclass(frozen=True)
class _RankableCategory:
    sort_key: tuple[Any, ...]
    value: RankedCategory | InterruptibleRankedCategory


def build_target_preparation(
    analysis: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    preference_compiler: PreferenceCompiler | None = None,
) -> dict[str, Any]:
    """Rank analysis categories and select one target for Repair.

    The function deliberately treats ``code_span`` as opaque Analysis output.
    It does not inspect code, validate paths, or use span shape in ranking.
    """
    summary = AnalysisSummaryInput.model_validate(analysis)
    if not isinstance(context, Mapping):
        raise TypeError('repair target context must be a mapping')
    target_context = RepairTargetContext.model_validate({
        'user_guidance': context.get('user_guidance', []),
        'preference': context.get('preference', []),
    })
    cards = _build_category_cards(summary)
    if not cards:
        preference = _empty_preference(summary)
        return _dump_result({
            'source_hash': summary.source_hash,
            'status': 'blocked',
            'blocked_reason': 'blocked_no_categories',
            'selected_category': None,
            'ranked_categories': [],
            'excluded_category_ids': [],
            'preference_interpretation': preference,
            'warnings': [],
        })

    preference, warnings = _compile_user_preference(
        summary,
        target_context,
        preference_compiler,
    )
    ranked, excluded = _rank_categories(cards, summary, preference)
    if not ranked:
        return _dump_result({
            'source_hash': summary.source_hash,
            'status': 'blocked',
            'blocked_reason': 'blocked_all_categories_excluded',
            'selected_category': None,
            'ranked_categories': [],
            'excluded_category_ids': excluded,
            'preference_interpretation': preference,
            'warnings': warnings,
        })
    return _dump_result({
        'source_hash': summary.source_hash,
        'status': 'ready',
        'blocked_reason': '',
        'selected_category': ranked[0].category_id,
        'ranked_categories': ranked,
        'excluded_category_ids': excluded,
        'preference_interpretation': preference,
        'warnings': warnings,
    })


def _build_category_cards(summary: AnalysisSummaryInput) -> tuple[_CategoryCard, ...]:
    return tuple(
        _CategoryCard(category_id, summary.categories[category_id])
        for category_id in sorted(summary.categories)
    )


def _compile_user_preference(
    summary: AnalysisSummaryInput,
    context: RepairTargetContext,
    compiler: PreferenceCompiler | None,
) -> tuple[CompiledPreference, list[TargetWarning]]:
    category_ids = set(summary.categories)
    metric_ids = set(summary.all_case_metric_averages)
    direct_categories: list[CategoryDirective] = []
    direct_metrics: list[MetricDirective] = []
    unresolved: list[PreferenceText] = [
        PreferenceText(
            evidence=PreferenceEvidenceRef(source='user_guidance', index=index),
            text=text,
        )
        for index, text in enumerate(context.user_guidance)
    ]

    for index, text in enumerate(context.preference):
        evidence = PreferenceEvidenceRef(source='preference', index=index)
        if text in category_ids:
            if any(item.category_id == text for item in direct_categories):
                continue
            direct_categories.append(CategoryDirective(
                category_id=text,
                tier='prefer',
                order=index,
                evidence=evidence,
            ))
        elif text in metric_ids:
            if any(item.metric_id == text for item in direct_metrics):
                continue
            direct_metrics.append(MetricDirective(
                metric_id=text,
                order=index,
                evidence=evidence,
            ))
        else:
            unresolved.append(PreferenceText(evidence=evidence, text=text))

    resolved = PreferenceResolution()
    warnings: list[TargetWarning] = []
    resolution_succeeded = False
    if unresolved:
        if compiler is None:
            warnings.extend(_fallback_warnings(compiler_unavailable=True))
        else:
            request = PreferenceCompileRequest(
                texts=unresolved,
                category_options=[
                    PreferenceCategoryOption(
                        category_id=card.category_id,
                        summary=_clip_text(card.category.summary, _PREFERENCE_SUMMARY_LIMIT),
                        analysis=_clip_text(card.category.analysis, _PREFERENCE_ANALYSIS_LIMIT),
                    )
                    for card in _build_category_cards(summary)
                ],
                metric_ids=sorted(metric_ids),
            )
            try:
                raw_resolution = compiler(request)
                resolved = (
                    raw_resolution
                    if isinstance(raw_resolution, PreferenceResolution)
                    else PreferenceResolution.model_validate(raw_resolution)
                )
                validate_preference_resolution(resolved, request)
                resolution_succeeded = True
            except Exception as exc:  # noqa: BLE001 - compiler adapters are untrusted
                _LOGGER.warning(
                    'Repair preference compiler failed; error_type=%s',
                    type(exc).__name__,
                )
                warnings.extend(_fallback_warnings())

    merged_categories, category_warnings = _merge_category_directives(
        direct_categories,
        resolved.category_directives if resolution_succeeded else [],
    )
    warnings.extend(category_warnings)
    merged_metrics, metric_warnings = _merge_metric_directives(
        direct_metrics,
        resolved.metric_directives if resolution_succeeded else [],
    )
    warnings.extend(metric_warnings)

    has_direct = bool(direct_categories or direct_metrics)
    source = (
        'mixed' if has_direct and resolution_succeeded
        else 'resolver' if resolution_succeeded
        else 'deterministic' if has_direct
        else 'none'
    )
    preference = CompiledPreference(
        source=source,
        category_directives=merged_categories,
        metric_directives=merged_metrics,
        metric_weights=_metric_weights(sorted(metric_ids), merged_metrics),
    )
    return preference, sorted(
        warnings,
        key=lambda item: (
            item.code,
            item.category_id or '',
            item.metric_id or '',
            item.message,
        ),
    )


def _rank_categories(
    cards: tuple[_CategoryCard, ...],
    summary: AnalysisSummaryInput,
    preference: CompiledPreference,
) -> tuple[list[RankedCategory], list[str]]:
    category_ids = {card.category_id for card in cards}
    directive_ids = {
        directive.category_id
        for directive in preference.category_directives
    }
    unknown_directives = sorted(directive_ids - category_ids)
    if unknown_directives:
        raise ValueError(
            'compiled preference contains unknown categories: '
            f'{", ".join(unknown_directives)}'
        )
    directives = {
        directive.category_id: directive
        for directive in preference.category_directives
    }
    excluded = sorted(
        (
            directive
            for directive in preference.category_directives
            if directive.tier == 'exclude'
        ),
        key=lambda item: (item.order, item.category_id),
    )
    excluded_ids = [directive.category_id for directive in excluded]
    excluded_set = set(excluded_ids)
    rankable: list[_RankableCategory] = []
    global_metrics = summary.all_case_metric_averages
    global_metric_ids = set(global_metrics)
    weight_ids = set(preference.metric_weights)
    if weight_ids != global_metric_ids:
        missing = sorted(global_metric_ids - weight_ids)
        unknown = sorted(weight_ids - global_metric_ids)
        raise ValueError(
            'compiled metric weights must exactly cover global metrics; '
            f'missing={missing}, unknown={unknown}'
        )

    for card in cards:
        if card.category_id in excluded_set:
            continue
        directive = directives.get(card.category_id)
        user_tier = directive.tier if directive is not None else 'normal'
        if user_tier == 'exclude':
            continue
        user_order = directive.order if directive is not None else None
        common_metrics = sorted(global_metric_ids & set(card.category.metric_averages))
        missing_metrics = sorted(global_metric_ids - set(common_metrics))
        raw_metric_gaps = {
            metric_id: max(
                0.0,
                float(global_metrics[metric_id])
                - float(card.category.metric_averages[metric_id]),
            )
            for metric_id in common_metrics
        }
        metric_gaps = {
            metric_id: _round_score(raw_metric_gaps[metric_id])
            for metric_id in common_metrics
        }
        total_weight = fsum(preference.metric_weights[metric_id] for metric_id in common_metrics)
        raw_weighted_gap = (
            fsum(
                preference.metric_weights[metric_id] * raw_metric_gaps[metric_id]
                for metric_id in common_metrics
            ) / total_weight
            if total_weight
            else 0.0
        )
        weighted_gap = _round_score(raw_weighted_gap)
        raw_average_drop = float(card.category.all_case_average_drop)
        average_drop = _round_score(raw_average_drop)
        case_count = len(card.category.cases)
        reason_codes = [
            *([f'user_{user_tier}'] if user_tier != 'normal' else []),
            'analysis_average_drop',
            *(['analysis_metric_gap'] if common_metrics else []),
            'analysis_case_count',
            'stable_category_id',
        ]
        value = RankedCategory(
            category_id=card.category_id,
            rank=1,
            signals=RankingSignals(
                user_tier=user_tier,
                user_order=user_order,
                all_case_average_drop=average_drop,
                metric_gaps=metric_gaps,
                weighted_metric_gap=weighted_gap,
                compared_metric_ids=common_metrics,
                missing_metric_ids=missing_metrics,
                case_count=case_count,
            ),
            preference_evidence=directive.evidence if directive is not None else None,
            reason_codes=reason_codes,
        )
        rankable.append(_RankableCategory(
            sort_key=(
                _TIER_ORDER[user_tier],
                user_order if user_order is not None else _MISSING_USER_ORDER,
                -raw_average_drop,
                -raw_weighted_gap,
                -case_count,
                card.category_id,
            ),
            value=value,
        ))

    rankable.sort(key=lambda item: item.sort_key)
    ranked = [
        item.value.model_copy(update={'rank': rank})
        for rank, item in enumerate(rankable, start=1)
    ]
    return ranked, excluded_ids


def _empty_preference(summary: AnalysisSummaryInput) -> CompiledPreference:
    metric_ids = sorted(summary.all_case_metric_averages)
    return CompiledPreference(
        source='none',
        category_directives=[],
        metric_directives=[],
        metric_weights={metric_id: 1.0 for metric_id in metric_ids},
    )


def _merge_category_directives(
    direct: list[CategoryDirective],
    resolved: list[CategoryDirective],
) -> tuple[list[CategoryDirective], list[TargetWarning]]:
    direct_ids = {item.category_id for item in direct}
    warnings = [
        TargetWarning(
            code='preference_conflict_explicit_wins',
            category_id=item.category_id,
            message='Explicit preference entry overrides semantic guidance for this category.',
        )
        for item in resolved
        if item.category_id in direct_ids
    ]
    combined = [
        *direct,
        *(item for item in resolved if item.category_id not in direct_ids),
    ]
    canonical: list[CategoryDirective] = []
    for tier in ('must', 'prefer', 'defer', 'exclude'):
        items = sorted(
            (item for item in combined if item.tier == tier),
            key=lambda item: _directive_sort_key(item, item.category_id),
        )
        canonical.extend(
            item.model_copy(update={'order': order})
            for order, item in enumerate(items)
        )
    return canonical, warnings


def _merge_metric_directives(
    direct: list[MetricDirective],
    resolved: list[MetricDirective],
) -> tuple[list[MetricDirective], list[TargetWarning]]:
    direct_ids = {item.metric_id for item in direct}
    warnings = [
        TargetWarning(
            code='preference_conflict_explicit_wins',
            metric_id=item.metric_id,
            message='Explicit preference entry overrides semantic guidance for this metric.',
        )
        for item in resolved
        if item.metric_id in direct_ids
    ]
    combined = [
        *direct,
        *(item for item in resolved if item.metric_id not in direct_ids),
    ]
    combined.sort(key=lambda item: _directive_sort_key(item, item.metric_id))
    return [
        item.model_copy(update={'order': order})
        for order, item in enumerate(combined)
    ], warnings


def _metric_weights(
    metric_ids: list[str],
    directives: list[MetricDirective],
) -> dict[str, float]:
    weights = {metric_id: 1.0 for metric_id in metric_ids}
    for position, directive in enumerate(sorted(
        directives,
        key=lambda item: (item.order, item.metric_id),
    )):
        weights[directive.metric_id] = max(1.0, 4.0 - position)
    return {metric_id: weights[metric_id] for metric_id in sorted(weights)}


def _directive_sort_key(
    directive: CategoryDirective | MetricDirective,
    target_id: str,
) -> tuple[int, int, int, str]:
    source_order = 0 if directive.evidence.source == 'preference' else 1
    return (
        source_order,
        directive.evidence.index,
        directive.order,
        target_id,
    )


def _fallback_warnings(
    *,
    compiler_unavailable: bool = False,
) -> list[TargetWarning]:
    return [
        TargetWarning(
            code=(
                'preference_compiler_unavailable'
                if compiler_unavailable
                else 'preference_compiler_failed'
            ),
            message=(
                'Semantic preference compiler is not configured.'
                if compiler_unavailable
                else 'Semantic user preference could not be compiled.'
            ),
        ),
        TargetWarning(
            code='preference_fallback_used',
            message=(
                'Unresolved preference text was ignored; validated exact preferences '
                'and numeric signals were used.'
            ),
        ),
    ]


def _round_score(value: float) -> float:
    return round(value, 4)


def _clip_text(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit - 1].rstrip() + '…'


def _dump_result(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return RepairTargetPreparation.model_validate(value).model_dump(mode='json')
    except ValidationError as exc:
        raise ValueError(f'repair.target_preparation output contract failed: {exc}') from exc


def build_target_ranking_preview(
    analysis: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    ranking_id: str,
    preference_compiler: PreferenceCompiler | None = None,
) -> dict[str, Any]:
    """Create a non-committed category ranking at the interrupt boundary.

    ``preference`` is available for the initial, soft ranking. ``user_guidance``
    is intentionally rejected here because guidance only exists after the user
    interrupts a visible preview.
    """
    summary = AnalysisSummaryInput.model_validate(analysis)
    if not isinstance(context, Mapping):
        raise TypeError('repair target preview context must be a mapping')
    target_context = RepairTargetContext.model_validate({
        'user_guidance': context.get('user_guidance', []),
        'preference': context.get('preference', []),
    })
    if target_context.user_guidance:
        raise ValueError(
            'initial target preview cannot contain user_guidance; '
            'submit it through a guidance interrupt'
        )

    if not summary.categories:
        intent = CompiledRankingIntent(
            preference_resolution=PreferenceResolution(),
            guidance_resolution=PreferenceResolution(),
            metric_weights={
                metric_id: 1.0
                for metric_id in sorted(summary.all_case_metric_averages)
            },
        )
        return _dump_ranking_state({
            'ranking_id': ranking_id,
            'source_hash': summary.source_hash,
            'revision': 1,
            'status': 'blocked',
            'blocked_reason': 'blocked_no_categories',
            'provisional_selected_category': None,
            'selected_category': None,
            'ranked_categories': [],
            'excluded_category_ids': [],
            'ranking_interpretation': intent,
            'preference': target_context.preference,
            'user_guidance': [],
            'processed_event_ids': [],
            'processed_event_fingerprints': {},
            'warnings': [],
        })

    preference_resolution, warnings = _compile_ranking_source(
        summary,
        target_context.preference,
        source='preference',
        mode='initial_preference',
        start_index=0,
        compiler=preference_compiler,
    )
    intent = _build_ranking_intent(
        summary,
        preference_resolution=preference_resolution,
        guidance_resolution=PreferenceResolution(),
    )
    return _build_preview_state(
        summary,
        ranking_id=ranking_id,
        revision=1,
        intent=intent,
        preference=target_context.preference,
        user_guidance=[],
        processed_event_ids=[],
        processed_event_fingerprints={},
        warnings=warnings,
    )


def apply_target_guidance_interrupt(
    analysis: Mapping[str, Any],
    current_state: Mapping[str, Any] | TargetRankingState,
    interrupt: Mapping[str, Any] | TargetGuidanceInterrupt,
    *,
    preference_compiler: PreferenceCompiler | None = None,
) -> dict[str, Any]:
    """Accept one interrupt, append its guidance and produce a new preview."""
    summary = AnalysisSummaryInput.model_validate(analysis)
    state = TargetRankingState.model_validate(current_state)
    command = TargetGuidanceInterrupt.model_validate(interrupt)
    _validate_transition_identity(summary, state, command)

    if _is_idempotent_retry(state, command):
        return state.model_dump(mode='json')
    if state.status != 'awaiting_interrupt':
        raise ValueError('ranking_closed: guidance requires an awaiting preview')
    if command.base_revision != state.revision:
        raise ValueError(
            f'stale_revision: expected {state.revision}, '
            f'got {command.base_revision}'
        )

    combined_guidance = (
        list(command.user_guidance)
        if command.reset_guidance
        else [*state.user_guidance, *command.user_guidance]
    )
    if len(combined_guidance) > 64:
        raise ValueError(
            'guidance_capacity_exceeded: active guidance is limited to 64 '
            'items; retry with reset_guidance=true'
        )
    RepairTargetContext.model_validate({
        'preference': state.preference,
        'user_guidance': combined_guidance,
    })

    if command.user_guidance:
        new_resolution, new_warnings = _compile_ranking_source(
            summary,
            command.user_guidance,
            source='user_guidance',
            mode='interrupt_guidance',
            start_index=0 if command.reset_guidance else len(state.user_guidance),
            compiler=preference_compiler,
        )
    else:
        new_resolution, new_warnings = PreferenceResolution(), []
    guidance_resolution = _merge_guidance_resolutions(
        (
            PreferenceResolution()
            if command.reset_guidance
            else state.ranking_interpretation.guidance_resolution
        ),
        new_resolution,
    )
    override_warnings = _guidance_override_warnings(
        state.ranking_interpretation.preference_resolution,
        new_resolution,
    )
    intent = _build_ranking_intent(
        summary,
        preference_resolution=(
            state.ranking_interpretation.preference_resolution
        ),
        guidance_resolution=guidance_resolution,
    )
    return _build_preview_state(
        summary,
        ranking_id=state.ranking_id,
        revision=state.revision + 1,
        intent=intent,
        preference=state.preference,
        user_guidance=combined_guidance,
        processed_event_ids=[
            *state.processed_event_ids,
            command.event_id,
        ],
        processed_event_fingerprints={
            **state.processed_event_fingerprints,
            command.event_id: _event_fingerprint(command),
        },
        warnings=_merge_target_warnings(
            (
                _without_guidance_warnings(state.warnings)
                if command.reset_guidance
                else state.warnings
            ),
            new_warnings,
            override_warnings,
        ),
    )


def commit_target_ranking(
    analysis: Mapping[str, Any],
    current_state: Mapping[str, Any] | TargetRankingState,
    request: Mapping[str, Any] | TargetCommitRequest,
) -> dict[str, Any]:
    """Atomically freeze the currently displayed rank 1 without reranking."""
    summary = AnalysisSummaryInput.model_validate(analysis)
    state = TargetRankingState.model_validate(current_state)
    command = TargetCommitRequest.model_validate(request)
    _validate_transition_identity(summary, state, command)

    if _is_idempotent_retry(state, command):
        return state.model_dump(mode='json')
    if state.status != 'awaiting_interrupt':
        raise ValueError('ranking_closed: commit requires an awaiting preview')
    if command.base_revision != state.revision:
        raise ValueError(
            f'stale_revision: expected {state.revision}, '
            f'got {command.base_revision}'
        )
    if state.provisional_selected_category is None:
        raise ValueError('ranking_not_committable: preview has no category')
    if command.expected_category_id != state.provisional_selected_category:
        raise ValueError(
            'category_changed: expected_category_id does not match '
            'the current preview'
        )

    payload = state.model_dump(mode='python')
    payload.update({
        'status': 'committed',
        'blocked_reason': '',
        'selected_category': state.provisional_selected_category,
        'processed_event_ids': [
            *state.processed_event_ids,
            command.event_id,
        ],
        'processed_event_fingerprints': {
            **state.processed_event_fingerprints,
            command.event_id: _event_fingerprint(command),
        },
    })
    return _dump_ranking_state(payload)


def target_ranking_to_legacy_preparation(
    current_state: Mapping[str, Any] | TargetRankingState,
) -> dict[str, Any]:
    """Adapt a committed or terminal v2 state to the existing v1 contract."""
    state = TargetRankingState.model_validate(current_state)
    if state.status == 'awaiting_interrupt':
        raise ValueError(
            'ranking_not_committed: preview cannot enter Repair execution'
        )
    compiled = _legacy_compiled_preference(state)
    if state.status == 'blocked':
        return _dump_result({
            'source_hash': state.source_hash,
            'status': 'blocked',
            'blocked_reason': state.blocked_reason,
            'selected_category': None,
            'ranked_categories': [],
            'excluded_category_ids': [],
            'preference_interpretation': compiled,
            'warnings': state.warnings,
        })

    effective_categories = {
        item.category_id: item
        for item in compiled.category_directives
    }
    ranked: list[RankedCategory] = []
    for item in state.ranked_categories:
        effective = effective_categories.get(item.category_id)
        user_tier = effective.tier if effective is not None else 'normal'
        reason_codes = [
            *([f'user_{user_tier}'] if user_tier != 'normal' else []),
            'analysis_average_drop',
            *(
                ['analysis_metric_gap']
                if item.signals.compared_metric_ids
                else []
            ),
            'analysis_case_count',
            'stable_category_id',
        ]
        ranked.append(RankedCategory(
            category_id=item.category_id,
            rank=item.rank,
            signals=RankingSignals(
                user_tier=user_tier,
                user_order=effective.order if effective is not None else None,
                all_case_average_drop=item.signals.all_case_average_drop,
                metric_gaps=item.signals.metric_gaps,
                weighted_metric_gap=item.signals.weighted_metric_gap,
                compared_metric_ids=item.signals.compared_metric_ids,
                missing_metric_ids=item.signals.missing_metric_ids,
                case_count=item.signals.case_count,
            ),
            preference_evidence=(
                effective.evidence if effective is not None else None
            ),
            reason_codes=reason_codes,
        ))
    return _dump_result({
        'source_hash': state.source_hash,
        'status': 'ready',
        'blocked_reason': '',
        'selected_category': state.selected_category,
        'ranked_categories': ranked,
        'excluded_category_ids': state.excluded_category_ids,
        'preference_interpretation': compiled,
        'warnings': state.warnings,
    })


def _compile_ranking_source(
    summary: AnalysisSummaryInput,
    texts: list[str],
    *,
    source: Literal['preference', 'user_guidance'],
    mode: PreferenceCompileMode,
    start_index: int,
    compiler: PreferenceCompiler | None,
) -> tuple[PreferenceResolution, list[TargetWarning]]:
    category_ids = set(summary.categories)
    metric_ids = set(summary.all_case_metric_averages)
    direct_categories: list[CategoryDirective] = []
    direct_metrics: list[MetricDirective] = []
    unresolved: list[PreferenceText] = []

    for offset, text in enumerate(texts):
        index = start_index + offset
        evidence = PreferenceEvidenceRef(source=source, index=index)
        if text in category_ids:
            if any(item.category_id == text for item in direct_categories):
                continue
            direct_categories.append(CategoryDirective(
                category_id=text,
                tier='prefer',
                order=offset,
                evidence=evidence,
            ))
        elif text in metric_ids:
            if any(item.metric_id == text for item in direct_metrics):
                continue
            direct_metrics.append(MetricDirective(
                metric_id=text,
                order=offset,
                evidence=evidence,
            ))
        else:
            unresolved.append(PreferenceText(evidence=evidence, text=text))

    resolved = PreferenceResolution()
    warnings: list[TargetWarning] = []
    if unresolved:
        if compiler is None:
            warnings.extend(
                _ranking_fallback_warnings(source, compiler_unavailable=True)
            )
        else:
            request = PreferenceCompileRequest(
                mode=mode,
                texts=unresolved,
                category_options=[
                    PreferenceCategoryOption(
                        category_id=card.category_id,
                        summary=_clip_text(
                            card.category.summary,
                            _PREFERENCE_SUMMARY_LIMIT,
                        ),
                        analysis=_clip_text(
                            card.category.analysis,
                            _PREFERENCE_ANALYSIS_LIMIT,
                        ),
                    )
                    for card in _build_category_cards(summary)
                ],
                metric_ids=sorted(metric_ids),
            )
            try:
                raw_resolution = compiler(request)
                resolved = (
                    raw_resolution
                    if isinstance(raw_resolution, PreferenceResolution)
                    else PreferenceResolution.model_validate(raw_resolution)
                )
                validate_preference_resolution(resolved, request)
            except Exception as exc:  # noqa: BLE001 - compiler adapters are untrusted
                _LOGGER.warning(
                    'Repair %s compiler failed; error_type=%s',
                    source,
                    type(exc).__name__,
                )
                warnings.extend(_ranking_fallback_warnings(source))
                resolved = PreferenceResolution()

    resolution, conflict_warnings = _merge_same_source_resolution(
        direct_categories,
        direct_metrics,
        resolved,
        source=source,
    )
    warnings.extend(conflict_warnings)
    return resolution, _merge_target_warnings(warnings)


def _merge_same_source_resolution(
    direct_categories: list[CategoryDirective],
    direct_metrics: list[MetricDirective],
    resolved: PreferenceResolution,
    *,
    source: Literal['preference', 'user_guidance'],
) -> tuple[PreferenceResolution, list[TargetWarning]]:
    category_ids = {item.category_id for item in direct_categories}
    metric_ids = {item.metric_id for item in direct_metrics}
    warning_code = (
        'guidance_conflict_explicit_wins'
        if source == 'user_guidance'
        else 'preference_conflict_explicit_wins'
    )
    warnings = [
        *(
            TargetWarning(
                code=warning_code,
                category_id=item.category_id,
                message=(
                    'Exact input overrides semantic output for this category.'
                ),
            )
            for item in resolved.category_directives
            if item.category_id in category_ids
        ),
        *(
            TargetWarning(
                code=warning_code,
                metric_id=item.metric_id,
                message='Exact input overrides semantic output for this metric.',
            )
            for item in resolved.metric_directives
            if item.metric_id in metric_ids
        ),
    ]
    categories = [
        *direct_categories,
        *(
            item
            for item in resolved.category_directives
            if item.category_id not in category_ids
        ),
    ]
    metrics = [
        *direct_metrics,
        *(
            item
            for item in resolved.metric_directives
            if item.metric_id not in metric_ids
        ),
    ]
    return PreferenceResolution(
        category_directives=_canonical_category_directives(
            categories,
        ),
        metric_directives=_canonical_metric_directives(
            metrics,
        ),
    ), warnings


def _merge_guidance_resolutions(
    existing: PreferenceResolution,
    new: PreferenceResolution,
) -> PreferenceResolution:
    new_category_ids = {
        item.category_id
        for item in new.category_directives
    }
    categories: list[CategoryDirective] = []
    for tier in ('must', 'prefer', 'defer', 'exclude'):
        current_batch = sorted(
            (item for item in new.category_directives if item.tier == tier),
            key=lambda item: (
                item.evidence.index,
                item.order,
                item.category_id,
            ),
        )
        older_batches = sorted(
            (
                item
                for item in existing.category_directives
                if item.tier == tier
                and item.category_id not in new_category_ids
            ),
            key=lambda item: (
                item.order,
                item.evidence.index,
                item.category_id,
            ),
        )
        categories.extend(
            item.model_copy(update={'order': order})
            for order, item in enumerate([*current_batch, *older_batches])
        )

    new_metric_ids = {
        item.metric_id
        for item in new.metric_directives
    }
    current_metrics = sorted(
        new.metric_directives,
        key=lambda item: (
            item.evidence.index,
            item.order,
            item.metric_id,
        ),
    )
    older_metrics = sorted(
        (
            item
            for item in existing.metric_directives
            if item.metric_id not in new_metric_ids
        ),
        key=lambda item: (
            item.order,
            item.evidence.index,
            item.metric_id,
        ),
    )
    metrics = [
        item.model_copy(update={'order': order})
        for order, item in enumerate([*current_metrics, *older_metrics])
    ]
    return PreferenceResolution(
        category_directives=categories,
        metric_directives=metrics,
    )


def _canonical_category_directives(
    directives: list[CategoryDirective],
) -> list[CategoryDirective]:
    canonical: list[CategoryDirective] = []
    for tier in ('must', 'prefer', 'defer', 'exclude'):
        items = sorted(
            (item for item in directives if item.tier == tier),
            key=lambda item: _layer_directive_sort_key(item, item.category_id),
        )
        canonical.extend(
            item.model_copy(update={'order': order})
            for order, item in enumerate(items)
        )
    return canonical


def _canonical_metric_directives(
    directives: list[MetricDirective],
) -> list[MetricDirective]:
    items = sorted(
        directives,
        key=lambda item: _layer_directive_sort_key(item, item.metric_id),
    )
    return [
        item.model_copy(update={'order': order})
        for order, item in enumerate(items)
    ]


def _layer_directive_sort_key(
    directive: CategoryDirective | MetricDirective,
    target_id: str,
) -> tuple[int, int, str]:
    return directive.evidence.index, directive.order, target_id


def _build_ranking_intent(
    summary: AnalysisSummaryInput,
    *,
    preference_resolution: PreferenceResolution,
    guidance_resolution: PreferenceResolution,
) -> CompiledRankingIntent:
    guidance_metric_ids = {
        item.metric_id
        for item in guidance_resolution.metric_directives
    }
    effective_metrics = [
        *guidance_resolution.metric_directives,
        *(
            item
            for item in preference_resolution.metric_directives
            if item.metric_id not in guidance_metric_ids
        ),
    ]
    effective_metrics = [
        item.model_copy(update={'order': order})
        for order, item in enumerate(effective_metrics)
    ]
    return CompiledRankingIntent(
        preference_resolution=preference_resolution,
        guidance_resolution=guidance_resolution,
        metric_weights=_metric_weights(
            sorted(summary.all_case_metric_averages),
            effective_metrics,
        ),
    )


def _build_preview_state(
    summary: AnalysisSummaryInput,
    *,
    ranking_id: str,
    revision: int,
    intent: CompiledRankingIntent,
    preference: list[str],
    user_guidance: list[str],
    processed_event_ids: list[str],
    processed_event_fingerprints: dict[str, str],
    warnings: list[TargetWarning],
) -> dict[str, Any]:
    ranked, excluded = _rank_interruptible_categories(summary, intent)
    provisional = ranked[0].category_id if ranked else None
    blocked_reason = '' if ranked else 'blocked_all_categories_excluded'
    return _dump_ranking_state({
        'ranking_id': ranking_id,
        'source_hash': summary.source_hash,
        'revision': revision,
        'status': 'awaiting_interrupt',
        'blocked_reason': blocked_reason,
        'provisional_selected_category': provisional,
        'selected_category': None,
        'ranked_categories': ranked,
        'excluded_category_ids': excluded,
        'ranking_interpretation': intent,
        'preference': preference,
        'user_guidance': user_guidance,
        'processed_event_ids': processed_event_ids,
        'processed_event_fingerprints': processed_event_fingerprints,
        'warnings': _merge_target_warnings(warnings),
    })


def _rank_interruptible_categories(
    summary: AnalysisSummaryInput,
    intent: CompiledRankingIntent,
) -> tuple[list[InterruptibleRankedCategory], list[str]]:
    preference_categories = {
        item.category_id: item
        for item in intent.preference_resolution.category_directives
    }
    guidance_categories = {
        item.category_id: item
        for item in intent.guidance_resolution.category_directives
    }
    unknown = sorted(
        (
            set(preference_categories)
            | set(guidance_categories)
        ) - set(summary.categories)
    )
    if unknown:
        raise ValueError(
            'compiled ranking intent contains unknown categories: '
            f'{", ".join(unknown)}'
        )

    excluded_directives = []
    for category_id in sorted(summary.categories):
        effective = guidance_categories.get(
            category_id,
            preference_categories.get(category_id),
        )
        if effective is not None and effective.tier == 'exclude':
            excluded_directives.append(effective)
    excluded_directives.sort(key=lambda item: (item.order, item.category_id))
    excluded_ids = [item.category_id for item in excluded_directives]
    excluded_set = set(excluded_ids)

    global_metrics = summary.all_case_metric_averages
    global_metric_ids = set(global_metrics)
    if set(intent.metric_weights) != global_metric_ids:
        raise ValueError(
            'compiled metric weights must exactly cover global metrics'
        )

    rankable: list[_RankableCategory] = []
    for card in _build_category_cards(summary):
        if card.category_id in excluded_set:
            continue
        preference = preference_categories.get(card.category_id)
        guidance = guidance_categories.get(card.category_id)
        preference_tier = (
            preference.tier if preference is not None else 'normal'
        )
        guidance_tier = guidance.tier if guidance is not None else 'normal'
        common_metrics = sorted(
            global_metric_ids & set(card.category.metric_averages)
        )
        missing_metrics = sorted(global_metric_ids - set(common_metrics))
        raw_metric_gaps = {
            metric_id: max(
                0.0,
                float(global_metrics[metric_id])
                - float(card.category.metric_averages[metric_id]),
            )
            for metric_id in common_metrics
        }
        metric_gaps = {
            metric_id: _round_score(raw_metric_gaps[metric_id])
            for metric_id in common_metrics
        }
        total_weight = fsum(
            intent.metric_weights[metric_id]
            for metric_id in common_metrics
        )
        raw_weighted_gap = (
            fsum(
                intent.metric_weights[metric_id]
                * raw_metric_gaps[metric_id]
                for metric_id in common_metrics
            ) / total_weight
            if total_weight
            else 0.0
        )
        raw_average_drop = float(card.category.all_case_average_drop)
        reason_codes = [
            *(
                [f'guidance_{guidance_tier}']
                if guidance_tier != 'normal'
                else []
            ),
            *(
                [f'preference_{preference_tier}']
                if preference_tier != 'normal'
                else []
            ),
            'analysis_average_drop',
            *(['analysis_metric_gap'] if common_metrics else []),
            'analysis_case_count',
            'stable_category_id',
        ]
        value = InterruptibleRankedCategory(
            category_id=card.category_id,
            rank=1,
            signals=InterruptibleRankingSignals(
                guidance_tier=guidance_tier,
                guidance_order=(
                    guidance.order if guidance is not None else None
                ),
                preference_tier=preference_tier,
                preference_order=(
                    preference.order if preference is not None else None
                ),
                all_case_average_drop=_round_score(raw_average_drop),
                metric_gaps=metric_gaps,
                weighted_metric_gap=_round_score(raw_weighted_gap),
                compared_metric_ids=common_metrics,
                missing_metric_ids=missing_metrics,
                case_count=len(card.category.cases),
            ),
            preference_evidence=(
                preference.evidence if preference is not None else None
            ),
            guidance_evidence=(
                guidance.evidence if guidance is not None else None
            ),
            reason_codes=reason_codes,
        )
        rankable.append(_RankableCategory(
            sort_key=(
                _TIER_ORDER[guidance_tier],
                guidance.order if guidance is not None else _MISSING_USER_ORDER,
                _TIER_ORDER[preference_tier],
                (
                    preference.order
                    if preference is not None
                    else _MISSING_USER_ORDER
                ),
                -raw_average_drop,
                -raw_weighted_gap,
                -len(card.category.cases),
                card.category_id,
            ),
            value=value,
        ))

    rankable.sort(key=lambda item: item.sort_key)
    ranked = [
        item.value.model_copy(update={'rank': rank})
        for rank, item in enumerate(rankable, start=1)
    ]
    return ranked, excluded_ids


def _guidance_override_warnings(
    preference: PreferenceResolution,
    new_guidance: PreferenceResolution,
) -> list[TargetWarning]:
    preference_category_ids = {
        item.category_id for item in preference.category_directives
    }
    preference_metric_ids = {
        item.metric_id for item in preference.metric_directives
    }
    return [
        *(
            TargetWarning(
                code='guidance_overrode_preference',
                category_id=item.category_id,
                message=(
                    'Interrupt guidance overrides the initial preference '
                    'for this category.'
                ),
            )
            for item in new_guidance.category_directives
            if item.category_id in preference_category_ids
        ),
        *(
            TargetWarning(
                code='guidance_overrode_preference',
                metric_id=item.metric_id,
                message=(
                    'Interrupt guidance overrides the initial preference '
                    'for this metric.'
                ),
            )
            for item in new_guidance.metric_directives
            if item.metric_id in preference_metric_ids
        ),
    ]


def _ranking_fallback_warnings(
    source: Literal['preference', 'user_guidance'],
    *,
    compiler_unavailable: bool = False,
) -> list[TargetWarning]:
    is_guidance = source == 'user_guidance'
    return [
        TargetWarning(
            code=(
                'guidance_compiler_unavailable'
                if is_guidance and compiler_unavailable
                else 'guidance_compiler_failed'
                if is_guidance
                else 'preference_compiler_unavailable'
                if compiler_unavailable
                else 'preference_compiler_failed'
            ),
            message=(
                'Semantic guidance compiler is not configured.'
                if is_guidance and compiler_unavailable
                else 'Semantic interrupt guidance could not be compiled.'
                if is_guidance
                else 'Semantic preference compiler is not configured.'
                if compiler_unavailable
                else 'Semantic preference could not be compiled.'
            ),
        ),
        TargetWarning(
            code=(
                'guidance_fallback_used'
                if is_guidance
                else 'preference_fallback_used'
            ),
            message=(
                'Unresolved interrupt guidance was not applied; the previous '
                'visible ranking was preserved where deterministic guidance '
                'was unavailable.'
                if is_guidance
                else 'Unresolved preference text was ignored; validated exact '
                'preferences and numeric signals were used.'
            ),
        ),
    ]


def _merge_target_warnings(
    *groups: list[TargetWarning],
) -> list[TargetWarning]:
    limit_already_reached = any(
        item.code == 'warning_limit_reached'
        for group in groups
        for item in group
    )
    unique = {
        (
            item.code,
            item.category_id or '',
            item.metric_id or '',
            item.message,
        ): item
        for group in groups
        for item in group
        if item.code != 'warning_limit_reached'
    }
    keys = sorted(unique)
    if (
        len(keys) <= _MAX_TARGET_WARNINGS
        and not limit_already_reached
    ):
        return [unique[key] for key in keys]

    # Preserve all preference/global warnings before guidance-specific detail.
    # The legal input maximum can otherwise make a later interrupt evict the
    # original preference diagnostics, which a reset cannot reconstruct.
    retained_keys = sorted(
        keys,
        key=lambda key: (key[0].startswith('guidance_'), key),
    )[:_MAX_TARGET_WARNINGS - 1]
    return [
        *[unique[key] for key in sorted(retained_keys)],
        TargetWarning(
            code='warning_limit_reached',
            message='Additional target-ranking warnings were omitted.',
        ),
    ]


def _without_guidance_warnings(
    warnings: list[TargetWarning],
) -> list[TargetWarning]:
    return [
        item
        for item in warnings
        if (
            not item.code.startswith('guidance_')
            and item.code != 'warning_limit_reached'
        )
    ]


def _event_fingerprint(
    command: TargetGuidanceInterrupt | TargetCommitRequest,
) -> str:
    payload = json.dumps(
        command.model_dump(mode='json'),
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    )
    return sha256(payload.encode()).hexdigest()


def _is_idempotent_retry(
    state: TargetRankingState,
    command: TargetGuidanceInterrupt | TargetCommitRequest,
) -> bool:
    fingerprint = state.processed_event_fingerprints.get(command.event_id)
    if fingerprint is None:
        return False
    if fingerprint != _event_fingerprint(command):
        raise ValueError(
            'idempotency_conflict: event_id was already used for a '
            'different command'
        )
    return True


def _validate_transition_identity(
    summary: AnalysisSummaryInput,
    state: TargetRankingState,
    command: TargetGuidanceInterrupt | TargetCommitRequest,
) -> None:
    if command.ranking_id != state.ranking_id:
        raise ValueError('foreign_session: ranking_id does not match')
    if command.source_hash != state.source_hash:
        raise ValueError('source_changed: command source_hash does not match')
    if summary.source_hash != state.source_hash:
        raise ValueError('source_changed: Analysis source_hash does not match')


def _legacy_compiled_preference(
    state: TargetRankingState,
) -> CompiledPreference:
    intent = state.ranking_interpretation
    preference_categories = {
        item.category_id: item
        for item in intent.preference_resolution.category_directives
    }
    guidance_categories = {
        item.category_id: item
        for item in intent.guidance_resolution.category_directives
    }
    effective_categories = {
        **preference_categories,
        **guidance_categories,
    }
    categories = _legacy_category_directives(
        state,
        list(effective_categories.values()),
    )
    metrics = _legacy_metric_directives(intent)
    return CompiledPreference(
        source=_legacy_compiled_source(
            state,
            categories=categories,
            metrics=metrics,
        ),
        category_directives=categories,
        metric_directives=metrics,
        metric_weights=intent.metric_weights,
    )


def _legacy_category_directives(
    state: TargetRankingState,
    directives: list[CategoryDirective],
) -> list[CategoryDirective]:
    ordered_category_ids = [
        item.category_id
        for item in state.ranked_categories
    ]
    ordered_category_ids.extend(state.excluded_category_ids)
    positions = {
        category_id: index
        for index, category_id in enumerate(ordered_category_ids)
    }
    result: list[CategoryDirective] = []
    for tier in ('must', 'prefer', 'defer', 'exclude'):
        tier_items = sorted(
            (
                item
                for item in directives
                if item.tier == tier
            ),
            key=lambda item: (
                positions.get(
                    item.category_id,
                    _MISSING_USER_ORDER,
                ),
                item.category_id,
            ),
        )
        result.extend(
            item.model_copy(update={'order': order})
            for order, item in enumerate(tier_items)
        )
    return result


def _legacy_metric_directives(
    intent: CompiledRankingIntent,
) -> list[MetricDirective]:
    guidance_metric_ids = {
        item.metric_id
        for item in intent.guidance_resolution.metric_directives
    }
    effective_metrics = [
        *intent.guidance_resolution.metric_directives,
        *(
            item
            for item in intent.preference_resolution.metric_directives
            if item.metric_id not in guidance_metric_ids
        ),
    ]
    return [
        item.model_copy(update={'order': order})
        for order, item in enumerate(effective_metrics)
    ]


def _legacy_compiled_source(
    state: TargetRankingState,
    *,
    categories: list[CategoryDirective],
    metrics: list[MetricDirective],
) -> Literal['none', 'deterministic', 'resolver', 'mixed']:
    sources: set[str] = set()
    for directive in categories:
        sources.add(
            _legacy_directive_source(
                state,
                directive=directive,
                target_id=directive.category_id,
            )
        )
    for directive in metrics:
        sources.add(
            _legacy_directive_source(
                state,
                directive=directive,
                target_id=directive.metric_id,
            )
        )
    if not sources:
        return 'none'
    if len(sources) > 1:
        return 'mixed'
    return 'deterministic' if 'deterministic' in sources else 'resolver'


def _legacy_directive_source(
    state: TargetRankingState,
    *,
    directive: CategoryDirective | MetricDirective,
    target_id: str,
) -> Literal['deterministic', 'resolver']:
    evidence = directive.evidence
    texts = (
        state.preference
        if evidence.source == 'preference'
        else state.user_guidance
    )
    return (
        'deterministic'
        if texts[evidence.index] == target_id
        else 'resolver'
    )


def _dump_ranking_state(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return TargetRankingState.model_validate(value).model_dump(mode='json')
    except ValidationError as exc:
        raise ValueError(f'repair.target_ranking output contract failed: {exc}') from exc


__all__ = [
    'PreferenceCompiler',
    'apply_target_guidance_interrupt',
    'build_target_preparation',
    'build_target_ranking_preview',
    'commit_target_ranking',
    'target_ranking_to_legacy_preparation',
]
