from __future__ import annotations

from typing import Annotated, Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

MetricScore = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
Identifier = Annotated[
    StrictStr,
    Field(
        min_length=1,
        max_length=256,
        pattern=r'^\S(?:.*\S)?$',
    ),
]
TraceReference = Annotated[StrictStr, Field(min_length=1, max_length=2048)]
EventFingerprint = Annotated[
    StrictStr,
    Field(pattern=r'^[0-9a-f]{64}$'),
]
PreferenceTier = Literal['must', 'prefer', 'defer', 'exclude']
RankingTier = Literal['must', 'prefer', 'normal', 'defer']
PreferenceCompileMode = Literal[
    'legacy',
    'initial_preference',
    'interrupt_guidance',
]
WarningCode = Literal[
    'preference_compiler_unavailable',
    'preference_compiler_failed',
    'preference_fallback_used',
    'preference_conflict_explicit_wins',
    'guidance_compiler_unavailable',
    'guidance_compiler_failed',
    'guidance_fallback_used',
    'guidance_conflict_explicit_wins',
    'guidance_overrode_preference',
    'warning_limit_reached',
]
ReasonCode = Literal[
    'user_must',
    'user_prefer',
    'user_defer',
    'analysis_average_drop',
    'analysis_metric_gap',
    'analysis_case_count',
    'stable_category_id',
]
RankingReasonCode = Literal[
    'guidance_must',
    'guidance_prefer',
    'guidance_defer',
    'preference_prefer',
    'preference_defer',
    'analysis_average_drop',
    'analysis_metric_gap',
    'analysis_case_count',
    'stable_category_id',
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class AnalysisCategory(StrictModel):
    metric_averages: dict[Identifier, MetricScore] = Field(max_length=128)
    all_case_average_drop: MetricScore
    code_span: list[dict[StrictStr, Any]]
    summary: StrictStr = Field(min_length=1, max_length=1000)
    analysis: StrictStr = Field(min_length=1, max_length=10000)
    cases: dict[Identifier, TraceReference] = Field(min_length=1)

    @field_validator('summary', 'analysis', mode='before')
    @classmethod
    def normalize_text(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            raise ValueError('value must contain non-whitespace text')
        return text

    @field_validator('metric_averages')
    @classmethod
    def validate_metric_names(cls, value: dict[str, float]) -> dict[str, float]:
        _require_clean_mapping_keys(value, 'metric name')
        return value

    @field_validator('cases')
    @classmethod
    def validate_cases(cls, value: dict[str, str]) -> dict[str, str]:
        _require_clean_mapping_keys(value, 'case id')
        for case_id, trace_ref in value.items():
            if not trace_ref or trace_ref != trace_ref.strip():
                raise ValueError(f'trace ref for case {case_id!r} must be non-empty and trimmed')
        return value


class AnalysisSummaryInput(StrictModel):
    source_hash: StrictStr = Field(min_length=1, max_length=256)
    all_case_metric_averages: dict[Identifier, MetricScore] = Field(
        min_length=1,
        max_length=128,
    )
    categories: dict[Identifier, AnalysisCategory] = Field(max_length=200)

    @field_validator('source_hash', mode='before')
    @classmethod
    def normalize_source_hash(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            raise ValueError('source_hash must contain non-whitespace text')
        return text

    @field_validator('all_case_metric_averages')
    @classmethod
    def validate_metric_names(cls, value: dict[str, float]) -> dict[str, float]:
        _require_clean_mapping_keys(value, 'metric name')
        return value

    @field_validator('categories')
    @classmethod
    def validate_category_ids(cls, value: dict[str, AnalysisCategory]) -> dict[str, AnalysisCategory]:
        _require_clean_mapping_keys(value, 'category id')
        return value

    @model_validator(mode='after')
    def validate_category_metrics(self) -> Self:
        global_metrics = set(self.all_case_metric_averages)
        ambiguous_ids = sorted(global_metrics & set(self.categories))
        if ambiguous_ids:
            raise ValueError(
                'category and metric identifiers must not overlap: '
                f'{", ".join(ambiguous_ids)}'
            )
        for category_id, category in self.categories.items():
            unknown = sorted(set(category.metric_averages) - global_metrics)
            if unknown:
                raise ValueError(
                    f'category {category_id!r} contains metrics missing from '
                    f'all_case_metric_averages: {", ".join(unknown)}'
                )
        return self


class RepairTargetContext(StrictModel):
    user_guidance: list[StrictStr] = Field(default_factory=list, max_length=64)
    preference: list[StrictStr] = Field(default_factory=list, max_length=64)

    @field_validator('user_guidance', 'preference')
    @classmethod
    def normalize_text_items(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        for raw in value:
            text = raw.strip()
            if not text:
                raise ValueError('context text items must contain non-whitespace text')
            if len(text) > 2000:
                raise ValueError('context text items must not exceed 2000 characters')
            result.append(text)
        return result


class PreferenceEvidenceRef(StrictModel):
    source: Literal['user_guidance', 'preference']
    index: StrictInt = Field(ge=0)


class PreferenceText(StrictModel):
    evidence: PreferenceEvidenceRef
    text: StrictStr = Field(min_length=1, max_length=2000)


class PreferenceCategoryOption(StrictModel):
    category_id: Identifier
    summary: StrictStr = Field(min_length=1, max_length=1000)
    analysis: StrictStr = Field(min_length=1, max_length=10000)


class PreferenceCompileRequest(StrictModel):
    mode: PreferenceCompileMode = 'legacy'
    texts: list[PreferenceText] = Field(max_length=128)
    category_options: list[PreferenceCategoryOption] = Field(max_length=200)
    metric_ids: list[Identifier] = Field(max_length=128)

    @model_validator(mode='after')
    def validate_unique_options(self) -> Self:
        _require_unique(
            (option.category_id for option in self.category_options),
            'category option',
        )
        _require_unique(self.metric_ids, 'metric id')
        _require_unique(
            (
                f'{item.evidence.source}[{item.evidence.index}]'
                for item in self.texts
            ),
            'preference evidence',
        )
        expected_source = {
            'initial_preference': 'preference',
            'interrupt_guidance': 'user_guidance',
        }.get(self.mode)
        if expected_source is not None:
            invalid_sources = sorted({
                item.evidence.source
                for item in self.texts
                if item.evidence.source != expected_source
            })
            if invalid_sources:
                raise ValueError(
                    f'{self.mode} requests only accept {expected_source} evidence'
                )
        return self


class CategoryDirective(StrictModel):
    category_id: Identifier
    tier: PreferenceTier
    order: StrictInt = Field(ge=0)
    evidence: PreferenceEvidenceRef


class MetricDirective(StrictModel):
    metric_id: Identifier
    order: StrictInt = Field(ge=0)
    evidence: PreferenceEvidenceRef


class PreferenceResolution(StrictModel):
    category_directives: list[CategoryDirective] = Field(default_factory=list, max_length=200)
    metric_directives: list[MetricDirective] = Field(default_factory=list, max_length=128)

    @model_validator(mode='after')
    def validate_unique_targets(self) -> Self:
        _require_unique(
            (directive.category_id for directive in self.category_directives),
            'category directive',
        )
        _require_unique(
            (directive.metric_id for directive in self.metric_directives),
            'metric directive',
        )
        return self


class CompiledPreference(PreferenceResolution):
    source: Literal['none', 'deterministic', 'resolver', 'mixed']
    metric_weights: dict[Identifier, FiniteFloat] = Field(max_length=128)

    @field_validator('metric_weights')
    @classmethod
    def validate_metric_weights(cls, value: dict[str, float]) -> dict[str, float]:
        _require_clean_mapping_keys(value, 'metric id')
        for metric_id, weight in value.items():
            if weight <= 0.0 or weight > 4.0:
                raise ValueError(f'metric weight for {metric_id!r} must be in (0, 4]')
        return value

    @model_validator(mode='after')
    def validate_canonical_order(self) -> Self:
        tier_order = {
            'must': 0,
            'prefer': 1,
            'defer': 2,
            'exclude': 3,
        }
        actual_tiers = [
            tier_order[item.tier]
            for item in self.category_directives
        ]
        if actual_tiers != sorted(actual_tiers):
            raise ValueError('compiled category directives must use canonical tier order')
        for tier in ('must', 'prefer', 'defer', 'exclude'):
            orders = [
                item.order
                for item in self.category_directives
                if item.tier == tier
            ]
            if orders != list(range(len(orders))):
                raise ValueError(
                    f'compiled {tier} category directive order must start at 0 '
                    'and be contiguous'
                )
        metric_orders = [item.order for item in self.metric_directives]
        if metric_orders != list(range(len(metric_orders))):
            raise ValueError(
                'compiled metric directive order must start at 0 and be contiguous'
            )
        return self


class CompiledRankingIntent(StrictModel):
    """Auditable preference and interrupt-guidance layers for v2 ranking."""

    preference_resolution: PreferenceResolution = Field(
        default_factory=PreferenceResolution
    )
    guidance_resolution: PreferenceResolution = Field(
        default_factory=PreferenceResolution
    )
    metric_weights: dict[Identifier, FiniteFloat] = Field(max_length=128)

    @field_validator('metric_weights')
    @classmethod
    def validate_metric_weights(cls, value: dict[str, float]) -> dict[str, float]:
        _require_clean_mapping_keys(value, 'metric id')
        for metric_id, weight in value.items():
            if weight <= 0.0 or weight > 4.0:
                raise ValueError(f'metric weight for {metric_id!r} must be in (0, 4]')
        return value

    @model_validator(mode='after')
    def validate_layers(self) -> Self:
        _validate_canonical_resolution(
            self.preference_resolution,
            label='preference',
            expected_source='preference',
            allowed_tiers={'prefer', 'defer'},
        )
        _validate_canonical_resolution(
            self.guidance_resolution,
            label='guidance',
            expected_source='user_guidance',
            allowed_tiers={'must', 'prefer', 'defer', 'exclude'},
        )
        return self


class RankingSignals(StrictModel):
    user_tier: RankingTier
    user_order: StrictInt | None = Field(default=None, ge=0)
    all_case_average_drop: MetricScore
    metric_gaps: dict[Identifier, MetricScore] = Field(max_length=128)
    weighted_metric_gap: MetricScore
    compared_metric_ids: list[Identifier] = Field(max_length=128)
    missing_metric_ids: list[Identifier] = Field(max_length=128)
    case_count: StrictInt = Field(ge=1)


class RankedCategory(StrictModel):
    category_id: Identifier
    rank: StrictInt = Field(ge=1)
    signals: RankingSignals
    preference_evidence: PreferenceEvidenceRef | None = None
    reason_codes: list[ReasonCode] = Field(max_length=7)


class InterruptibleRankingSignals(StrictModel):
    guidance_tier: RankingTier
    guidance_order: StrictInt | None = Field(default=None, ge=0)
    preference_tier: RankingTier
    preference_order: StrictInt | None = Field(default=None, ge=0)
    all_case_average_drop: MetricScore
    metric_gaps: dict[Identifier, MetricScore] = Field(max_length=128)
    weighted_metric_gap: MetricScore
    compared_metric_ids: list[Identifier] = Field(max_length=128)
    missing_metric_ids: list[Identifier] = Field(max_length=128)
    case_count: StrictInt = Field(ge=1)


class InterruptibleRankedCategory(StrictModel):
    category_id: Identifier
    rank: StrictInt = Field(ge=1)
    signals: InterruptibleRankingSignals
    preference_evidence: PreferenceEvidenceRef | None = None
    guidance_evidence: PreferenceEvidenceRef | None = None
    reason_codes: list[RankingReasonCode] = Field(max_length=9)


class TargetWarning(StrictModel):
    code: WarningCode
    message: StrictStr = Field(min_length=1, max_length=500)
    category_id: Identifier | None = None
    metric_id: Identifier | None = None

    @model_validator(mode='after')
    def validate_warning_target(self) -> Self:
        has_category = self.category_id is not None
        has_metric = self.metric_id is not None
        targeted_codes = {
            'preference_conflict_explicit_wins',
            'guidance_conflict_explicit_wins',
            'guidance_overrode_preference',
        }
        if self.code in targeted_codes:
            if has_category == has_metric:
                raise ValueError('targeted warning requires exactly one target id')
        elif has_category or has_metric:
            raise ValueError('only targeted warnings can identify a target')
        return self


class RepairTargetPreparation(StrictModel):
    id: Literal['repair.target_preparation'] = 'repair.target_preparation'
    schema_version: Literal['1'] = '1'
    source_hash: StrictStr = Field(min_length=1, max_length=256)
    status: Literal['ready', 'blocked']
    blocked_reason: Literal[
        '',
        'blocked_no_categories',
        'blocked_all_categories_excluded',
    ]
    selected_category: Identifier | None
    ranked_categories: list[RankedCategory] = Field(max_length=200)
    excluded_category_ids: list[Identifier] = Field(max_length=200)
    preference_interpretation: CompiledPreference
    warnings: list[TargetWarning] = Field(max_length=400)

    @model_validator(mode='after')
    def validate_result(self) -> Self:
        ranked_ids = [item.category_id for item in self.ranked_categories]
        _require_unique(ranked_ids, 'ranked category')
        _require_unique(self.excluded_category_ids, 'excluded category')
        overlap = sorted(set(ranked_ids) & set(self.excluded_category_ids))
        if overlap:
            raise ValueError(f'categories cannot be both ranked and excluded: {", ".join(overlap)}')
        expected_excluded = [
            item.category_id
            for item in self.preference_interpretation.category_directives
            if item.tier == 'exclude'
        ]
        if self.excluded_category_ids != expected_excluded:
            raise ValueError(
                'excluded_category_ids must match ordered exclude preference directives'
            )
        known_category_ids = set(ranked_ids) | set(self.excluded_category_ids)
        directive_ids = {
            item.category_id
            for item in self.preference_interpretation.category_directives
        }
        ghost_categories = sorted(directive_ids - known_category_ids)
        if ghost_categories:
            raise ValueError(
                'category directives reference categories absent from the result: '
                f'{", ".join(ghost_categories)}'
            )
        metric_ids = set(self.preference_interpretation.metric_weights)
        directive_metric_ids = {
            item.metric_id
            for item in self.preference_interpretation.metric_directives
        }
        ghost_metrics = sorted(directive_metric_ids - metric_ids)
        if ghost_metrics:
            raise ValueError(
                'metric directives lack corresponding metric weights: '
                f'{", ".join(ghost_metrics)}'
            )

        category_directives = {
            item.category_id: item
            for item in self.preference_interpretation.category_directives
        }
        for item in self.ranked_categories:
            directive = category_directives.get(item.category_id)
            expected_tier = directive.tier if directive is not None else 'normal'
            expected_order = directive.order if directive is not None else None
            expected_evidence = directive.evidence if directive is not None else None
            if (
                item.signals.user_tier != expected_tier
                or item.signals.user_order != expected_order
                or item.preference_evidence != expected_evidence
            ):
                raise ValueError(
                    f'ranked category {item.category_id!r} does not match '
                    'its preference directive'
                )
            compared = set(item.signals.compared_metric_ids)
            missing = set(item.signals.missing_metric_ids)
            if set(item.signals.metric_gaps) != compared:
                raise ValueError('metric_gaps keys must match compared_metric_ids')
            if compared & missing or compared | missing != metric_ids:
                raise ValueError(
                    'compared_metric_ids and missing_metric_ids must partition '
                    'the weighted metrics'
                )

        expected_ranks = list(range(1, len(self.ranked_categories) + 1))
        actual_ranks = [item.rank for item in self.ranked_categories]
        if actual_ranks != expected_ranks:
            raise ValueError('ranked category ranks must be contiguous and start at 1')

        if self.status == 'ready':
            if self.blocked_reason:
                raise ValueError('ready result cannot have blocked_reason')
            if not self.selected_category or not self.ranked_categories:
                raise ValueError('ready result requires a selected and ranked category')
            if self.selected_category != self.ranked_categories[0].category_id:
                raise ValueError('selected_category must match rank 1')
        else:
            if not self.blocked_reason:
                raise ValueError('blocked result requires blocked_reason')
            if self.selected_category is not None:
                raise ValueError('blocked result cannot have selected_category')
            if self.ranked_categories:
                raise ValueError('blocked result cannot have ranked_categories')
            if (
                self.blocked_reason == 'blocked_no_categories'
                and self.excluded_category_ids
            ):
                raise ValueError('blocked_no_categories cannot have excluded categories')
            if (
                self.blocked_reason == 'blocked_all_categories_excluded'
                and not self.excluded_category_ids
            ):
                raise ValueError('blocked_all_categories_excluded requires excluded categories')
        return self


class TargetRankingState(StrictModel):
    """Server-owned v2 state for preview, interrupt and commit transitions."""

    id: Literal['repair.target_ranking'] = 'repair.target_ranking'
    schema_version: Literal['2'] = '2'
    ranking_id: Identifier
    source_hash: StrictStr = Field(min_length=1, max_length=256)
    revision: StrictInt = Field(ge=1)
    status: Literal['awaiting_interrupt', 'committed', 'blocked']
    blocked_reason: Literal[
        '',
        'blocked_no_categories',
        'blocked_all_categories_excluded',
    ]
    provisional_selected_category: Identifier | None
    selected_category: Identifier | None
    ranked_categories: list[InterruptibleRankedCategory] = Field(max_length=200)
    excluded_category_ids: list[Identifier] = Field(max_length=200)
    ranking_interpretation: CompiledRankingIntent
    preference: list[StrictStr] = Field(default_factory=list, max_length=64)
    user_guidance: list[StrictStr] = Field(default_factory=list, max_length=64)
    processed_event_ids: list[Identifier] = Field(default_factory=list)
    processed_event_fingerprints: dict[Identifier, EventFingerprint] = Field(
        default_factory=dict,
    )
    warnings: list[TargetWarning] = Field(default_factory=list, max_length=400)

    @field_validator('source_hash', mode='before')
    @classmethod
    def normalize_source_hash(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            raise ValueError('source_hash must contain non-whitespace text')
        return text

    @field_validator('preference', 'user_guidance')
    @classmethod
    def normalize_context_text(cls, value: list[str]) -> list[str]:
        return _normalize_text_items(value, label='ranking context')

    @model_validator(mode='after')
    def validate_state(self) -> Self:
        ranked_ids = [item.category_id for item in self.ranked_categories]
        _require_unique(ranked_ids, 'ranked category')
        _require_unique(self.excluded_category_ids, 'excluded category')
        _require_unique(self.processed_event_ids, 'processed event id')
        if set(self.processed_event_fingerprints) != set(self.processed_event_ids):
            raise ValueError(
                'processed_event_fingerprints keys must match processed_event_ids'
            )
        overlap = sorted(set(ranked_ids) & set(self.excluded_category_ids))
        if overlap:
            raise ValueError(
                f'categories cannot be both ranked and excluded: {", ".join(overlap)}'
            )

        expected_ranks = list(range(1, len(self.ranked_categories) + 1))
        if [item.rank for item in self.ranked_categories] != expected_ranks:
            raise ValueError('ranked category ranks must be contiguous and start at 1')

        preference_categories = {
            item.category_id: item
            for item in self.ranking_interpretation.preference_resolution.category_directives
        }
        guidance_categories = {
            item.category_id: item
            for item in self.ranking_interpretation.guidance_resolution.category_directives
        }
        known_category_ids = set(ranked_ids) | set(self.excluded_category_ids)
        directive_ids = set(preference_categories) | set(guidance_categories)
        ghost_categories = sorted(directive_ids - known_category_ids)
        if ghost_categories:
            raise ValueError(
                'ranking directives reference categories absent from the state: '
                f'{", ".join(ghost_categories)}'
            )
        effective_categories = {
            category_id: guidance_categories.get(
                category_id,
                preference_categories.get(category_id),
            )
            for category_id in directive_ids
        }
        expected_excluded = [
            item.category_id
            for item in sorted(
                (
                    directive
                    for directive in effective_categories.values()
                    if directive is not None and directive.tier == 'exclude'
                ),
                key=lambda item: (item.order, item.category_id),
            )
        ]
        if self.excluded_category_ids != expected_excluded:
            raise ValueError(
                'excluded_category_ids must match effective exclude directives'
            )

        metric_ids = set(self.ranking_interpretation.metric_weights)
        directive_metric_ids = {
            item.metric_id
            for item in (
                *self.ranking_interpretation.preference_resolution.metric_directives,
                *self.ranking_interpretation.guidance_resolution.metric_directives,
            )
        }
        ghost_metrics = sorted(directive_metric_ids - metric_ids)
        if ghost_metrics:
            raise ValueError(
                'ranking directives reference metrics absent from metric_weights: '
                f'{", ".join(ghost_metrics)}'
            )
        for item in self.ranked_categories:
            preference = preference_categories.get(item.category_id)
            guidance = guidance_categories.get(item.category_id)
            if (
                item.signals.preference_tier
                != (preference.tier if preference is not None else 'normal')
                or item.signals.preference_order
                != (preference.order if preference is not None else None)
                or item.preference_evidence
                != (preference.evidence if preference is not None else None)
            ):
                raise ValueError(
                    f'ranked category {item.category_id!r} does not match '
                    'its preference directive'
                )
            if (
                item.signals.guidance_tier
                != (guidance.tier if guidance is not None else 'normal')
                or item.signals.guidance_order
                != (guidance.order if guidance is not None else None)
                or item.guidance_evidence
                != (guidance.evidence if guidance is not None else None)
            ):
                raise ValueError(
                    f'ranked category {item.category_id!r} does not match '
                    'its guidance directive'
                )
            compared = set(item.signals.compared_metric_ids)
            missing = set(item.signals.missing_metric_ids)
            if set(item.signals.metric_gaps) != compared:
                raise ValueError('metric_gaps keys must match compared_metric_ids')
            if compared & missing or compared | missing != metric_ids:
                raise ValueError(
                    'compared_metric_ids and missing_metric_ids must partition '
                    'the weighted metrics'
                )

        _validate_evidence_indexes(
            self.ranking_interpretation.preference_resolution,
            source='preference',
            item_count=len(self.preference),
        )
        _validate_evidence_indexes(
            self.ranking_interpretation.guidance_resolution,
            source='user_guidance',
            item_count=len(self.user_guidance),
        )

        if self.status == 'blocked':
            if self.blocked_reason != 'blocked_no_categories':
                raise ValueError('blocked state is reserved for no-category analysis')
            if (
                self.provisional_selected_category is not None
                or self.selected_category is not None
                or self.ranked_categories
                or self.excluded_category_ids
            ):
                raise ValueError('blocked state cannot contain a category selection')
        elif self.status == 'awaiting_interrupt':
            if self.selected_category is not None:
                raise ValueError('preview state cannot contain a committed category')
            if self.ranked_categories:
                expected = self.ranked_categories[0].category_id
                if self.provisional_selected_category != expected:
                    raise ValueError('provisional category must match rank 1')
                if self.blocked_reason:
                    raise ValueError('ranked preview cannot have blocked_reason')
            else:
                if self.provisional_selected_category is not None:
                    raise ValueError('empty preview cannot have a provisional category')
                if self.blocked_reason != 'blocked_all_categories_excluded':
                    raise ValueError(
                        'empty preview must explain that all categories are excluded'
                    )
                if not self.excluded_category_ids:
                    raise ValueError(
                        'all-categories-excluded preview requires excluded categories'
                    )
        else:
            if self.blocked_reason:
                raise ValueError('committed state cannot have blocked_reason')
            if not self.ranked_categories:
                raise ValueError('committed state requires ranked categories')
            expected = self.ranked_categories[0].category_id
            if (
                self.provisional_selected_category != expected
                or self.selected_category != expected
            ):
                raise ValueError('committed category must match rank 1')
        return self


class TargetGuidanceInterrupt(StrictModel):
    id: Literal['repair.target_guidance_interrupt'] = (
        'repair.target_guidance_interrupt'
    )
    schema_version: Literal['2'] = '2'
    event_id: Identifier
    ranking_id: Identifier
    source_hash: StrictStr = Field(min_length=1, max_length=256)
    base_revision: StrictInt = Field(ge=1)
    reset_guidance: StrictBool = False
    user_guidance: list[StrictStr] = Field(default_factory=list, max_length=64)

    @field_validator('source_hash', mode='before')
    @classmethod
    def normalize_source_hash(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            raise ValueError('source_hash must contain non-whitespace text')
        return text

    @field_validator('user_guidance')
    @classmethod
    def normalize_guidance(cls, value: list[str]) -> list[str]:
        return _normalize_text_items(value, label='interrupt guidance')

    @model_validator(mode='after')
    def validate_interrupt_action(self) -> Self:
        if not self.reset_guidance and not self.user_guidance:
            raise ValueError(
                'guidance interrupt requires user_guidance unless '
                'reset_guidance is true'
            )
        return self


class TargetCommitRequest(StrictModel):
    id: Literal['repair.target_commit'] = 'repair.target_commit'
    schema_version: Literal['2'] = '2'
    event_id: Identifier
    ranking_id: Identifier
    source_hash: StrictStr = Field(min_length=1, max_length=256)
    base_revision: StrictInt = Field(ge=1)
    expected_category_id: Identifier

    @field_validator('source_hash', mode='before')
    @classmethod
    def normalize_source_hash(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            raise ValueError('source_hash must contain non-whitespace text')
        return text


def _normalize_text_items(value: list[str], *, label: str) -> list[str]:
    result: list[str] = []
    for raw in value:
        text = raw.strip()
        if not text:
            raise ValueError(f'{label} items must contain non-whitespace text')
        if len(text) > 2000:
            raise ValueError(f'{label} items must not exceed 2000 characters')
        result.append(text)
    return result


def _validate_canonical_resolution(
    resolution: PreferenceResolution,
    *,
    label: str,
    expected_source: Literal['preference', 'user_guidance'],
    allowed_tiers: set[str],
) -> None:
    tier_order = {
        'must': 0,
        'prefer': 1,
        'defer': 2,
        'exclude': 3,
    }
    actual_tiers = [
        tier_order[item.tier]
        for item in resolution.category_directives
    ]
    if actual_tiers != sorted(actual_tiers):
        raise ValueError(f'{label} category directives must use canonical tier order')
    for directive in resolution.category_directives:
        if directive.evidence.source != expected_source:
            raise ValueError(
                f'{label} category directives require {expected_source} evidence'
            )
        if directive.tier not in allowed_tiers:
            raise ValueError(
                f'{label} category directive tier {directive.tier!r} is not allowed'
            )
    for tier in ('must', 'prefer', 'defer', 'exclude'):
        orders = [
            item.order
            for item in resolution.category_directives
            if item.tier == tier
        ]
        if orders and orders != list(range(len(orders))):
            raise ValueError(
                f'{label} {tier} category directive order must start at 0 '
                'and be contiguous'
            )
    for directive in resolution.metric_directives:
        if directive.evidence.source != expected_source:
            raise ValueError(
                f'{label} metric directives require {expected_source} evidence'
            )
    metric_orders = [item.order for item in resolution.metric_directives]
    if metric_orders and metric_orders != list(range(len(metric_orders))):
        raise ValueError(
            f'{label} metric directive order must start at 0 and be contiguous'
        )


def _validate_evidence_indexes(
    resolution: PreferenceResolution,
    *,
    source: Literal['preference', 'user_guidance'],
    item_count: int,
) -> None:
    directives = [
        *resolution.category_directives,
        *resolution.metric_directives,
    ]
    for directive in directives:
        if directive.evidence.source != source:
            raise ValueError(f'{source} resolution contains foreign evidence')
        if directive.evidence.index >= item_count:
            raise ValueError(
                f'{source} evidence index {directive.evidence.index} is out of range'
            )


def _require_clean_mapping_keys(value: dict[str, Any], label: str) -> None:
    for key in value:
        if not key or key != key.strip():
            raise ValueError(f'{label} must be non-empty and trimmed')


def validate_preference_resolution(
    resolution: PreferenceResolution,
    request: PreferenceCompileRequest,
) -> PreferenceResolution:
    category_ids = {item.category_id for item in request.category_options}
    metric_ids = set(request.metric_ids)
    evidence_refs = {
        (item.evidence.source, item.evidence.index)
        for item in request.texts
    }
    for directive in resolution.category_directives:
        if directive.category_id not in category_ids:
            raise ValueError(
                'preference compiler returned unknown category: '
                f'{directive.category_id}'
            )
        _require_evidence(directive.evidence, evidence_refs)
        if (
            request.mode == 'initial_preference'
            and directive.tier not in {'prefer', 'defer'}
        ):
            raise ValueError(
                'initial preference may only produce prefer or defer category tiers'
            )
    for directive in resolution.metric_directives:
        if directive.metric_id not in metric_ids:
            raise ValueError(
                'preference compiler returned unknown metric: '
                f'{directive.metric_id}'
            )
        _require_evidence(directive.evidence, evidence_refs)
    return resolution


def _require_evidence(
    evidence: PreferenceEvidenceRef,
    allowed: set[tuple[str, int]],
) -> None:
    if (evidence.source, evidence.index) not in allowed:
        raise ValueError(
            'preference compiler returned unavailable evidence: '
            f'{evidence.source}[{evidence.index}]'
        )


def _require_unique(values: Any, label: str) -> None:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    if duplicate:
        raise ValueError(f'duplicate {label}: {", ".join(sorted(duplicate))}')


__all__ = [
    'AnalysisCategory',
    'AnalysisSummaryInput',
    'CategoryDirective',
    'CompiledPreference',
    'CompiledRankingIntent',
    'EventFingerprint',
    'Identifier',
    'InterruptibleRankedCategory',
    'InterruptibleRankingSignals',
    'MetricDirective',
    'PreferenceCategoryOption',
    'PreferenceCompileMode',
    'PreferenceCompileRequest',
    'PreferenceEvidenceRef',
    'PreferenceResolution',
    'PreferenceText',
    'RankedCategory',
    'RepairTargetContext',
    'RepairTargetPreparation',
    'TargetCommitRequest',
    'TargetGuidanceInterrupt',
    'TargetRankingState',
    'TargetWarning',
    'validate_preference_resolution',
]
