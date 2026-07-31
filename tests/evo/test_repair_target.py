from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from evo.operations.repair.target import build_target_preparation
from evo.operations.repair.target_contracts import (
    PreferenceCompileRequest,
    RepairTargetPreparation,
)


def _analysis() -> dict[str, Any]:
    return {
        'source_hash': 'analysis-sha256',
        'all_case_metric_averages': {
            'answer_correctness': 0.8,
            'faithfulness': 0.75,
            'retrieval_recall': 0.9,
        },
        'categories': {
            'category_01': {
                'metric_averages': {
                    'answer_correctness': 0.3,
                    'faithfulness': 0.45,
                    'retrieval_recall': 0.75,
                },
                'all_case_average_drop': 0.32,
                'code_span': [
                    {'file': 'algorithm/retrieve.py', 'start_line': 10, 'end_line': 20},
                ],
                'summary': 'Relevant documents are not recalled.',
                'analysis': 'The retrieval stage filters potentially useful documents too early.',
                'cases': {
                    'case_01': 'trace://case_01',
                    'case_02': 'trace://case_02',
                },
            },
            'category_02': {
                'metric_averages': {
                    'answer_correctness': 0.7,
                    'faithfulness': 0.65,
                    'retrieval_recall': 0.68,
                },
                'all_case_average_drop': 0.27,
                'code_span': [
                    {'file': 'algorithm/generate.py', 'symbol': 'answer'},
                ],
                'summary': 'The generated answer misses a qualification.',
                'analysis': 'The generation instruction does not preserve a source qualification.',
                'cases': {
                    'case_03': 'trace://case_03',
                    'case_04': 'trace://case_04',
                    'case_05': 'trace://case_05',
                },
            },
        },
    }


def _category(
    *,
    metrics: dict[str, float],
    drop: float = 0.2,
    case_count: int = 1,
) -> dict[str, Any]:
    return {
        'metric_averages': metrics,
        'all_case_average_drop': drop,
        'code_span': [],
        'summary': 'Category summary.',
        'analysis': 'Category analysis.',
        'cases': {
            f'case_{index}': f'trace://case_{index}'
            for index in range(case_count)
        },
    }


def _ranked_ids(result: dict[str, Any]) -> list[str]:
    return [item['category_id'] for item in result['ranked_categories']]


def test_numeric_ranking_selects_largest_drop_then_metric_gap() -> None:
    result = build_target_preparation(_analysis(), {})

    assert result['status'] == 'ready'
    assert result['selected_category'] == 'category_01'
    assert _ranked_ids(result) == ['category_01', 'category_02']
    assert result['ranked_categories'][0]['signals']['weighted_metric_gap'] == 0.3167
    assert result['ranked_categories'][1]['signals']['weighted_metric_gap'] == 0.14
    assert result['preference_interpretation']['source'] == 'none'
    assert result['warnings'] == []


def test_explicit_category_preference_has_priority_without_calling_compiler() -> None:
    calls = 0

    def compiler(_: object) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise AssertionError('compiler must not be called for exact identifiers')

    result = build_target_preparation(
        _analysis(),
        {'preference': ['category_02']},
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_02'
    assert calls == 0
    assert result['preference_interpretation']['source'] == 'deterministic'
    assert result['ranked_categories'][0]['signals']['user_tier'] == 'prefer'
    assert result['ranked_categories'][0]['preference_evidence'] == {
        'source': 'preference',
        'index': 0,
    }


def test_semantic_category_preference_is_validated_and_applied() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        assert request.metric_ids == [
            'answer_correctness',
            'faithfulness',
            'retrieval_recall',
        ]
        return {
            'category_directives': [
                {
                    'category_id': 'category_02',
                    'tier': 'must',
                    'order': 0,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                },
                {
                    'category_id': 'category_01',
                    'tier': 'defer',
                    'order': 0,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                },
            ],
            'metric_directives': [],
        }

    result = build_target_preparation(
        _analysis(),
        {'user_guidance': ['先处理回答中限定条件丢失的问题']},
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_02'
    assert _ranked_ids(result) == ['category_02', 'category_01']
    assert result['preference_interpretation']['source'] == 'resolver'
    assert result['warnings'] == []


def test_text_and_exact_preferences_keep_their_original_array_order() -> None:
    def compiler(_: object) -> dict[str, Any]:
        return {
            'category_directives': [
                {
                    'category_id': 'category_02',
                    'tier': 'prefer',
                    'order': 0,
                    'evidence': {'source': 'preference', 'index': 0},
                },
            ],
            'metric_directives': [],
        }

    result = build_target_preparation(
        _analysis(),
        {'preference': ['先修复生成问题', 'category_01']},
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_02'
    assert [
        item['category_id']
        for item in result['preference_interpretation']['category_directives']
    ] == ['category_02', 'category_01']


def test_excluded_category_is_not_ranked() -> None:
    def compiler(_: object) -> dict[str, Any]:
        return {
            'category_directives': [
                {
                    'category_id': 'category_01',
                    'tier': 'exclude',
                    'order': 0,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                },
            ],
            'metric_directives': [],
        }

    result = build_target_preparation(
        _analysis(),
        {'user_guidance': ['不要修改检索相关问题']},
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_02'
    assert result['excluded_category_ids'] == ['category_01']
    assert _ranked_ids(result) == ['category_02']


def test_all_categories_excluded_returns_explicit_blocked_result() -> None:
    def compiler(_: object) -> dict[str, Any]:
        return {
            'category_directives': [
                {
                    'category_id': category_id,
                    'tier': 'exclude',
                    'order': order,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                }
                for order, category_id in enumerate(('category_02', 'category_01'))
            ],
            'metric_directives': [],
        }

    result = build_target_preparation(
        _analysis(),
        {'user_guidance': ['这两类问题都不要修复']},
        preference_compiler=compiler,
    )

    assert result['status'] == 'blocked'
    assert result['blocked_reason'] == 'blocked_all_categories_excluded'
    assert result['selected_category'] is None
    assert result['ranked_categories'] == []
    assert result['excluded_category_ids'] == ['category_02', 'category_01']


def test_metric_preference_changes_weighted_gap_order_when_primary_drop_ties() -> None:
    analysis = {
        'source_hash': 'metric-focus',
        'all_case_metric_averages': {'precision': 0.9, 'recall': 0.9},
        'categories': {
            'precision_problem': _category(metrics={'precision': 0.1, 'recall': 0.9}),
            'recall_problem': _category(metrics={'precision': 0.9, 'recall': 0.1}),
        },
    }

    baseline = build_target_preparation(analysis, {})
    recall_first = build_target_preparation(analysis, {'preference': ['recall']})

    assert baseline['selected_category'] == 'precision_problem'
    assert recall_first['selected_category'] == 'recall_problem'
    assert recall_first['preference_interpretation']['metric_weights'] == {
        'precision': 1.0,
        'recall': 4.0,
    }


@pytest.mark.parametrize(
    ('categories', 'expected'),
    [
        (
            {
                'category_a': _category(metrics={'metric': 0.4}, drop=0.2),
                'category_b': _category(metrics={'metric': 0.4}, drop=0.3),
            },
            'category_b',
        ),
        (
            {
                'category_a': _category(metrics={'metric': 0.5}, drop=0.2),
                'category_b': _category(metrics={'metric': 0.4}, drop=0.2),
            },
            'category_b',
        ),
        (
            {
                'category_a': _category(metrics={'metric': 0.4}, case_count=1),
                'category_b': _category(metrics={'metric': 0.4}, case_count=2),
            },
            'category_b',
        ),
        (
            {
                'category_b': _category(metrics={'metric': 0.4}),
                'category_a': _category(metrics={'metric': 0.4}),
            },
            'category_a',
        ),
    ],
    ids=['drop', 'metric-gap', 'case-count', 'category-id'],
)
def test_numeric_tie_breakers_are_stable(
    categories: dict[str, dict[str, Any]],
    expected: str,
) -> None:
    result = build_target_preparation({
        'source_hash': 'tie-breaker',
        'all_case_metric_averages': {'metric': 0.8},
        'categories': categories,
    }, {})

    assert result['selected_category'] == expected


def test_sorting_uses_unrounded_scores_and_only_rounds_output() -> None:
    result = build_target_preparation({
        'source_hash': 'precision',
        'all_case_metric_averages': {'metric': 0.8},
        'categories': {
            'category_a': _category(metrics={'metric': 0.4}, drop=0.50003),
            'category_z': _category(metrics={'metric': 0.4}, drop=0.50004),
        },
    }, {})

    assert result['selected_category'] == 'category_z'
    assert [
        item['signals']['all_case_average_drop']
        for item in result['ranked_categories']
    ] == [0.5, 0.5]


def test_mapping_insertion_order_does_not_change_output() -> None:
    original = _analysis()
    reordered = deepcopy(original)
    reordered['all_case_metric_averages'] = dict(
        reversed(list(reordered['all_case_metric_averages'].items()))
    )
    reordered['categories'] = dict(reversed(list(reordered['categories'].items())))
    for category in reordered['categories'].values():
        category['metric_averages'] = dict(reversed(list(category['metric_averages'].items())))
        category['cases'] = dict(reversed(list(category['cases'].items())))

    assert build_target_preparation(original, {}) == build_target_preparation(reordered, {})


def test_code_span_content_does_not_affect_selection_or_output() -> None:
    changed = _analysis()
    changed['categories']['category_01']['code_span'] = [
        {
            'arbitrary_analysis_owned_shape': {
                'path': 'completely/different.py',
                'lines': [1, 999],
            },
        },
    ]

    assert build_target_preparation(_analysis(), {}) == build_target_preparation(changed, {})
    assert 'code_span' not in str(build_target_preparation(changed, {}))


def test_compiler_failure_uses_numeric_fallback_with_stable_warnings() -> None:
    def compiler(_: object) -> dict[str, Any]:
        raise RuntimeError('provider-specific and unstable error')

    result = build_target_preparation(
        _analysis(),
        {'user_guidance': ['优先修复用户最关心的类别']},
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_01'
    assert result['preference_interpretation']['source'] == 'none'
    assert [item['code'] for item in result['warnings']] == [
        'preference_compiler_failed',
        'preference_fallback_used',
    ]
    assert 'provider-specific' not in str(result)


def test_compiler_failure_keeps_valid_exact_preferences() -> None:
    def compiler(_: object) -> dict[str, Any]:
        raise RuntimeError('failed')

    result = build_target_preparation(
        _analysis(),
        {'preference': ['category_02', '以及其他自然语言偏好']},
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_02'
    assert result['preference_interpretation']['source'] == 'deterministic'
    assert [item['category_id'] for item in result[
        'preference_interpretation'
    ]['category_directives']] == ['category_02']


def test_unresolved_text_without_compiler_reports_unavailable_fallback() -> None:
    result = build_target_preparation(
        _analysis(),
        {'user_guidance': ['优先修复召回问题']},
    )

    assert result['selected_category'] == 'category_01'
    assert [item['code'] for item in result['warnings']] == [
        'preference_compiler_unavailable',
        'preference_fallback_used',
    ]


@pytest.mark.parametrize(
    'resolution',
    [
        {
            'category_directives': [
                {
                    'category_id': 'invented_category',
                    'tier': 'must',
                    'order': 0,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                },
            ],
            'metric_directives': [],
        },
        {
            'category_directives': [],
            'metric_directives': [
                {
                    'metric_id': 'invented_metric',
                    'order': 0,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                },
            ],
        },
        {
            'category_directives': [
                {
                    'category_id': 'category_02',
                    'tier': 'must',
                    'order': 0,
                    'evidence': {'source': 'preference', 'index': 99},
                },
            ],
            'metric_directives': [],
        },
        {
            'category_directives': [],
            'metric_directives': [],
            'reason': 'free-form fields are forbidden',
        },
    ],
    ids=['unknown-category', 'unknown-metric', 'unknown-evidence', 'extra-field'],
)
def test_invalid_compiler_output_cannot_pollute_ranking(
    resolution: dict[str, Any],
) -> None:
    result = build_target_preparation(
        _analysis(),
        {'user_guidance': ['自然语言偏好']},
        preference_compiler=lambda _: resolution,
    )

    assert result['selected_category'] == 'category_01'
    assert result['preference_interpretation']['category_directives'] == []
    assert result['preference_interpretation']['metric_directives'] == []
    assert len(result['warnings']) == 2


def test_explicit_identifier_wins_over_conflicting_semantic_directive() -> None:
    def compiler(_: object) -> dict[str, Any]:
        return {
            'category_directives': [
                {
                    'category_id': 'category_02',
                    'tier': 'exclude',
                    'order': 0,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                },
            ],
            'metric_directives': [],
        }

    result = build_target_preparation(
        _analysis(),
        {
            'user_guidance': ['不要处理生成问题'],
            'preference': ['category_02'],
        },
        preference_compiler=compiler,
    )

    assert result['selected_category'] == 'category_02'
    assert result['excluded_category_ids'] == []
    assert result['preference_interpretation']['source'] == 'mixed'
    assert [item['code'] for item in result['warnings']] == [
        'preference_conflict_explicit_wins',
    ]


def test_metric_conflict_warning_identifies_the_metric() -> None:
    def compiler(_: object) -> dict[str, Any]:
        return {
            'category_directives': [],
            'metric_directives': [
                {
                    'metric_id': 'retrieval_recall',
                    'order': 0,
                    'evidence': {'source': 'preference', 'index': 1},
                },
            ],
        }

    result = build_target_preparation(
        _analysis(),
        {'preference': ['retrieval_recall', '尤其关注召回率']},
        preference_compiler=compiler,
    )

    assert result['warnings'] == [{
        'code': 'preference_conflict_explicit_wins',
        'message': 'Explicit preference entry overrides semantic guidance for this metric.',
        'category_id': None,
        'metric_id': 'retrieval_recall',
    }]


def test_metric_conflict_warning_order_does_not_depend_on_resolver_list_order() -> None:
    def compiler(reverse: bool):
        directives = [
            {
                'metric_id': metric_id,
                'order': order,
                'evidence': {'source': 'preference', 'index': 2},
            }
            for order, metric_id in enumerate(('faithfulness', 'answer_correctness'))
        ]
        if reverse:
            directives.reverse()
        return lambda _: {
            'category_directives': [],
            'metric_directives': directives,
        }

    context = {
        'preference': [
            'answer_correctness',
            'faithfulness',
            '同时关注这两个指标',
        ],
    }
    first = build_target_preparation(
        _analysis(),
        context,
        preference_compiler=compiler(False),
    )
    second = build_target_preparation(
        _analysis(),
        context,
        preference_compiler=compiler(True),
    )

    assert first == second
    assert [
        item['metric_id']
        for item in first['warnings']
    ] == ['answer_correctness', 'faithfulness']


def test_duplicate_explicit_identifiers_preserve_original_evidence_index() -> None:
    result = build_target_preparation(
        _analysis(),
        {'preference': ['category_02', 'category_02']},
    )

    directive = result['preference_interpretation']['category_directives'][0]
    assert directive['evidence'] == {'source': 'preference', 'index': 0}
    assert len(result['preference_interpretation']['category_directives']) == 1


def test_missing_category_metric_is_recorded_and_not_fabricated() -> None:
    analysis = _analysis()
    del analysis['categories']['category_01']['metric_averages']['faithfulness']

    result = build_target_preparation(analysis, {})
    signals = result['ranked_categories'][0]['signals']

    assert signals['missing_metric_ids'] == ['faithfulness']
    assert signals['compared_metric_ids'] == ['answer_correctness', 'retrieval_recall']
    assert 'faithfulness' not in signals['metric_gaps']


def test_category_metric_missing_from_global_metrics_is_rejected() -> None:
    analysis = _analysis()
    analysis['categories']['category_01']['metric_averages']['unknown'] = 0.4

    with pytest.raises(ValidationError, match='metrics missing'):
        build_target_preparation(analysis, {})


def test_category_and_metric_identifiers_cannot_be_ambiguous() -> None:
    analysis = {
        'source_hash': 'ambiguous',
        'all_case_metric_averages': {'shared_id': 0.8},
        'categories': {
            'shared_id': _category(metrics={'shared_id': 0.4}),
        },
    }

    with pytest.raises(ValidationError, match='must not overlap'):
        build_target_preparation(analysis, {})


def test_identifier_and_collection_limits_fail_at_input_boundary() -> None:
    too_many_metrics = {
        f'metric_{index:03d}': 0.8
        for index in range(129)
    }
    analysis = {
        'source_hash': 'too-many-metrics',
        'all_case_metric_averages': too_many_metrics,
        'categories': {},
    }

    with pytest.raises(ValidationError):
        build_target_preparation(analysis, {})

    analysis = _analysis()
    category = analysis['categories'].pop('category_01')
    analysis['categories']['c' * 257] = category
    with pytest.raises(ValidationError):
        build_target_preparation(analysis, {})

    analysis = {
        'source_hash': 'too-many-categories',
        'all_case_metric_averages': {'metric': 0.8},
        'categories': {
            f'category_{index:03d}': _category(metrics={'metric': 0.4})
            for index in range(201)
        },
    }
    with pytest.raises(ValidationError):
        build_target_preparation(analysis, {})


@pytest.mark.parametrize('invalid_score', [float('nan'), float('inf'), -0.1, 1.1, True, '0.4'])
def test_invalid_metric_scores_are_rejected(invalid_score: object) -> None:
    analysis = _analysis()
    analysis['all_case_metric_averages']['answer_correctness'] = invalid_score

    with pytest.raises(ValidationError):
        build_target_preparation(analysis, {})


def test_empty_categories_return_explicit_blocked_result() -> None:
    analysis = _analysis()
    analysis['categories'] = {}

    def compiler(_: object) -> dict[str, Any]:
        raise AssertionError('compiler must not run without categories')

    result = build_target_preparation(
        analysis,
        {'user_guidance': ['anything']},
        preference_compiler=compiler,
    )

    assert result['status'] == 'blocked'
    assert result['blocked_reason'] == 'blocked_no_categories'
    assert result['selected_category'] is None
    assert result['ranked_categories'] == []
    assert result['preference_interpretation']['source'] == 'none'
    assert result['preference_interpretation']['metric_weights'] == {
        'answer_correctness': 1.0,
        'faithfulness': 1.0,
        'retrieval_recall': 1.0,
    }
    assert result['warnings'] == []


def test_empty_category_metrics_are_allowed_and_report_all_metrics_missing() -> None:
    analysis = {
        'source_hash': 'missing-all-category-metrics',
        'all_case_metric_averages': {'metric': 0.8},
        'categories': {
            'category': _category(metrics={}),
        },
    }

    result = build_target_preparation(analysis, {})
    signals = result['ranked_categories'][0]['signals']

    assert signals['metric_gaps'] == {}
    assert signals['weighted_metric_gap'] == 0.0
    assert signals['compared_metric_ids'] == []
    assert signals['missing_metric_ids'] == ['metric']
    assert 'analysis_metric_gap' not in result['ranked_categories'][0]['reason_codes']


def test_metric_gap_is_clamped_at_zero() -> None:
    analysis = {
        'source_hash': 'negative-gap',
        'all_case_metric_averages': {'metric': 0.4},
        'categories': {
            'category': _category(metrics={'metric': 0.8}),
        },
    }

    result = build_target_preparation(analysis, {})

    assert result['ranked_categories'][0]['signals']['metric_gaps'] == {'metric': 0.0}
    assert result['ranked_categories'][0]['signals']['weighted_metric_gap'] == 0.0


def test_non_mapping_context_has_clear_boundary_error() -> None:
    with pytest.raises(TypeError, match='context must be a mapping'):
        build_target_preparation(_analysis(), [])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    'context',
    [
        {'user_guidance': None},
        {'preference': None},
        {'user_guidance': ['guidance'] * 65},
        {'preference': ['preference'] * 65},
    ],
)
def test_context_null_and_oversized_lists_are_rejected(
    context: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        build_target_preparation(_analysis(), context)


def test_analysis_text_is_trimmed_before_length_validation() -> None:
    analysis = _analysis()
    analysis['source_hash'] = '  normalized-hash  '
    analysis['categories']['category_01']['summary'] = '  ' + ('s' * 1000) + '  '
    captured: list[PreferenceCompileRequest] = []

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        captured.append(request)
        return {'category_directives': [], 'metric_directives': []}

    result = build_target_preparation(
        analysis,
        {'user_guidance': ['neutral guidance']},
        preference_compiler=compiler,
    )

    assert result['source_hash'] == 'normalized-hash'
    category = next(
        item
        for item in captured[0].category_options
        if item.category_id == 'category_01'
    )
    assert category.summary == ('s' * 499) + '…'


def test_unknown_analysis_fields_and_invalid_code_span_shape_are_rejected() -> None:
    analysis = _analysis()
    analysis['unexpected'] = True
    with pytest.raises(ValidationError):
        build_target_preparation(analysis, {})

    analysis = _analysis()
    analysis['categories']['category_01']['code_span'] = {'not': 'a list'}
    with pytest.raises(ValidationError):
        build_target_preparation(analysis, {})


def test_output_contract_rejects_inconsistent_blocked_state() -> None:
    valid = build_target_preparation(_analysis(), {})
    invalid = deepcopy(valid)
    invalid.update({
        'status': 'blocked',
        'blocked_reason': 'blocked_no_categories',
        'selected_category': None,
    })

    with pytest.raises(ValidationError, match='cannot have ranked_categories'):
        RepairTargetPreparation.model_validate(invalid)


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (
            lambda result: result['preference_interpretation'][
                'category_directives'
            ].append({
                'category_id': 'ghost',
                'tier': 'prefer',
                'order': 0,
                'evidence': {'source': 'preference', 'index': 0},
            }),
            'absent from the result',
        ),
        (
            lambda result: result['preference_interpretation'][
                'metric_directives'
            ].append({
                'metric_id': 'ghost_metric',
                'order': 0,
                'evidence': {'source': 'preference', 'index': 0},
            }),
            'lack corresponding metric weights',
        ),
    ],
    ids=['ghost-category', 'ghost-metric'],
)
def test_output_contract_rejects_ghost_directives(
    mutate: Any,
    message: str,
) -> None:
    invalid = build_target_preparation(_analysis(), {})
    mutate(invalid)

    with pytest.raises(ValidationError, match=message):
        RepairTargetPreparation.model_validate(invalid)


def test_output_contract_rejects_ranked_preference_signal_mismatch() -> None:
    invalid = build_target_preparation(
        _analysis(),
        {'preference': ['category_02']},
    )
    invalid['ranked_categories'][0]['signals']['user_tier'] = 'normal'

    with pytest.raises(ValidationError, match='does not match'):
        RepairTargetPreparation.model_validate(invalid)


def test_output_contract_rejects_excluded_category_order_mismatch() -> None:
    def compiler(_: object) -> dict[str, Any]:
        return {
            'category_directives': [
                {
                    'category_id': category_id,
                    'tier': 'exclude',
                    'order': order,
                    'evidence': {'source': 'user_guidance', 'index': 0},
                }
                for order, category_id in enumerate(('category_02', 'category_01'))
            ],
            'metric_directives': [],
        }

    invalid = build_target_preparation(
        _analysis(),
        {'user_guidance': ['exclude both']},
        preference_compiler=compiler,
    )
    invalid['excluded_category_ids'].reverse()

    with pytest.raises(ValidationError, match='ordered exclude'):
        RepairTargetPreparation.model_validate(invalid)


def test_unrelated_repair_context_fields_are_outside_target_contract_view() -> None:
    result = build_target_preparation(
        _analysis(),
        {
            'user_guidance': [],
            'preference': [],
            'web_search': {'enabled': True},
            'code_search': {'enabled': True},
            'demo': {'anything': 'ignored by target preparation'},
        },
    )

    assert result['status'] == 'ready'
    assert result['selected_category'] == 'category_01'
