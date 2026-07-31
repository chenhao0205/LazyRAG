from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from evo.operations.repair.target import (
    apply_target_guidance_interrupt,
    build_target_ranking_preview,
    commit_target_ranking,
    target_ranking_to_legacy_preparation,
)
from evo.operations.repair.target_contracts import PreferenceCompileRequest


def _analysis() -> dict[str, Any]:
    return {
        'source_hash': 'analysis-sha256',
        'all_case_metric_averages': {
            'answer_correctness': 0.8,
            'retrieval_recall': 0.9,
        },
        'categories': {
            'category_01': {
                'metric_averages': {
                    'answer_correctness': 0.3,
                    'retrieval_recall': 0.5,
                },
                'all_case_average_drop': 0.4,
                'code_span': [
                    {
                        'file': 'algorithm/retrieve.py',
                        'start_line': 10,
                        'end_line': 20,
                    },
                ],
                'summary': 'Relevant documents are not recalled.',
                'analysis': 'Retrieval filters useful documents too early.',
                'cases': {
                    'case_01': 'trace://case_01',
                    'case_02': 'trace://case_02',
                },
            },
            'category_02': {
                'metric_averages': {
                    'answer_correctness': 0.7,
                    'retrieval_recall': 0.8,
                },
                'all_case_average_drop': 0.1,
                'code_span': [
                    {
                        'file': 'algorithm/generate.py',
                        'symbol': 'answer',
                    },
                ],
                'summary': 'The answer misses a qualification.',
                'analysis': 'Generation does not preserve source qualifications.',
                'cases': {
                    'case_03': 'trace://case_03',
                },
            },
        },
    }


def _warning_boundary_analysis() -> dict[str, Any]:
    metrics = {
        f'metric_{index:03d}': 0.8
        for index in range(128)
    }
    return {
        'source_hash': 'warning-boundary',
        'all_case_metric_averages': metrics,
        'categories': {
            f'category_{index:03d}': {
                'metric_averages': {},
                'all_case_average_drop': 0.5,
                'code_span': [],
                'summary': f'Category {index} summary.',
                'analysis': f'Category {index} analysis.',
                'cases': {
                    f'case_{index:03d}': f'trace://case_{index:03d}',
                },
            }
            for index in range(200)
        },
    }


def _ranked_ids(state: dict[str, Any]) -> list[str]:
    return [item['category_id'] for item in state['ranked_categories']]


def _interrupt(
    state: dict[str, Any],
    *,
    event_id: str,
    guidance: list[str],
    reset_guidance: bool = False,
    base_revision: int | None = None,
    ranking_id: str | None = None,
    source_hash: str | None = None,
) -> dict[str, Any]:
    return {
        'event_id': event_id,
        'ranking_id': ranking_id or state['ranking_id'],
        'source_hash': source_hash or state['source_hash'],
        'base_revision': (
            state['revision'] if base_revision is None else base_revision
        ),
        'reset_guidance': reset_guidance,
        'user_guidance': guidance,
    }


def _commit(
    state: dict[str, Any],
    *,
    event_id: str = 'commit-1',
    expected_category_id: str | None = None,
) -> dict[str, Any]:
    return {
        'event_id': event_id,
        'ranking_id': state['ranking_id'],
        'source_hash': state['source_hash'],
        'base_revision': state['revision'],
        'expected_category_id': (
            expected_category_id
            or state['provisional_selected_category']
        ),
    }


def _category_directive(
    request: PreferenceCompileRequest,
    *,
    category_id: str,
    tier: str,
) -> dict[str, Any]:
    return {
        'category_directives': [
            {
                'category_id': category_id,
                'tier': tier,
                'order': 0,
                'evidence': request.texts[0].evidence.model_dump(),
            },
        ],
        'metric_directives': [],
    }


def test_initial_preview_is_uncommitted_revision_one() -> None:
    state = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )

    assert state['revision'] == 1
    assert state['status'] == 'awaiting_interrupt'
    assert state['selected_category'] is None
    assert state['provisional_selected_category'] == 'category_01'
    assert _ranked_ids(state) == ['category_01', 'category_02']


def test_initial_preview_rejects_guidance_before_showing_the_ranking() -> None:
    with pytest.raises(
        ValueError,
        match='initial target preview cannot contain user_guidance',
    ):
        build_target_ranking_preview(
            _analysis(),
            {'user_guidance': ['优先处理生成模块']},
            ranking_id='ranking-1',
        )


def test_initial_preference_is_a_soft_ranking_signal() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        assert request.mode == 'initial_preference'
        assert [item.text for item in request.texts] == ['长期偏好生成质量']
        assert {
            item.evidence.source for item in request.texts
        } == {'preference'}
        return _category_directive(
            request,
            category_id='category_02',
            tier='prefer',
        )

    state = build_target_ranking_preview(
        _analysis(),
        {'preference': ['长期偏好生成质量']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )

    assert _ranked_ids(state) == ['category_02', 'category_01']
    assert state['ranked_categories'][0]['signals']['preference_tier'] == 'prefer'
    assert state['ranked_categories'][0]['signals']['guidance_tier'] == 'normal'


def test_interrupt_guidance_overrides_numeric_and_preference_ranking() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        if request.mode == 'initial_preference':
            return _category_directive(
                request,
                category_id='category_01',
                tier='prefer',
            )
        assert request.mode == 'interrupt_guidance'
        return _category_directive(
            request,
            category_id='category_02',
            tier='must',
        )

    initial = build_target_ranking_preview(
        _analysis(),
        {'preference': ['长期优先检索问题']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        initial,
        _interrupt(
            initial,
            event_id='interrupt-1',
            guidance=['本轮必须优先处理生成模块'],
        ),
        preference_compiler=compiler,
    )

    assert _ranked_ids(initial) == ['category_01', 'category_02']
    assert interrupted['revision'] == 2
    assert interrupted['status'] == 'awaiting_interrupt'
    assert interrupted['selected_category'] is None
    assert interrupted['provisional_selected_category'] == 'category_02'
    assert _ranked_ids(interrupted) == ['category_02', 'category_01']


def test_latest_interrupt_wins_when_guidance_conflicts() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        text = request.texts[0].text
        if text == '先处理生成模块':
            return _category_directive(
                request,
                category_id='category_02',
                tier='must',
            )
        assert text == '生成模块稍后处理'
        return _category_directive(
            request,
            category_id='category_02',
            tier='defer',
        )

    initial = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    first = apply_target_guidance_interrupt(
        _analysis(),
        initial,
        _interrupt(
            initial,
            event_id='interrupt-1',
            guidance=['先处理生成模块'],
        ),
        preference_compiler=compiler,
    )
    second = apply_target_guidance_interrupt(
        _analysis(),
        first,
        _interrupt(
            first,
            event_id='interrupt-2',
            guidance=['生成模块稍后处理'],
        ),
        preference_compiler=compiler,
    )

    assert first['provisional_selected_category'] == 'category_02'
    assert second['revision'] == 3
    assert second['status'] == 'awaiting_interrupt'
    assert second['provisional_selected_category'] == 'category_01'
    directive = second['ranking_interpretation']['guidance_resolution'][
        'category_directives'
    ][0]
    assert directive['category_id'] == 'category_02'
    assert directive['tier'] == 'defer'
    assert directive['evidence'] == {
        'source': 'user_guidance',
        'index': 1,
    }


def test_guidance_preserves_order_within_one_interrupt_batch() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-ordered-batch',
            guidance=[
                'category_02',
                'category_01',
                'retrieval_recall',
                'answer_correctness',
            ],
        ),
    )

    assert _ranked_ids(interrupted) == ['category_02', 'category_01']
    directives = interrupted['ranking_interpretation'][
        'guidance_resolution'
    ]
    assert [
        item['category_id']
        for item in directives['category_directives']
    ] == ['category_02', 'category_01']
    assert [
        item['metric_id']
        for item in directives['metric_directives']
    ] == ['retrieval_recall', 'answer_correctness']
    assert interrupted['ranking_interpretation']['metric_weights'] == {
        'answer_correctness': 3.0,
        'retrieval_recall': 4.0,
    }


def test_new_interrupt_batch_precedes_older_guidance_in_the_same_tier() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    first = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-old',
            guidance=['category_01'],
        ),
    )
    second = apply_target_guidance_interrupt(
        _analysis(),
        first,
        _interrupt(
            first,
            event_id='interrupt-new',
            guidance=['category_02'],
        ),
    )

    assert _ranked_ids(second) == ['category_02', 'category_01']
    directives = second['ranking_interpretation']['guidance_resolution'][
        'category_directives'
    ]
    assert [
        (item['category_id'], item['order'], item['evidence']['index'])
        for item in directives
    ] == [
        ('category_02', 0, 1),
        ('category_01', 1, 0),
    ]


def test_commit_freezes_visible_ranking_without_recompiling() -> None:
    compiler_calls = 0

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        nonlocal compiler_calls
        compiler_calls += 1
        return _category_directive(
            request,
            category_id='category_02',
            tier='prefer',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {'preference': ['长期偏好生成质量']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )
    committed = commit_target_ranking(
        _analysis(),
        preview,
        _commit(preview),
    )

    assert compiler_calls == 1
    assert committed['status'] == 'committed'
    assert committed['revision'] == preview['revision']
    assert committed['ranked_categories'] == preview['ranked_categories']
    assert (
        committed['selected_category']
        == preview['provisional_selected_category']
    )


def test_legacy_adapter_requires_commit_and_preserves_committed_target() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )

    with pytest.raises(ValueError, match='ranking_not_committed'):
        target_ranking_to_legacy_preparation(preview)

    committed = commit_target_ranking(
        _analysis(),
        preview,
        _commit(preview),
    )
    legacy = target_ranking_to_legacy_preparation(committed)

    assert legacy['status'] == 'ready'
    assert legacy['selected_category'] == 'category_01'
    assert _ranked_ids(legacy) == ['category_01', 'category_02']


def test_legacy_adapter_uses_reindexed_effective_directive_orders() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        category_id = (
            'category_01'
            if request.mode == 'initial_preference'
            else 'category_02'
        )
        return _category_directive(
            request,
            category_id=category_id,
            tier='prefer',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {'preference': ['长期优先检索问题']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-1',
            guidance=['本轮优先处理生成模块'],
        ),
        preference_compiler=compiler,
    )
    committed = commit_target_ranking(
        _analysis(),
        interrupted,
        _commit(interrupted),
    )

    legacy = target_ranking_to_legacy_preparation(committed)

    assert legacy['preference_interpretation']['source'] == 'resolver'
    assert [
        item['category_id']
        for item in legacy['preference_interpretation'][
            'category_directives'
        ]
    ] == ['category_02', 'category_01']
    directives = {
        item['category_id']: item
        for item in legacy['preference_interpretation'][
            'category_directives'
        ]
    }
    for ranked in legacy['ranked_categories']:
        directive = directives[ranked['category_id']]
        assert ranked['signals']['user_order'] == directive['order']
        assert ranked['preference_evidence'] == directive['evidence']


def test_legacy_adapter_preserves_exact_source_and_effective_metric_order() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {
            'preference': [
                'category_01',
                'answer_correctness',
            ],
        },
        ranking_id='ranking-1',
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-exact',
            guidance=[
                'category_02',
                'retrieval_recall',
            ],
        ),
    )
    committed = commit_target_ranking(
        _analysis(),
        interrupted,
        _commit(interrupted),
    )

    legacy = target_ranking_to_legacy_preparation(committed)
    interpretation = legacy['preference_interpretation']

    assert interpretation['source'] == 'deterministic'
    assert [
        (item['category_id'], item['order'])
        for item in interpretation['category_directives']
    ] == [
        ('category_02', 0),
        ('category_01', 1),
    ]
    assert [
        (item['metric_id'], item['order'])
        for item in interpretation['metric_directives']
    ] == [
        ('retrieval_recall', 0),
        ('answer_correctness', 1),
    ]
    assert interpretation['metric_weights'] == {
        'answer_correctness': 3.0,
        'retrieval_recall': 4.0,
    }


def test_legacy_adapter_reports_mixed_exact_and_semantic_provenance() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        return _category_directive(
            request,
            category_id='category_02',
            tier='prefer',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {'preference': ['category_01']},
        ranking_id='ranking-1',
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-semantic',
            guidance=['本轮优先处理生成模块'],
        ),
        preference_compiler=compiler,
    )
    committed = commit_target_ranking(
        _analysis(),
        interrupted,
        _commit(interrupted),
    )

    legacy = target_ranking_to_legacy_preparation(committed)

    assert legacy['preference_interpretation']['source'] == 'mixed'


def test_legacy_adapter_preserves_latest_exclude_order() -> None:
    analysis = _analysis()
    analysis['categories']['category_03'] = {
        'metric_averages': {
            'answer_correctness': 0.75,
            'retrieval_recall': 0.85,
        },
        'all_case_average_drop': 0.05,
        'code_span': [],
        'summary': 'A third repair category.',
        'analysis': 'This category remains available after exclusions.',
        'cases': {
            'case_04': 'trace://case_04',
        },
    }

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        category_id = (
            'category_02'
            if request.texts[0].text == '排除第二类'
            else 'category_01'
        )
        return _category_directive(
            request,
            category_id=category_id,
            tier='exclude',
        )

    preview = build_target_ranking_preview(
        analysis,
        {},
        ranking_id='ranking-1',
    )
    first = apply_target_guidance_interrupt(
        analysis,
        preview,
        _interrupt(
            preview,
            event_id='interrupt-exclude-second',
            guidance=['排除第二类'],
        ),
        preference_compiler=compiler,
    )
    second = apply_target_guidance_interrupt(
        analysis,
        first,
        _interrupt(
            first,
            event_id='interrupt-exclude-first',
            guidance=['排除第一类'],
        ),
        preference_compiler=compiler,
    )
    committed = commit_target_ranking(
        analysis,
        second,
        _commit(second),
    )

    legacy = target_ranking_to_legacy_preparation(committed)

    assert legacy['selected_category'] == 'category_03'
    assert legacy['excluded_category_ids'] == [
        'category_01',
        'category_02',
    ]
    assert [
        item['category_id']
        for item in legacy['preference_interpretation'][
            'category_directives'
        ]
    ] == [
        'category_01',
        'category_02',
    ]


def test_warning_boundary_is_capped_and_reset_restores_preference_warnings() -> None:
    analysis = _warning_boundary_analysis()

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        evidence = request.texts[-1].evidence.model_dump()
        return {
            'category_directives': [
                {
                    'category_id': option.category_id,
                    'tier': 'prefer',
                    'order': order,
                    'evidence': evidence,
                }
                for order, option in enumerate(request.category_options)
            ],
            'metric_directives': [
                {
                    'metric_id': metric_id,
                    'order': order,
                    'evidence': evidence,
                }
                for order, metric_id in enumerate(request.metric_ids)
            ],
        }

    preview = build_target_ranking_preview(
        analysis,
        {
            'preference': [
                *[
                    f'category_{index:03d}'
                    for index in range(63)
                ],
                'semantic preference',
            ],
        },
        ranking_id='warning-boundary-ranking',
        preference_compiler=compiler,
    )
    interrupted = apply_target_guidance_interrupt(
        analysis,
        preview,
        _interrupt(
            preview,
            event_id='warning-boundary-interrupt',
            guidance=[
                *[
                    f'category_{index:03d}'
                    for index in range(63, 126)
                ],
                'semantic guidance',
            ],
        ),
        preference_compiler=compiler,
    )

    assert len(preview['warnings']) == 63
    assert len(interrupted['warnings']) == 400
    assert interrupted['warnings'][-1] == {
        'code': 'warning_limit_reached',
        'message': 'Additional target-ranking warnings were omitted.',
        'category_id': None,
        'metric_id': None,
    }

    reset = apply_target_guidance_interrupt(
        analysis,
        interrupted,
        _interrupt(
            interrupted,
            event_id='warning-boundary-reset',
            guidance=[],
            reset_guidance=True,
        ),
        preference_compiler=compiler,
    )

    assert len(reset['warnings']) == 63
    assert {
        item['code']
        for item in reset['warnings']
    } == {'preference_conflict_explicit_wins'}


def test_stale_interrupt_is_rejected_before_compiler_runs() -> None:
    compiler_calls = 0

    def compiler(_: PreferenceCompileRequest) -> dict[str, Any]:
        nonlocal compiler_calls
        compiler_calls += 1
        raise AssertionError('stale commands must not reach the compiler')

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )

    with pytest.raises(ValueError, match=r'stale_revision: expected 1, got 2'):
        apply_target_guidance_interrupt(
            _analysis(),
            preview,
            _interrupt(
                preview,
                event_id='interrupt-stale',
                guidance=['优先处理生成模块'],
                base_revision=2,
            ),
            preference_compiler=compiler,
        )

    assert compiler_calls == 0


def test_stale_commit_is_rejected_without_changing_the_preview() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    stale_commit = {
        **_commit(preview),
        'base_revision': preview['revision'] + 1,
    }

    with pytest.raises(ValueError, match=r'stale_revision: expected 1, got 2'):
        commit_target_ranking(
            _analysis(),
            preview,
            stale_commit,
        )

    assert preview['status'] == 'awaiting_interrupt'
    assert preview['selected_category'] is None


def test_changed_analysis_source_is_rejected_before_compiler_runs() -> None:
    compiler_calls = 0

    def compiler(_: PreferenceCompileRequest) -> dict[str, Any]:
        nonlocal compiler_calls
        compiler_calls += 1
        raise AssertionError('changed Analysis must not reach the compiler')

    analysis = _analysis()
    preview = build_target_ranking_preview(
        analysis,
        {},
        ranking_id='ranking-1',
    )
    changed = deepcopy(analysis)
    changed['source_hash'] = 'new-analysis-sha256'

    with pytest.raises(ValueError, match='source_changed'):
        apply_target_guidance_interrupt(
            changed,
            preview,
            _interrupt(
                preview,
                event_id='interrupt-old-source',
                guidance=['优先处理生成模块'],
            ),
            preference_compiler=compiler,
        )

    assert compiler_calls == 0


@pytest.mark.parametrize(
    ('overrides', 'error'),
    [
        ({'ranking_id': 'foreign-ranking'}, 'foreign_session'),
        ({'source_hash': 'foreign-source'}, 'source_changed'),
    ],
)
def test_interrupt_rejects_wrong_transition_identity(
    overrides: dict[str, str],
    error: str,
) -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )

    with pytest.raises(ValueError, match=error):
        apply_target_guidance_interrupt(
            _analysis(),
            preview,
            _interrupt(
                preview,
                event_id='interrupt-foreign',
                guidance=['优先处理生成模块'],
                **overrides,
            ),
        )


def test_interrupt_rejects_noncanonical_event_id() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )

    with pytest.raises(ValueError, match='event_id'):
        apply_target_guidance_interrupt(
            _analysis(),
            preview,
            _interrupt(
                preview,
                event_id=' interrupt-with-spaces ',
                guidance=['category_02'],
            ),
        )


def test_commit_checks_expected_category_against_visible_preview() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )

    with pytest.raises(ValueError, match='category_changed'):
        commit_target_ranking(
            _analysis(),
            preview,
            _commit(
                preview,
                expected_category_id='category_02',
            ),
        )


def test_interrupt_after_commit_is_rejected() -> None:
    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    committed = commit_target_ranking(
        _analysis(),
        preview,
        _commit(preview),
    )

    with pytest.raises(ValueError, match='ranking_closed'):
        apply_target_guidance_interrupt(
            _analysis(),
            committed,
            _interrupt(
                committed,
                event_id='interrupt-after-commit',
                guidance=['改为优先处理生成模块'],
            ),
        )


def test_repeated_interrupt_and_commit_events_are_idempotent() -> None:
    compiler_calls = 0

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        nonlocal compiler_calls
        compiler_calls += 1
        return _category_directive(
            request,
            category_id='category_02',
            tier='must',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    interrupt = _interrupt(
        preview,
        event_id='interrupt-once',
        guidance=['本轮优先处理生成模块'],
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        interrupt,
        preference_compiler=compiler,
    )
    repeated_interrupt = apply_target_guidance_interrupt(
        _analysis(),
        interrupted,
        interrupt,
        preference_compiler=compiler,
    )

    assert compiler_calls == 1
    assert repeated_interrupt == interrupted

    commit = _commit(interrupted, event_id='commit-once')
    committed = commit_target_ranking(_analysis(), interrupted, commit)
    repeated_commit = commit_target_ranking(_analysis(), committed, commit)

    assert repeated_commit == committed


def test_reused_event_id_with_changed_payload_is_rejected() -> None:
    compiler_calls = 0

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        nonlocal compiler_calls
        compiler_calls += 1
        return _category_directive(
            request,
            category_id='category_02',
            tier='must',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    applied = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='event-reused',
            guidance=['先处理生成模块'],
        ),
        preference_compiler=compiler,
    )

    with pytest.raises(ValueError, match='idempotency_conflict'):
        apply_target_guidance_interrupt(
            _analysis(),
            applied,
            _interrupt(
                applied,
                event_id='event-reused',
                guidance=['改为处理检索模块'],
            ),
            preference_compiler=compiler,
        )
    with pytest.raises(ValueError, match='idempotency_conflict'):
        commit_target_ranking(
            _analysis(),
            applied,
            _commit(applied, event_id='event-reused'),
        )

    assert compiler_calls == 1


def test_reset_guidance_restores_preference_baseline_without_recompiling() -> None:
    compiler_calls = 0

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        nonlocal compiler_calls
        compiler_calls += 1
        if request.mode == 'initial_preference':
            return _category_directive(
                request,
                category_id='category_02',
                tier='prefer',
            )
        return _category_directive(
            request,
            category_id='category_02',
            tier='exclude',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {'preference': ['长期优先生成模块']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )
    original_interrupt = _interrupt(
        preview,
        event_id='interrupt-exclude-preferred',
        guidance=['本轮不要处理生成模块'],
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        original_interrupt,
        preference_compiler=compiler,
    )
    reset = apply_target_guidance_interrupt(
        _analysis(),
        interrupted,
        _interrupt(
            interrupted,
            event_id='interrupt-reset',
            guidance=[],
            reset_guidance=True,
        ),
        preference_compiler=compiler,
    )

    assert compiler_calls == 2
    assert interrupted['provisional_selected_category'] == 'category_01'
    assert reset['revision'] == 3
    assert reset['provisional_selected_category'] == 'category_02'
    assert reset['user_guidance'] == []
    assert reset['ranking_interpretation']['guidance_resolution'] == {
        'category_directives': [],
        'metric_directives': [],
    }
    assert all(
        not warning['code'].startswith('guidance_')
        for warning in reset['warnings']
    )
    assert reset['processed_event_ids'] == [
        'interrupt-exclude-preferred',
        'interrupt-reset',
    ]
    assert apply_target_guidance_interrupt(
        _analysis(),
        reset,
        original_interrupt,
        preference_compiler=compiler,
    ) == reset
    assert compiler_calls == 2


def test_reset_with_new_guidance_restarts_evidence_indexes() -> None:
    seen_indexes: list[list[int]] = []

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        seen_indexes.append([
            item.evidence.index
            for item in request.texts
        ])
        return _category_directive(
            request,
            category_id='category_02',
            tier='must',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    first = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-before-reset',
            guidance=['先处理生成模块'],
        ),
        preference_compiler=compiler,
    )
    reset = apply_target_guidance_interrupt(
        _analysis(),
        first,
        _interrupt(
            first,
            event_id='interrupt-reset-with-new',
            guidance=['重新确认生成模块优先'],
            reset_guidance=True,
        ),
        preference_compiler=compiler,
    )

    assert seen_indexes == [[0], [0]]
    assert reset['user_guidance'] == ['重新确认生成模块优先']
    evidence = reset['ranking_interpretation']['guidance_resolution'][
        'category_directives'
    ][0]['evidence']
    assert evidence == {'source': 'user_guidance', 'index': 0}


def test_full_guidance_batch_can_reset_and_then_commit() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        evidence = request.texts[0].evidence.model_dump()
        return {
            'category_directives': [
                {
                    'category_id': category_id,
                    'tier': 'exclude',
                    'order': order,
                    'evidence': evidence,
                }
                for order, category_id in enumerate(
                    ('category_01', 'category_02')
                )
            ],
            'metric_directives': [],
        }

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    excluded = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-full',
            guidance=[
                f'排除指令 {index}'
                for index in range(64)
            ],
        ),
        preference_compiler=compiler,
    )
    reset = apply_target_guidance_interrupt(
        _analysis(),
        excluded,
        _interrupt(
            excluded,
            event_id='interrupt-reset-full',
            guidance=[],
            reset_guidance=True,
        ),
    )
    committed = commit_target_ranking(
        _analysis(),
        reset,
        _commit(reset, event_id='commit-after-reset'),
    )

    assert excluded['blocked_reason'] == 'blocked_all_categories_excluded'
    assert len(excluded['user_guidance']) == 64
    assert reset['provisional_selected_category'] == 'category_01'
    assert committed['status'] == 'committed'
    assert committed['selected_category'] == 'category_01'


def test_failed_guidance_compilation_advances_to_recoverable_preview() -> None:
    def failing_compiler(_: PreferenceCompileRequest) -> dict[str, Any]:
        raise RuntimeError('deterministic test failure')

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    interrupted = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-failed',
            guidance=['无法解析的本轮重点'],
        ),
        preference_compiler=failing_compiler,
    )

    assert interrupted['revision'] == 2
    assert interrupted['status'] == 'awaiting_interrupt'
    assert interrupted['user_guidance'] == ['无法解析的本轮重点']
    assert _ranked_ids(interrupted) == _ranked_ids(preview)
    assert (
        interrupted['provisional_selected_category']
        == preview['provisional_selected_category']
    )
    assert {warning['code'] for warning in interrupted['warnings']} == {
        'guidance_compiler_failed',
        'guidance_fallback_used',
    }


def test_all_excluded_preview_can_be_recovered_by_later_guidance() -> None:
    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        text = request.texts[0].text
        if text == '本轮两个类别都不要处理':
            evidence = request.texts[0].evidence.model_dump()
            return {
                'category_directives': [
                    {
                        'category_id': category_id,
                        'tier': 'exclude',
                        'order': order,
                        'evidence': evidence,
                    }
                    for order, category_id in enumerate(
                        ('category_01', 'category_02')
                    )
                ],
                'metric_directives': [],
            }
        assert text == '恢复并优先处理生成模块'
        return _category_directive(
            request,
            category_id='category_02',
            tier='must',
        )

    preview = build_target_ranking_preview(
        _analysis(),
        {},
        ranking_id='ranking-1',
    )
    excluded = apply_target_guidance_interrupt(
        _analysis(),
        preview,
        _interrupt(
            preview,
            event_id='interrupt-exclude',
            guidance=['本轮两个类别都不要处理'],
        ),
        preference_compiler=compiler,
    )

    assert excluded['status'] == 'awaiting_interrupt'
    assert excluded['revision'] == 2
    assert excluded['blocked_reason'] == 'blocked_all_categories_excluded'
    assert excluded['provisional_selected_category'] is None
    assert excluded['ranked_categories'] == []
    assert excluded['excluded_category_ids'] == [
        'category_01',
        'category_02',
    ]

    recovered = apply_target_guidance_interrupt(
        _analysis(),
        excluded,
        _interrupt(
            excluded,
            event_id='interrupt-recover',
            guidance=['恢复并优先处理生成模块'],
        ),
        preference_compiler=compiler,
    )

    assert recovered['status'] == 'awaiting_interrupt'
    assert recovered['revision'] == 3
    assert recovered['blocked_reason'] == ''
    assert recovered['provisional_selected_category'] == 'category_02'
    assert _ranked_ids(recovered) == ['category_02']
    assert recovered['excluded_category_ids'] == ['category_01']


def test_code_span_is_not_exposed_to_compiler_or_used_for_ranking() -> None:
    seen_requests: list[dict[str, Any]] = []

    def compiler(request: PreferenceCompileRequest) -> dict[str, Any]:
        payload = request.model_dump(mode='json')
        seen_requests.append(payload)
        assert all(
            set(option) == {'category_id', 'summary', 'analysis'}
            for option in payload['category_options']
        )
        assert 'code_span' not in repr(payload)
        return {
            'category_directives': [],
            'metric_directives': [],
        }

    original = _analysis()
    changed_spans = deepcopy(original)
    changed_spans['categories']['category_01']['code_span'] = [
        {
            'file': 'completely/different.py',
            'symbol': 'unrelated_symbol',
            'arbitrary': {'nested': ['opaque', 'data']},
        },
    ]
    changed_spans['categories']['category_02']['code_span'] = []

    original_state = build_target_ranking_preview(
        original,
        {'preference': ['偏好内容需要语义解析']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )
    changed_state = build_target_ranking_preview(
        changed_spans,
        {'preference': ['偏好内容需要语义解析']},
        ranking_id='ranking-1',
        preference_compiler=compiler,
    )

    assert len(seen_requests) == 2
    assert changed_state == original_state
