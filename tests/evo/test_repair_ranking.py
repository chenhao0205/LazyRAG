from __future__ import annotations

import json
from copy import deepcopy

from evo.operations.operation import _repair_input
from evo.operations.repair.ranking import rerank_repair_groups


def _group(
    group_id: str,
    category: str,
    block: str,
    *,
    candidate_files: list[str] | None = None,
) -> dict[str, object]:
    return {
        'group_id': group_id,
        'issue_category': category,
        'function_block_id': block,
        'affected_block': block,
        'issue_type': f'{block}_failure',
        'failure_mode': f'{block}_failed',
        'candidate_files': candidate_files or [f'evo/{block}.py'],
        'adjacent_blocks': [],
    }


def test_no_guidance_preserves_analysis_order() -> None:
    queue = [_group('generation', 'generation', 'llm_generate'), _group('retrieval', 'retrieval', 'retrieve')]

    ranked = rerank_repair_groups(queue, [])

    assert [item['group_id'] for item in ranked] == ['generation', 'retrieval']


def test_chinese_category_guidance_moves_matching_group_first() -> None:
    queue = [_group('generation', 'generation', 'llm_generate'), _group('retrieval', 'retrieval', 'retrieve')]

    ranked = rerank_repair_groups(queue, ['请优先检查检索和召回逻辑'])

    assert [item['group_id'] for item in ranked] == ['retrieval', 'generation']


def test_candidate_filename_has_strong_match() -> None:
    queue = [
        _group('retrieve', 'retrieval', 'retrieve', candidate_files=['evo/retriever.py']),
        _group('rerank', 'retrieval', 'rerank', candidate_files=['evo/rerank.py']),
    ]

    ranked = rerank_repair_groups(queue, 'please inspect rerank.py first')

    assert [item['group_id'] for item in ranked] == ['rerank', 'retrieve']


def test_negated_target_is_not_promoted() -> None:
    queue = [_group('generation', 'generation', 'llm_generate'), _group('retrieval', 'retrieval', 'retrieve')]

    ranked = rerank_repair_groups(queue, '不要修改生成模块，优先检查检索模块')

    assert [item['group_id'] for item in ranked] == ['retrieval', 'generation']


def test_irrelevant_guidance_keeps_stable_order_and_does_not_mutate_input() -> None:
    queue = [_group('first', 'retrieval', 'retrieve'), _group('second', 'generation', 'llm_generate')]
    original = deepcopy(queue)

    first = rerank_repair_groups(queue, '保持公开 API 向后兼容')
    second = rerank_repair_groups(queue, '保持公开 API 向后兼容')

    assert first == second
    assert [item['group_id'] for item in first] == ['first', 'second']
    assert queue == original
    assert first[0] is not queue[0]


def test_new_attempt_uses_latest_guidance_without_waiting() -> None:
    analysis = {
        'repair_group_queue': [
            _group('generation', 'generation', 'llm_generate'),
            _group('rerank', 'retrieval', 'rerank', candidate_files=['evo/rerank.py']),
        ],
    }

    baseline = _repair_input('run-1', analysis, {'candidate_source_dir': '/tmp/source'})
    guided = _repair_input(
        'run-1',
        analysis,
        {'user_guidance': ['优先修复 rerank.py'], 'candidate_source_dir': '/tmp/source'},
    )

    assert json.loads(baseline.objective)['group_id'] == 'generation'
    assert json.loads(guided.objective)['group_id'] == 'rerank'
    assert guided.case_scope == 'evo/rerank.py'
    assert guided.guidance == '优先修复 rerank.py'
