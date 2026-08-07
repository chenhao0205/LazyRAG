from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from unidiff import PatchSet

from evo import artifacts as A
from evo.artifact_runtime import (
    AggregateValue,
    Operation,
    OperationContext,
    OperationResult,
    all_items,
    each,
    keyed,
    one,
    operation,
    partitioned,
    record_process,
    scalar,
)

from .abtest.candidate import async_candidate_rag_answer, candidate_service, finalize_candidate
from .abtest.comparison import compare_abtest
from .analysis.classify import classify_case
from .analysis.cluster import cluster_traces
from .analysis.summary import build_analysis_summary
from .analysis.trace_summary import build_trace_summary
from .dataset.operations import dataset_operations
from .eval.answer import async_answer_case
from .eval.judge import judge_case
from .public_contracts import RepairPatch, build_eval_summary_root, dump_contract, require_mapping as _mapping
from .repair.capabilities import DefaultCapabilityFactory
from .repair.contracts import RepairInput
from .repair.opencode import OpenCodeAdapter
from .repair.session import RepairSession
from evo.llm import LazyLLMClient


@operation(
    op_id='eval.answer',
    inputs={
        'case': each(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'dataset': one(A.EVAL_DATASET),
        'target_config': one(A.EVAL_TARGET_CONFIG),
        'approval': one(A.APPROVAL_DATASET),
    },
    outputs={'answer': partitioned(A.EVAL_RAG_ANSWER)},
    max_concurrency=4,
)
async def eval_answer_operation(ctx: OperationContext, case: object, dataset: object, target_config: object,
                                approval: object) -> OperationResult:
    await ctx.record('eval.answer_requested', status='started', case_id=ctx.partition_key)
    answer = await async_answer_case(
        _mapping(case, 'case'),
        _mapping(target_config, 'target_config'),
    )
    return await _recorded_result(
        ctx, 'eval.answer_received', {'answer': answer}, case_id=ctx.partition_key,
        answer_status=answer.get('status'), trace_id=answer.get('trace_id'),
    )


@operation(
    op_id='eval.judge',
    inputs={
        'case': each(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'answer': keyed(A.EVAL_RAG_ANSWER),
        'policy': one(A.EVAL_POLICY),
    },
    outputs={'judge': partitioned(A.EVAL_JUDGE_RESULT)},
    max_concurrency=4,
)
async def eval_judge_operation(ctx: OperationContext, case: object, answer: object, policy: object) -> OperationResult:
    judge = judge_case(
        _mapping(case, 'case'),
        _mapping(answer, 'answer'),
        _mapping(policy, 'policy'),
    )
    return await _recorded_result(
        ctx, 'eval.case_judged', {'judge': judge}, case_id=ctx.partition_key,
        quality_label=judge.get('quality_label'), failure_type=judge.get('failure_type'),
        overall_score=judge.get('overall_score'),
    )


@operation(
    op_id='eval.summary',
    inputs={'judges': all_items(A.EVAL_JUDGE_RESULT, over=A.EVAL_CASE_REQUESTS)},
    outputs={'summary': scalar(A.EVAL_SUMMARY)},
)
async def eval_summary_operation(ctx: OperationContext, judges: object) -> OperationResult:
    values = _partition_values(judges, 'judges')
    failures = _failure_summary(judges)
    summary = build_eval_summary_root(ctx.run_id, values, failures['failed_cases'])
    return await _recorded_result(
        ctx, 'eval.summary_built', {'summary': summary},
        current=len(values), total=len(values) + failures['failed_case_num'], case_count=len(values),
        failed_case_count=failures['failed_case_num'],
    )


@operation(
    op_id='analysis.trace_summary',
    inputs={
        'case': each(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'answer': keyed(A.EVAL_RAG_ANSWER),
        'eval_summary': one(A.EVAL_SUMMARY),
        'approval': one(A.APPROVAL_EVAL),
    },
    outputs={'summary': partitioned(A.ANALYSIS_TRACE_SUMMARY)},
    max_concurrency=4,
)
async def trace_summary_operation(ctx: OperationContext, case: object, answer: object, eval_summary: object,
                                  approval: object) -> OperationResult:
    _mapping(eval_summary, 'eval_summary')
    summary = build_trace_summary(_mapping(case, 'case'), _mapping(answer, 'answer'))
    return await _recorded_result(
        ctx, 'analysis.trace_summarized', {'summary': summary}, case_id=ctx.partition_key,
        retrieval_step_count=len(summary.get('retrieval_steps') or ()),
        error_stage_count=len(summary.get('error_stages') or ()),
    )


@operation(
    op_id='analysis.classify_case',
    inputs={
        'case': each(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'answer': keyed(A.EVAL_RAG_ANSWER),
        'judge': keyed(A.EVAL_JUDGE_RESULT),
        'trace': keyed(A.ANALYSIS_TRACE_SUMMARY),
    },
    outputs={'classification': partitioned(A.ANALYSIS_CASE_CLASSIFICATION)},
    max_concurrency=4,
)
async def classify_case_operation(ctx: OperationContext, case: object, answer: object, judge: object, trace: object
                                  ) -> OperationResult:
    classification = classify_case(
        _mapping(case, 'case'), _mapping(answer, 'answer'),
        _mapping(judge, 'judge'), _mapping(trace, 'trace'),
    )
    return await _recorded_result(
        ctx, 'analysis.case_classified', {'classification': classification}, case_id=ctx.partition_key,
        issue_type=classification.get('issue_type'), failure_mode=classification.get('failure_mode'),
    )


@operation(
    op_id='analysis.trace_clusters',
    inputs={
        'classifications': all_items(
            A.ANALYSIS_CASE_CLASSIFICATION,
            over=A.EVAL_CASE_REQUESTS,
        ),
    },
    outputs={'clusters': scalar(A.ANALYSIS_TRACE_CLUSTERS)},
)
async def trace_clusters_operation(ctx: OperationContext, classifications: object) -> OperationResult:
    values = _partition_values(classifications, 'classifications')
    clusters = cluster_traces(values) | _failure_summary(classifications)
    return await _recorded_result(
        ctx, 'analysis.traces_clustered', {'clusters': clusters},
        case_count=len(values), cluster_count=len(clusters.get('clusters') or ()),
    )


@operation(
    op_id='analysis.summary',
    inputs={
        'classifications': all_items(
            A.ANALYSIS_CASE_CLASSIFICATION,
            over=A.EVAL_CASE_REQUESTS,
        ),
        'clusters': one(A.ANALYSIS_TRACE_CLUSTERS),
    },
    outputs={'summary': scalar(A.ANALYSIS_SUMMARY)},
)
async def analysis_summary_operation(ctx: OperationContext, classifications: object, clusters: object
                                     ) -> OperationResult:
    values = _partition_values(classifications, 'classifications')
    summary = build_analysis_summary(ctx.run_id, values, _mapping(clusters, 'clusters')) | _failure_summary(
        classifications,
    )
    return await _recorded_result(
        ctx, 'analysis.summary_built', {'summary': summary},
        case_count=len(values), repair_group_count=len(summary.get('repair_group_queue') or ()),
    )


@operation(
    op_id='repair.session',
    inputs={
        'analysis': one(A.ANALYSIS_SUMMARY),
        'policy': one(A.REPAIR_POLICY),
        'approval': one(A.APPROVAL_ANALYSIS),
    },
    outputs={'patch': scalar(A.REPAIR_VERIFIED_PATCH)},
    timeout=7200.0,
)
@record_process
async def repair_session_operation(ctx: OperationContext, analysis: object, policy: object, approval: object
                                   ) -> OperationResult:
    del approval
    policy_value = _mapping(policy, 'policy')
    repair_input = _repair_input(ctx.run_id, _mapping(analysis, 'analysis'), policy_value)
    llm_config = policy_value.get('llm_config')
    client = LazyLLMClient(
        llm_config=llm_config if isinstance(llm_config, Mapping) else None,
        model='evo_llm',
    )
    session = RepairSession(
        OpenCodeAdapter(client, int(policy_value.get('model_timeout_seconds') or 120)),
        DefaultCapabilityFactory(),
    )
    result = await asyncio.to_thread(session.run, repair_input)
    if result.status != 'success':
        raise RuntimeError(f'repair session did not complete: {result.status}: {result.summary}')
    patch = _verified_patch(ctx.run_id, result.patch_ref)
    return await _recorded_result(
        ctx,
        'repair.session_completed',
        {'patch': patch},
        status=result.status,
        file_count=len(patch['diff']),
    )


@operation(
    op_id='abtest.candidate_service',
    inputs={
        'config': one(A.ABTEST_CANDIDATE_CONFIG),
        'patch': one(A.REPAIR_VERIFIED_PATCH),
        'approval': one(A.APPROVAL_REPAIR),
    },
    outputs={'service': scalar(A.ABTEST_CANDIDATE_SERVICE)},
)
async def candidate_service_operation(ctx: OperationContext, config: object, patch: object,
                                      approval: object) -> OperationResult:
    await ctx.record('abtest.candidate_service_starting', status='started')
    service = candidate_service(_mapping(config, 'config'), _mapping(patch, 'patch'), ctx)
    return await _recorded_result(
        ctx, 'abtest.candidate_service_ready', {'service': service}, status=service.get('status'),
        service_kind=service.get('service_kind'), algorithm_id=service.get('algorithm_id'),
    )


@operation(
    op_id='abtest.candidate_rag_answer',
    inputs={
        'case': each(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'service': one(A.ABTEST_CANDIDATE_SERVICE),
    },
    outputs={'answer': partitioned(A.ABTEST_CANDIDATE_RAG_ANSWER)},
    max_concurrency=4,
)
async def candidate_answer_operation(ctx: OperationContext, case: object, service: object) -> OperationResult:
    await ctx.record('abtest.candidate_answer_requested', status='started', case_id=ctx.partition_key)
    answer = await async_candidate_rag_answer(
        _mapping(case, 'case'),
        _mapping(service, 'service'),
    )
    return await _recorded_result(
        ctx, 'abtest.candidate_answer_received', {'answer': answer}, case_id=ctx.partition_key,
        answer_status=answer.get('status'), trace_id=answer.get('trace_id'),
    )


@operation(
    op_id='abtest.candidate_judge',
    inputs={
        'case': each(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'answer': keyed(A.ABTEST_CANDIDATE_RAG_ANSWER),
        'policy': one(A.EVAL_POLICY),
    },
    outputs={'judge': partitioned(A.ABTEST_CANDIDATE_JUDGE_RESULT)},
    max_concurrency=4,
)
async def candidate_judge_operation(ctx: OperationContext, case: object, answer: object, policy: object
                                    ) -> OperationResult:
    judge = judge_case(
        _mapping(case, 'case'),
        _mapping(answer, 'answer'),
        _mapping(policy, 'policy'),
    )
    return await _recorded_result(
        ctx, 'abtest.candidate_case_judged', {'judge': judge}, case_id=ctx.partition_key,
        quality_label=judge.get('quality_label'), failure_type=judge.get('failure_type'),
        overall_score=judge.get('overall_score'),
    )


@operation(
    op_id='abtest.candidate_eval_summary',
    inputs={
        'judges': all_items(
            A.ABTEST_CANDIDATE_JUDGE_RESULT,
            over=A.EVAL_CASE_REQUESTS,
        ),
    },
    outputs={'summary': scalar(A.ABTEST_CANDIDATE_EVAL_SUMMARY)},
)
async def candidate_summary_operation(ctx: OperationContext, judges: object) -> OperationResult:
    values = _partition_values(judges, 'judges')
    failures = _failure_summary(judges)
    summary = build_eval_summary_root(ctx.run_id, values, failures['failed_cases'])
    return await _recorded_result(
        ctx, 'abtest.candidate_summary_built', {'summary': summary},
        current=len(values), total=len(values) + failures['failed_case_num'], case_count=len(values),
        failed_case_count=failures['failed_case_num'],
    )


@operation(
    op_id='abtest.compare',
    inputs={
        'baseline': one(A.EVAL_SUMMARY),
        'candidate': one(A.ABTEST_CANDIDATE_EVAL_SUMMARY),
        'service': one(A.ABTEST_CANDIDATE_SERVICE),
    },
    outputs={'comparison': scalar(A.ABTEST_COMPARISON)},
)
async def compare_abtest_operation(ctx: OperationContext, baseline: object, candidate: object, service: object
                                   ) -> OperationResult:
    comparison = compare_abtest(
        ctx.run_id,
        _mapping(baseline, 'baseline'),
        _mapping(candidate, 'candidate'),
        _mapping(service, 'service'),
    )
    finalize_candidate(_mapping(service, 'service'), comparison)
    return await _recorded_result(
        ctx, 'abtest.comparison_completed', {'comparison': comparison},
        verdict=comparison.get('verdict'), status=comparison.get('status'),
    )


_EVO_OPERATIONS: tuple[Operation, ...] = (
    *dataset_operations(),
    eval_answer_operation,
    eval_judge_operation,
    eval_summary_operation,
    trace_summary_operation,
    classify_case_operation,
    trace_clusters_operation,
    analysis_summary_operation,
    repair_session_operation,
    candidate_service_operation,
    candidate_answer_operation,
    candidate_judge_operation,
    candidate_summary_operation,
    compare_abtest_operation,
)


def evo_operations() -> tuple[Operation, ...]:
    return _EVO_OPERATIONS


async def _recorded_result(ctx: OperationContext, event_type: str, values: Mapping[str, object], *,
                           current: int | None = None, total: int | None = None, **data: object) -> OperationResult:
    await ctx.record(event_type, status='completed', data=data, current=current, total=total)
    return OperationResult(values)


def _partition_values(value: object, name: str) -> tuple[Mapping[str, Any], ...]:
    values = tuple(_mapping(value, name).values())
    if not values:
        raise ValueError(f'{name} has no successful cases')
    if not all(isinstance(item, Mapping) for item in values):
        raise ValueError(f'{name} must contain mappings')
    return values


def _failure_summary(value: object) -> dict[str, object]:
    failures = [] if not isinstance(value, AggregateValue) else [
        {
            'case_id': failure.case_id,
            'operation_id': failure.operation_id,
            'attempt_id': failure.attempt_id,
            'error_kind': failure.error.kind,
            'error_message': failure.error.message,
        }
        for failure in value.failures.values()
    ]
    return {
        'failed_case_num': len(failures),
        'failed_cases': failures,
        'completed_with_problems': bool(failures),
    }


def _repair_input(run_id: str, analysis: Mapping[str, Any], policy: Mapping[str, Any]) -> RepairInput:
    queue = analysis.get('repair_group_queue')
    group = queue[0] if isinstance(queue, list) and queue and isinstance(queue[0], Mapping) else None
    if group is None:
        raise ValueError('analysis has no repairable group')
    candidate_files = [str(path).strip() for path in group.get('candidate_files') or () if str(path).strip()]
    if not candidate_files:
        raise ValueError('repair group has no candidate files')
    guidance = policy.get('user_guidance')
    guidance_text = '\n'.join(str(item) for item in guidance) if isinstance(guidance, list) else str(guidance or '')
    source_ref = str(
        policy.get('candidate_source_dir')
        or os.getenv('LAZYMIND_EVO_SOURCE')
        or '/app'
    )
    budget = policy.get('repair_budget')
    configured_constraints = policy.get('constraints')
    constraints = dict(configured_constraints) if isinstance(configured_constraints, Mapping) else {}
    if 'test_commands' in policy:
        constraints['test_commands'] = policy['test_commands']
    return RepairInput(
        run_id=run_id,
        objective=json.dumps(dict(group), ensure_ascii=False, sort_keys=True, default=str),
        guidance=guidance_text,
        source_ref=source_ref,
        case_scope='\n'.join(candidate_files),
        constraints=constraints,
        budget=dict(budget) if isinstance(budget, Mapping) else {'turns': 50, 'seconds': 7200},
    )


def _verified_patch(run_id: str, patch_ref: str) -> dict[str, Any]:
    path = Path(patch_ref)
    text = path.read_text(encoding='utf-8')
    diff = {}
    for patched in PatchSet(text.splitlines(True)):
        source = str(patched.source_file).removeprefix('a/')
        target = str(patched.target_file).removeprefix('b/')
        name = source if target == '/dev/null' else target or source
        if name:
            diff[name] = str(patched)
    if not diff:
        raise ValueError('repair result has no patch')
    workspace_ref = path.parent.parent / 'sandbox/source'
    return dump_contract(RepairPatch, {
        'run_id': run_id,
        'algo_id': '',
        'candidate_algo_id': '',
        'status': 'verified',
        'workspace_ref': str(workspace_ref),
        'diff': diff,
    })


__all__ = [
    'analysis_summary_operation', 'candidate_answer_operation', 'candidate_judge_operation',
    'candidate_service_operation', 'candidate_summary_operation',
    'classify_case_operation', 'compare_abtest_operation', 'eval_answer_operation',
    'eval_judge_operation', 'eval_summary_operation', 'evo_operations', 'repair_session_operation',
    'trace_clusters_operation', 'trace_summary_operation',
]
