from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import replace

from evo.artifact_runtime import (
    ArtifactKey,
    ArtifactRetryRequest,
    CaseFailure,
    DefinitionError,
    RuntimeProgress,
    RuntimeSnapshot,
)

from .definition import FlowDefinition
from .state import FlowSnapshot, StageProgress, StageStatus


def project_flow(definition: FlowDefinition, runtime: RuntimeSnapshot,
                 retries: Iterable[ArtifactRetryRequest] = ()) -> FlowSnapshot:
    if not isinstance(definition, FlowDefinition):
        raise TypeError('definition must be FlowDefinition')
    if not isinstance(runtime, RuntimeSnapshot):
        raise TypeError('runtime must be RuntimeSnapshot')

    requests = tuple(retries)
    if not all(isinstance(request, ArtifactRetryRequest) for request in requests):
        raise TypeError('retries must contain ArtifactRetryRequest values')

    refs = tuple(
        (
            runtime.completed_artifacts.get(stage.result_key),
            None if stage.approval_key is None else runtime.completed_artifacts.get(
                stage.approval_key
            ),
        )
        for stage in definition.stages
    )
    artifact_index = next(
        (
            index
            for index, stage in enumerate(definition.stages)
            if refs[index][0] is None or stage.approval_key is not None and refs[index][1] is None
        ),
        None,
    )
    approval_index = (
        artifact_index
        if artifact_index is not None
        and refs[artifact_index][0] is not None
        and definition.stages[artifact_index].approval_key is not None
        else None
    )
    active_index = _active_index(definition, runtime, requests)
    stage_details = tuple(
        _progress(definition, index, runtime)
        for index in range(len(definition.stages))
    )
    failure_index = next(
        (
            index
            for index, (progress, _) in enumerate(stage_details)
            if progress.case_total > 0 and progress.case_failed == progress.case_total
        ),
        None,
    )
    frontier = min(
        (index for index in (artifact_index, active_index, failure_index) if index is not None),
        default=None,
    )
    if failure_index is None and frontier is not None and refs[frontier][0] is None and runtime.status == 'failed':
        failure_index = frontier

    stages: list[StageProgress] = []
    for index, stage in enumerate(definition.stages):
        progress, failures = stage_details[index]
        if runtime.status == 'failed' and failure_index == index and progress.pending:
            failed = progress.failed + progress.pending
            progress = replace(
                progress,
                failed=failed,
                pending=0,
                percentage=0.0 if progress.total == 0 else round(
                    (progress.completed + failed) * 100 / progress.total,
                    2,
                ),
                case_failed=progress.case_failed + progress.case_pending,
                case_pending=0,
            )
        stages.append(StageProgress(
            stage.name,
            stage.result_key,
            refs[index][0],
            stage.approval_key,
            refs[index][1],
            _stage_status(index, frontier, active_index, approval_index, failure_index, runtime),
            tuple(operation.spec.op_id for operation in definition.stage_operations(index)),
            progress,
            failures,
            runtime.error if failure_index == index else None,
        ))
    stage_values = tuple(stages)
    total = sum(stage.progress.total for stage in stage_values)
    completed = sum(stage.progress.completed for stage in stage_values)
    running = sum(stage.progress.running for stage in stage_values)
    failed = sum(stage.progress.failed for stage in stage_values)
    pending = sum(stage.progress.pending for stage in stage_values)
    return FlowSnapshot(
        runtime,
        stage_values,
        replace(
            runtime.progress,
            total=total,
            completed=completed,
            running=running,
            failed=failed,
            pending=pending,
            percentage=0.0 if total == 0 else round((completed + failed) * 100 / total, 2),
        ),
        runtime.case_failures,
    )


def _active_index(definition: FlowDefinition, runtime: RuntimeSnapshot, retries: tuple[ArtifactRetryRequest, ...]
                  ) -> int | None:
    indices = [
        _required_stage(
            definition.stage_index_for_operation(attempt.operation_id),
            f'operation {attempt.operation_id}',
        )
        for attempt in runtime.active_attempts
    ]
    indices.extend(
        _required_stage(
            definition.stage_index_for_artifact(request.artifact_key.artifact_id),
            f'artifact {request.artifact_key.artifact_id}',
        )
        for request in retries
        if request.status == 'pending'
    )
    if runtime.status == 'cancelled':
        indices.extend(
            _required_stage(
                definition.stage_index_for_artifact(request.artifact_key.artifact_id),
                f'artifact {request.artifact_key.artifact_id}',
            )
            for request in retries
            if (
                request.status == 'cancelled'
                and runtime.completed_artifacts.get(request.artifact_key) == request.base_ref
            )
        )
    return min(indices, default=None)


def _required_stage(index: int | None, subject: str) -> int:
    if index is None:
        raise DefinitionError(f'{subject} does not belong to a flow stage')
    return index


def _stage_status(index: int, frontier: int | None, active: int | None, approval: int | None, failure: int | None,
                  runtime: RuntimeSnapshot) -> StageStatus:
    if frontier is None or index < frontier:
        return 'completed'
    if index > frontier:
        return 'pending'
    if active == frontier:
        if runtime.status == 'created':
            return 'pending'
        return 'running' if runtime.status == 'completed' else runtime.status
    if runtime.status in {'cancelling', 'cancelled'}:
        return runtime.status
    if failure == frontier:
        return 'failed'
    if approval == frontier:
        return 'awaiting_approval'
    if runtime.status in {'pausing', 'paused'}:
        return runtime.status
    return 'pending'


def _progress(definition: FlowDefinition, stage_index: int,
              runtime: RuntimeSnapshot) -> tuple[RuntimeProgress, tuple[CaseFailure, ...]]:
    operations = definition.stage_operations(stage_index)
    operation_ids = {operation.spec.op_id for operation in operations}
    failures_by_case: dict[str, list[CaseFailure]] = {}
    for failure in runtime.case_failures:
        failures_by_case.setdefault(failure.case_id, []).append(failure)
    active = {
        (attempt.operation_id, attempt.partition_key)
        for attempt in runtime.active_attempts
        if attempt.operation_id in operation_ids
    }
    operation_states: Counter[str] = Counter()
    case_states: dict[str, list[str]] = {}
    stage_failure_attempts: set[str] = set()
    for operation in operations:
        if operation.spec.driver_input is None:
            partition_keys = ('',)
        else:
            partitions = runtime.partition_sets.get(ArtifactKey.scalar(operation.spec.partition_set_id))
            partition_keys = () if partitions is None else partitions.keys
        dependencies = definition.dependencies_by_operation_id[operation.spec.op_id]
        for partition_key in partition_keys:
            identity = (operation.spec.op_id, partition_key)
            blocking_failures = tuple(
                failure
                for failure in failures_by_case.get(partition_key, ())
                if (
                    failure.operation_id == operation.spec.op_id
                    or any(key.artifact_id in dependencies for key in failure.output_keys)
                )
            )
            output_keys: list[ArtifactKey] = []
            for output in operation.spec.outputs.values():
                if partition_key or output.mode == 'scalar':
                    output_keys.append(output.key_for(partition_key))
                    continue
                partitions = runtime.partition_sets.get(ArtifactKey.scalar(output.partition_set_id))
                if partitions is not None:
                    output_keys.extend(output.key_for(case_id) for case_id in partitions.keys)
            if identity in active:
                status = 'running'
            elif output_keys and all(key in runtime.completed_artifacts for key in output_keys):
                status = 'completed'
            elif blocking_failures:
                status = 'failed'
            else:
                status = 'pending'
            operation_states[status] += 1
            if partition_key:
                case_states.setdefault(partition_key, []).append(status)
                stage_failure_attempts.update(failure.attempt_id for failure in blocking_failures)

    cases: Counter[str] = Counter()
    for states in case_states.values():
        if 'running' in states:
            cases['running'] += 1
        elif 'failed' in states:
            cases['failed'] += 1
        elif states and all(status == 'completed' for status in states):
            cases['completed'] += 1
        else:
            cases['pending'] += 1
    total = sum(operation_states.values())
    finished = operation_states['completed'] + operation_states['failed']
    return (
        RuntimeProgress(
            total,
            operation_states['completed'],
            operation_states['running'],
            operation_states['failed'],
            operation_states['pending'],
            0.0 if total == 0 else round(finished * 100 / total, 2),
            len(case_states),
            cases['completed'],
            cases['running'],
            cases['failed'],
            cases['pending'],
        ),
        tuple(failure for failure in runtime.case_failures if failure.attempt_id in stage_failure_attempts),
    )


__all__ = ['project_flow']
