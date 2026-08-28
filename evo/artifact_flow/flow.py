from __future__ import annotations

import asyncio
from collections.abc import Iterable, Mapping
from pathlib import Path

from evo.artifact_runtime import (
    ArtifactCommit,
    ArtifactDraft,
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    ArtifactRetryRequest,
    ArtifactRuntime,
    AttemptSnapshot,
    DefinitionError,
    EventLevel,
    EventStatus,
    OperationResult,
    PartitionGuard,
    PartitionSet,
    RecordedOperationEvent,
    RUN_CONFIGURATION_ARTIFACT_ID,
    RunConfiguration,
    RunHistory,
    RuntimeSnapshot,
)

from .definition import FlowDefinition
from .projection import project_flow
from .state import ArtifactUpdate, FlowCaseSnapshot, FlowRunHistory, FlowSnapshot, StageProgress, StageSnapshot


_RESERVED_PRODUCERS = ('operation:', 'runtime:', 'user:approval', 'user:artifact-update')


class ArtifactFlow:
    def __init__(self, runtime: ArtifactRuntime, definition: FlowDefinition) -> None:
        if not isinstance(runtime, ArtifactRuntime):
            raise TypeError('runtime must be ArtifactRuntime')
        if not isinstance(definition, FlowDefinition):
            raise TypeError('definition must be FlowDefinition')
        self._runtime = runtime
        self.definition = definition
        self._approval_keys = frozenset(
            stage.approval_key for stage in definition.stages if stage.approval_key is not None
        )
        self._content_update_forbidden_ids = frozenset({
            RUN_CONFIGURATION_ARTIFACT_ID,
            *definition.partition_set_by_artifact.values(),
        })

    @classmethod
    async def open(cls, root: str | Path, definition: FlowDefinition, *, max_concurrency: int = 4,
                   terminate_timeout: float = 1.0) -> ArtifactFlow:
        if not isinstance(definition, FlowDefinition):
            raise TypeError('definition must be FlowDefinition')
        runtime = await ArtifactRuntime.open(
            root,
            definition.runtime_definition,
            max_concurrency=max_concurrency,
            terminate_timeout=terminate_timeout,
        )
        return cls(runtime, definition)

    async def create(self, run_id: str, initial_commit: ArtifactCommit | None = None, *,
                     configuration: Mapping[str, object] | RunConfiguration | None = None) -> FlowSnapshot:
        if initial_commit is not None:
            self._validate_user_commit(initial_commit)
        return await self._project(await self._runtime.create(
            run_id,
            initial_commit,
            configuration=configuration,
        ))

    async def start(self, run_id: str) -> FlowSnapshot:
        return await self._project(await self._runtime.start(run_id))

    async def approve(self, run_id: str, stage: str) -> FlowSnapshot:
        current = await self.snapshot(run_id)
        progress = current.stages[self.definition.stage_index(stage)]
        if progress.approval_key is None:
            raise DefinitionError(f'flow stage does not require approval: {stage}')
        if progress.result_ref is None:
            raise DefinitionError(f'flow stage is not complete: {stage}')
        if progress.approved:
            return current
        pending = current.pending_approval
        if pending is None or pending.stage != stage:
            raise DefinitionError(f'flow is not awaiting approval for: {stage}')
        result_ref = progress.result_ref
        approval_key = progress.approval_key

        approval = await self._runtime.head(run_id, approval_key)
        commit = ArtifactCommit(
            f'approval:{progress.stage}:{result_ref.key.artifact_id}:{result_ref.version}',
            'user:approval',
            (ArtifactDraft(
                approval_key,
                {
                    'stage': progress.stage,
                    'result': {
                        'artifact_id': result_ref.key.artifact_id,
                        'version': result_ref.version,
                    },
                },
                (result_ref,),
            ),),
            {
                result_ref.key: result_ref,
                approval_key: None if approval is None else approval.ref,
            },
        )
        return await self._project(await self._runtime.commit(run_id, commit))

    async def commit(self, run_id: str, commit: ArtifactCommit) -> FlowSnapshot:
        self._validate_user_commit(commit)
        await self._validate_structure_commit(run_id, commit)
        return await self._project(await self._runtime.commit(run_id, commit))

    async def commit_structure_with_values(
        self,
        run_id: str,
        commit: ArtifactCommit,
        *,
        value_keys: Iterable[ArtifactKey],
    ) -> FlowSnapshot:
        """Atomically replace a partition topology and its declared value snapshot."""
        self._validate_user_commit(commit)
        await self._validate_structure_commit(
            run_id,
            commit,
            value_keys=frozenset(value_keys),
        )
        return await self._project(await self._runtime.commit(run_id, commit))

    async def commit_values(self, run_id: str, commit: ArtifactCommit) -> FlowSnapshot:
        """Write existing artifact content with a composite CAS.

        `commit()` is reserved for PartitionSet structure changes.
        `update_artifacts()` only compares the keys it writes. Dataset apply
        needs both: write only the values that changed, while treating the
        whole snapshot that produced those values as a single precondition.
        """
        self._validate_user_commit(commit)
        self._validate_content_commit(commit)
        expected_by_key = commit.expected_heads
        records = await asyncio.gather(*(
            self._runtime.record(run_id, expected_by_key[write.key])
            for write in commit.writes
        ))
        missing = tuple(
            expected_by_key[write.key]
            for write, record in zip(commit.writes, records, strict=True)
            if record is None
        )
        if missing:
            names = ', '.join(
                f'{ref.key.artifact_id}@v{ref.version}' for ref in missing if ref is not None
            )
            raise DefinitionError(f'content commit targets do not exist: {names}')
        return await self._project(await self._runtime.commit(run_id, ArtifactCommit(
            commit.commit_id,
            commit.producer,
            tuple(
                ArtifactDraft(write.key, write.value, record.input_refs)
                for write, record in zip(commit.writes, records, strict=True)
                if record is not None
            ),
            dict(expected_by_key),
            self._partition_guards((*commit.output_keys, *expected_by_key)),
        )))

    async def configuration(self, run_id: str) -> RunConfiguration:
        return await self._runtime.configuration(run_id)

    async def update_configuration(self, run_id: str, configuration: Mapping[str, object] | RunConfiguration, *,
                                   request_id: str, base_version: int | None = None) -> FlowSnapshot:
        return await self._project(await self._runtime.update_configuration(
            run_id,
            configuration,
            request_id=request_id,
            base_version=base_version,
        ))

    async def rerun_artifact(self, run_id: str, artifact_key: ArtifactKey, *, request_id: str) -> FlowSnapshot:
        return await self._rerun_keys(run_id, (artifact_key,), request_id, 'artifact')

    async def rerun_stage(self, run_id: str, stage: str, *, request_id: str) -> FlowSnapshot:
        stage_index = self.definition.stage_index(stage)
        operation_ids = tuple(operation.spec.op_id for operation in self.definition.stage_entry_operations(stage_index))
        command_id = f'flow-rerun:stage:{stage}:{_request_id(request_id)}'
        return await self._project(await self._runtime.rerun_operations(
            run_id,
            operation_ids,
            request_id=command_id,
        ))

    async def rerun_case(self, run_id: str, case_id: str, *, request_id: str, from_stage: str = '',
                         from_artifact: ArtifactKey | None = None) -> FlowSnapshot:
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError('case_id must be non-empty')
        if bool(from_stage.strip()) == (from_artifact is not None):
            raise DefinitionError('rerun_case requires exactly one of from_stage or from_artifact')
        if from_artifact is not None:
            if not isinstance(from_artifact, ArtifactKey) or from_artifact.partition_key != case_id:
                raise DefinitionError('from_artifact must identify the requested case')
            keys = (from_artifact,)
            namespace = f'case:{case_id}:artifact'
        else:
            operations = self.definition.stage_case_entry_operations(
                self.definition.stage_index(from_stage)
            )
            if not operations:
                raise DefinitionError(f'flow stage has no effective case rerun entry: {from_stage}[{case_id}]')
            command_id = f'flow-rerun:case:{case_id}:stage:{from_stage}:{_request_id(request_id)}'
            return await self._project(await self._runtime.rerun_operations(
                run_id,
                (operation.spec.op_id for operation in operations),
                request_id=command_id,
                case_ids=(case_id,),
            ))
        return await self._rerun_keys(run_id, keys, request_id, namespace)

    async def retry_failed_case(self, run_id: str, case_id: str, *, request_id: str) -> FlowSnapshot:
        child_id = f'flow-case-retry:{_request_id(request_id)}:{case_id}'
        return await self._project(await self._runtime.retry_case(run_id, case_id, request_id=child_id))

    async def update_artifacts(self, run_id: str, updates: Iterable[ArtifactUpdate], *, request_id: str) -> FlowSnapshot:
        values = tuple(updates)
        if not values or not all(isinstance(update, ArtifactUpdate) for update in values):
            raise TypeError('updates must contain ArtifactUpdate values')
        keys = tuple(update.target_ref.key for update in values)
        if len(set(keys)) != len(keys):
            raise DefinitionError('one artifact update request cannot write the same key twice')
        forbidden = tuple(key for key in keys if key in self._approval_keys)
        if forbidden:
            names = ', '.join(key.artifact_id for key in forbidden)
            raise DefinitionError(f'approval artifacts require approve(): {names}')
        structural = sorted({key.artifact_id for key in keys if key.artifact_id in self._content_update_forbidden_ids})
        if structural:
            raise DefinitionError(f'artifacts require their dedicated update API: {", ".join(structural)}')
        unknown_partitioned = sorted({
            key.artifact_id
            for key in keys
            if key.partition_key and key.artifact_id not in self.definition.partition_set_by_artifact
        })
        if unknown_partitioned:
            raise DefinitionError(f'unknown partitioned artifacts: {", ".join(unknown_partitioned)}')

        records = await asyncio.gather(*(self._runtime.record(run_id, update.target_ref) for update in values))
        missing = tuple(update.target_ref for update, record in zip(values, records, strict=True) if record is None)
        if missing:
            names = ', '.join(f'{ref.key.artifact_id}@v{ref.version}' for ref in missing)
            raise DefinitionError(f'artifact update targets do not exist: {names}')
        guards = self._partition_guards(keys)
        command_id = _request_id(request_id)
        return await self._project(await self._runtime.commit(run_id, ArtifactCommit(
            f'flow-update:{command_id}',
            'user:artifact-update',
            tuple(
                ArtifactDraft(update.target_ref.key, update.value, record.input_refs)
                for update, record in zip(values, records, strict=True)
                if record is not None
            ),
            {update.target_ref.key: update.target_ref for update in values},
            guards,
        )))

    async def pause(self, run_id: str) -> FlowSnapshot:
        return await self._project(await self._runtime.pause(run_id))

    async def resume(self, run_id: str) -> FlowSnapshot:
        return await self._project(await self._runtime.resume(run_id))

    async def retry_stage(self, run_id: str, stage: str, *, request_id: str) -> FlowSnapshot:
        target_index = self.definition.stage_index(stage)
        return await self._project(await self._runtime.retry_operations(
            run_id,
            (operation.spec.op_id for operation in self.definition.stage_operations(target_index)),
            request_id=f'flow-stage-retry:{_request_id(request_id)}:{stage}',
        ))

    async def cancel(self, run_id: str) -> FlowSnapshot:
        return await self._project(await self._runtime.cancel(run_id))

    async def wait_until_boundary(self, run_id: str, *, timeout: float = 10.0) -> FlowSnapshot:
        snapshot = await self._runtime.wait_until_settled(run_id, timeout=timeout)
        return await self._project(snapshot)

    async def snapshot(self, run_id: str) -> FlowSnapshot:
        return await self._project(await self._runtime.snapshot(run_id))

    async def stage_snapshot(self, run_id: str, stage: str) -> StageSnapshot:
        return (await self.run_history(run_id)).stages[self.definition.stage_index(stage)]

    async def case_snapshot(self, run_id: str, case_id: str) -> FlowCaseSnapshot:
        case = await self._runtime.case_snapshot(run_id, case_id)
        indices = tuple(dict.fromkeys(
            index
            for operation in case.operations
            if (index := self.definition.stage_index_for_operation(operation.operation_id)) is not None
        ))
        stages = tuple(self.definition.stages[index].name for index in indices)
        active = next(
            (
                self.definition.stages[index].name
                for operation in case.operations
                if operation.status != 'succeeded'
                if (index := self.definition.stage_index_for_operation(operation.operation_id)) is not None
            ),
            stages[-1] if stages else '',
        )
        return FlowCaseSnapshot(
            case,
            stages,
            active,
            case.artifact_records,
            case.attempts,
            case.operation_events,
            case.retries,
        )

    async def case_operation_statuses(self, run_id: str, case_ids: Iterable[str],
                                      operation_ids: Iterable[str]) -> dict[str, dict[str, str]]:
        return await self._runtime.case_operation_statuses(run_id, case_ids, operation_ids)

    async def run_history(self, run_id: str) -> FlowRunHistory:
        history = await self._runtime.run_history(run_id)
        snapshot = project_flow(self.definition, history.snapshot, history.retry_requests)
        return FlowRunHistory(
            snapshot,
            history,
            tuple(
                _stage_snapshot(self.definition, index, snapshot.stages[index], history)
                for index in range(len(self.definition.stages))
            ),
        )

    async def submit_external_result(self, run_id: str, attempt_id: str, result: OperationResult) -> FlowSnapshot:
        return await self._project(await self._runtime.submit_attempt_result(run_id, attempt_id, result))

    async def read(self, run_id: str, ref: ArtifactRef) -> object:
        return await self._runtime.read(run_id, ref)

    async def read_many(self, run_id: str, refs: Iterable[ArtifactRef]) -> Mapping[ArtifactRef, object]:
        return await self._runtime.read_many(run_id, refs)

    async def record(self, run_id: str, ref: ArtifactRef) -> ArtifactRecord | None:
        return await self._runtime.record(run_id, ref)

    async def head(self, run_id: str, key: ArtifactKey) -> ArtifactRecord | None:
        return await self._runtime.head(run_id, key)

    async def history(self, run_id: str, key: ArtifactKey) -> tuple[ArtifactRecord, ...]:
        return await self._runtime.history(run_id, key)

    async def attempts(self, run_id: str) -> tuple[AttemptSnapshot, ...]:
        return await self._runtime.attempts(run_id)

    async def operation_events(self, run_id: str, *, stage: str = '', attempt_id: str = '', operation_id: str = '',
                               case_id: str | None = None, event_type: str = '', level: EventLevel | None = None,
                               status: EventStatus | None = None, after: int = 0, limit: int | None = None
                               ) -> tuple[RecordedOperationEvent, ...]:
        if stage:
            stage_operation_ids = tuple(
                operation.spec.op_id
                for operation in self.definition.stage_operations(self.definition.stage_index(stage))
            )
            if operation_id and operation_id not in stage_operation_ids:
                return ()
        else:
            stage_operation_ids = ()
        return await self._runtime.operation_events(
            run_id,
            attempt_id=attempt_id,
            operation_id=operation_id,
            operation_ids=() if operation_id else stage_operation_ids,
            case_id=case_id,
            event_type=event_type,
            level=level,
            status=status,
            after=after,
            limit=limit,
        )

    async def retry_requests(self, run_id: str) -> tuple[ArtifactRetryRequest, ...]:
        return await self._runtime.retry_requests(run_id)

    async def run_ids(self) -> tuple[str, ...]:
        return await self._runtime.run_ids()

    async def has_run(self, run_id: str) -> bool:
        return await self._runtime.has_run(run_id)

    async def release(self, run_id: str) -> None:
        await self._runtime.release(run_id)

    async def delete_run(self, run_id: str) -> None:
        await self._runtime.delete_run(run_id)

    async def close(self) -> None:
        await self._runtime.close()

    async def _project(self, runtime: RuntimeSnapshot) -> FlowSnapshot:
        retries = await self._runtime.retry_requests(runtime.run_id)
        return project_flow(self.definition, runtime, retries)

    def _partition_guards(self, keys: Iterable[ArtifactKey]) -> tuple[PartitionGuard, ...]:
        return tuple(dict.fromkeys(
            PartitionGuard(
                ArtifactKey.scalar(self.definition.partition_set_by_artifact[key.artifact_id]),
                key.partition_key,
            )
            for key in keys
            if key.partition_key and key.artifact_id in self.definition.partition_set_by_artifact
        ))

    def _validate_user_commit(self, commit: ArtifactCommit) -> None:
        if not isinstance(commit, ArtifactCommit):
            raise TypeError('commit must be ArtifactCommit')
        if commit.producer.startswith(_RESERVED_PRODUCERS):
            raise DefinitionError(f'artifact producer is reserved for Flow or Runtime: {commit.producer}')
        forbidden = sorted(
            (write.key for write in commit.writes if write.key in self._approval_keys),
            key=lambda key: (key.artifact_id, key.partition_key),
        )
        if forbidden:
            names = ', '.join(key.artifact_id for key in forbidden)
            raise DefinitionError(f'approval artifacts require approve(): {names}')
        if any(write.key.artifact_id == RUN_CONFIGURATION_ARTIFACT_ID for write in commit.writes):
            raise DefinitionError('run configuration requires update_configuration()')

    def _validate_content_commit(self, commit: ArtifactCommit) -> None:
        structural = sorted({
            write.key.artifact_id
            for write in commit.writes
            if write.key.artifact_id in self._content_update_forbidden_ids
        })
        if structural:
            raise DefinitionError(f'artifacts require their dedicated update API: {", ".join(structural)}')
        if not set(commit.output_keys).issubset(commit.expected_heads):
            raise DefinitionError('content commits must compare every write')
        if any(commit.expected_heads.get(key) is None for key in (*commit.output_keys, *commit.expected_heads)):
            raise DefinitionError('content commits can only update existing artifacts')
        unknown_partitioned = sorted({
            key.artifact_id
            for key in (*commit.output_keys, *commit.expected_heads)
            if key.partition_key and key.artifact_id not in self.definition.partition_set_by_artifact
        })
        if unknown_partitioned:
            raise DefinitionError(f'unknown partitioned artifacts: {", ".join(unknown_partitioned)}')

    async def _validate_structure_commit(
        self,
        run_id: str,
        commit: ArtifactCommit,
        *,
        value_keys: frozenset[ArtifactKey] = frozenset(),
    ) -> None:
        partition_set_ids = frozenset(self.definition.partition_set_by_artifact.values())
        set_writes = {
            write.key: write.value
            for write in commit.writes
            if write.key.artifact_id in partition_set_ids
        }
        if not set_writes:
            raise DefinitionError('commit() is reserved for atomic case structure changes')
        if any(key.partition_key or not isinstance(value, PartitionSet) for key, value in set_writes.items()):
            raise DefinitionError('case structure writes must contain scalar PartitionSet values')
        if set(commit.expected_heads) != set(commit.output_keys):
            raise DefinitionError('case structure commits must compare every write and no unrelated artifact')
        if any(key.partition_key for key in value_keys):
            raise DefinitionError('atomic value keys must be scalar artifacts')
        if not value_keys.issubset(commit.output_keys):
            raise DefinitionError('atomic value keys must be committed values')

        base_refs = tuple(commit.expected_heads[key] for key in set_writes)
        if any(ref is None for ref in base_refs):
            raise DefinitionError('case structure can only update an existing PartitionSet')
        try:
            current_sets = await asyncio.gather(*(
                self._runtime.read(run_id, ref) for ref in base_refs if ref is not None
            ))
        except KeyError as exc:
            raise DefinitionError('case structure base version does not exist') from exc
        additions: dict[str, frozenset[str]] = {}
        for (key, value), current in zip(set_writes.items(), current_sets, strict=True):
            if not isinstance(current, PartitionSet):
                raise DefinitionError(f'current case structure is invalid: {key.artifact_id}')
            if current == value:
                raise DefinitionError(f'case structure does not change: {key.artifact_id}')
            additions[key.artifact_id] = frozenset(value.keys) - frozenset(current.keys)

        added_seeds: dict[str, set[str]] = {artifact_id: set() for artifact_id in additions}
        for write in commit.writes:
            if write.key in set_writes:
                continue
            if write.key in value_keys:
                if commit.expected_heads[write.key] is None:
                    raise DefinitionError('atomic values must update existing artifacts')
                continue
            set_id = self.definition.partition_set_by_artifact.get(write.key.artifact_id)
            if not write.key.partition_key or set_id not in additions:
                raise DefinitionError('case structure commits cannot write unrelated artifacts')
            if write.key.partition_key not in additions[set_id]:
                raise DefinitionError('case structure commits can only seed newly added cases')
            added_seeds[set_id].add(write.key.partition_key)
        missing = sorted(
            f'{set_id}[{case_id}]'
            for set_id, case_ids in additions.items()
            for case_id in case_ids - added_seeds[set_id]
        )
        if missing:
            raise DefinitionError(f'new cases require an initial artifact: {", ".join(missing)}')

    async def _rerun_keys(self, run_id: str, keys: tuple[ArtifactKey, ...], request_id: str, namespace: str
                          ) -> FlowSnapshot:
        forbidden = tuple(key for key in keys if key in self._approval_keys)
        if forbidden:
            names = ', '.join(key.artifact_id for key in forbidden)
            raise DefinitionError(f'approval artifacts require approve(): {names}')
        retry_id = f'flow-rerun:{namespace}:{_request_id(request_id)}'
        return await self._project(await self._runtime.rerun_artifacts(run_id, keys, request_id=retry_id))


def _request_id(request_id: str) -> str:
    if not isinstance(request_id, str) or not request_id.strip():
        raise DefinitionError('request_id must be non-empty')
    return request_id.strip()


def _stage_snapshot(definition: FlowDefinition, stage_index: int, progress: StageProgress, history: RunHistory
                    ) -> StageSnapshot:
    stage = definition.stages[stage_index]
    operation_ids = set(progress.operation_ids)
    attempts = tuple(attempt for attempt in history.attempts if attempt.operation_id in operation_ids)
    attempt_ids = {attempt.attempt_id for attempt in attempts}
    artifacts = tuple(
        record
        for record in history.artifacts
        if (
            definition.stage_index_for_artifact(record.ref.key.artifact_id) == stage_index
            or record.ref.key in {stage.result_key, stage.approval_key}
        )
    )
    results = tuple(record for record in artifacts if record.ref.key == stage.result_key)
    approvals = tuple(record for record in artifacts if record.ref.key == stage.approval_key)
    approval_by_result = {
        ref: approval
        for approval in approvals
        for ref in approval.input_refs
        if ref.key == stage.result_key
    }
    return StageSnapshot(
        progress,
        tuple(operation for operation in history.operations if operation.operation_id in operation_ids),
        attempts,
        artifacts,
        tuple(event for event in history.operation_events if event.attempt_id in attempt_ids),
        tuple(
            request
            for request in history.retry_requests
            if definition.stage_index_for_artifact(request.artifact_key.artifact_id) == stage_index
        ),
        tuple((result, approval_by_result.get(result.ref)) for result in results),
    )


__all__ = ['ArtifactFlow']
