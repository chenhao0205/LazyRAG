from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar
from weakref import WeakSet

from .artifact import (
    RUN_CONFIGURATION_ARTIFACT_ID,
    ArtifactCommit,
    ArtifactDraft,
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
)
from .errors import (
    DefinitionError,
    _as_exception,
    _integer,
    _positive_number,
    _text,
)
from .operation import Operation, OperationResult
from .planning import (
    RuntimeDefinition,
    compile_operations,
    obsolete_retries,
    plan_next,
    project_runtime_snapshot,
)
from .session import RunSession, _load_case_failures
from .state import (
    ArtifactRetryRequest,
    AttemptSnapshot,
    CaseFailure,
    CaseOperationSnapshot,
    CaseSnapshot,
    EventLevel,
    EventStatus,
    RecordedOperationEvent,
    OperationDefinitionSnapshot,
    RunConfiguration,
    RunHistory,
    RuntimeSnapshot,
)
from .store import ArtifactStore


_ACTIVE_STATUSES = frozenset({'running', 'pausing', 'paused', 'cancelling'})
_T = TypeVar('_T')


@dataclass(frozen=True, slots=True)
class _SessionEntry:
    session: RunSession
    task: asyncio.Task[None]


class ArtifactRuntime:
    def __init__(self, store: ArtifactStore, definition: RuntimeDefinition, *, max_concurrency: int,
                 terminate_timeout: float) -> None:
        self._store = store
        self._definition = definition
        self._max_run_concurrency = max_concurrency
        self._terminate_timeout = terminate_timeout
        self._sessions: dict[str, _SessionEntry] = {}
        self._reported_session_tasks: WeakSet[asyncio.Task[None]] = WeakSet()
        self._run_locks: dict[str, asyncio.Lock] = {}
        self._activity_lock = asyncio.Lock()
        self._lifecycle = asyncio.Condition()
        self._close_lock = asyncio.Lock()
        self._active_accesses = 0
        self._closing = False
        self._closed = False

    @classmethod
    async def open(cls, root: str | Path, operations: Sequence[Operation] | RuntimeDefinition, *,
                   max_concurrency: int = 4, terminate_timeout: float = 1.0) -> ArtifactRuntime:
        _integer(max_concurrency, 'max_concurrency', minimum=1)
        _positive_number(terminate_timeout, 'terminate_timeout')
        definition = operations if isinstance(operations, RuntimeDefinition) else compile_operations(operations)
        store = await ArtifactStore.open(root)
        try:
            await store.recover_runs()
            for run_id in await store.run_ids():
                artifacts = await store.snapshot(run_id, definition.partition_set_ids)
                retries = await store.retry_requests(run_id, pending_only=True)
                for request in obsolete_retries(definition, artifacts, retries):
                    await store.cancel_retry(run_id, request.request_id)
        except BaseException:
            await store.close()
            raise
        return cls(
            store,
            definition,
            max_concurrency=max_concurrency,
            terminate_timeout=terminate_timeout,
        )

    async def __aenter__(self) -> Self:
        async with self._access():
            return self

    async def __aexit__(self, _exc_type: type[BaseException] | None, _exc: BaseException | None,
                        _traceback: TracebackType | None) -> None:
        await self.close()

    async def create(self, run_id: str, initial_commit: ArtifactCommit | None = None, *,
                     configuration: Mapping[str, object] | RunConfiguration | None = None) -> RuntimeSnapshot:
        _text(run_id, 'run_id')
        configured_commit = _with_run_configuration(initial_commit, configuration)
        if configured_commit is not None:
            self._definition.validate_commit(configured_commit)
        async with self._access(), self._activity_lock, self._run_lock(run_id):
            await self._ensure_active_slot(run_id)
            await self._store.create_run(run_id, configured_commit)
            try:
                return await self._inspect(run_id)
            except BaseException:
                await self._store.delete_run(run_id)
                raise

    async def start(self, run_id: str) -> RuntimeSnapshot:
        return await self._session_command(run_id, lambda session: session.enter_running('created'), claim_active=True)

    async def pause(self, run_id: str) -> RuntimeSnapshot:
        return await self._session_command(run_id, RunSession.pause)

    async def resume(self, run_id: str) -> RuntimeSnapshot:
        return await self._session_command(run_id, lambda session: session.enter_running('paused'), claim_active=True)

    async def cancel(self, run_id: str) -> RuntimeSnapshot:
        return await self._session_command(run_id, RunSession.cancel)

    async def commit(self, run_id: str, commit: ArtifactCommit) -> RuntimeSnapshot:
        if not isinstance(commit, ArtifactCommit):
            raise TypeError('commit must be ArtifactCommit')
        return await self._session_command(
            run_id,
            lambda session: session.commit(commit),
            claim_active=True,
        )

    async def rerun_artifacts(self, run_id: str, artifact_keys: Iterable[ArtifactKey], *, request_id: str
                              ) -> RuntimeSnapshot:
        keys = tuple(artifact_keys)
        if not keys or not all(isinstance(key, ArtifactKey) for key in keys):
            raise TypeError('artifact_keys must contain ArtifactKey values')
        _text(request_id, 'rerun request_id')
        return await self._session_command(
            run_id,
            lambda session: session.recompute(keys, (), (), request_id),
            claim_active=True,
        )

    async def rerun_operations(self, run_id: str, operation_ids: Iterable[str], *, request_id: str,
                               case_ids: Iterable[str] = ()) -> RuntimeSnapshot:
        operations = tuple(dict.fromkeys(operation_ids))
        cases = tuple(dict.fromkeys(case_ids))
        if not operations:
            raise DefinitionError('operation rerun requires at least one operation')
        for operation_id in operations:
            _text(operation_id, 'operation_id')
        for case_id in cases:
            _text(case_id, 'case_id')
        _text(request_id, 'operation rerun request_id')
        return await self._session_command(
            run_id,
            lambda session: session.recompute(
                (),
                operations,
                cases,
                request_id,
                failures_only=False,
            ),
            claim_active=True,
        )

    async def retry_case(self, run_id: str, case_id: str, *, request_id: str) -> RuntimeSnapshot:
        _text(case_id, 'case_id')
        _text(request_id, 'case retry request_id')
        return await self._session_command(
            run_id,
            lambda session: session.recompute(
                (),
                (),
                (case_id,),
                request_id,
                failures_only=True,
            ),
            claim_active=True,
        )

    async def retry_operations(self, run_id: str, operation_ids: Iterable[str], *, request_id: str) -> RuntimeSnapshot:
        operations = tuple(dict.fromkeys(operation_ids))
        if not operations:
            raise DefinitionError('operation retry requires at least one operation')
        for operation_id in operations:
            _text(operation_id, 'operation_id')
        _text(request_id, 'operation retry request_id')
        return await self._session_command(
            run_id,
            lambda session: session.recompute(
                (),
                operations,
                (),
                request_id,
                failures_only=True,
            ),
            claim_active=True,
        )

    async def submit_attempt_result(self, run_id: str, attempt_id: str, result: OperationResult) -> RuntimeSnapshot:
        _text(attempt_id, 'attempt result attempt_id')
        if not isinstance(result, OperationResult):
            raise TypeError('attempt result must be OperationResult')
        return await self._session_command(
            run_id,
            lambda session: session.submit_attempt_result(attempt_id, result),
            claim_active=True,
        )

    async def configuration(self, run_id: str) -> RunConfiguration:
        record = await self.head(
            run_id,
            ArtifactKey.scalar(RUN_CONFIGURATION_ARTIFACT_ID),
        )
        if record is None:
            raise DefinitionError(f'run has no configuration: {run_id}')
        value = await self.read(run_id, record.ref)
        if not isinstance(value, RunConfiguration):
            raise DefinitionError('run configuration artifact has an invalid value')
        return value

    async def update_configuration(self, run_id: str, configuration: Mapping[str, object] | RunConfiguration, *,
                                   request_id: str, base_version: int | None = None) -> RuntimeSnapshot:
        _text(request_id, 'configuration request_id')
        value = configuration if isinstance(configuration, RunConfiguration) else RunConfiguration(configuration)
        key = ArtifactKey.scalar(RUN_CONFIGURATION_ARTIFACT_ID)
        if base_version is None:
            current = await self.head(run_id, key)
            if current is None:
                raise DefinitionError(f'run has no configuration: {run_id}')
            expected = current.ref
        else:
            expected = ArtifactRef(key, base_version)
        return await self.commit(
            run_id,
            ArtifactCommit(
                f'configuration:{request_id}',
                f'runtime:configuration:{request_id}',
                (ArtifactDraft(key, value),),
                {key: expected},
            ),
        )

    async def case_snapshot(self, run_id: str, case_id: str) -> CaseSnapshot:
        _text(case_id, 'case_id')
        async with self._access():
            history = await self._run_history(run_id)
            snapshot = history.snapshot
            memberships = sorted(
                (
                    key,
                    partitions,
                )
                for key, partitions in snapshot.partition_sets.items()
                if case_id in partitions
            )
            if not memberships:
                raise DefinitionError(f'case is not active: {case_id}')
            attempts = history.attempts
            operation_events = history.operation_events
            partition_set_ids = {key.artifact_id for key, _ in memberships}
            operations = tuple(
                operation
                for operation in self._definition.operations
                if (
                    operation.spec.driver_input is not None
                    and operation.spec.partition_set_id in partition_set_ids
                )
            )
            case_failures = tuple(
                failure
                for failure in snapshot.case_failures
                if failure.case_id == case_id
            )
            latest_event_by_attempt: dict[str, RecordedOperationEvent] = {}
            for event in operation_events:
                latest_event_by_attempt[event.attempt_id] = event
            operation_snapshots = tuple(
                _case_operation_snapshot(
                    operation,
                    case_id,
                    snapshot,
                    attempts,
                    case_failures,
                    latest_event_by_attempt,
                )
                for operation in operations
            )
            artifacts = {
                key: ref
                for key, ref in snapshot.completed_artifacts.items()
                if key.partition_key == case_id
            }
            if case_failures:
                status = 'failed'
            elif any(item.status == 'running' for item in operation_snapshots):
                status = 'running'
            elif (
                operation_snapshots
                and all(item.status == 'succeeded' for item in operation_snapshots)
                or not operation_snapshots and artifacts
            ):
                status = 'completed'
            else:
                status = 'pending'
            return CaseSnapshot(
                run_id,
                case_id,
                memberships[0][1].keys.index(case_id) + 1,
                status,
                operation_snapshots,
                artifacts,
                case_failures,
                tuple(record for record in history.artifacts if record.ref.key.partition_key == case_id),
                tuple(attempt for attempt in attempts if attempt.partition_key == case_id),
                tuple(event for event in operation_events if event.partition_key == case_id),
                tuple(request for request in history.retry_requests if request.artifact_key.partition_key == case_id),
            )

    async def snapshot(self, run_id: str) -> RuntimeSnapshot:
        return await self._query(self._inspect, run_id)

    async def wait_for_status(self, run_id: str, statuses: str | tuple[str, ...], *, timeout: float = 10.0
                              ) -> RuntimeSnapshot:
        async with self._access(), self._run_lock(run_id):
            session = await self._session(run_id)
        return await session.wait_for_status(statuses, timeout=timeout)

    async def wait_until_settled(self, run_id: str, *, timeout: float = 10.0) -> RuntimeSnapshot:
        async with self._access(), self._run_lock(run_id):
            session = await self._session(run_id)
        return await session.wait_until_settled(timeout=timeout)

    async def attempts(self, run_id: str) -> tuple[AttemptSnapshot, ...]:
        return await self._query(self._store.attempts, run_id)

    async def operation_events(self, run_id: str, *, attempt_id: str = '', operation_id: str = '',
                               operation_ids: Iterable[str] = (),
                               case_id: str | None = None, event_type: str = '', level: EventLevel | None = None,
                               status: EventStatus | None = None, after: int = 0, limit: int | None = None
                               ) -> tuple[RecordedOperationEvent, ...]:
        async with self._access():
            return await self._store.operation_events(
                run_id,
                attempt_id=attempt_id,
                operation_id=operation_id,
                operation_ids=operation_ids,
                partition_key=case_id,
                event_type=event_type,
                level=level,
                status=status,
                after=after,
                limit=limit,
            )

    async def retry_requests(self, run_id: str) -> tuple[ArtifactRetryRequest, ...]:
        return await self._query(self._store.retry_requests, run_id)

    async def run_history(self, run_id: str) -> RunHistory:
        async with self._access():
            return await self._run_history(run_id)

    async def _run_history(self, run_id: str) -> RunHistory:
        inspection, artifacts, operation_events, retries = await asyncio.gather(
            self._store.inspect(run_id, self._definition.partition_set_ids),
            self._store.artifact_records(run_id),
            self._store.operation_events(run_id),
            self._store.retry_requests(run_id),
        )
        snapshot = await self._inspect(run_id, inspection)
        return RunHistory(
            snapshot,
            tuple(
                OperationDefinitionSnapshot(
                    operation.spec.op_id,
                    tuple(
                        (name, binding.artifact_id, binding.mode, binding.partition_set_id)
                        for name, binding in operation.spec.inputs.items()
                    ),
                    tuple(
                        (name, output.artifact_id, output.mode)
                        for name, output in operation.spec.outputs.items()
                    ),
                    operation.spec.execution,
                    operation.spec.max_concurrency,
                    operation.spec.timeout,
                )
                for operation in self._definition.operations
            ),
            artifacts,
            inspection[2],
            operation_events,
            retries,
        )

    async def read(self, run_id: str, ref: ArtifactRef) -> object:
        return await self._query(self._store.read, run_id, ref)

    async def read_many(self, run_id: str, refs: Iterable[ArtifactRef]) -> Mapping[ArtifactRef, object]:
        return await self._query(self._store.read_many, run_id, refs)

    async def record(self, run_id: str, ref: ArtifactRef) -> ArtifactRecord | None:
        return await self._query(self._store.record, run_id, ref)

    async def head(self, run_id: str, key: ArtifactKey) -> ArtifactRecord | None:
        return await self._query(self._store.head, run_id, key)

    async def history(self, run_id: str, key: ArtifactKey) -> tuple[ArtifactRecord, ...]:
        return await self._query(self._store.history, run_id, key)

    async def run_ids(self) -> tuple[str, ...]:
        return await self._query(self._store.run_ids)

    async def has_run(self, run_id: str) -> bool:
        _text(run_id, 'run_id')
        async with self._access():
            return await self._store.run_state(run_id) is not None

    async def release(self, run_id: str) -> None:
        _text(run_id, 'run_id')
        async with self._access(), self._run_lock(run_id):
            entry = self._current_entry(run_id)
            if entry is None:
                await self._require_run(run_id)
                return
            await entry.session.release()
            await entry.task
            if self._sessions.get(run_id) is entry:
                del self._sessions[run_id]

    async def delete_run(self, run_id: str) -> None:
        _text(run_id, 'run_id')
        async with self._access(), self._run_lock(run_id):
            entry = self._current_entry(run_id)
            if entry is not None:
                await entry.session.release()
                await entry.task
                if self._sessions.get(run_id) is entry:
                    del self._sessions[run_id]
            else:
                state = await self._require_run(run_id)
                if state.status in _ACTIVE_STATUSES:
                    snapshot = await self._inspect(run_id)
                    if (
                        state.status != 'running'
                        or snapshot.running
                        or snapshot.ready_count
                        or not snapshot.awaiting_artifacts
                    ):
                        raise RuntimeError('cannot delete a run with active persisted state')
            await self._store.delete_run(run_id)

    async def close(self) -> None:
        async with self._close_lock:
            async with self._lifecycle:
                await self._lifecycle.wait_for(lambda: not self._closing)
                if self._closed:
                    return
                self._closing = True
                await self._lifecycle.wait_for(lambda: self._active_accesses == 0)

            entries = tuple(self._sessions.items())
            results = await asyncio.gather(
                *(entry.session.close() for _, entry in entries),
                return_exceptions=True,
            )
            failures: list[Exception] = []
            closed_entries: list[tuple[str, _SessionEntry]] = []
            for (run_id, entry), result in zip(entries, results, strict=True):
                if isinstance(result, BaseException):
                    failures.append(_as_exception(result))
                else:
                    closed_entries.append((run_id, entry))

            task_results = await asyncio.gather(
                *(entry.task for _, entry in closed_entries),
                return_exceptions=True,
            )
            failures.extend(
                _as_exception(result)
                for result in task_results
                if isinstance(result, BaseException)
            )
            for run_id, entry in closed_entries:
                if self._sessions.get(run_id) is entry:
                    del self._sessions[run_id]

            try:
                if failures:
                    raise ExceptionGroup('artifact runtime failed to close cleanly', failures)
                await self._store.close()
            except BaseException:
                async with self._lifecycle:
                    self._closing = False
                    self._lifecycle.notify_all()
                raise
            self._sessions.clear()
            async with self._lifecycle:
                self._closed = True
                self._closing = False
                self._lifecycle.notify_all()

    async def _session_command(self, run_id: str, command: Callable[[RunSession], Awaitable[RuntimeSnapshot]], *,
                               claim_active: bool = False) -> RuntimeSnapshot:
        _text(run_id, 'run_id')
        async with self._access(), self._activity_lock, self._run_lock(run_id):
            if claim_active:
                await self._ensure_active_slot(run_id)
            session = await self._session(run_id)
            return await command(session)

    async def _ensure_active_slot(self, run_id: str) -> None:
        other = next(
            (
                active_run_id
                for active_run_id in await self._store.active_run_ids()
                if active_run_id != run_id
            ),
            None,
        )
        if other is not None:
            raise DefinitionError(
                f'active run {other} must complete or be cancelled first'
            )

    async def _session(self, run_id: str) -> RunSession:
        entry = self._current_entry(run_id)
        if entry is None:
            await self._require_run(run_id)
            session = RunSession(
                run_id,
                self._definition,
                self._store,
                max_concurrency=self._max_run_concurrency,
                terminate_timeout=self._terminate_timeout,
            )
            task = asyncio.create_task(session.serve(), name=f'artifact-run:{run_id}')
            entry = _SessionEntry(session, task)
            self._sessions[run_id] = entry
            task.add_done_callback(
                lambda completed, key=run_id, current=entry:
                self._discard_session(key, current)
            )
        try:
            await entry.session.wait_ready()
        except BaseException:
            if self._sessions.get(run_id) is entry:
                del self._sessions[run_id]
            await asyncio.gather(entry.task, return_exceptions=True)
            raise
        if entry.task.done():
            self._consume_session_task(run_id, entry)
        return entry.session

    async def _inspect(self, run_id: str, inspection: tuple | None = None) -> RuntimeSnapshot:
        state, artifacts, attempts, retries = inspection or await self._store.inspect(
            run_id, self._definition.partition_set_ids,
        )
        decision = plan_next(self._definition, artifacts, retries)
        failures = await _load_case_failures(
            self._store,
            run_id,
            decision.failure_refs,
        )
        return project_runtime_snapshot(
            run_id,
            state.status,
            state.error,
            self._definition,
            decision,
            attempts,
            failures,
        )

    async def _require_run(self, run_id: str):
        state = await self._store.run_state(run_id)
        if state is None:
            raise DefinitionError(f'run not found: {run_id}')
        return state

    async def _query(self, query: Callable[..., Awaitable[_T]], *args: object) -> _T:
        async with self._access():
            return await query(*args)

    def _current_entry(self, run_id: str) -> _SessionEntry | None:
        entry = self._sessions.get(run_id)
        if entry is not None and entry.task.done():
            self._consume_session_task(run_id, entry)
            return None
        return entry

    def _run_lock(self, run_id: str) -> asyncio.Lock:
        return self._run_locks.setdefault(run_id, asyncio.Lock())

    @asynccontextmanager
    async def _access(self) -> AsyncIterator[None]:
        async with self._lifecycle:
            if self._closed:
                raise RuntimeError('artifact runtime is closed')
            if self._closing:
                raise RuntimeError('artifact runtime is closing')
            self._active_accesses += 1
        try:
            yield
        finally:
            async with self._lifecycle:
                self._active_accesses -= 1
                if self._active_accesses == 0:
                    self._lifecycle.notify_all()

    def _consume_session_task(self, run_id: str, entry: _SessionEntry) -> None:
        if self._sessions.get(run_id) is entry:
            del self._sessions[run_id]
        if entry.task.cancelled():
            return
        error = entry.task.exception()
        if error is not None:
            raise RuntimeError(f'artifact run session failed: {run_id}') from error

    def _discard_session(self, run_id: str, entry: _SessionEntry) -> None:
        if self._sessions.get(run_id) is entry:
            del self._sessions[run_id]
        task = entry.task
        if task in self._reported_session_tasks:
            return
        self._reported_session_tasks.add(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            task.get_loop().call_exception_handler({
                'message': f'artifact run session failed: {run_id}',
                'exception': error,
                'task': task,
            })


def _with_run_configuration(initial_commit: ArtifactCommit | None,
                            configuration: Mapping[str, object] | RunConfiguration | None) -> ArtifactCommit | None:
    if initial_commit is not None and not isinstance(initial_commit, ArtifactCommit):
        raise TypeError('initial_commit must be ArtifactCommit or None')
    if configuration is None:
        return initial_commit
    value = configuration if isinstance(configuration, RunConfiguration) else RunConfiguration(configuration)
    key = ArtifactKey.scalar(RUN_CONFIGURATION_ARTIFACT_ID)
    if initial_commit is None:
        return ArtifactCommit(
            'run-configuration',
            'runtime:create',
            (ArtifactDraft(key, value),),
            {key: None},
        )
    if key in initial_commit.output_keys or key in initial_commit.expected_heads:
        raise DefinitionError('initial_commit must not write the reserved run configuration')
    return ArtifactCommit(
        initial_commit.commit_id,
        initial_commit.producer,
        (ArtifactDraft(key, value), *initial_commit.writes),
        {key: None, **initial_commit.expected_heads},
        initial_commit.partition_guards,
    )


def _case_operation_snapshot(operation: Operation, case_id: str, snapshot: RuntimeSnapshot,
                             attempts: tuple[AttemptSnapshot, ...], failures: tuple[CaseFailure, ...],
                             latest_event_by_attempt: Mapping[str, RecordedOperationEvent]) -> CaseOperationSnapshot:
    operation_attempts = tuple(sorted(
        (
            attempt
            for attempt in attempts
            if (
                attempt.operation_id == operation.spec.op_id
                and attempt.partition_key == case_id
            )
        ),
        key=lambda item: item.created_at,
    ))
    latest = operation_attempts[-1] if operation_attempts else None
    failure = next(
        (
            item for item in failures
            if item.operation_id == operation.spec.op_id
        ),
        None,
    )
    output_ids = {output.artifact_id for output in operation.spec.outputs.values()}
    outputs = tuple(sorted(
        (
            ref
            for key, ref in snapshot.completed_artifacts.items()
            if (
                key.partition_key == case_id
                and key.artifact_id in output_ids
            )
        ),
        key=lambda ref: ref.key.artifact_id,
    ))
    active = any(
        attempt.status in {'scheduled', 'running', 'cancelling'}
        for attempt in operation_attempts
    )
    if active:
        status = 'running'
    elif len(outputs) == len(operation.spec.outputs):
        status = 'succeeded'
    elif failure is not None:
        status = 'failed'
    else:
        status = 'pending'
    return CaseOperationSnapshot(
        operation.spec.op_id,
        status,
        outputs,
        '' if latest is None else latest.attempt_id,
        sum(bool(attempt.retry_request_id) for attempt in operation_attempts),
        None if latest is None else latest_event_by_attempt.get(latest.attempt_id),
        None if failure is None else failure.error,
    )


__all__ = ['ArtifactRuntime']
