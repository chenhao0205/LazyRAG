from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import Literal

from .artifact import (
    ArtifactCommit,
    ArtifactDraft,
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    ArtifactSnapshot,
    failure_key,
)
from .errors import (
    DefinitionError,
    OperationExecutionError,
    OperationTimeoutError,
    _as_exception,
    _integer,
    _positive_number,
    _text,
)
from .execution import ExecutionCleanupError, ExecutionHandle, start_execution
from .operation import OperationContext, OperationInvocation, OperationResult
from .planning import (
    PlanComplete,
    PlanReady,
    PlanningResult,
    RuntimeDefinition,
    obsolete_retries,
    plan_next,
    project_runtime_snapshot,
)
from .state import (
    ArtifactRetryRequest,
    AttemptSnapshot,
    CaseFailure,
    OperationEvent,
    RunStatus,
    RuntimeErrorInfo,
    RuntimeSnapshot,
)
from .store import ArtifactStore


_CONTROL_PRIORITY = 0
_COMPLETION_PRIORITY = -1
_OPERATION_EVENT_PRIORITY = -2
_OPERATION_EVENT_CAPACITY = 256
_EDITABLE_STATUSES = frozenset({'created', 'paused', 'completed', 'running'})
_LOG = logging.getLogger(__name__)


class _TerminationFailure(ExceptionGroup):
    pass


@dataclass(frozen=True, slots=True)
class _Request:
    action: Callable[[], Awaitable[None]]
    reply: asyncio.Future[RuntimeSnapshot]
    flush_failure: bool = False


@dataclass(frozen=True, slots=True)
class _ExecutionEvent:
    attempt_id: str
    event: OperationEvent


@dataclass(frozen=True, slots=True)
class _ExecutionDone:
    attempt_id: str
    result: OperationResult | None
    error: BaseException | None


@dataclass(slots=True)
class _ActiveExecution:
    invocation: OperationInvocation
    attempt: AttemptSnapshot
    handle: ExecutionHandle
    waiter: asyncio.Task[None]


_Event = _Request | _ExecutionEvent | _ExecutionDone


class RunSession:
    def __init__(self, run_id: str, definition: RuntimeDefinition, store: ArtifactStore, *, max_concurrency: int,
                 terminate_timeout: float) -> None:
        _text(run_id, 'run_id')
        if not isinstance(definition, RuntimeDefinition):
            raise TypeError('definition must be RuntimeDefinition')
        if not isinstance(store, ArtifactStore):
            raise TypeError('store must be ArtifactStore')
        _integer(max_concurrency, 'max_concurrency', minimum=1)
        _positive_number(terminate_timeout, 'terminate_timeout')

        self.run_id = run_id
        self._definition = definition
        self._store = store
        self._max_concurrency = max_concurrency
        self._terminate_timeout = terminate_timeout
        self._events: asyncio.PriorityQueue[tuple[int, int, _Event]] = asyncio.PriorityQueue()
        self._event_sequence = itertools.count()
        self._operation_event_slots = asyncio.Semaphore(_OPERATION_EVENT_CAPACITY)
        self._ready = asyncio.Event()
        self._condition = asyncio.Condition()
        self._serve_task: asyncio.Task[None] | None = None
        self._initialization_error: BaseException | None = None
        self._stopping = False
        self._closed = False

        self._status: RunStatus = 'created'
        self._error: RuntimeErrorInfo | None = None
        self._failure_pending: RuntimeErrorInfo | None = None
        self._artifacts = ArtifactSnapshot()
        self._retries: tuple[ArtifactRetryRequest, ...] = ()
        self._decision: PlanningResult | None = None
        self._active: dict[str, _ActiveExecution] = {}
        self._case_failures: tuple[CaseFailure, ...] = ()
        self._attempts: tuple[AttemptSnapshot, ...] = ()
        self._snapshot = RuntimeSnapshot(run_id)

    async def serve(self) -> None:
        if self._serve_task is not None:
            raise RuntimeError('run session is already serving')
        self._serve_task = asyncio.current_task()
        try:
            await self._initialize()
        except Exception as exc:
            self._initialization_error = exc
            self._ready.set()
            self._closed = True
            await self._notify()
            raise
        self._ready.set()

        try:
            while not self._stopping:
                _, _, event = await self._events.get()
                try:
                    try:
                        if isinstance(event, _Request):
                            await self._handle_request(event)
                        elif isinstance(event, _ExecutionEvent):
                            execution = self._active.get(event.attempt_id)
                            if execution is not None and execution.attempt.status == 'running':
                                try:
                                    await self._store.append_operation_event(
                                        self.run_id, event.attempt_id, event.event,
                                    )
                                except Exception as exc:
                                    _LOG.warning('operation event was dropped: %s', exc)
                        else:
                            await self._handle_done(event)
                    except Exception as exc:
                        if self._status == 'failed':
                            try:
                                await self._flush_failure()
                            except Exception:
                                pass
                            await self._terminate_failed_siblings()
                            await self._publish()
                        else:
                            await self._fail_running(exc)
                finally:
                    if isinstance(event, _ExecutionEvent):
                        self._operation_event_slots.release()
                    self._events.task_done()
        except BaseException:
            await asyncio.gather(
                *(execution.handle.terminate() for execution in self._active.values()),
                return_exceptions=True,
            )
            raise
        finally:
            self._closed = True
            error = RuntimeError('run session is closed')
            while not self._events.empty():
                _, _, event = self._events.get_nowait()
                if isinstance(event, _Request) and not event.reply.done():
                    event.reply.set_exception(error)
                if isinstance(event, _ExecutionEvent):
                    self._operation_event_slots.release()
                self._events.task_done()
            await self._notify()

    async def wait_ready(self) -> None:
        await self._ready.wait()
        if self._initialization_error is not None:
            raise self._initialization_error

    async def enter_running(self, expected: Literal['created', 'paused']) -> RuntimeSnapshot:
        return await self._request(self._enter_running, expected, flush_failure=True)

    async def pause(self) -> RuntimeSnapshot:
        return await self._request(self._pause, flush_failure=True)

    async def cancel(self) -> RuntimeSnapshot:
        return await self._request(self._cancel, flush_failure=True)

    async def release(self) -> RuntimeSnapshot:
        return await self._request(self._release, flush_failure=True)

    async def close(self) -> RuntimeSnapshot:
        if self._closed:
            return self._snapshot
        return await self._request(self._close, flush_failure=True)

    async def commit(self, commit: ArtifactCommit) -> RuntimeSnapshot:
        return await self._request(self._commit_artifacts, commit)

    async def recompute(self, artifact_keys: tuple[ArtifactKey, ...], operation_ids: tuple[str, ...],
                        case_ids: tuple[str, ...], request_id: str, *, failures_only: bool = False) -> RuntimeSnapshot:
        return await self._request(
            self._recompute,
            artifact_keys,
            operation_ids,
            case_ids,
            request_id,
            'failure' if failures_only else 'rerun',
            flush_failure=True,
        )

    async def submit_attempt_result(self, attempt_id: str, result: OperationResult) -> RuntimeSnapshot:
        return await self._request(self._submit_attempt_result, attempt_id, result)

    async def wait_for_status(self, statuses: str | tuple[str, ...], *, timeout: float = 10.0) -> RuntimeSnapshot:
        expected = (statuses,) if isinstance(statuses, str) else tuple(statuses)
        if not expected:
            raise DefinitionError('statuses must not be empty')
        async with asyncio.timeout(timeout):
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._snapshot.status in expected or self._closed
                )
        if self._snapshot.status not in expected:
            raise RuntimeError('run session closed before reaching requested status')
        return self._snapshot

    async def wait_until_settled(self, *, timeout: float = 10.0) -> RuntimeSnapshot:
        async with asyncio.timeout(timeout):
            async with self._condition:
                await self._condition.wait_for(
                    lambda: self._settled() or self._closed
                )
        if not self._settled():
            raise RuntimeError('run session closed before becoming settled')
        return self._snapshot

    async def _initialize(self) -> None:
        state, artifacts, attempts, retries = await self._store.inspect(
            self.run_id,
            self._definition.partition_set_ids,
        )
        active_attempts = tuple(
            attempt
            for attempt in attempts
            if attempt.status in {'scheduled', 'running', 'cancelling'}
        )
        if active_attempts:
            raise RuntimeError('artifact store contains unrecovered execution attempts')
        self._status = state.status
        self._error = state.error
        self._artifacts = artifacts
        self._retries = retries
        self._attempts = attempts
        self._decision = plan_next(self._definition, self._artifacts, self._retries)
        await self._refresh_snapshot_data()

        if self._status == 'running':
            await self._schedule()
        else:
            await self._publish()

    async def _refresh_snapshot_data(self, *, include_failures: bool = True) -> None:
        if self._decision is None:
            self._case_failures = ()
            return
        self._attempts = await self._store.attempts(self.run_id)
        if include_failures:
            self._case_failures = await _load_case_failures(
                self._store,
                self.run_id,
                self._decision.failure_refs,
            )

    async def _request(self, action: Callable[..., Awaitable[None]], /, *args: object, flush_failure: bool = False
                       ) -> RuntimeSnapshot:
        if self._closed or self._stopping:
            raise RuntimeError('run session is closed')
        reply: asyncio.Future[RuntimeSnapshot] = asyncio.get_running_loop().create_future()
        await self._enqueue(
            _Request(partial(action, *args), reply, flush_failure),
            _CONTROL_PRIORITY,
        )
        return await reply

    async def _enqueue(self, event: _Event, priority: int) -> None:
        if self._closed or self._stopping:
            raise RuntimeError('run session is closed')
        await self._events.put((priority, next(self._event_sequence), event))

    async def _handle_request(self, request: _Request) -> None:
        try:
            if request.flush_failure:
                await self._flush_failure()
            await request.action()
        except Exception as exc:
            reply_error: Exception = exc
            if request.flush_failure and self._failure_pending is not None:
                try:
                    await self._flush_failure()
                except Exception as persistence_error:
                    reply_error = ExceptionGroup(
                        'command and failure persistence both failed',
                        [exc, persistence_error],
                    )
            if not request.reply.done():
                request.reply.set_exception(reply_error)
        else:
            if not request.reply.done():
                request.reply.set_result(self._snapshot)

    async def _pause(self) -> None:
        if self._status == 'paused':
            return
        if self._status == 'running':
            await self._persist_status('pausing')
        elif self._status != 'pausing':
            raise DefinitionError(f'cannot pause run from {self._status}')
        await self._terminate(tuple(self._active.values()), final='paused')
        self._status = 'paused'
        await self._publish()

    async def _recompute(self, artifact_keys: tuple[ArtifactKey, ...], operation_ids: tuple[str, ...],
                         case_ids: tuple[str, ...], request_id: str, mode: Literal['rerun', 'failure']) -> None:
        if self._status not in {*_EDITABLE_STATUSES, 'failed'}:
            raise DefinitionError(f'cannot recompute from {self._status}')
        if not (artifact_keys or operation_ids or case_ids):
            raise DefinitionError('recompute requires an artifact, operation, or case scope')

        decision = await self._require_decision('recompute')
        operations = {operation.spec.op_id: operation for operation in self._definition.operations}
        unknown = tuple(operation_id for operation_id in operation_ids if operation_id not in operations)
        if unknown:
            raise DefinitionError(f'unknown recompute operations: {", ".join(unknown)}')
        selected = list(artifact_keys)
        selected_invocations = {self._retry_invocation(key) for key in selected}
        for key in sorted(decision.view.records, key=lambda item: (item.artifact_id, item.partition_key)):
            operation = self._definition.producer_by_artifact.get(key.artifact_id)
            if operation is None or operation.spec.op_id not in operation_ids:
                continue
            partition_key = key.partition_key if operation.spec.driver_input else ''
            if case_ids and partition_key not in case_ids:
                continue
            identity = operation.spec.op_id, partition_key
            if identity not in selected_invocations:
                selected.append(key)
                selected_invocations.add(identity)

        failure_records: dict[ArtifactRef, ArtifactRecord] = {}
        for failure in self._case_failures:
            identity = failure.operation_id, failure.case_id
            if case_ids and failure.case_id not in case_ids:
                continue
            if operation_ids and failure.operation_id not in operation_ids:
                continue
            if artifact_keys and not operation_ids and identity not in selected_invocations:
                continue
            for output_key in failure.output_keys:
                record = decision.view.records.get(failure_key(output_key))
                if record is not None and record.producer.startswith('runtime:failure:'):
                    failure_records[record.ref] = record
        failed_attempt_ids = tuple(
            attempt.attempt_id
            for attempt in self._attempts
            if self._status == 'failed'
            and attempt.status == 'failed'
            and not attempt.partition_key
            and (not operation_ids or attempt.operation_id in operation_ids)
        )
        entries = await self._retry_entries(tuple(selected), request_id)
        command_id = f'recompute:{request_id}'
        scope = repr((
            mode,
            tuple((key.artifact_id, key.partition_key) for key in artifact_keys),
            operation_ids,
            case_ids,
        ))
        applied = await self._store.replace_pending_retries(
            self.run_id,
            entries,
            command_id=command_id,
            producer=f'runtime:recompute:{mode}:{request_id}',
            scope=scope,
            resolved_failures=failure_records.values(),
            required_failure_cases=case_ids if mode == 'failure' else (),
            failed_attempt_ids=failed_attempt_ids if mode == 'failure' else (),
            require_failures=mode == 'failure',
        )
        if applied is None:
            await self._refresh_plan()
            await self._publish()
            return

        targets = tuple(
            execution
            for execution in self._active.values()
            if (
                (
                    execution.invocation.operation.spec.op_id,
                    execution.invocation.partition_key,
                ) in selected_invocations
                or (
                    execution.invocation.operation.spec.op_id in operation_ids
                    and (not case_ids or execution.invocation.partition_key in case_ids)
                )
                or (mode == 'failure' and execution.invocation.partition_key in case_ids)
            )
        )
        if targets:
            await self._terminate(targets)
        previous_status = self._status
        await self._refresh_plan()
        if previous_status == 'failed':
            await self._enter_running()
        else:
            await self._continue_after_change(previous_status, cancel_invalidated=True)

    async def _cancel(self) -> None:
        if self._status == 'cancelled':
            return
        if self._status == 'completed':
            raise DefinitionError('cannot cancel run from completed')
        if self._status != 'cancelling':
            await self._persist_status('cancelling')
        await self._terminate(tuple(self._active.values()), final='cancelled')
        self._status = 'cancelled'
        self._retries = ()
        await self._publish()

    async def _release(self) -> None:
        if self._status in {'pausing', 'cancelling'} or self._active or not self._settled():
            raise RuntimeError('cannot release a run while it is executing')
        self._stopping = True

    async def _close(self) -> None:
        if self._status in {'running', 'pausing'}:
            if self._status == 'running':
                await self._persist_status('pausing')
            await self._terminate(tuple(self._active.values()), final='interrupted')
            self._status = 'paused'
        elif self._status == 'cancelling':
            await self._terminate(tuple(self._active.values()), final='cancelled')
            self._status = 'cancelled'
            self._retries = ()
        elif self._active:
            await self._terminate(tuple(self._active.values()))
        await self._publish()
        self._stopping = True

    async def _enter_running(self, expected: RunStatus | None = None) -> None:
        if expected is not None and self._status != expected:
            action = 'start' if expected == 'created' else 'resume'
            raise DefinitionError(f'cannot {action} run from {self._status}')
        await self._refresh_plan()
        await self._persist_status('running')
        await self._schedule()

    async def _commit_artifacts(self, commit: ArtifactCommit) -> None:
        # Dataset applies (topic rename, material edits, …) must work after a
        # failed run so the user can fix inputs and continue without an explicit
        # retry first. Recompute already allows 'failed'; keep commits aligned.
        if self._status not in {*_EDITABLE_STATUSES, 'failed'}:
            raise DefinitionError(f'cannot commit artifact from {self._status}')
        self._definition.validate_commit(commit)
        previous_status = self._status
        result = await self._store.commit(self.run_id, commit)
        if result.status == 'stale':
            raise DefinitionError('artifact commit precondition is stale')

        await self._refresh_plan()
        await self._continue_after_change(previous_status, cancel_invalidated=True)

    async def _continue_after_change(self, previous_status: RunStatus, *, cancel_invalidated: bool = False) -> None:
        if cancel_invalidated:
            try:
                if self._decision is not None:
                    targets = tuple(
                        execution
                        for execution in self._active.values()
                        if not execution.invocation.is_current(
                            self._artifacts.records,
                            self._decision.view.records,
                            self._decision.view.partition_sets,
                        )
                    )
                    if targets:
                        await self._terminate(targets)
            except _TerminationFailure:
                await self._publish()
                return
        if previous_status == 'failed':
            await self._enter_running()
        elif previous_status == 'completed' and not isinstance(self._decision, PlanComplete):
            await self._persist_status('running')
            await self._schedule()
        elif previous_status == 'running':
            await self._schedule()
        else:
            await self._publish()

    async def _submit_attempt_result(self, attempt_id: str, result: OperationResult) -> None:
        execution = self._active.get(attempt_id)
        if execution is None:
            attempts = await self._store.attempts(self.run_id)
            attempt = next(
                (item for item in attempts if item.attempt_id == attempt_id),
                None,
            )
            if attempt is None:
                raise DefinitionError(f'attempt not found: {attempt_id}')
            if attempt.status == 'succeeded':
                await self._publish()
                return
            raise DefinitionError(
                f'attempt {attempt_id} no longer accepts results: {attempt.status}'
            )
        if execution.attempt.status != 'running':
            raise DefinitionError(
                f'attempt {attempt_id} no longer accepts results: '
                f'{execution.attempt.status}'
            )

        commit = execution.invocation.artifact_commit(result)
        self._definition.validate_commit(commit)
        committed = await self._store.commit(
            self.run_id,
            commit,
            attempt_id=attempt_id,
        )
        try:
            await execution.handle.terminate()
            await asyncio.gather(execution.waiter, return_exceptions=True)
        except Exception as exc:
            await self._fail_run(exc)
            await self._publish()
            raise
        finally:
            self._active.pop(attempt_id, None)

        await self._refresh_plan()
        await self._schedule()
        if committed.status != 'ok':
            raise DefinitionError(f'attempt {attempt_id} result is stale')

    async def _retry_entries(self, artifact_keys: tuple[ArtifactKey, ...], request_id: str
                             ) -> tuple[tuple[str, ArtifactKey, ArtifactRef], ...]:
        decision = await self._require_decision('retry')

        entries: list[tuple[str, ArtifactKey, ArtifactRef]] = []
        seen: set[tuple[str, str]] = set()
        for key in artifact_keys:
            operation = self._definition.producer_by_artifact.get(key.artifact_id)
            if operation is None:
                raise DefinitionError(f'artifact has no producer operation: {key}')
            current = decision.view.records.get(key)
            if current is None:
                raise DefinitionError(f'artifact is not currently effective: {key}')
            invocation = self._retry_invocation(key)
            if invocation in seen:
                raise DefinitionError('one invocation cannot have multiple retry targets')
            seen.add(invocation)
            child_id = (
                f'{request_id}:{key.artifact_id}:'
                f'{key.partition_key or "_"}:{current.ref.version}'
            )
            entries.append((child_id, key, current.ref))
        return tuple(entries)

    async def _require_decision(self, purpose: str) -> PlanningResult:
        if self._decision is None:
            await self._refresh_plan()
        if self._decision is None:
            raise RuntimeError(f'{purpose} planning state is unavailable')
        return self._decision

    def _retry_invocation(self, key: ArtifactKey) -> tuple[str, str]:
        operation = self._definition.producer_by_artifact.get(key.artifact_id)
        if operation is None:
            raise DefinitionError(f'artifact has no producer operation: {key}')
        return (
            operation.spec.op_id,
            key.partition_key if operation.spec.driver_input else '',
        )

    async def _refresh_plan(self) -> None:
        self._artifacts = await self._store.snapshot(
            self.run_id,
            self._definition.partition_set_ids,
        )
        self._retries = await self._store.retry_requests(self.run_id, pending_only=True)
        if self._retries:
            obsolete = obsolete_retries(self._definition, self._artifacts, self._retries)
            for request in obsolete:
                await self._store.cancel_retry(self.run_id, request.request_id)
            if obsolete:
                obsolete_ids = {request.request_id for request in obsolete}
                self._retries = tuple(
                    request
                    for request in self._retries
                    if request.request_id not in obsolete_ids
                )
        self._decision = plan_next(self._definition, self._artifacts, self._retries)
        await self._refresh_snapshot_data()

    async def _schedule(self) -> None:
        if self._status != 'running':
            await self._publish()
            return
        if self._decision is None:
            await self._refresh_plan()

        if isinstance(self._decision, PlanComplete):
            if not self._active:
                await self._persist_status('completed')
            await self._publish()
            return
        if not isinstance(self._decision, PlanReady):
            await self._publish()
            return

        candidates = tuple(self._launch_candidates(self._decision))
        if await self._pause_for_generation_plan_gate(candidates):
            await self._publish()
            return

        for invocation in candidates:
            await self._launch(invocation)
            if self._status == 'failed':
                break
        await self._publish()

    async def _pause_for_generation_plan_gate(
        self,
        candidates: tuple[OperationInvocation, ...],
    ) -> bool:
        if not any(
            invocation.operation.spec.op_id == 'dataset.qaplan_plan'
            for invocation in candidates
        ):
            return False
        if self._status == 'paused':
            return True
        if not isinstance(self._decision, PlanReady):
            return False

        from evo import artifacts as runtime_artifacts
        from evo.operations.dataset.qaplan_capacity import default_lane_distribution_exceeds_capacity

        if any(
            self._decision.view.records.get(key) is not None
            for key in (
                ArtifactKey.scalar(runtime_artifacts.DATASET_QAPLAN_PLAN),
                ArtifactKey.scalar(runtime_artifacts.EVAL_CASE_REQUESTS),
            )
        ):
            return False

        keys = (
            ArtifactKey.scalar(runtime_artifacts.DATASET_IMPORT_CASES_MANIFEST),
            ArtifactKey.scalar(runtime_artifacts.DATASET_TOPIC_MANIFEST),
            ArtifactKey.scalar(runtime_artifacts.DATASET_QAPLAN_PLAN_PARAMS),
        )
        records = tuple(self._decision.view.records.get(key) for key in keys)
        if any(record is None for record in records):
            return False
        refs = tuple(record.ref for record in records)
        values = await self._store.read_many(self.run_id, refs)
        import_manifest, topic_manifest, plan_params = (
            values[ref] for ref in refs
        )
        if not default_lane_distribution_exceeds_capacity(
            import_manifest,
            topic_manifest,
            plan_params,
        ):
            return False
        await self._pause()
        return True

    def _launch_candidates(self, decision: PlanReady) -> Iterator[OperationInvocation]:
        active_invocations = {
            execution.invocation.invocation_id
            for execution in self._active.values()
        }
        per_operation = Counter(
            execution.invocation.operation.spec.op_id
            for execution in self._active.values()
        )
        remaining = self._max_concurrency - len(self._active)
        for invocation in decision.invocations:
            operation_id = invocation.operation.spec.op_id
            if remaining <= 0:
                return
            if invocation.invocation_id in active_invocations:
                continue
            if per_operation[operation_id] >= invocation.operation.spec.max_concurrency:
                continue
            yield invocation
            remaining -= 1
            active_invocations.add(invocation.invocation_id)
            per_operation[operation_id] += 1

    async def _launch(self, invocation: OperationInvocation) -> None:
        try:
            values = await self._store.read_many(self.run_id, invocation.value_refs())
            attempt_id = uuid.uuid4().hex
            attempt = await self._store.create_attempt(
                self.run_id,
                attempt_id,
                invocation.invocation_id,
                invocation.operation.spec.op_id,
                invocation.partition_key,
                invocation.lineage_refs(),
                tuple(key for key in invocation.output_keys.values() if key is not None),
                retry_request_id=invocation.retry_request_id,
            )
        except Exception as exc:
            await self._fail_run(exc)
            await self._terminate_failed_siblings()
            return
        try:
            attempt = await self._store.set_attempt_status(
                self.run_id,
                attempt_id,
                'running',
            )
            context = OperationContext(
                self.run_id,
                invocation.invocation_id,
                invocation.partition_key,
                self._reporter(attempt_id),
            )
            handle = await start_execution(
                invocation,
                context,
                invocation.bind_values(values),
                terminate_timeout=self._terminate_timeout,
            )
        except Exception as exc:
            await self._fail_attempt(attempt, exc)
            if attempt.partition_key:
                await self._refresh_plan()
            else:
                await self._terminate_failed_siblings()
            return

        waiter = asyncio.create_task(
            self._wait_execution(attempt_id, handle, invocation.operation.spec.timeout),
            name=f'artifact-attempt:{attempt_id}',
        )
        self._active[attempt_id] = _ActiveExecution(invocation, attempt, handle, waiter)

    def _reporter(self, attempt_id: str) -> Callable[[OperationEvent], Awaitable[None]]:
        async def report(event: OperationEvent) -> None:
            await self._operation_event_slots.acquire()
            try:
                await self._enqueue(
                    _ExecutionEvent(attempt_id, event),
                    _OPERATION_EVENT_PRIORITY,
                )
            except BaseException:
                self._operation_event_slots.release()
                raise
        return report

    async def _wait_execution(self, attempt_id: str, handle: ExecutionHandle, timeout: float | None) -> None:
        result = None
        error = None
        try:
            async with asyncio.timeout(timeout):
                result = await handle.wait()
        except TimeoutError:
            try:
                await handle.terminate()
            except Exception as exc:
                error = exc
            else:
                error = OperationTimeoutError(
                    f'operation exceeded its {timeout:g}s timeout'
                )
        except (asyncio.CancelledError, Exception) as exc:
            error = exc
        try:
            await self._enqueue(
                _ExecutionDone(attempt_id, result, error),
                _COMPLETION_PRIORITY,
            )
        except RuntimeError:
            return

    async def _handle_done(self, event: _ExecutionDone) -> None:
        execution = self._active.get(event.attempt_id)
        if execution is None:
            return
        if execution.attempt.status == 'cancelling':
            return
        if event.error is not None:
            if isinstance(event.error, asyncio.CancelledError):
                error = OperationExecutionError('operation ended without a cancellation request')
            else:
                error = _as_exception(event.error)
            execution.attempt = await self._fail_attempt(execution.attempt, error)
            cleanup_error = (
                event.error
                if isinstance(event.error, ExecutionCleanupError)
                and event.error.cleanup_pending
                else None
            )
            await self._complete_failed_execution(execution, cleanup_error)
            return

        try:
            if event.result is None:
                raise OperationExecutionError('operation returned no result')
            commit = execution.invocation.artifact_commit(event.result)
            self._definition.validate_commit(commit)
            await self._store.commit(
                self.run_id,
                commit,
                attempt_id=event.attempt_id,
            )
        except Exception as exc:
            execution.attempt = await self._fail_attempt(execution.attempt, exc)
            await self._complete_failed_execution(execution, None)
            return

        self._active.pop(event.attempt_id, None)
        await self._refresh_plan()
        await self._schedule()

    async def _complete_failed_execution(self, execution: _ActiveExecution, cleanup_error: ExecutionCleanupError | None
                                         ) -> None:
        if cleanup_error is not None:
            await self._fail_run(cleanup_error)
            await self._terminate_failed_siblings()
            await self._publish()
            return
        self._active.pop(execution.attempt.attempt_id, None)
        if execution.attempt.partition_key:
            await self._refresh_plan()
            await self._schedule()
        else:
            await self._terminate_failed_siblings()
            await self._publish()

    async def _fail_attempt(self, attempt: AttemptSnapshot, error: Exception) -> AttemptSnapshot:
        info = RuntimeErrorInfo(type(error).__name__, str(error) or type(error).__name__)
        if attempt.partition_key:
            failed = await self._store.set_attempt_status(
                self.run_id,
                attempt.attempt_id,
                'failed',
                error=info,
            )
            failure = CaseFailure(
                failed.attempt_id,
                failed.invocation_id,
                failed.operation_id,
                failed.partition_key,
                info,
                failed.input_refs,
                failed.output_keys,
                failed.finished_at if failed.finished_at is not None else time.time(),
            )
            failure_heads = {
                key: await self._store.head(self.run_id, key)
                for key in (failure_key(output) for output in failed.output_keys)
            }
            result = await self._store.commit(
                self.run_id,
                ArtifactCommit(
                    f'case-failure:{failed.attempt_id}',
                    f'runtime:failure:{failed.attempt_id}',
                    tuple(
                        ArtifactDraft(key, failure, failed.input_refs)
                        for key in failure_heads
                    ),
                    {
                        key: None if head is None else head.ref
                        for key, head in failure_heads.items()
                    },
                ),
            )
            if result.status == 'stale':
                raise RuntimeError('case failure changed before it could be recorded')
            return failed
        failed = await self._store.fail_attempt_and_run(self.run_id, attempt.attempt_id, info)
        self._status = 'failed'
        self._error = info
        return failed

    async def _fail_run(self, error: Exception) -> None:
        info = RuntimeErrorInfo(type(error).__name__, str(error) or type(error).__name__)
        self._status = 'failed'
        self._error = info
        self._failure_pending = info
        await self._flush_failure()

    async def _flush_failure(self) -> None:
        info = self._failure_pending
        if info is None:
            return
        await self._store.set_run_state(self.run_id, 'failed', error=info)
        self._failure_pending = None

    async def _fail_running(self, error: Exception) -> None:
        persistence_error: Exception | None = None
        try:
            await self._fail_run(error)
        except Exception as exc:
            persistence_error = exc
        await self._terminate_failed_siblings()
        await self._publish()
        if persistence_error is not None:
            raise persistence_error

    async def _terminate_failed_siblings(self) -> None:
        siblings = tuple(self._active.values())
        if not siblings:
            return
        try:
            await self._terminate(siblings)
        except _TerminationFailure:
            return

    async def _terminate(self, executions: tuple[_ActiveExecution, ...], *,
                         final: Literal['paused', 'cancelled', 'interrupted'] | None = None) -> None:
        for execution in executions:
            if execution.attempt.status in {'scheduled', 'running'}:
                try:
                    execution.attempt = await self._store.set_attempt_status(
                        self.run_id,
                        execution.attempt.attempt_id,
                        'cancelling',
                    )
                except Exception:
                    pass
        results = await asyncio.gather(*(
            self._terminate_execution(execution, final)
            for execution in executions
        ))
        failures = [error for error in results if error is not None]
        if not failures:
            try:
                if final in {'paused', 'interrupted'}:
                    await self._store.finish_stopping(self.run_id, 'paused')
                elif final == 'cancelled':
                    await self._store.finish_stopping(self.run_id, 'cancelled')
            except Exception as exc:
                failures.append(exc)
        if failures:
            failure = _TerminationFailure(
                'operation cleanup did not reach a verified terminal state',
                failures,
            )
            if self._status != 'failed':
                try:
                    await self._fail_run(failure)
                except Exception as exc:
                    failure = _TerminationFailure(
                        'operation cleanup and failure persistence both failed',
                        [failure, exc],
                    )
            await self._publish()
            raise failure

    async def _terminate_execution(self, execution: _ActiveExecution,
                                   final: Literal['paused', 'cancelled', 'interrupted'] | None) -> Exception | None:
        try:
            await execution.handle.terminate()
        except asyncio.CancelledError as error:
            return _as_exception(error)
        except Exception as error:
            return error
        await asyncio.gather(execution.waiter, return_exceptions=True)
        try:
            if execution.attempt.status != 'failed':
                execution.attempt = await self._store.set_attempt_status(
                    self.run_id,
                    execution.attempt.attempt_id,
                    'interrupted' if final == 'interrupted' else 'cancelled',
                )
        except Exception as error:
            return error
        self._active.pop(execution.attempt.attempt_id, None)
        return None

    async def _persist_status(self, status: RunStatus) -> None:
        if status == 'failed':
            raise ValueError('failed status must be persisted through _fail_run')
        await self._store.set_run_state(self.run_id, status)
        self._status = status
        self._error = None
        self._failure_pending = None

    async def _publish(self) -> None:
        await self._refresh_snapshot_data(include_failures=False)
        if self._decision is None:
            raise RuntimeError('runtime snapshot requires a planning decision')
        self._snapshot = project_runtime_snapshot(
            self.run_id,
            self._status,
            self._error,
            self._definition,
            self._decision,
            self._attempts,
            self._case_failures,
            active_attempts=(
                execution.attempt for execution in self._active.values()
            ),
        )
        await self._notify()

    def _settled(self) -> bool:
        if self._snapshot.status in {'created', 'paused', 'cancelled', 'failed', 'completed'}:
            return not self._snapshot.active_attempts
        return (
            self._snapshot.status == 'running'
            and not self._snapshot.running
            and self._snapshot.ready_count == 0
            and not self._snapshot.active_attempts
        )

    async def _notify(self) -> None:
        async with self._condition:
            self._condition.notify_all()


async def _load_case_failures(store: ArtifactStore, run_id: str, refs: tuple[ArtifactRef, ...]
                              ) -> tuple[CaseFailure, ...]:
    values = await store.read_many(run_id, refs)
    failures: dict[str, CaseFailure] = {}
    for ref in refs:
        failure = values[ref]
        if not isinstance(failure, CaseFailure):
            raise DefinitionError('failure artifact must contain CaseFailure')
        failures.setdefault(failure.attempt_id, failure)
    return tuple(sorted(
        failures.values(),
        key=lambda failure: (
            failure.case_id,
            failure.operation_id,
            failure.attempt_id,
        ),
    ))


__all__ = ['RunSession']
