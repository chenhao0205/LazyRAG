from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, get_args

import aiosqlite

from .artifact import (
    ArtifactCommit,
    ArtifactDraft,
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    ArtifactSnapshot,
    PartitionSet,
)
from .errors import DefinitionError, _integer, _known, _string, _text, _tuple_of
from .state import (
    ArtifactRetryRequest,
    AttemptSnapshot,
    AttemptStatus,
    EventLevel,
    EventStatus,
    OperationEvent,
    RecordedOperationEvent,
    RunStatus,
    RuntimeErrorInfo,
)


_SCHEMA_VERSION = 4
_SCHEMA_TABLES = frozenset({
    'artifacts', 'attempts', 'commits', 'operation_events', 'retry_requests', 'runs',
})
_Row = Mapping[str, object]
_RUN_STATUSES = frozenset(get_args(RunStatus))
_ACTIVE_ATTEMPT_STATUSES = ('scheduled', 'running', 'cancelling')
_ATTEMPT_TRANSITIONS = {
    'scheduled': frozenset({'running', 'cancelling', 'cancelled', 'failed', 'interrupted'}),
    'running': frozenset({'cancelling', 'cancelled', 'failed', 'interrupted'}),
    'cancelling': frozenset({'cancelled', 'interrupted'}),
    'cancelled': frozenset(),
    'succeeded': frozenset(),
    'failed': frozenset(),
    'interrupted': frozenset(),
    'discarded': frozenset(),
}


@dataclass(frozen=True)
class StoredRunState:
    status: RunStatus
    error: RuntimeErrorInfo | None = None


@dataclass(frozen=True)
class CommitResult:
    status: Literal['ok', 'stale']
    refs: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True, slots=True)
class _PreparedCommit:
    run_id: str
    command: ArtifactCommit
    payloads: tuple[bytes, ...]
    request_hash: str


class ArtifactStore:
    def __init__(self, connection: aiosqlite.Connection) -> None:
        self._connection = connection
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(cls, root: str | Path) -> ArtifactStore:
        path = Path(root)
        path.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(path / 'artifact-runtime.sqlite3')
        try:
            connection.row_factory = aiosqlite.Row
            await connection.execute('PRAGMA foreign_keys = ON')
            await connection.execute('PRAGMA journal_mode = WAL')
            await connection.execute('PRAGMA synchronous = FULL')
            store = cls(connection)
            await store._create_schema()
            return store
        except BaseException:
            await connection.close()
            raise

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            await self._connection.close()
            self._closed = True

    async def create_run(self, run_id: str, initial_commit: ArtifactCommit | None = None) -> StoredRunState:
        _text(run_id, 'run_id')
        prepared = None
        if initial_commit is not None:
            if not isinstance(initial_commit, ArtifactCommit):
                raise TypeError('initial_commit must be ArtifactCommit or None')
            if initial_commit.producer.startswith('operation:'):
                raise DefinitionError('initial commit cannot be produced by an operation')
            prepared = await asyncio.to_thread(_prepare_commit, run_id, initial_commit)

        async with self._transaction():
            try:
                await self._connection.execute(
                    """
                    INSERT INTO runs(run_id, status, error_kind, error_message)
                    VALUES (?, 'created', '', '')
                    """,
                    (run_id,),
                )
            except aiosqlite.IntegrityError as exc:
                raise DefinitionError(f'run already exists: {run_id}') from exc

            if prepared is not None:
                snapshot = ArtifactSnapshot()
                if not await self._commit_is_current(run_id, prepared.command, snapshot):
                    raise DefinitionError('initial artifact commit precondition is stale')
                result = await self._apply_commit(prepared)
                await self._write_receipt(prepared, result.refs)
        return StoredRunState('created')

    async def commit(self, run_id: str, commit: ArtifactCommit, *, attempt_id: str | None = None) -> CommitResult:
        _text(run_id, 'run_id')
        if not isinstance(commit, ArtifactCommit):
            raise TypeError('commit must be ArtifactCommit')
        if attempt_id is None and commit.producer.startswith('operation:'):
            raise DefinitionError('operation commit requires attempt_id')
        if attempt_id is not None:
            _text(attempt_id, 'attempt_id')
        prepared = await asyncio.to_thread(_prepare_commit, run_id, commit)
        return await self._commit(prepared, attempt_id)

    async def snapshot(self, run_id: str, partition_set_ids: Iterable[str] = ()) -> ArtifactSnapshot:
        _text(run_id, 'run_id')
        ids = frozenset(partition_set_ids)
        for artifact_id in ids:
            _text(artifact_id, 'partition set artifact_id')
        async with self._lock:
            await self._require_run(run_id)
            rows = await self._head_rows(run_id)
        return await asyncio.to_thread(_snapshot_from_rows, rows, ids)

    async def read(self, run_id: str, ref: ArtifactRef) -> object:
        _text(run_id, 'run_id')
        if not isinstance(ref, ArtifactRef):
            raise TypeError('ref must be ArtifactRef')
        async with self._lock:
            await self._require_run(run_id)
            payload = await self._payload(run_id, ref)
        if payload is None:
            raise KeyError(ref)
        return await asyncio.to_thread(pickle.loads, payload)

    async def read_many(self, run_id: str, refs: Iterable[ArtifactRef]) -> Mapping[ArtifactRef, object]:
        _text(run_id, 'run_id')
        requested = _tuple_of(refs, ArtifactRef, 'refs must contain ArtifactRef values')

        payloads: dict[ArtifactRef, bytes] = {}
        async with self._lock:
            await self._require_run(run_id)
            for offset in range(0, len(requested), 250):
                chunk = requested[offset:offset + 250]
                if not chunk:
                    continue
                placeholders = ','.join('(?, ?, ?)' for _ in chunk)
                parameters: list[object] = [run_id]
                for ref in chunk:
                    parameters.extend((
                        ref.key.artifact_id,
                        ref.key.partition_key,
                        ref.version,
                    ))
                rows = await self._connection.execute_fetchall(
                    f"""
                    SELECT artifact_id, partition_key, version, payload
                    FROM artifacts
                    WHERE run_id = ?
                      AND (artifact_id, partition_key, version) IN ({placeholders})
                    """,
                    parameters,
                )
                for row in rows:
                    ref = ArtifactRef(
                        ArtifactKey(row['artifact_id'], row['partition_key']),
                        row['version'],
                    )
                    payloads[ref] = row['payload']

        missing = next((ref for ref in requested if ref not in payloads), None)
        if missing is not None:
            raise DefinitionError(f'input artifact is missing: {missing}')
        return await asyncio.to_thread(
            lambda: {ref: pickle.loads(payloads[ref]) for ref in requested}
        )

    async def record(self, run_id: str, ref: ArtifactRef) -> ArtifactRecord | None:
        _text(run_id, 'run_id')
        if not isinstance(ref, ArtifactRef):
            raise TypeError('ref must be ArtifactRef')
        rows = await self._rows(
            run_id,
            """
            SELECT producer, input_refs_json FROM artifacts
            WHERE run_id = ? AND artifact_id = ? AND partition_key = ? AND version = ?
            """,
            (ref.key.artifact_id, ref.key.partition_key, ref.version),
        )
        row = rows[0] if rows else None
        return None if row is None else _record_from_row(ref, row)

    async def head(self, run_id: str, key: ArtifactKey) -> ArtifactRecord | None:
        _text(run_id, 'run_id')
        if not isinstance(key, ArtifactKey):
            raise TypeError('key must be ArtifactKey')
        rows = await self._rows(
            run_id,
            """
            SELECT version, producer, input_refs_json FROM artifacts
            WHERE run_id = ? AND artifact_id = ? AND partition_key = ?
            ORDER BY version DESC LIMIT 1
            """,
            (key.artifact_id, key.partition_key),
        )
        return None if not rows else _record_from_row(ArtifactRef(key, rows[0]['version']), rows[0])

    async def history(self, run_id: str, key: ArtifactKey) -> tuple[ArtifactRecord, ...]:
        _text(run_id, 'run_id')
        if not isinstance(key, ArtifactKey):
            raise TypeError('key must be ArtifactKey')
        rows = await self._rows(
            run_id,
            """
            SELECT version, producer, input_refs_json FROM artifacts
            WHERE run_id = ? AND artifact_id = ? AND partition_key = ?
            ORDER BY version
            """,
            (key.artifact_id, key.partition_key),
        )
        return tuple(
            _record_from_row(ArtifactRef(key, row['version']), row)
            for row in rows
        )

    async def artifact_records(self, run_id: str) -> tuple[ArtifactRecord, ...]:
        _text(run_id, 'run_id')
        rows = await self._rows(
            run_id,
            """
            SELECT artifact_id, partition_key, version, producer, input_refs_json
            FROM artifacts WHERE run_id = ?
            ORDER BY artifact_id, partition_key, version
            """,
        )
        return tuple(
            _record_from_row(
                ArtifactRef(
                    ArtifactKey(row['artifact_id'], row['partition_key']),
                    row['version'],
                ),
                row,
            )
            for row in rows
        )

    async def replace_pending_retries(
        self, run_id: str, entries: Iterable[tuple[str, ArtifactKey, ArtifactRef]], *, command_id: str, producer: str,
        scope: str, resolved_failures: Iterable[ArtifactRecord] = (), required_failure_cases: Iterable[str] = (),
        failed_attempt_ids: Iterable[str] = (), require_failures: bool = False,
    ) -> tuple[ArtifactRetryRequest, ...] | None:
        _text(run_id, 'run_id')
        _text(command_id, 'recompute command_id')
        _text(producer, 'recompute producer')
        _text(scope, 'recompute scope')
        requests = _validated_retry_entries(entries)
        failures = _tuple_of(
            resolved_failures,
            ArtifactRecord,
            'resolved_failures must contain ArtifactRecord values',
        )
        required_cases = tuple(dict.fromkeys(required_failure_cases))
        for case_id in required_cases:
            _text(case_id, 'required failure case_id')
        failed_attempts = tuple(dict.fromkeys(failed_attempt_ids))
        for attempt_id in failed_attempts:
            _text(attempt_id, 'failed attempt_id')
        fingerprint = hashlib.sha256(f'{run_id}\0{producer}\0{scope}'.encode()).hexdigest()
        marker = None
        if failures:
            marker = await asyncio.to_thread(
                _prepare_commit,
                run_id,
                ArtifactCommit(
                    command_id,
                    producer,
                    tuple(
                        ArtifactDraft(
                            record.ref.key,
                            {'command_id': command_id, 'failure_ref': record.ref},
                            record.input_refs,
                        )
                        for record in failures
                    ),
                    {record.ref.key: record.ref for record in failures},
                ),
            )

        results: list[ArtifactRetryRequest] = []
        async with self._transaction():
            await self._require_run(run_id)
            receipt = await self._fetchone(
                'SELECT request_hash FROM commits WHERE run_id = ? AND commit_id = ?',
                (run_id, command_id),
            )
            if receipt is not None:
                if receipt['request_hash'] != fingerprint:
                    raise DefinitionError(f'command id reused with different request: {command_id}')
                return None
            failed_cases = {record.ref.key.partition_key for record in failures}
            missing_cases = tuple(case_id for case_id in required_cases if case_id not in failed_cases)
            if missing_cases:
                raise DefinitionError(f'cases have no active failure: {", ".join(missing_cases)}')
            if failed_attempts:
                placeholders = ','.join('?' for _ in failed_attempts)
                rows = await self._connection.execute_fetchall(
                    f"""
                    SELECT attempt_id FROM attempts
                    WHERE run_id = ? AND status = 'failed'
                      AND attempt_id IN ({placeholders})
                    """,
                    (run_id, *failed_attempts),
                )
                if {row['attempt_id'] for row in rows} != set(failed_attempts):
                    raise DefinitionError('failed attempts changed before retry was applied')
            if require_failures and not (failures or failed_attempts):
                raise DefinitionError('recompute scope has no active case failure')
            rows = await self._connection.execute_fetchall(
                """
                SELECT * FROM retry_requests
                WHERE run_id = ? AND status = 'pending'
                """,
                (run_id,),
            )
            pending = {
                ArtifactKey(row['artifact_id'], row['partition_key']): row
                for row in rows
            }
            retained_ids: set[str] = set()
            missing: list[tuple[str, ArtifactKey, ArtifactRef]] = []
            for request_id, artifact_key, base_ref in requests:
                current = await self._head_ref(run_id, artifact_key)
                if current != base_ref:
                    raise DefinitionError(
                        'retry target is no longer the current artifact version'
                    )
                row = pending.get(artifact_key)
                if row is not None and int(row['base_version']) == base_ref.version:
                    retained_ids.add(str(row['request_id']))
                    results.append(_retry_request(row))
                else:
                    missing.append((request_id, artifact_key, base_ref))

            if retained_ids:
                placeholders = ','.join('?' for _ in retained_ids)
                await self._connection.execute(
                    f"""
                    UPDATE retry_requests SET status = 'cancelled'
                    WHERE run_id = ? AND status = 'pending'
                      AND request_id NOT IN ({placeholders})
                    """,
                    (run_id, *sorted(retained_ids)),
                )
            else:
                await self._cancel_pending_retries(run_id)

            for request_id, artifact_key, base_ref in missing:
                existing = await self._retry_row(run_id, request_id)
                if existing is not None:
                    raise DefinitionError(f'retry request id reused: {request_id}')
                results.append(await self._insert_retry(
                    run_id,
                    request_id,
                    artifact_key,
                    base_ref,
                ))
            refs: tuple[ArtifactRef, ...] = ()
            if marker is not None:
                rows = await self._head_rows(run_id)
                snapshot = await asyncio.to_thread(_snapshot_from_rows, rows, frozenset())
                if not await self._commit_is_current(run_id, marker.command, snapshot):
                    raise DefinitionError('case failures changed before retry was applied')
                refs = (await self._apply_commit(marker)).refs
            await self._connection.execute(
                'INSERT INTO commits(run_id, commit_id, request_hash, refs_json) VALUES (?, ?, ?, ?)',
                (run_id, command_id, fingerprint, _refs_json(refs)),
            )
        return tuple(results)

    async def _insert_retry(self, run_id: str, request_id: str, artifact_key: ArtifactKey, base_ref: ArtifactRef
                            ) -> ArtifactRetryRequest:
        created_at = time.time()
        await self._connection.execute(
            """
            INSERT INTO retry_requests(
              run_id, request_id, artifact_id, partition_key, base_version,
              status, created_at, result_version
            ) VALUES (?, ?, ?, ?, ?, 'pending', ?, NULL)
            """,
            (
                run_id, request_id, artifact_key.artifact_id,
                artifact_key.partition_key, base_ref.version, created_at,
            ),
        )
        return ArtifactRetryRequest(
            request_id,
            artifact_key,
            base_ref,
            'pending',
            created_at,
        )

    async def retry_requests(self, run_id: str, *, pending_only: bool = False) -> tuple[ArtifactRetryRequest, ...]:
        _text(run_id, 'run_id')
        statement = 'SELECT * FROM retry_requests WHERE run_id = ?'
        if pending_only:
            statement += " AND status = 'pending'"
        statement += ' ORDER BY created_at, request_id'
        rows = await self._rows(run_id, statement)
        return tuple(_retry_request(row) for row in rows)

    async def cancel_retry(self, run_id: str, request_id: str) -> ArtifactRetryRequest:
        _text(run_id, 'run_id')
        _text(request_id, 'retry request_id')
        async with self._transaction():
            row = await self._retry_row(run_id, request_id)
            if row is None:
                raise DefinitionError(f'retry request not found: {request_id}')
            if row['status'] == 'pending':
                await self._connection.execute(
                    """
                    UPDATE retry_requests SET status = 'cancelled'
                    WHERE run_id = ? AND request_id = ? AND status = 'pending'
                    """,
                    (run_id, request_id),
                )
                row = dict(row)
                row['status'] = 'cancelled'
        return _retry_request(row)

    async def set_run_state(self, run_id: str, status: RunStatus, *, error: RuntimeErrorInfo | None = None) -> None:
        _text(run_id, 'run_id')
        state = StoredRunState(status, error)
        error_kind = '' if state.error is None else state.error.kind
        error_message = '' if state.error is None else state.error.message
        async with self._transaction():
            cursor = await self._connection.execute(
                """
                UPDATE runs SET status = ?, error_kind = ?, error_message = ?
                WHERE run_id = ?
                """,
                (status, error_kind, error_message, run_id),
            )
            if cursor.rowcount != 1:
                raise DefinitionError(f'run not found: {run_id}')

    async def run_state(self, run_id: str) -> StoredRunState | None:
        _text(run_id, 'run_id')
        async with self._lock:
            self._require_open()
            row = await self._fetchone(
                'SELECT status, error_kind, error_message FROM runs WHERE run_id = ?',
                (run_id,),
            )
        return None if row is None else _run_state(row)

    async def run_ids(self) -> tuple[str, ...]:
        return await self._list_run_ids('SELECT run_id FROM runs ORDER BY run_id')

    async def active_run_ids(self) -> tuple[str, ...]:
        return await self._list_run_ids(
            """
            SELECT run_id FROM runs
            WHERE status IN ('running', 'pausing', 'paused', 'cancelling')
            ORDER BY run_id
            """
        )

    async def inspect(
        self, run_id: str, partition_set_ids: Iterable[str] = (),
    ) -> tuple[StoredRunState, ArtifactSnapshot, tuple[AttemptSnapshot, ...], tuple[ArtifactRetryRequest, ...],]:
        _text(run_id, 'run_id')
        ids = frozenset(partition_set_ids)
        async with self._lock:
            state_row = await self._fetchone(
                'SELECT status, error_kind, error_message FROM runs WHERE run_id = ?',
                (run_id,),
            )
            if state_row is None:
                raise DefinitionError(f'run not found: {run_id}')
            artifact_rows = await self._head_rows(run_id)
            attempt_rows = await self._connection.execute_fetchall(
                'SELECT * FROM attempts WHERE run_id = ? ORDER BY created_at, attempt_id',
                (run_id,),
            )
            retry_rows = await self._connection.execute_fetchall(
                """
                SELECT * FROM retry_requests
                WHERE run_id = ? AND status = 'pending'
                ORDER BY created_at, request_id
                """,
                (run_id,),
            )
        snapshot = await asyncio.to_thread(_snapshot_from_rows, artifact_rows, ids)
        return (
            _run_state(state_row),
            snapshot,
            tuple(_attempt_snapshot(row) for row in attempt_rows),
            tuple(_retry_request(row) for row in retry_rows),
        )

    async def create_attempt(self, run_id: str, attempt_id: str, invocation_id: str, operation_id: str,
                             partition_key: str, input_refs: Iterable[ArtifactRef] = (),
                             output_keys: Iterable[ArtifactKey] = (), *, retry_request_id: str = '') -> AttemptSnapshot:
        for value, name in (
            (run_id, 'run_id'),
            (attempt_id, 'attempt_id'),
            (invocation_id, 'invocation_id'),
            (operation_id, 'operation_id'),
        ):
            _text(value, name)
        _string(partition_key, 'partition_key')
        _string(retry_request_id, 'retry_request_id')
        inputs = tuple(input_refs)
        outputs = tuple(output_keys)
        created_at = time.time()
        snapshot = AttemptSnapshot(
            attempt_id,
            invocation_id,
            operation_id,
            partition_key,
            'scheduled',
            created_at,
            input_refs=inputs,
            output_keys=outputs,
            retry_request_id=retry_request_id,
        )

        async with self._transaction():
            await self._require_run(run_id)
            if retry_request_id:
                row = await self._retry_row(run_id, retry_request_id)
                if row is None or row['status'] != 'pending':
                    raise DefinitionError(f'pending retry request not found: {retry_request_id}')
                retry_key = ArtifactKey(row['artifact_id'], row['partition_key'])
                if retry_key not in outputs:
                    raise DefinitionError('retry target must be an invocation output')
            try:
                await self._connection.execute(
                    """
                    INSERT INTO attempts(
                      run_id, attempt_id, invocation_id, operation_id, partition_key,
                      retry_request_id, status, created_at, started_at, finished_at,
                      error_kind, error_message, input_refs_json, output_keys_json
                    ) VALUES (?, ?, ?, ?, ?, ?, 'scheduled', ?, NULL, NULL, '', '', ?, ?)
                    """,
                    (
                        run_id, attempt_id, invocation_id, operation_id, partition_key,
                        retry_request_id or None, created_at, _refs_json(inputs),
                        json.dumps([_key_data(key) for key in outputs], separators=(',', ':')),
                    ),
                )
            except aiosqlite.IntegrityError as exc:
                raise DefinitionError(f'attempt conflicts with existing execution: {attempt_id}') from exc
        return snapshot

    async def set_attempt_status(self, run_id: str, attempt_id: str, status: AttemptStatus, *,
                                 error: RuntimeErrorInfo | None = None) -> AttemptSnapshot:
        _text(run_id, 'run_id')
        _text(attempt_id, 'attempt_id')
        if status in {'succeeded', 'discarded'}:
            raise DefinitionError(f'{status} attempt status is owned by artifact commit')
        async with self._transaction():
            row = await self._attempt_row(run_id, attempt_id)
            if row is None:
                raise DefinitionError(f'attempt not found: {attempt_id}')
            current = row['status']
            if current == status:
                snapshot = _attempt_snapshot(row)
                if error is not None and error != snapshot.error:
                    raise DefinitionError('attempt terminal state cannot change its error')
                return snapshot
            updated = _attempt_status_values(row, status, error)
            await self._update_attempt(run_id, attempt_id, updated)
        return _attempt_snapshot(updated)

    async def fail_attempt_and_run(self, run_id: str, attempt_id: str, error: RuntimeErrorInfo) -> AttemptSnapshot:
        if not isinstance(error, RuntimeErrorInfo):
            raise TypeError('error must be RuntimeErrorInfo')
        async with self._transaction():
            row = await self._attempt_row(run_id, attempt_id)
            if row is None:
                raise DefinitionError(f'attempt not found: {attempt_id}')
            updated = _attempt_status_values(row, 'failed', error)
            await self._update_attempt(run_id, attempt_id, updated)
            cursor = await self._connection.execute(
                """
                UPDATE runs SET status = 'failed', error_kind = ?, error_message = ?
                WHERE run_id = ?
                """,
                (error.kind, error.message, run_id),
            )
            if cursor.rowcount != 1:
                raise DefinitionError(f'run not found: {run_id}')
        return _attempt_snapshot(updated)

    async def finish_stopping(self, run_id: str, status: Literal['paused', 'cancelled']) -> None:
        async with self._transaction():
            await self._require_run(run_id)
            await self._connection.execute(
                """
                UPDATE attempts SET status = 'cancelled', finished_at = ?
                WHERE run_id = ? AND status IN ('scheduled', 'running', 'cancelling')
                """,
                (time.time(), run_id),
            )
            await self._connection.execute(
                "UPDATE runs SET status = ?, error_kind = '', error_message = '' WHERE run_id = ?",
                (status, run_id),
            )
            if status == 'cancelled':
                await self._cancel_pending_retries(run_id)

    async def attempts(self, run_id: str) -> tuple[AttemptSnapshot, ...]:
        _text(run_id, 'run_id')
        rows = await self._rows(
            run_id, 'SELECT * FROM attempts WHERE run_id = ? ORDER BY created_at, attempt_id'
        )
        return tuple(_attempt_snapshot(row) for row in rows)

    async def append_operation_event(self, run_id: str, attempt_id: str, event: OperationEvent
                                     ) -> RecordedOperationEvent:
        _text(run_id, 'run_id')
        _text(attempt_id, 'attempt_id')
        if not isinstance(event, OperationEvent):
            raise TypeError('event must be OperationEvent')
        created_at = time.time()
        data_json = json.dumps(dict(event.data), ensure_ascii=False, sort_keys=True, allow_nan=False)
        async with self._transaction():
            row = await self._attempt_row(run_id, attempt_id)
            if row is None:
                raise DefinitionError(f'attempt not found: {attempt_id}')
            if row['status'] != 'running':
                raise DefinitionError('operation events can only be appended to a running attempt')
            sequence_row = await self._fetchone(
                """
                SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence
                FROM operation_events WHERE run_id = ?
                """,
                (run_id,),
            )
            assert sequence_row is not None
            sequence = int(sequence_row['sequence'])
            await self._connection.execute(
                """
                INSERT INTO operation_events(
                  run_id, sequence, attempt_id, event_type, level, status, message,
                  current_value, total_value, data_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, sequence, attempt_id, event.event_type, event.level, event.status,
                    event.message, event.current, event.total, data_json, created_at,
                ),
            )
        return RecordedOperationEvent(
            run_id,
            attempt_id,
            cast(str, row['invocation_id']),
            cast(str, row['operation_id']),
            cast(str, row['partition_key']),
            sequence,
            event,
            created_at,
        )

    async def operation_events(self, run_id: str, *, attempt_id: str = '', operation_id: str = '',
                               operation_ids: Iterable[str] = (),
                               partition_key: str | None = None, event_type: str = '', level: EventLevel | None = None,
                               status: EventStatus | None = None, after: int = 0, limit: int | None = None
                               ) -> tuple[RecordedOperationEvent, ...]:
        _text(run_id, 'run_id')
        _integer(after, 'operation event cursor')
        if limit is not None:
            _integer(limit, 'operation event limit', minimum=1)
        filters = ['event.run_id = ?', 'event.sequence > ?']
        parameters: list[object] = [after]
        if attempt_id:
            _text(attempt_id, 'attempt_id')
            filters.append('event.attempt_id = ?')
            parameters.append(attempt_id)
        selected_operations = tuple(dict.fromkeys(operation_ids))
        for selected in selected_operations:
            _text(selected, 'operation_id')
        if operation_id and selected_operations:
            raise DefinitionError('operation_id and operation_ids are mutually exclusive')
        if operation_id:
            _text(operation_id, 'operation_id')
            filters.append('attempt.operation_id = ?')
            parameters.append(operation_id)
        elif selected_operations:
            placeholders = ','.join('?' for _ in selected_operations)
            filters.append(f'attempt.operation_id IN ({placeholders})')
            parameters.extend(selected_operations)
        if partition_key is not None:
            _string(partition_key, 'partition_key')
            filters.append('attempt.partition_key = ?')
            parameters.append(partition_key)
        if event_type:
            _text(event_type, 'event_type')
            filters.append('event.event_type = ?')
            parameters.append(event_type)
        if level is not None:
            _known(level, get_args(EventLevel), 'operation event level')
            filters.append('event.level = ?')
            parameters.append(level)
        if status is not None:
            _known(status, get_args(EventStatus), 'operation event status')
            filters.append('event.status = ?')
            parameters.append(status)
        statement = f"""
            SELECT event.*, attempt.invocation_id, attempt.operation_id, attempt.partition_key
            FROM operation_events AS event
            JOIN attempts AS attempt
              ON attempt.run_id = event.run_id AND attempt.attempt_id = event.attempt_id
            WHERE {' AND '.join(filters)}
            ORDER BY event.sequence
        """
        if limit is not None:
            statement += ' LIMIT ?'
            parameters.append(limit)
        rows = await self._rows(run_id, statement, tuple(parameters))
        return tuple(
            RecordedOperationEvent(
                cast(str, row['run_id']),
                cast(str, row['attempt_id']),
                cast(str, row['invocation_id']),
                cast(str, row['operation_id']),
                cast(str, row['partition_key']),
                cast(int, row['sequence']),
                OperationEvent(
                    cast(str, row['event_type']),
                    cast(EventLevel, row['level']),
                    cast(str | None, row['status']),
                    cast(str, row['message']),
                    json.loads(cast(str, row['data_json'])),
                    cast(int | None, row['current_value']),
                    cast(int | None, row['total_value']),
                ),
                cast(float, row['created_at']),
            )
            for row in rows
        )

    async def recover_runs(self) -> tuple[str, ...]:
        recovered: list[str] = []
        async with self._transaction():
            rows = await self._connection.execute_fetchall(
                """
                SELECT run_id, status FROM runs AS run
                WHERE status IN ('running', 'pausing', 'cancelling')
                   OR (
                     status = 'failed' AND EXISTS (
                       SELECT 1 FROM attempts AS attempt
                       WHERE attempt.run_id = run.run_id
                         AND attempt.status IN ('scheduled', 'running', 'cancelling')
                     )
                   )
                ORDER BY run_id
                """
            )
            now = time.time()
            for row in rows:
                run_id = row['run_id']
                cancelling = row['status'] == 'cancelling'
                attempt_status = 'cancelled' if cancelling else 'interrupted'
                await self._connection.execute(
                    """
                    UPDATE attempts SET status = ?, finished_at = ?
                    WHERE run_id = ? AND status IN ('scheduled', 'running', 'cancelling')
                    """,
                    (attempt_status, now, run_id),
                )
                if row['status'] != 'failed':
                    run_status = 'cancelled' if cancelling else 'paused'
                    await self._connection.execute(
                        "UPDATE runs SET status = ?, error_kind = '', error_message = '' WHERE run_id = ?",
                        (run_status, run_id),
                    )
                    if cancelling:
                        await self._cancel_pending_retries(run_id)
                recovered.append(run_id)
        return tuple(recovered)

    async def delete_run(self, run_id: str) -> None:
        _text(run_id, 'run_id')
        async with self._transaction():
            cursor = await self._connection.execute('DELETE FROM runs WHERE run_id = ?', (run_id,))
            if cursor.rowcount != 1:
                raise DefinitionError(f'run not found: {run_id}')

    async def _commit(self, prepared: _PreparedCommit, attempt_id: str | None) -> CommitResult:
        async with self._transaction():
            await self._require_run(prepared.run_id)
            attempt = None
            if attempt_id is not None:
                attempt = await self._attempt_row(prepared.run_id, attempt_id)
                if attempt is None:
                    raise DefinitionError(f'attempt not found: {attempt_id}')
                _validate_attempt_commit(attempt, prepared.command)

            replay = await self._replay(prepared)
            if replay is not None:
                if attempt is not None:
                    if attempt['status'] == 'running':
                        await self._finish_attempt_commit(prepared.run_id, attempt, replay)
                    elif attempt['status'] != 'succeeded':
                        raise DefinitionError(
                            f'replayed commit conflicts with attempt state: '
                            f'{attempt["attempt_id"]} is {attempt["status"]}'
                        )
                return replay
            if attempt is not None and attempt['status'] != 'running':
                return CommitResult('stale')

            if attempt is not None and attempt['retry_request_id']:
                request_id = cast(str, attempt['retry_request_id'])
                row = await self._retry_row(prepared.run_id, request_id)
                if row is None or row['status'] != 'pending':
                    raise DefinitionError(f'pending retry request not found: {request_id}')
                key = ArtifactKey(cast(str, row['artifact_id']), cast(str, row['partition_key']))
                base = ArtifactRef(key, cast(int, row['base_version']))
                if prepared.command.expected_heads.get(key) != base:
                    raise DefinitionError('retry commit must compare against its requested base version')

            retry_conflict = await self._pending_retry_conflict(prepared, attempt)
            if retry_conflict is not None:
                if attempt is None:
                    raise DefinitionError(
                        f'artifact has pending retry {retry_conflict}; '
                        'cancel the run or wait for it'
                    )
                result = CommitResult('stale')
            else:
                rows = await self._head_rows(prepared.run_id)
                snapshot = await asyncio.to_thread(_snapshot_from_rows, rows, frozenset())
                if not await self._commit_is_current(prepared.run_id, prepared.command, snapshot):
                    result = CommitResult('stale')
                else:
                    result = await self._apply_commit(prepared)
                    await self._write_receipt(prepared, result.refs)

            if attempt is not None:
                await self._finish_attempt_commit(prepared.run_id, attempt, result)
            return result

    async def _commit_is_current(self, run_id: str, commit: ArtifactCommit, snapshot: ArtifactSnapshot) -> bool:
        for key, expected in commit.expected_heads.items():
            current = snapshot.records.get(key)
            if (None if current is None else current.ref) != expected:
                return False

        effective = snapshot.effective_records()
        if any(
            effective.get(ref.key) is None or effective[ref.key].ref != ref
            for write in commit.writes
            for ref in write.input_refs
        ):
            return False

        new_partition_sets = {
            write.key: write.value
            for write in commit.writes
            if isinstance(write.value, PartitionSet)
        }
        for guard in commit.partition_guards:
            partitions = new_partition_sets.get(guard.partition_set_key)
            if partitions is None:
                record = effective.get(guard.partition_set_key)
                if record is None:
                    return False
                payload = await self._payload(run_id, record.ref)
                if payload is None:
                    return False
                partitions = await asyncio.to_thread(pickle.loads, payload)
            if not isinstance(partitions, PartitionSet) or guard.partition_key not in partitions:
                return False
        return True

    async def _apply_commit(self, prepared: _PreparedCommit) -> CommitResult:
        records: list[ArtifactRecord] = []
        for write, payload in zip(prepared.command.writes, prepared.payloads, strict=True):
            current = await self._head_ref(prepared.run_id, write.key)
            ref = ArtifactRef(write.key, 1 if current is None else current.version + 1)
            record = ArtifactRecord(ref, prepared.command.producer, write.input_refs)
            await self._connection.execute(
                """
                INSERT INTO artifacts(
                  run_id, artifact_id, partition_key, version,
                  producer, input_refs_json, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prepared.run_id, ref.key.artifact_id, ref.key.partition_key,
                    ref.version, record.producer, _refs_json(record.input_refs), payload,
                ),
            )
            records.append(record)
        return CommitResult('ok', tuple(record.ref for record in records))

    async def _write_receipt(self, prepared: _PreparedCommit, refs: tuple[ArtifactRef, ...]) -> None:
        await self._connection.execute(
            """
            INSERT INTO commits(run_id, commit_id, request_hash, refs_json)
            VALUES (?, ?, ?, ?)
            """,
            (prepared.run_id, prepared.command.commit_id, prepared.request_hash, _refs_json(refs)),
        )

    async def _replay(self, prepared: _PreparedCommit) -> CommitResult | None:
        rows = await self._connection.execute_fetchall(
            'SELECT request_hash, refs_json FROM commits WHERE run_id = ? AND commit_id = ?',
            (prepared.run_id, prepared.command.commit_id),
        )
        row = rows[0] if rows else None
        if row is None:
            return None
        if row['request_hash'] != prepared.request_hash:
            raise DefinitionError(f'commit id reused with different request: {prepared.command.commit_id}')
        return CommitResult('ok', _refs_from_json(row['refs_json']))

    async def _pending_retry_conflict(self, prepared: _PreparedCommit, attempt: _Row | None) -> str | None:
        allowed = '' if attempt is None else cast(str | None, attempt['retry_request_id']) or ''
        for write in prepared.command.writes:
            row = await self._fetchone(
                """
                SELECT request_id FROM retry_requests
                WHERE run_id = ? AND artifact_id = ? AND partition_key = ? AND status = 'pending'
                """,
                (prepared.run_id, write.key.artifact_id, write.key.partition_key),
            )
            if row is not None and row['request_id'] != allowed:
                return cast(str, row['request_id'])
        return None

    async def _finish_attempt_commit(self, run_id: str, attempt: _Row, result: CommitResult) -> None:
        status = 'succeeded' if result.status == 'ok' else 'discarded'
        cursor = await self._connection.execute(
            """
            UPDATE attempts SET status = ?, finished_at = ?
            WHERE run_id = ? AND attempt_id = ? AND status = 'running'
            """,
            (status, time.time(), run_id, attempt['attempt_id']),
        )
        if cursor.rowcount != 1:
            raise DefinitionError(f'attempt is no longer running: {attempt["attempt_id"]}')

        request_id = cast(str | None, attempt['retry_request_id']) or ''
        if result.status == 'ok' and request_id:
            row = await self._retry_row(run_id, request_id)
            if row is None or row['status'] != 'pending':
                raise DefinitionError(f'pending retry request not found: {request_id}')
            key = ArtifactKey(row['artifact_id'], row['partition_key'])
            ref = next((ref for ref in result.refs if ref.key == key), None)
            if ref is None:
                raise DefinitionError('retry operation did not write its target artifact')
            await self._connection.execute(
                """
                UPDATE retry_requests SET status = 'fulfilled', result_version = ?
                WHERE run_id = ? AND request_id = ? AND status = 'pending'
                """,
                (ref.version, run_id, request_id),
            )

    async def _cancel_pending_retries(self, run_id: str) -> None:
        await self._connection.execute(
            "UPDATE retry_requests SET status = 'cancelled' "
            "WHERE run_id = ? AND status = 'pending'",
            (run_id,),
        )

    async def _head_rows(self, run_id: str) -> list[aiosqlite.Row]:
        rows = await self._connection.execute_fetchall(
            """
            WITH heads AS (
              SELECT artifact_id, partition_key, MAX(version) AS version
              FROM artifacts WHERE run_id = ? GROUP BY artifact_id, partition_key
            )
            SELECT a.artifact_id, a.partition_key, a.version,
                   a.producer, a.input_refs_json, a.payload
            FROM artifacts a JOIN heads h
              ON h.artifact_id = a.artifact_id
             AND h.partition_key = a.partition_key
             AND h.version = a.version
            WHERE a.run_id = ?
            """,
            (run_id, run_id),
        )
        return list(rows)

    async def _rows(self, run_id: str, statement: str, parameters: Sequence[object] = ()) -> list[aiosqlite.Row]:
        async with self._lock:
            await self._require_run(run_id)
            rows = await self._connection.execute_fetchall(statement, (run_id, *parameters))
            return list(rows)

    async def _list_run_ids(self, statement: str) -> tuple[str, ...]:
        async with self._lock:
            self._require_open()
            rows = await self._connection.execute_fetchall(statement)
            return tuple(row['run_id'] for row in rows)

    async def _fetchone(self, statement: str, parameters: Sequence[object] = ()) -> aiosqlite.Row | None:
        cursor = await self._connection.execute(statement, parameters)
        return await cursor.fetchone()

    async def _head_ref(self, run_id: str, key: ArtifactKey) -> ArtifactRef | None:
        row = await self._fetchone(
            """
            SELECT MAX(version) AS version FROM artifacts
            WHERE run_id = ? AND artifact_id = ? AND partition_key = ?
            """,
            (run_id, key.artifact_id, key.partition_key),
        )
        assert row is not None
        return None if row['version'] is None else ArtifactRef(key, row['version'])

    async def _payload(self, run_id: str, ref: ArtifactRef) -> bytes | None:
        row = await self._fetchone(
            """
            SELECT payload FROM artifacts
            WHERE run_id = ? AND artifact_id = ? AND partition_key = ? AND version = ?
            """,
            (run_id, ref.key.artifact_id, ref.key.partition_key, ref.version),
        )
        return None if row is None else row['payload']

    async def _attempt_row(self, run_id: str, attempt_id: str) -> aiosqlite.Row | None:
        return await self._fetchone(
            'SELECT * FROM attempts WHERE run_id = ? AND attempt_id = ?',
            (run_id, attempt_id),
        )

    async def _retry_row(self, run_id: str, request_id: str) -> aiosqlite.Row | None:
        return await self._fetchone(
            'SELECT * FROM retry_requests WHERE run_id = ? AND request_id = ?',
            (run_id, request_id),
        )

    async def _update_attempt(self, run_id: str, attempt_id: str, values: _Row) -> None:
        await self._connection.execute(
            """
            UPDATE attempts SET status = ?, started_at = ?, finished_at = ?,
                                error_kind = ?, error_message = ?
            WHERE run_id = ? AND attempt_id = ?
            """,
            (
                values['status'], values['started_at'], values['finished_at'],
                values['error_kind'], values['error_message'], run_id, attempt_id,
            ),
        )

    async def _require_run(self, run_id: str) -> None:
        self._require_open()
        if await self._fetchone('SELECT 1 FROM runs WHERE run_id = ?', (run_id,)) is None:
            raise DefinitionError(f'run not found: {run_id}')

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError('artifact store is closed')

    async def _create_schema(self) -> None:
        row = await self._fetchone('PRAGMA user_version')
        assert row is not None
        version = int(row[0])
        if version == _SCHEMA_VERSION:
            await self._validate_schema()
            return
        if version != 0:
            raise DefinitionError(
                f'unsupported artifact store schema version: {version}; delete and recreate it'
            )
        row = await self._fetchone(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
        )
        if row is not None:
            raise DefinitionError('unversioned artifact store; delete and recreate it')

        await self._connection.executescript(
            f"""
            BEGIN IMMEDIATE;
            CREATE TABLE runs(
              run_id TEXT PRIMARY KEY,
              status TEXT NOT NULL CHECK(status IN (
                'created', 'running', 'pausing', 'paused',
                'cancelling', 'cancelled', 'failed', 'completed'
              )),
              error_kind TEXT NOT NULL,
              error_message TEXT NOT NULL,
              CHECK(
                (status = 'failed' AND trim(error_kind) != '' AND trim(error_message) != '')
                OR (status != 'failed' AND error_kind = '' AND error_message = '')
              )
            );
            CREATE TABLE artifacts(
              run_id TEXT NOT NULL,
              artifact_id TEXT NOT NULL,
              partition_key TEXT NOT NULL,
              version INTEGER NOT NULL CHECK(version > 0),
              producer TEXT NOT NULL,
              input_refs_json TEXT NOT NULL,
              payload BLOB NOT NULL,
              PRIMARY KEY(run_id, artifact_id, partition_key, version),
              FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            CREATE TABLE commits(
              run_id TEXT NOT NULL,
              commit_id TEXT NOT NULL,
              request_hash TEXT NOT NULL,
              refs_json TEXT NOT NULL,
              PRIMARY KEY(run_id, commit_id),
              FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            CREATE TABLE retry_requests(
              run_id TEXT NOT NULL,
              request_id TEXT NOT NULL,
              artifact_id TEXT NOT NULL,
              partition_key TEXT NOT NULL,
              base_version INTEGER NOT NULL CHECK(base_version > 0),
              status TEXT NOT NULL CHECK(status IN ('pending', 'fulfilled', 'cancelled')),
              created_at REAL NOT NULL,
              result_version INTEGER,
              PRIMARY KEY(run_id, request_id),
              FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
              FOREIGN KEY(run_id, artifact_id, partition_key, base_version)
                REFERENCES artifacts(run_id, artifact_id, partition_key, version),
              FOREIGN KEY(run_id, artifact_id, partition_key, result_version)
                REFERENCES artifacts(run_id, artifact_id, partition_key, version),
              CHECK(
                (status = 'fulfilled' AND result_version IS NOT NULL AND result_version > base_version)
                OR (status != 'fulfilled' AND result_version IS NULL)
              )
            );
            CREATE UNIQUE INDEX pending_retry_by_artifact
              ON retry_requests(run_id, artifact_id, partition_key)
              WHERE status = 'pending';
            CREATE TABLE attempts(
              run_id TEXT NOT NULL,
              attempt_id TEXT NOT NULL,
              invocation_id TEXT NOT NULL,
              operation_id TEXT NOT NULL,
              partition_key TEXT NOT NULL,
              retry_request_id TEXT,
              status TEXT NOT NULL CHECK(status IN (
                'scheduled', 'running', 'cancelling', 'cancelled',
                'succeeded', 'failed', 'interrupted', 'discarded'
              )),
              created_at REAL NOT NULL,
              started_at REAL,
              finished_at REAL,
              error_kind TEXT NOT NULL,
              error_message TEXT NOT NULL,
              input_refs_json TEXT NOT NULL,
              output_keys_json TEXT NOT NULL,
              PRIMARY KEY(run_id, attempt_id),
              FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE,
              FOREIGN KEY(run_id, retry_request_id)
                REFERENCES retry_requests(run_id, request_id),
              CHECK(
                (status = 'failed' AND trim(error_kind) != '' AND trim(error_message) != '')
                OR (status != 'failed' AND error_kind = '' AND error_message = '')
              ),
              CHECK(
                (status IN ('scheduled', 'running', 'cancelling') AND finished_at IS NULL)
                OR (status IN ('cancelled', 'succeeded', 'failed', 'interrupted', 'discarded')
                    AND finished_at IS NOT NULL)
              ),
              CHECK(status NOT IN ('running', 'succeeded', 'discarded')
                    OR started_at IS NOT NULL)
            );
            CREATE UNIQUE INDEX active_attempt_by_invocation
              ON attempts(run_id, invocation_id)
              WHERE status IN ('scheduled', 'running', 'cancelling');
            CREATE TABLE operation_events(
              run_id TEXT NOT NULL,
              sequence INTEGER NOT NULL CHECK(sequence > 0),
              attempt_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              level TEXT NOT NULL CHECK(level IN ('debug', 'info', 'warning', 'error')),
              status TEXT CHECK(status IS NULL OR status IN ('started', 'running', 'completed', 'failed', 'skipped')),
              message TEXT NOT NULL,
              current_value INTEGER,
              total_value INTEGER,
              data_json TEXT NOT NULL,
              created_at REAL NOT NULL,
              PRIMARY KEY(run_id, sequence),
              FOREIGN KEY(run_id, attempt_id)
                REFERENCES attempts(run_id, attempt_id) ON DELETE CASCADE,
              CHECK(current_value IS NULL OR current_value >= 0),
              CHECK(total_value IS NULL OR total_value >= 0),
              CHECK(current_value IS NULL OR total_value IS NULL OR current_value <= total_value)
            );
            CREATE INDEX operation_events_by_attempt ON operation_events(run_id, attempt_id, sequence);
            PRAGMA user_version = {_SCHEMA_VERSION};
            COMMIT;
            """
        )

    async def _validate_schema(self) -> None:
        rows = await self._connection.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = frozenset(row[0] for row in rows)
        if tables != _SCHEMA_TABLES:
            raise DefinitionError('invalid artifact store schema; delete and recreate it')
        if await self._fetchone('PRAGMA foreign_key_check') is not None:
            raise DefinitionError('artifact store contains invalid references')

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        async with self._lock:
            self._require_open()
            try:
                await self._connection.execute('BEGIN IMMEDIATE')
                yield
                await self._connection.commit()
            except BaseException:
                rollback = asyncio.create_task(self._connection.rollback())
                while not rollback.done():
                    try:
                        await asyncio.shield(rollback)
                    except asyncio.CancelledError:
                        continue
                await rollback
                raise


def _validated_retry_entries(entries: Iterable[tuple[str, ArtifactKey, ArtifactRef]]
                             ) -> tuple[tuple[str, ArtifactKey, ArtifactRef], ...]:
    requests = tuple(entries)
    request_ids: set[str] = set()
    artifact_keys: set[ArtifactKey] = set()
    for request_id, artifact_key, base_ref in requests:
        _text(request_id, 'retry request_id')
        if not isinstance(artifact_key, ArtifactKey):
            raise TypeError('artifact_key must be ArtifactKey')
        if not isinstance(base_ref, ArtifactRef) or base_ref.key != artifact_key:
            raise DefinitionError('base_ref must identify artifact_key')
        if request_id in request_ids:
            raise DefinitionError(f'duplicate retry request id: {request_id}')
        if artifact_key in artifact_keys:
            raise DefinitionError(f'duplicate retry artifact: {artifact_key}')
        request_ids.add(request_id)
        artifact_keys.add(artifact_key)
    return requests


def _run_state(row: _Row) -> StoredRunState:
    status = cast(RunStatus, row['status'])
    error = (
        RuntimeErrorInfo(cast(str, row['error_kind']), cast(str, row['error_message']))
        if status == 'failed' else None
    )
    return StoredRunState(status, error)


def _record_from_row(ref: ArtifactRef, row: _Row) -> ArtifactRecord:
    return ArtifactRecord(ref, cast(str, row['producer']), _refs_from_json(cast(str, row['input_refs_json'])))


def _snapshot_from_rows(rows: Iterable[_Row], partition_set_ids: frozenset[str]) -> ArtifactSnapshot:
    records: dict[ArtifactKey, ArtifactRecord] = {}
    partition_sets: dict[ArtifactKey, PartitionSet] = {}
    for row in rows:
        key = ArtifactKey(cast(str, row['artifact_id']), cast(str, row['partition_key']))
        ref = ArtifactRef(key, cast(int, row['version']))
        records[key] = _record_from_row(ref, row)
        if key.artifact_id in partition_set_ids and not key.partition_key:
            value = pickle.loads(cast(bytes, row['payload']))
            if not isinstance(value, PartitionSet):
                raise DefinitionError(f'{key.artifact_id} must contain a PartitionSet value')
            partition_sets[key] = value
    return ArtifactSnapshot(records, partition_sets)


def _attempt_status_values(row: _Row, status: AttemptStatus, error: RuntimeErrorInfo | None) -> dict[str, object]:
    current = cast(str, row['status'])
    if current != status and status not in _ATTEMPT_TRANSITIONS.get(current, frozenset()):
        raise DefinitionError(f'cannot transition attempt from {current} to {status}')
    if status == 'failed' and error is None:
        raise DefinitionError('failed attempt requires error details')
    if status != 'failed' and error is not None:
        raise DefinitionError('attempt error is only valid for failed status')
    values = dict(row)
    now = time.time()
    if status == 'running' and values['started_at'] is None:
        values['started_at'] = now
    if status in {'cancelled', 'succeeded', 'failed', 'interrupted', 'discarded'}:
        values['finished_at'] = now
    values.update({
        'status': status,
        'error_kind': '' if error is None else error.kind,
        'error_message': '' if error is None else error.message,
    })
    return values


def _validate_attempt_commit(attempt: _Row, commit: ArtifactCommit) -> None:
    attempt_id = attempt['attempt_id']
    if attempt['invocation_id'] != commit.commit_id:
        raise DefinitionError(f'attempt {attempt_id} does not belong to commit {commit.commit_id}')
    expected_producer = f'operation:{attempt["operation_id"]}'
    if commit.producer != expected_producer:
        raise DefinitionError(f'attempt {attempt_id} requires producer {expected_producer}')
    input_refs = _refs_from_json(cast(str, attempt['input_refs_json']))
    if any(write.input_refs != input_refs for write in commit.writes):
        raise DefinitionError(f'attempt {attempt_id} input refs do not match commit lineage')
    outputs = {
        ArtifactKey(item[0], item[1])
        for item in json.loads(cast(str, attempt['output_keys_json']))
    }
    if not outputs.issubset(commit.output_keys):
        raise DefinitionError(f'attempt {attempt_id} declared outputs are missing from commit')
    partition_key = cast(str, attempt['partition_key'])
    if partition_key and not any(guard.partition_key == partition_key for guard in commit.partition_guards):
        raise DefinitionError(f'attempt {attempt_id} partition guard does not match commit')


def _attempt_snapshot(row: _Row) -> AttemptSnapshot:
    status = cast(AttemptStatus, row['status'])
    error = (
        RuntimeErrorInfo(cast(str, row['error_kind']), cast(str, row['error_message']))
        if status == 'failed' else None
    )
    output_keys = tuple(
        ArtifactKey(item[0], item[1])
        for item in json.loads(cast(str, row['output_keys_json']))
    )
    return AttemptSnapshot(
        cast(str, row['attempt_id']),
        cast(str, row['invocation_id']),
        cast(str, row['operation_id']),
        cast(str, row['partition_key']),
        status,
        cast(float, row['created_at']),
        cast(float | None, row['started_at']),
        cast(float | None, row['finished_at']),
        error,
        _refs_from_json(cast(str, row['input_refs_json'])),
        output_keys,
        cast(str | None, row['retry_request_id']) or '',
    )


def _retry_request(row: _Row) -> ArtifactRetryRequest:
    key = ArtifactKey(cast(str, row['artifact_id']), cast(str, row['partition_key']))
    result_version = cast(int | None, row['result_version'])
    return ArtifactRetryRequest(
        cast(str, row['request_id']),
        key,
        ArtifactRef(key, cast(int, row['base_version'])),
        cast(Literal['pending', 'fulfilled', 'cancelled'], row['status']),
        cast(float, row['created_at']),
        None if result_version is None else ArtifactRef(key, result_version),
    )


def _prepare_commit(run_id: str, commit: ArtifactCommit) -> _PreparedCommit:
    payloads = tuple(
        pickle.dumps(write.value, protocol=pickle.HIGHEST_PROTOCOL)
        for write in commit.writes
    )
    return _PreparedCommit(run_id, commit, payloads, _commit_fingerprint(run_id, commit, payloads))


def _commit_fingerprint(run_id: str, commit: ArtifactCommit, payloads: tuple[bytes, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(run_id.encode())
    digest.update(commit.producer.encode())
    for write, payload in zip(commit.writes, payloads, strict=True):
        digest.update(json.dumps(_key_data(write.key), separators=(',', ':')).encode())
        digest.update(_refs_json(write.input_refs).encode())
        digest.update(payload)
    guards = [
        [_key_data(guard.partition_set_key), guard.partition_key]
        for guard in sorted(commit.partition_guards)
    ]
    digest.update(json.dumps(guards, separators=(',', ':')).encode())
    return digest.hexdigest()


def _key_data(key: ArtifactKey) -> list[str]:
    return [key.artifact_id, key.partition_key]


def _ref_data(ref: ArtifactRef | None) -> list[object] | None:
    return None if ref is None else [ref.key.artifact_id, ref.key.partition_key, ref.version]


def _refs_json(refs: Iterable[ArtifactRef]) -> str:
    return json.dumps([_ref_data(ref) for ref in refs], separators=(',', ':'))


def _refs_from_json(value: str) -> tuple[ArtifactRef, ...]:
    return tuple(
        ArtifactRef(ArtifactKey(item[0], item[1]), item[2])
        for item in json.loads(value)
    )


__all__ = ['ArtifactStore', 'CommitResult', 'StoredRunState']
