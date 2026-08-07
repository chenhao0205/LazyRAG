from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, get_args

from .artifact import ArtifactKey, ArtifactRecord, ArtifactRef, PartitionSet
from .errors import (
    DefinitionError,
    _integer,
    _known,
    _number,
    _string,
    _text,
    _tuple_of,
    _unique,
)


RunStatus = Literal[
    'created',
    'running',
    'pausing',
    'paused',
    'cancelling',
    'cancelled',
    'failed',
    'completed',
]

AttemptStatus = Literal[
    'scheduled',
    'running',
    'cancelling',
    'cancelled',
    'succeeded',
    'failed',
    'interrupted',
    'discarded',
]

RetryStatus = Literal['pending', 'fulfilled', 'cancelled']
CaseOperationStatus = Literal['pending', 'running', 'succeeded', 'failed']
CaseStatus = Literal['pending', 'running', 'completed', 'failed']
EventLevel = Literal['debug', 'info', 'warning', 'error']
EventStatus = Literal['started', 'running', 'completed', 'failed', 'skipped']

_EVENT_COLLECTION_LIMIT = 50
_EVENT_DATA_LIMIT = 16 * 1024
_EVENT_DEPTH_LIMIT = 4
_EVENT_TYPE_LIMIT = 256
_SECRET_KEY = re.compile(r'(api[_-]?key|token|secret|password|authorization|llm_config)', re.I)
_SECRET_VALUE = re.compile(
    r'(?i)\b(authorization|api[_-]?key|token|secret|password)\b\s*[:=]\s*(?:bearer\s+)?[^\s,;)\]}]+'
)
_BEARER = re.compile(r'(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}')
_URL_SECRET = re.compile(r'(?i)([?&](?:api[_-]?key|token|secret|password)=)[^&#\s]+')


@dataclass(frozen=True)
class RunConfiguration:
    values: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = dict(self.values)
        for key in values:
            _text(key, 'run configuration key')
        try:
            json.dumps(values, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise DefinitionError('run configuration must be JSON-serializable') from exc
        object.__setattr__(self, 'values', values)


@dataclass(frozen=True)
class InvocationSnapshot:
    invocation_id: str
    operation_id: str
    partition_key: str = ''


@dataclass(frozen=True)
class RuntimeErrorInfo:
    kind: str
    message: str

    def __post_init__(self) -> None:
        _text(self.kind, 'runtime error kind')
        _text(self.message, 'runtime error message')


@dataclass(frozen=True)
class OperationEvent:
    event_type: str
    level: EventLevel = 'info'
    status: EventStatus | None = None
    message: str = ''
    data: Mapping[str, object] = field(default_factory=dict)
    current: int | None = None
    total: int | None = None

    def __post_init__(self) -> None:
        _text(self.event_type, 'operation event_type')
        if len(self.event_type.encode()) > _EVENT_TYPE_LIMIT:
            raise DefinitionError('operation event_type must not exceed 256 bytes')
        _known(self.level, get_args(EventLevel), 'operation event level')
        if self.status is not None:
            _known(self.status, get_args(EventStatus), 'operation event status')
        _string(self.message, 'operation event message')
        for name, value in (('current', self.current), ('total', self.total)):
            if value is not None:
                _integer(value, f'operation event {name}')
        if self.current is not None and self.total is not None and self.current > self.total:
            raise DefinitionError('operation event current cannot exceed total')
        message = _event_value(self.message)
        data = _event_value(dict(self.data))
        assert isinstance(message, str) and isinstance(data, dict)
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if len(encoded.encode()) > _EVENT_DATA_LIMIT:
            summary = encoded.encode()[:_EVENT_DATA_LIMIT].decode('utf-8', 'ignore')
            data = {'truncated': True, 'summary': summary}
        object.__setattr__(self, 'message', message)
        object.__setattr__(self, 'data', MappingProxyType(data))


@dataclass(frozen=True)
class AttemptSnapshot:
    attempt_id: str
    invocation_id: str
    operation_id: str
    partition_key: str
    status: AttemptStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: RuntimeErrorInfo | None = None
    input_refs: tuple[ArtifactRef, ...] = ()
    output_keys: tuple[ArtifactKey, ...] = ()
    retry_request_id: str = ''

    def __post_init__(self) -> None:
        _text(self.attempt_id, 'attempt_id')
        _text(self.invocation_id, 'invocation_id')
        _text(self.operation_id, 'operation_id')
        _string(self.partition_key, 'partition_key')
        _known(self.status, get_args(AttemptStatus), 'attempt status')
        for name, value in (
            ('created_at', self.created_at),
            ('started_at', self.started_at),
            ('finished_at', self.finished_at),
        ):
            _number(value, name, optional=True)
        if self.status == 'failed' and self.error is None:
            raise DefinitionError('failed attempt requires error details')
        if self.status != 'failed' and self.error is not None:
            raise DefinitionError('attempt error details are only valid for failed status')
        input_refs = _tuple_of(self.input_refs, ArtifactRef,
                               'attempt input_refs must contain ArtifactRef values')
        output_keys = _tuple_of(self.output_keys, ArtifactKey,
                                'attempt output_keys must contain ArtifactKey values')
        _string(self.retry_request_id, 'retry_request_id')
        object.__setattr__(self, 'input_refs', input_refs)
        object.__setattr__(self, 'output_keys', output_keys)


@dataclass(frozen=True)
class ArtifactRetryRequest:
    request_id: str
    artifact_key: ArtifactKey
    base_ref: ArtifactRef
    status: RetryStatus
    created_at: float
    result_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        _text(self.request_id, 'retry request_id')
        if not isinstance(self.artifact_key, ArtifactKey):
            raise TypeError('retry artifact_key must be ArtifactKey')
        if not isinstance(self.base_ref, ArtifactRef):
            raise TypeError('retry base_ref must be ArtifactRef')
        if self.base_ref.key != self.artifact_key:
            raise DefinitionError('retry base_ref must identify artifact_key')
        _known(self.status, get_args(RetryStatus), 'retry status')
        _number(self.created_at, 'retry created_at')
        if self.status == 'fulfilled':
            if not isinstance(self.result_ref, ArtifactRef):
                raise DefinitionError('fulfilled retry requires result_ref')
            if self.result_ref.key != self.artifact_key:
                raise DefinitionError('retry result_ref must identify artifact_key')
            if self.result_ref.version <= self.base_ref.version:
                raise DefinitionError('retry result_ref must be newer than base_ref')
        elif self.result_ref is not None:
            raise DefinitionError('only fulfilled retry can contain result_ref')


@dataclass(frozen=True)
class RecordedOperationEvent:
    run_id: str
    attempt_id: str
    invocation_id: str
    operation_id: str
    partition_key: str
    sequence: int
    event: OperationEvent
    created_at: float

    def __post_init__(self) -> None:
        _text(self.run_id, 'operation event run_id')
        _text(self.attempt_id, 'attempt_id')
        _text(self.invocation_id, 'operation event invocation_id')
        _text(self.operation_id, 'operation event operation_id')
        _string(self.partition_key, 'operation event partition_key')
        _integer(self.sequence, 'operation event sequence', minimum=1)
        if not isinstance(self.event, OperationEvent):
            raise TypeError('event must be OperationEvent')
        _number(self.created_at, 'created_at')


@dataclass(frozen=True)
class CaseFailure:
    attempt_id: str
    invocation_id: str
    operation_id: str
    case_id: str
    error: RuntimeErrorInfo
    input_refs: tuple[ArtifactRef, ...]
    output_keys: tuple[ArtifactKey, ...]
    failed_at: float

    def __post_init__(self) -> None:
        _text(self.attempt_id, 'case failure attempt_id')
        _text(self.invocation_id, 'case failure invocation_id')
        _text(self.operation_id, 'case failure operation_id')
        _text(self.case_id, 'case failure case_id')
        if not isinstance(self.error, RuntimeErrorInfo):
            raise TypeError('case failure error must be RuntimeErrorInfo')
        inputs = _tuple_of(self.input_refs, ArtifactRef,
                           'case failure input_refs must contain ArtifactRef values')
        outputs = _tuple_of(self.output_keys, ArtifactKey,
                            'case failure output_keys must contain ArtifactKey values',
                            nonempty=True)
        if any(key.partition_key != self.case_id for key in outputs):
            raise DefinitionError('case failure output keys must identify its case')
        _number(self.failed_at, 'case failure failed_at')
        object.__setattr__(self, 'input_refs', inputs)
        object.__setattr__(self, 'output_keys', outputs)


@dataclass(frozen=True)
class RuntimeProgress:
    total: int = 0
    completed: int = 0
    running: int = 0
    failed: int = 0
    pending: int = 0
    percentage: float = 0.0
    case_total: int = 0
    case_completed: int = 0
    case_running: int = 0
    case_failed: int = 0
    case_pending: int = 0

    def __post_init__(self) -> None:
        counts = (
            self.total,
            self.completed,
            self.running,
            self.failed,
            self.pending,
            self.case_total,
            self.case_completed,
            self.case_running,
            self.case_failed,
            self.case_pending,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in counts
        ):
            raise DefinitionError('runtime progress counts must be non-negative integers')
        if self.completed + self.running + self.failed + self.pending != self.total:
            raise DefinitionError('runtime progress operation counts must sum to total')
        if (
            self.case_completed
            + self.case_running
            + self.case_failed
            + self.case_pending
            != self.case_total
        ):
            raise DefinitionError('runtime progress case counts must sum to case_total')
        if (
            not isinstance(self.percentage, (int, float))
            or isinstance(self.percentage, bool)
            or not 0 <= self.percentage <= 100
        ):
            raise DefinitionError('runtime progress percentage must be between 0 and 100')


@dataclass(frozen=True)
class CaseOperationSnapshot:
    operation_id: str
    status: CaseOperationStatus
    output_refs: tuple[ArtifactRef, ...] = ()
    latest_attempt_id: str = ''
    retry_count: int = 0
    latest_event: RecordedOperationEvent | None = None
    error: RuntimeErrorInfo | None = None

    def __post_init__(self) -> None:
        _text(self.operation_id, 'case operation_id')
        _known(self.status, get_args(CaseOperationStatus), 'case operation status')
        outputs = _tuple_of(self.output_refs, ArtifactRef,
                            'case operation output_refs must contain ArtifactRef values')
        _string(self.latest_attempt_id, 'case latest_attempt_id')
        _integer(self.retry_count, 'case retry_count')
        if self.latest_event is not None and not isinstance(self.latest_event, RecordedOperationEvent):
            raise TypeError('case latest_event must be RecordedOperationEvent or None')
        if self.error is not None and not isinstance(self.error, RuntimeErrorInfo):
            raise TypeError('case operation error must be RuntimeErrorInfo or None')
        if self.status == 'failed' and self.error is None:
            raise DefinitionError('failed case operation requires error details')
        object.__setattr__(self, 'output_refs', outputs)


@dataclass(frozen=True)
class CaseSnapshot:
    run_id: str
    case_id: str
    display_index: int
    status: CaseStatus
    operations: tuple[CaseOperationSnapshot, ...]
    artifacts: Mapping[ArtifactKey, ArtifactRef] = field(default_factory=dict)
    failures: tuple[CaseFailure, ...] = ()
    artifact_records: tuple[ArtifactRecord, ...] = ()
    attempts: tuple[AttemptSnapshot, ...] = ()
    operation_events: tuple[RecordedOperationEvent, ...] = ()
    retries: tuple[ArtifactRetryRequest, ...] = ()

    def __post_init__(self) -> None:
        _text(self.run_id, 'case snapshot run_id')
        _text(self.case_id, 'case snapshot case_id')
        _integer(self.display_index, 'case display_index', minimum=1)
        _known(self.status, get_args(CaseStatus), 'case status')
        operations = _tuple_of(self.operations, CaseOperationSnapshot,
                               'case operations must contain CaseOperationSnapshot values')
        _unique(operations, 'case operation ids must be unique',
                key=lambda item: item.operation_id)
        artifacts = dict(self.artifacts)
        for key, ref in artifacts.items():
            if (
                not isinstance(key, ArtifactKey)
                or not isinstance(ref, ArtifactRef)
                or ref.key != key
                or key.partition_key != self.case_id
            ):
                raise DefinitionError('case artifacts must identify this case')
        failures = _tuple_of(self.failures, CaseFailure,
                             'case failures must contain CaseFailure values')
        if any(failure.case_id != self.case_id for failure in failures):
            raise DefinitionError('case failures must identify this case')
        records = _tuple_of(self.artifact_records, ArtifactRecord,
                            'case artifact_records must contain ArtifactRecord values')
        attempts = _tuple_of(self.attempts, AttemptSnapshot,
                             'case attempts must contain AttemptSnapshot values')
        events = _tuple_of(self.operation_events, RecordedOperationEvent,
                           'case operation_events must contain RecordedOperationEvent values')
        retries = _tuple_of(self.retries, ArtifactRetryRequest,
                            'case retries must contain ArtifactRetryRequest values')
        if any(record.ref.key.partition_key != self.case_id for record in records):
            raise DefinitionError('case artifact_records must identify this case')
        if any(attempt.partition_key != self.case_id for attempt in attempts):
            raise DefinitionError('case attempts must identify this case')
        if any(event.partition_key != self.case_id for event in events):
            raise DefinitionError('case operation_events must identify this case')
        if any(request.artifact_key.partition_key != self.case_id for request in retries):
            raise DefinitionError('case retries must identify this case')
        object.__setattr__(self, 'operations', operations)
        object.__setattr__(self, 'artifacts', MappingProxyType(artifacts))
        object.__setattr__(self, 'failures', failures)
        object.__setattr__(self, 'artifact_records', records)
        object.__setattr__(self, 'attempts', attempts)
        object.__setattr__(self, 'operation_events', events)
        object.__setattr__(self, 'retries', retries)


@dataclass(frozen=True)
class RuntimeSnapshot:
    run_id: str
    status: RunStatus = 'created'
    running: tuple[InvocationSnapshot, ...] = ()
    ready_count: int = 0
    completed_artifacts: Mapping[ArtifactKey, ArtifactRef] = field(default_factory=dict)
    partition_sets: Mapping[ArtifactKey, PartitionSet] = field(default_factory=dict)
    error: RuntimeErrorInfo | None = None
    active_attempts: tuple[AttemptSnapshot, ...] = ()
    awaiting_artifacts: tuple[ArtifactKey, ...] = ()
    case_failures: tuple[CaseFailure, ...] = ()
    progress: RuntimeProgress = field(default_factory=RuntimeProgress)

    def __post_init__(self) -> None:
        _text(self.run_id, 'run_id')
        _known(self.status, get_args(RunStatus), 'run status')

        running = _tuple_of(self.running, InvocationSnapshot,
                            'running must contain InvocationSnapshot values')
        _unique(running, 'running invocation ids must be unique',
                key=lambda item: item.invocation_id)

        _integer(self.ready_count, 'ready_count')

        completed = dict(self.completed_artifacts)
        for key, ref in completed.items():
            if not isinstance(key, ArtifactKey) or not isinstance(ref, ArtifactRef):
                raise TypeError('completed_artifacts must map ArtifactKey to ArtifactRef')
            if key != ref.key:
                raise DefinitionError('completed artifact key must match its ref')

        partition_sets = dict(self.partition_sets)
        for key, partitions in partition_sets.items():
            if not isinstance(key, ArtifactKey) or key.partition_key:
                raise TypeError('partition_sets keys must be scalar ArtifactKey values')
            if not isinstance(partitions, PartitionSet):
                raise TypeError('partition_sets values must be PartitionSet')

        if self.error is not None and not isinstance(self.error, RuntimeErrorInfo):
            raise TypeError('error must be RuntimeErrorInfo or None')
        if self.status == 'failed' and self.error is None:
            raise DefinitionError('failed runtime snapshot requires error details')
        if self.status != 'failed' and self.error is not None:
            raise DefinitionError('runtime error details are only valid for failed status')
        if self.status in {'created', 'paused', 'cancelled', 'failed', 'completed'} and running:
            raise DefinitionError(f'{self.status} runtime snapshot cannot contain running invocations')
        if self.status in {'cancelled', 'failed', 'completed'} and self.ready_count:
            raise DefinitionError(f'{self.status} runtime snapshot cannot contain ready invocations')

        attempts = _tuple_of(self.active_attempts, AttemptSnapshot,
                             'attempts must contain AttemptSnapshot values')
        _unique(attempts, 'attempt ids must be unique', key=lambda attempt: attempt.attempt_id)

        awaiting = _tuple_of(self.awaiting_artifacts, ArtifactKey,
                             'awaiting_artifacts must contain ArtifactKey values')
        _unique(awaiting, 'awaiting artifact keys must be unique')

        failures = _tuple_of(self.case_failures, CaseFailure,
                             'case_failures must contain CaseFailure values')
        _unique(failures, 'case failure attempt ids must be unique',
                key=lambda failure: failure.attempt_id)

        if not isinstance(self.progress, RuntimeProgress):
            raise TypeError('runtime progress must be RuntimeProgress')

        object.__setattr__(self, 'running', running)
        object.__setattr__(self, 'completed_artifacts', MappingProxyType(completed))
        object.__setattr__(self, 'partition_sets', MappingProxyType(partition_sets))
        object.__setattr__(self, 'active_attempts', attempts)
        object.__setattr__(self, 'awaiting_artifacts', awaiting)
        object.__setattr__(self, 'case_failures', failures)


@dataclass(frozen=True)
class OperationDefinitionSnapshot:
    operation_id: str
    inputs: tuple[tuple[str, str, str, str], ...]
    outputs: tuple[tuple[str, str, str], ...]
    execution: str
    max_concurrency: int
    timeout: float | None = None


@dataclass(frozen=True)
class RunHistory:
    snapshot: RuntimeSnapshot
    operations: tuple[OperationDefinitionSnapshot, ...]
    artifacts: tuple[ArtifactRecord, ...]
    attempts: tuple[AttemptSnapshot, ...]
    operation_events: tuple[RecordedOperationEvent, ...]
    retry_requests: tuple[ArtifactRetryRequest, ...]


__all__ = [
    'ArtifactRetryRequest', 'AttemptSnapshot', 'AttemptStatus', 'CaseFailure',
    'CaseOperationSnapshot', 'CaseSnapshot', 'InvocationSnapshot', 'OperationDefinitionSnapshot',
    'EventLevel', 'EventStatus', 'OperationEvent', 'RecordedOperationEvent', 'RetryStatus',
    'RunConfiguration', 'RunHistory',
    'RunStatus', 'RuntimeErrorInfo', 'RuntimeProgress', 'RuntimeSnapshot',
]


def _event_value(value: object, depth: int = 0) -> object:
    if depth > _EVENT_DEPTH_LIMIT:
        return '<truncated>'
    if isinstance(value, Mapping):
        return {
            str(key): '<redacted>' if _SECRET_KEY.search(str(key)) else _event_value(item, depth + 1)
            for key, item in list(value.items())[:_EVENT_COLLECTION_LIMIT]
        }
    if isinstance(value, (list, tuple)):
        return [_event_value(item, depth + 1) for item in value[:_EVENT_COLLECTION_LIMIT]]
    if isinstance(value, str):
        text = _URL_SECRET.sub(r'\1<redacted>', _BEARER.sub('bearer <redacted>', _SECRET_VALUE.sub(
            lambda match: f'{match.group(1)}=<redacted>', value,
        )))
        encoded = text.encode()
        return text if len(encoded) <= _EVENT_DATA_LIMIT else encoded[:_EVENT_DATA_LIMIT].decode('utf-8', 'ignore')
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return _event_value(repr(value), depth)
