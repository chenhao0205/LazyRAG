from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

from evo.artifact_runtime import (
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    ArtifactRetryRequest,
    AttemptSnapshot,
    CaseFailure,
    CaseSnapshot,
    OperationDefinitionSnapshot,
    RecordedOperationEvent,
    RunHistory,
    RuntimeErrorInfo,
    RuntimeProgress,
    RuntimeSnapshot,
)


FlowStatus = Literal[
    'idle',
    'running',
    'pausing',
    'paused',
    'awaiting_approval',
    'cancelling',
    'cancelled',
    'failed',
    'completed',
]

StageStatus = Literal[
    'pending',
    'running',
    'pausing',
    'paused',
    'awaiting_approval',
    'cancelling',
    'cancelled',
    'failed',
    'completed',
]


@dataclass(frozen=True)
class ArtifactUpdate:
    target_ref: ArtifactRef
    value: object

    def __post_init__(self) -> None:
        if not isinstance(self.target_ref, ArtifactRef):
            raise TypeError('artifact update target_ref must be ArtifactRef')


@dataclass(frozen=True)
class StageProgress:
    stage: str
    result_key: ArtifactKey
    result_ref: ArtifactRef | None = None
    approval_key: ArtifactKey | None = None
    approval_ref: ArtifactRef | None = None
    status: StageStatus = 'pending'
    operation_ids: tuple[str, ...] = ()
    progress: RuntimeProgress = RuntimeProgress()
    failures: tuple[CaseFailure, ...] = ()
    error: RuntimeErrorInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError('stage must be non-empty')
        if not isinstance(self.result_key, ArtifactKey):
            raise TypeError('result_key must be ArtifactKey')
        if self.result_ref is not None and self.result_ref.key != self.result_key:
            raise ValueError('result_ref must identify result_key')
        if self.approval_key is not None and not isinstance(self.approval_key, ArtifactKey):
            raise TypeError('approval_key must be ArtifactKey or None')
        if self.approval_ref is not None and (
            self.approval_key is None or self.approval_ref.key != self.approval_key
        ):
            raise ValueError('approval_ref must identify approval_key')
        if self.status not in {
            'pending', 'running', 'pausing', 'paused', 'awaiting_approval',
            'cancelling', 'cancelled', 'failed', 'completed',
        }:
            raise ValueError(f'unknown stage status: {self.status}')
        operation_ids = tuple(self.operation_ids)
        if not all(isinstance(operation_id, str) and operation_id.strip() for operation_id in operation_ids):
            raise TypeError('operation_ids must contain non-empty strings')
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError('operation_ids must be unique')
        if not isinstance(self.progress, RuntimeProgress):
            raise TypeError('progress must be RuntimeProgress')
        failures = tuple(self.failures)
        if not all(isinstance(failure, CaseFailure) for failure in failures):
            raise TypeError('failures must contain CaseFailure values')
        if self.error is not None and not isinstance(self.error, RuntimeErrorInfo):
            raise TypeError('error must be RuntimeErrorInfo or None')
        object.__setattr__(self, 'operation_ids', operation_ids)
        object.__setattr__(self, 'failures', failures)

    @property
    def completed(self) -> bool:
        return self.status in {'awaiting_approval', 'completed'}

    @property
    def approved(self) -> bool:
        return self.status == 'completed' and self.approval_key is not None and self.approval_ref is not None


@dataclass(frozen=True)
class StageSnapshot:
    progress: StageProgress
    operations: tuple[OperationDefinitionSnapshot, ...] = ()
    attempts: tuple[AttemptSnapshot, ...] = ()
    artifacts: tuple[ArtifactRecord, ...] = ()
    operation_events: tuple[RecordedOperationEvent, ...] = ()
    retries: tuple[ArtifactRetryRequest, ...] = ()
    versions: tuple[tuple[ArtifactRecord, ArtifactRecord | None], ...] = ()


@dataclass(frozen=True)
class FlowCaseSnapshot:
    runtime: CaseSnapshot
    stages: tuple[str, ...]
    current_stage: str
    artifacts: tuple[ArtifactRecord, ...] = ()
    attempts: tuple[AttemptSnapshot, ...] = ()
    operation_events: tuple[RecordedOperationEvent, ...] = ()
    retries: tuple[ArtifactRetryRequest, ...] = ()


@dataclass(frozen=True)
class FlowRunHistory:
    snapshot: FlowSnapshot
    runtime: RunHistory
    stages: tuple[StageSnapshot, ...]


@dataclass(frozen=True)
class FlowSnapshot:
    runtime: RuntimeSnapshot
    stages: tuple[StageProgress, ...]
    progress: RuntimeProgress
    failures: tuple[CaseFailure, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, RuntimeSnapshot):
            raise TypeError('runtime must be RuntimeSnapshot')
        stages = tuple(self.stages)
        if not stages or not all(isinstance(stage, StageProgress) for stage in stages):
            raise TypeError('stages must contain StageProgress values')
        if len({stage.stage for stage in stages}) != len(stages):
            raise ValueError('stage progress names must be unique')
        if not isinstance(self.progress, RuntimeProgress):
            raise TypeError('progress must be RuntimeProgress')
        failures = tuple(self.failures)
        if not all(isinstance(failure, CaseFailure) for failure in failures):
            raise TypeError('failures must contain CaseFailure values')
        object.__setattr__(self, 'stages', stages)
        object.__setattr__(self, 'failures', failures)

    @property
    def run_id(self) -> str:
        return self.runtime.run_id

    @property
    def status(self) -> FlowStatus:
        if self.runtime.status == 'created':
            return 'idle'
        if self.runtime.status in {'pausing', 'paused', 'cancelling', 'cancelled'}:
            return cast(FlowStatus, self.runtime.status)
        if self.current_progress.status in {'failed', 'awaiting_approval'}:
            return cast(FlowStatus, self.current_progress.status)
        if all(stage.status == 'completed' for stage in self.stages):
            return 'completed'
        if self.runtime.status == 'completed':
            return 'running'
        return cast(FlowStatus, self.runtime.status)

    @property
    def pending_approval(self) -> StageProgress | None:
        return next((stage for stage in self.stages if stage.status == 'awaiting_approval'), None)

    @property
    def current_stage(self) -> str:
        return self.current_progress.stage

    @property
    def current_progress(self) -> StageProgress:
        return next((stage for stage in self.stages if stage.status != 'completed'), self.stages[-1])


__all__ = [
    'ArtifactUpdate', 'FlowCaseSnapshot', 'FlowRunHistory', 'FlowSnapshot', 'FlowStatus',
    'StageProgress', 'StageSnapshot', 'StageStatus',
]
