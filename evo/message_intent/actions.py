from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from evo import artifacts as A
from evo.artifact_flow import ArtifactFlow
from evo.artifact_flow.state import ArtifactUpdate
from evo.artifact_runtime import ArtifactKey, ArtifactRecord, ArtifactRef

from .schemas import CaseAction, FlowAction, PlannedAction, QueryAction, RepairGuidanceAction


@dataclass(frozen=True)
class PreparedAction:
    action: PlannedAction
    command_id: str
    summary: str
    needs_confirmation: bool


class ActionExecutor:
    def __init__(self, flow: ArtifactFlow, thread_id: str) -> None:
        if not isinstance(flow, ArtifactFlow):
            raise TypeError('flow must be ArtifactFlow')
        if not isinstance(thread_id, str) or not thread_id.strip():
            raise ValueError('thread_id must be non-empty')
        self.flow = flow
        self.thread_id = thread_id

    async def prepare(self, action: PlannedAction, source_message_id: str) -> PreparedAction:
        if isinstance(action, (FlowAction, QueryAction, CaseAction, RepairGuidanceAction)):
            stage = getattr(action, 'stage', '')
            if stage and stage not in A.STEPS:
                raise ValueError(f'unknown flow stage: {stage}')
            return PreparedAction(
                action,
                _command_id(self.thread_id, source_message_id, action),
                _summary(action),
                isinstance(action, FlowAction) and action.command == 'cancel',
            )
        raise ValueError(f'action cannot be executed: {action.kind}')

    async def execute(self, prepared: PreparedAction) -> object:
        action = prepared.action
        if isinstance(action, FlowAction):
            return await self._execute_flow(action, prepared.command_id)
        if isinstance(action, QueryAction):
            return await self._execute_query(action)
        if isinstance(action, CaseAction):
            if action.command == 'rerun':
                return await self.flow.rerun_case(
                    self.thread_id,
                    action.case_id,
                    from_stage=action.stage,
                    request_id=prepared.command_id,
                )
            return await self.flow.retry_failed_case(
                self.thread_id,
                action.case_id,
                request_id=prepared.command_id,
            )
        if isinstance(action, RepairGuidanceAction):
            key = ArtifactKey.scalar(A.REPAIR_POLICY)
            record = await self.flow.head(self.thread_id, key)
            if record is None:
                raise ValueError('repair policy is not available')
            current = await self.flow.read(self.thread_id, record.ref)
            if not isinstance(current, Mapping):
                raise ValueError('repair policy must be an object')
            raw_guidance = current.get('user_guidance') or []
            if not isinstance(raw_guidance, (list, tuple)):
                raise ValueError('repair policy user_guidance must be a list')
            guidance = [str(item).strip() for item in raw_guidance if str(item).strip()]
            if len(guidance) >= 100:
                raise ValueError('repair policy user_guidance limit reached')
            guidance.append(action.message.strip())
            return await self.flow.update_artifacts(
                self.thread_id,
                (ArtifactUpdate(record.ref, {**current, 'user_guidance': guidance}),),
                request_id=prepared.command_id,
            )
        raise TypeError('prepared action is not executable')

    async def _execute_flow(self, action: FlowAction, command_id: str) -> object:
        if action.command == 'start':
            return await self.flow.start(self.thread_id)
        if action.command == 'approve':
            return await self.flow.approve(self.thread_id, action.stage)
        if action.command == 'pause':
            return await self.flow.pause(self.thread_id)
        if action.command == 'resume':
            return await self.flow.resume(self.thread_id)
        if action.command == 'rerun':
            return await self.flow.rerun_stage(self.thread_id, action.stage, request_id=command_id)
        if action.command == 'retry':
            if action.stage:
                return await self.flow.retry_stage(self.thread_id, action.stage, request_id=command_id)
            snapshot = await self.flow.snapshot(self.thread_id)
            return await self.flow.retry_stage(self.thread_id, snapshot.current_stage, request_id=command_id)
        return await self.flow.cancel(self.thread_id)

    async def _execute_query(self, action: QueryAction) -> object:
        if action.query == 'progress':
            return await self.flow.snapshot(self.thread_id)
        if action.query == 'run_history':
            return await self.flow.run_history(self.thread_id)
        if action.query == 'stage_snapshot':
            return await self.flow.stage_snapshot(self.thread_id, action.stage)
        if action.query == 'case_snapshot':
            return await self.flow.case_snapshot(self.thread_id, action.case_id)
        if action.query == 'operation_events':
            return await self.flow.operation_events(
                self.thread_id,
                stage=action.stage,
                case_id=action.case_id or None,
                event_type=action.event_type,
                level=action.level,
                limit=action.limit,
            )
        if action.query == 'stage_result':
            return await self._read_artifact(ArtifactKey.scalar(A.ROOTS[action.stage]), None)
        key = ArtifactKey(action.artifact_id, action.partition_key)
        if action.query == 'artifact':
            return await self._read_artifact(key, action.version)
        records = await self.flow.history(self.thread_id, key)
        return {'artifact': _artifact_key_data(key), 'versions': [_record_data(record) for record in records]}

    async def _read_artifact(self, key: ArtifactKey, version: int | None) -> Mapping[str, object]:
        record = (
            await self.flow.head(self.thread_id, key)
            if version is None
            else await self.flow.record(self.thread_id, ArtifactRef(key, version))
        )
        if record is None:
            target = f'{key.artifact_id}[{key.partition_key}]' if key.partition_key else key.artifact_id
            raise ValueError(f'artifact version not found: {target}@{version or "head"}')
        return {**_record_data(record), 'value': await self.flow.read(self.thread_id, record.ref)}


def intent_catalog() -> Mapping[str, object]:
    artifact_ids = sorted({
        value
        for name in A.__all__
        if isinstance((value := getattr(A, name)), str)
    })
    return MappingProxyType({
        'stages': tuple({
            'name': stage,
            'result_artifact': A.ROOTS[stage],
            'requires_approval': stage in A.APPROVALS,
        } for stage in A.STEPS),
        'artifact_ids': tuple(artifact_ids),
        'partitioned_artifacts': dict(A.PARTITION_SET_BY_ARTIFACT),
    })


def _summary(action: PlannedAction) -> str:
    if isinstance(action, FlowAction):
        return {
            'start': '启动流程',
            'approve': f'批准 {action.stage} 阶段',
            'pause': '暂停流程',
            'resume': '恢复流程',
            'rerun': f'重新执行 {action.stage} 阶段',
            'retry': f'重试 {action.stage} 阶段失败项' if action.stage else '重试当前阶段失败项',
            'cancel': '终止流程',
        }[action.command]
    if isinstance(action, QueryAction):
        return '查询流程或产物'
    if isinstance(action, CaseAction):
        return (
            f'从 {action.stage} 阶段重新执行 case {action.case_id}'
            if action.command == 'rerun'
            else f'重试失败 case {action.case_id}'
        )
    if isinstance(action, RepairGuidanceAction):
        return '补充 Repair 修复观察与方向'
    raise ValueError(f'action has no summary: {action.kind}')


def _command_id(thread_id: str, message_id: str, action: PlannedAction) -> str:
    payload = json.dumps(
        action.model_dump(mode='json'), ensure_ascii=False, sort_keys=True, separators=(',', ':'),
    ).encode()
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f'message:{thread_id}:{message_id}:{digest}'


def _artifact_key_data(key: ArtifactKey) -> Mapping[str, str]:
    return {'artifact_id': key.artifact_id, 'partition_key': key.partition_key}


def _artifact_ref_data(ref: ArtifactRef) -> Mapping[str, object]:
    return {**_artifact_key_data(ref.key), 'version': ref.version}


def _record_data(record: ArtifactRecord) -> Mapping[str, object]:
    return {
        'ref': _artifact_ref_data(record.ref),
        'producer': record.producer,
        'input_refs': [_artifact_ref_data(ref) for ref in record.input_refs],
    }


__all__ = ['ActionExecutor', 'PreparedAction', 'intent_catalog']
