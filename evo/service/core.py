from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evo import artifacts as A
from evo.artifact_flow import ArtifactFlow, ArtifactUpdate, FlowDefinition, FlowSnapshot
from evo.artifact_runtime import (
    RUN_CONFIGURATION_ARTIFACT_ID,
    ArtifactCommit,
    ArtifactDraft,
    ArtifactKey,
    ArtifactRef,
    DefinitionError,
    OperationResult,
    PartitionSet,
)
from evo.message_intent import MessageIntent, MessageRequest, MessageTurnResult
from evo.operations import evo_flow_definition
from evo.repair_guidance import canonicalize_guidance_policy_update, guidance_snapshot
from evo.repair_model import EvoModelConfigError, resolve_evo_model

from .contracts import (
    ArtifactUpdateBody,
    AutomaticUpdateBody,
    CaseRerunBody,
    CaseStructureBody,
    CommandRequest,
    ConfigurationUpdateBody,
    ControlRequest,
    ExternalResultBody,
    RetryRequest,
    ServiceError,
    ThreadCreate,
)
from .projections import ProjectionService
from .public import public_message_history, public_thread_state, public_value
from .router import RouterService


_STAGES = tuple(A.STEPS)
_FIRST_FRAME_TIMEOUT = 60.0
_THREAD_ID_ATTEMPTS = 32
_AUTO_WAIT_TIMEOUT = 30.0
_AUTO_STOPPED = frozenset({'idle', 'cancelled', 'failed', 'completed'})
_CONFIG_ARTIFACTS = {
    'run_config': A.RUN_CONFIG,
    'source_config': A.CORPUS_SOURCE_CONFIG,
    'target_config': A.EVAL_TARGET_CONFIG,
    'eval_policy': A.EVAL_POLICY,
    'repair_policy': A.REPAIR_POLICY,
    'candidate_config': A.ABTEST_CANDIDATE_CONFIG,
}
_CONFIG_ARTIFACT_IDS = frozenset(_CONFIG_ARTIFACTS.values())


logger = logging.getLogger(__name__)


class EvoService:
    def __init__(self, root: str | Path, definition: FlowDefinition, flow: ArtifactFlow) -> None:
        self.root = Path(root)
        self.definition = definition
        self.flow = flow
        self.messages = MessageIntent(self.root, flow)
        self.projections = ProjectionService(flow, definition)
        self.router = RouterService(self.root, flow)
        self._control_locks: dict[str, asyncio.Lock] = {}
        self._message_locks: dict[str, asyncio.Lock] = {}
        self._auto_tasks: dict[str, asyncio.Task[None]] = {}
        self._closing = False

    @classmethod
    async def open(cls, root: str | Path, definition: FlowDefinition | None = None, *, max_concurrency: int = 4,
                   terminate_timeout: float = 1.0) -> EvoService:
        root = Path(root)
        definition = definition or evo_flow_definition()
        flow = await ArtifactFlow.open(
            root / 'artifact-runtime',
            definition,
            max_concurrency=max_concurrency,
            terminate_timeout=terminate_timeout,
        )
        service = cls(root, definition, flow)
        await asyncio.to_thread(service.router.reconcile_unpublished)
        await asyncio.to_thread(service.router.reconcile_published)
        return service

    async def create_thread(self, request: ThreadCreate | Mapping[str, Any]) -> dict[str, Any]:
        request = (
            request
            if isinstance(request, ThreadCreate)
            else ThreadCreate.model_validate(request)
        )
        try:
            resolve_evo_model(request.llm_config.get('evo_llm'))
        except EvoModelConfigError as exc:
            raise ServiceError(422, exc.detail()) from exc
        thread_id = await self._new_thread_id()
        seed = _seed_values(thread_id, request)
        keys = tuple(ArtifactKey.scalar(artifact_id) for artifact_id in A.SEEDS)
        commit = ArtifactCommit(
            f'create:{thread_id}',
            'user:create',
            tuple(
                ArtifactDraft(key, seed[key.artifact_id])
                for key in keys
            ),
            {key: None for key in keys},
        )
        snapshot = await self.flow.create(
            thread_id,
            commit,
            configuration={'mode': 'interactive', 'automatic': request.automatic},
        )
        return _public_thread(request, snapshot)

    async def _new_thread_id(self) -> str:
        for _ in range(_THREAD_ID_ATTEMPTS):
            thread_id = f'thr-{uuid.uuid4().hex[:8]}'
            if not await self.flow.has_run(thread_id):
                return thread_id
        raise ServiceError(503, 'unable to allocate thread_id')

    async def list_threads(self, page_size: int, page_token: str, status: str = '') -> dict[str, Any]:
        offset = _page_offset(page_token)
        items = []
        for thread_id in sorted(await self.flow.run_ids()):
            item = await self.public_thread(thread_id, include_inputs=False)
            if not status or item['status'] == status:
                items.append(item)
        page = items[offset:offset + page_size]
        next_offset = offset + page_size
        return {
            'items': page,
            'next_page_token': str(next_offset) if next_offset < len(items) else '',
        }

    async def public_thread(self, thread_id: str, *, include_inputs: bool = True) -> dict[str, Any]:
        snapshot, metadata, configuration = await asyncio.gather(
            self.flow.snapshot(thread_id),
            self._run_config(thread_id),
            self.flow.configuration(thread_id),
        )
        configuration_ref = snapshot.runtime.completed_artifacts.get(
            ArtifactKey.scalar(RUN_CONFIGURATION_ARTIFACT_ID)
        )
        item = {
            'thread_id': thread_id,
            'mode': str(configuration.values.get('mode') or 'interactive'),
            'automatic': bool(configuration.values.get('automatic')),
            'automatic_version': 0 if configuration_ref is None else configuration_ref.version,
            'title': str(metadata.get('title') or ''),
            **public_thread_state(snapshot),
        }
        if include_inputs:
            item['inputs'] = public_value(metadata.get('inputs') or {})
            item['retryable'] = snapshot.status == 'failed'
        return item

    async def start(self, thread_id: str, request: CommandRequest | Mapping[str, Any]) -> dict[str, str]:
        request = _command(request)
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            snapshot, automatic = await asyncio.gather(
                self.flow.snapshot(thread_id),
                self._auto_enabled(thread_id),
            )
            if snapshot.status == 'paused':
                raise ServiceError(409, 'paused thread requires resume')
            if snapshot.status != 'idle':
                raise ServiceError(409, 'thread has already been started')
            target = request.until_step or _STAGES[0]
            if target != _STAGES[0]:
                raise ServiceError(422, f'start runs through {_STAGES[0]}')
            await self.flow.start(thread_id)
            if automatic:
                self._ensure_auto_task(thread_id)
            return _accepted(thread_id, request.command_id, 'start')

    async def continue_thread(self, thread_id: str, request: CommandRequest | Mapping[str, Any]) -> dict[str, str]:
        request = _command(request)
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            if await self._auto_enabled(thread_id):
                snapshot = await self.flow.snapshot(thread_id)
                if snapshot.status == 'idle':
                    raise ServiceError(409, 'thread has not been started')
                if snapshot.status in {'cancelled', 'failed'}:
                    raise ServiceError(409, f'cannot continue thread from {snapshot.status}')
                if snapshot.status == 'paused':
                    await self.flow.resume(thread_id)
                self._ensure_auto_task(thread_id)
                return _accepted(thread_id, request.command_id, 'continue')

            snapshot = await self.flow.snapshot(thread_id)
            pending = snapshot.pending_approval
            if pending is None:
                raise ServiceError(409, 'thread is not awaiting approval')
            next_stage = _next_stage(pending.stage)
            target = request.until_step or next_stage
            if target != next_stage:
                raise ServiceError(
                    422,
                    f'continue from {pending.stage} runs through {next_stage}',
                )
            await self.flow.approve(thread_id, pending.stage)
            return _accepted(thread_id, request.command_id, 'continue')

    async def retry(self, thread_id: str, request: RetryRequest | Mapping[str, Any]) -> dict[str, str]:
        request = _retry_request(request)
        _validate_retry_stage(request.stage)
        command_id = (
            request.command_id.strip()
            or f'retry:{thread_id}:{time.time_ns()}'
        )

        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            snapshot = await self.flow.snapshot(thread_id)
            stage = request.stage or snapshot.current_stage
            await self.flow.retry_stage(thread_id, stage, request_id=command_id)
            await self._continue_automatic(thread_id)
            return _accepted(thread_id, command_id, 'retry')

    async def rerun_stage(self, thread_id: str, stage: str, request: ControlRequest | Mapping[str, Any]
                          ) -> dict[str, str]:
        request = _control_request(request)
        _validate_retry_stage(stage)
        command_id = request.command_id.strip() or f'rerun-stage:{thread_id}:{stage}:{time.time_ns()}'
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            await self.flow.rerun_stage(thread_id, stage, request_id=command_id)
            await self._continue_automatic(thread_id)
            return _accepted(thread_id, command_id, 'rerun-stage')

    async def rerun_case(self, thread_id: str, case_id: str, request: CaseRerunBody | Mapping[str, Any]
                         ) -> dict[str, str]:
        request = request if isinstance(request, CaseRerunBody) else CaseRerunBody.model_validate(request)
        command_id = request.command_id.strip() or f'rerun-case:{thread_id}:{case_id}:{time.time_ns()}'
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            await self.flow.rerun_case(
                thread_id,
                case_id,
                request_id=command_id,
                from_stage=request.stage,
                from_artifact=(ArtifactKey.partition(request.artifact_id, case_id) if request.artifact_id else None),
            )
            await self._continue_automatic(thread_id)
            return _accepted(thread_id, command_id, 'rerun-case')

    async def retry_case(self, thread_id: str, case_id: str, request: ControlRequest | Mapping[str, Any]
                         ) -> dict[str, str]:
        request = _control_request(request)
        command_id = request.command_id.strip() or f'retry-case:{thread_id}:{case_id}:{time.time_ns()}'
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            await self.flow.retry_failed_case(thread_id, case_id, request_id=command_id)
            await self._continue_automatic(thread_id)
            return _accepted(thread_id, command_id, 'retry-case')

    async def pause(self, thread_id: str, request: ControlRequest | Mapping[str, Any]) -> dict[str, str]:
        request = _control_request(request)
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            await self._stop_auto_task(thread_id)
            snapshot = await self.flow.snapshot(thread_id)
            if snapshot.status != 'paused':
                await self.flow.pause(thread_id)
            return _accepted(thread_id, request.command_id, 'pause')

    async def resume(self, thread_id: str, request: ControlRequest | Mapping[str, Any]) -> dict[str, str]:
        request = _control_request(request)
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            await self.flow.resume(thread_id)
            await self._continue_automatic(thread_id)
            return _accepted(thread_id, request.command_id, 'resume')

    async def cancel(self, thread_id: str, request: ControlRequest | Mapping[str, Any]) -> dict[str, str]:
        request = _control_request(request)
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            snapshot = await self.flow.snapshot(thread_id)
            if snapshot.status != 'cancelled':
                await self.flow.cancel(thread_id)
            return _accepted(thread_id, request.command_id, 'cancel')

    async def update_artifacts(self, thread_id: str, request: ArtifactUpdateBody | Mapping[str, Any]) -> dict[str, Any]:
        request = request if isinstance(request, ArtifactUpdateBody) else ArtifactUpdateBody.model_validate(request)
        config_targets = sorted({item.artifact_id for item in request.updates} & _CONFIG_ARTIFACT_IDS)
        if config_targets:
            raise ServiceError(422, f'configuration requires the configuration endpoint: {", ".join(config_targets)}')
        await self.flow.update_artifacts(
            thread_id,
            tuple(
                ArtifactUpdate(
                    ArtifactRef(ArtifactKey(item.artifact_id, item.partition_key), item.base_version),
                    item.value,
                )
                for item in request.updates
            ),
            request_id=request.request_id,
        )
        await self._continue_automatic(thread_id)
        return await self.public_thread(thread_id, include_inputs=False)

    async def set_automatic(self, thread_id: str, request: AutomaticUpdateBody | Mapping[str, Any]) -> dict[str, Any]:
        request = request if isinstance(request, AutomaticUpdateBody) else AutomaticUpdateBody.model_validate(request)
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            configuration = await self.flow.configuration(thread_id)
            try:
                await self.flow.update_configuration(
                    thread_id,
                    {**configuration.values, 'mode': 'interactive', 'automatic': request.enabled},
                    request_id=request.request_id,
                    base_version=request.base_version,
                )
            except DefinitionError as exc:
                if str(exc) != 'artifact commit precondition is stale':
                    raise
                raise ServiceError(409, 'automatic base_version is stale') from exc
            if request.enabled:
                self._ensure_auto_task(thread_id)
            else:
                await self._stop_auto_task(thread_id)
        return await self.public_thread(thread_id, include_inputs=False)

    async def update_configuration(self, thread_id: str, request: ConfigurationUpdateBody | Mapping[str, Any]
                                   ) -> dict[str, Any]:
        request = (
            request
            if isinstance(request, ConfigurationUpdateBody)
            else ConfigurationUpdateBody.model_validate(request)
        )
        key = ArtifactKey.scalar(_CONFIG_ARTIFACTS[request.target])
        base_policy = None
        if request.target == 'repair_policy':
            base_ref = ArtifactRef(key, request.base_version)
            base_record = await self.flow.record(thread_id, base_ref)
            if base_record is None:
                raise ServiceError(409, 'repair_policy base_version is stale')
            base_policy = await self.flow.read(thread_id, base_ref)
            if not isinstance(base_policy, Mapping):
                raise ServiceError(409, 'repair_policy base_version is invalid')
        value = _configuration_value(request, base_policy)
        await self.flow.update_artifacts(
            thread_id,
            (ArtifactUpdate(ArtifactRef(key, request.base_version), value),),
            request_id=request.request_id,
        )
        await self._continue_automatic(thread_id)
        return await self.public_thread(thread_id, include_inputs=False)

    async def update_cases(self, thread_id: str, request: CaseStructureBody | Mapping[str, Any]) -> dict[str, Any]:
        request = request if isinstance(request, CaseStructureBody) else CaseStructureBody.model_validate(request)
        set_key = ArtifactKey.scalar(request.partition_set_id)
        writes = (
            ArtifactDraft(set_key, PartitionSet(tuple(request.case_ids))),
            *(
                ArtifactDraft(ArtifactKey.partition(seed.artifact_id, seed.case_id), seed.value)
                for seed in request.seeds
            ),
        )
        await self.flow.commit(
            thread_id,
            ArtifactCommit(
                f'case-structure:{request.request_id}',
                'user:case-structure',
                writes,
                {
                    set_key: ArtifactRef(set_key, request.base_version),
                    **{write.key: None for write in writes[1:]},
                },
            ),
        )
        await self._continue_automatic(thread_id)
        return await self.public_thread(thread_id, include_inputs=False)

    async def submit_external_result(self, thread_id: str, attempt_id: str,
                                     request: ExternalResultBody | Mapping[str, Any]) -> dict[str, str]:
        request = request if isinstance(request, ExternalResultBody) else ExternalResultBody.model_validate(request)
        await self.flow.submit_external_result(thread_id, attempt_id, OperationResult(request.values))
        await self._continue_automatic(thread_id)
        return {'status': 'accepted', 'thread_id': thread_id, 'attempt_id': attempt_id}

    async def message(self, thread_id: str, request: MessageRequest) -> MessageTurnResult:
        lock = self._message_locks.setdefault(thread_id, asyncio.Lock())
        async with lock:
            result = await self.messages.run('user', thread_id, request)
        snapshot, automatic = await asyncio.gather(
            self.flow.snapshot(thread_id),
            self._auto_enabled(thread_id),
        )
        if automatic and snapshot.status not in _AUTO_STOPPED:
            self._ensure_auto_task(thread_id)
        elif not automatic:
            await self._stop_auto_task(thread_id)
        return result

    async def message_history(self, thread_id: str, page_size: int, page_token: str) -> dict[str, Any]:
        await self.flow.snapshot(thread_id)
        history = await asyncio.to_thread(
            self.messages.history,
            thread_id,
            page_size,
            page_token,
        )
        return public_message_history(history)

    async def delete_thread(self, thread_id: str) -> dict[str, Any]:
        async with self._control_locks.setdefault(thread_id, asyncio.Lock()):
            automatic = await self._auto_enabled(thread_id)
            await self._stop_auto_task(thread_id)
            try:
                await self.flow.release(thread_id)
            except BaseException:
                if automatic:
                    self._ensure_auto_task(thread_id)
                raise
            lock = self._message_locks.setdefault(thread_id, asyncio.Lock())
            async with lock:
                await self.router.delete_thread(thread_id)
                await self.messages.delete_thread(thread_id)
                await self.flow.delete_run(thread_id)
            return {
                'thread_id': thread_id,
                'deleted': True,
                'message': 'thread deleted',
            }

    async def close(self) -> None:
        self._closing = True
        tasks = tuple(self._auto_tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._auto_tasks.clear()
        await self.flow.close()

    async def _run_config(self, thread_id: str) -> Mapping[str, Any]:
        record = await self.flow.head(thread_id, ArtifactKey.scalar(A.RUN_CONFIG))
        if record is None:
            return {}
        value = await self.flow.read(thread_id, record.ref)
        return value if isinstance(value, Mapping) else {}

    async def _auto_enabled(self, thread_id: str) -> bool:
        return bool((await self.flow.configuration(thread_id)).values.get('automatic'))

    async def _continue_automatic(self, thread_id: str) -> None:
        if await self._auto_enabled(thread_id):
            self._ensure_auto_task(thread_id)

    def _ensure_auto_task(self, thread_id: str) -> None:
        if self._closing:
            return
        current = self._auto_tasks.get(thread_id)
        if current is not None and not current.done():
            return
        task = asyncio.create_task(
            self._drive_auto(thread_id),
            name=f'evo-auto:{thread_id}',
        )
        self._auto_tasks[thread_id] = task
        task.add_done_callback(
            lambda completed, run_id=thread_id: self._auto_task_done(
                run_id,
                completed,
            )
        )

    async def _drive_auto(self, thread_id: str) -> None:
        while not self._closing:
            lock = self._control_locks.setdefault(thread_id, asyncio.Lock())
            async with lock:
                snapshot = await self.flow.snapshot(thread_id)
                if snapshot.status in _AUTO_STOPPED:
                    return
                if snapshot.status == 'paused':
                    return
                if snapshot.status == 'awaiting_approval':
                    pending = snapshot.pending_approval
                    if pending is None:
                        raise RuntimeError('awaiting approval without pending stage')
                    await self.flow.approve(thread_id, pending.stage)
                    continue

            try:
                await self.flow.wait_until_boundary(
                    thread_id,
                    timeout=_AUTO_WAIT_TIMEOUT,
                )
            except TimeoutError:
                continue

    def _auto_task_done(self, thread_id: str, task: asyncio.Task[None]) -> None:
        if self._auto_tasks.get(thread_id) is task:
            del self._auto_tasks[thread_id]
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                'auto runner failed for %s',
                thread_id,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _stop_auto_task(self, thread_id: str) -> None:
        task = self._auto_tasks.pop(thread_id, None)
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _seed_values(thread_id: str, request: ThreadCreate) -> dict[str, object]:
    inputs = request.inputs.model_dump()
    llm_config = dict(request.llm_config)
    target_config = {
        'router_chat_url': request.inputs.router_chat_url,
        'router_admin_url': request.inputs.router_admin_url,
        'algorithm_id': request.inputs.algorithm_id,
        'llm_config': llm_config,
        'case_deadline_seconds': request.inputs.case_deadline_seconds,
        'first_frame_timeout_seconds': _FIRST_FRAME_TIMEOUT,
    }
    return {
        A.RUN_CONFIG: {
            'thread_id': thread_id,
            'title': request.title,
            'inputs': inputs,
            'num_case': request.inputs.num_case,
            'llm_config': llm_config,
        },
        A.CORPUS_SOURCE_CONFIG: {
            'kb_id': request.inputs.kb_id,
            'csv_data': request.inputs.csv_data,
            'target_case_count': request.inputs.num_case,
            'min_case_count': request.inputs.num_case,
        },
        A.EVAL_TARGET_CONFIG: target_config,
        A.EVAL_POLICY: {'judge_llm_config': llm_config},
        A.REPAIR_POLICY: {
            'llm_config': llm_config,
            'thread_id': thread_id,
            'workspace_namespace': thread_id,
        },
        A.ABTEST_CANDIDATE_CONFIG: {
            'router_chat_url': request.inputs.router_chat_url,
            'router_admin_url': request.inputs.router_admin_url,
            'llm_config': llm_config,
            'case_deadline_seconds': request.inputs.case_deadline_seconds,
            'first_frame_timeout_seconds': _FIRST_FRAME_TIMEOUT,
        },
    }


def _public_thread(request: ThreadCreate, snapshot: FlowSnapshot) -> dict[str, Any]:
    return {
        'thread_id': snapshot.run_id,
        'mode': request.mode,
        'automatic': request.automatic,
        'automatic_version': 1,
        'title': request.title,
        **public_thread_state(snapshot),
    }


def _command(request: CommandRequest | Mapping[str, Any]) -> CommandRequest:
    return request if isinstance(request, CommandRequest) else CommandRequest.model_validate(request)


def _control_request(request: ControlRequest | Mapping[str, Any]) -> ControlRequest:
    return request if isinstance(request, ControlRequest) else ControlRequest.model_validate(request)


def _configuration_value(
    request: ConfigurationUpdateBody,
    base_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = dict(request.value)
    if request.target == 'repair_policy':
        try:
            value = (
                canonicalize_guidance_policy_update(base_policy, value)
                if base_policy is not None
                else value
            )
            guidance_snapshot(value)
        except ValueError as exc:
            raise ServiceError(422, f'invalid repair guidance: {exc}') from exc
    return value


def _retry_request(request: RetryRequest | Mapping[str, Any]) -> RetryRequest:
    return request if isinstance(request, RetryRequest) else RetryRequest.model_validate(request)


def _accepted(thread_id: str, command_id: str, command: str) -> dict[str, str]:
    return {
        'status': 'accepted',
        'thread_id': thread_id,
        'command_id': command_id.strip() or f'{command}:{thread_id}:{time.time_ns()}',
    }


def _page_offset(token: str) -> int:
    if not str(token or '0').isdigit():
        raise ServiceError(422, 'page_token must be a non-negative integer offset')
    return int(token or 0)


def _validate_retry_stage(stage: str) -> None:
    if stage and stage not in _STAGES:
        raise ServiceError(422, f'stage must be one of: {", ".join(_STAGES)}')


def _next_stage(stage: str) -> str:
    index = _STAGES.index(stage)
    return _STAGES[index + 1] if index + 1 < len(_STAGES) else ''


__all__ = ['EvoService']
