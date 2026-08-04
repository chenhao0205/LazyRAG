from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from evo import artifacts as A
from evo.artifact_flow import ArtifactFlow, FlowDefinition, FlowStage
from evo.artifact_runtime import (
    ArtifactCommit,
    ArtifactDraft,
    ArtifactKey,
    ArtifactRef,
    OperationResult,
    one,
    operation,
    scalar,
)
from evo.message_intent.actions import ActionExecutor, PreparedAction
from evo.message_intent.schemas import RepairGuidanceAction
from evo.repair_guidance import active_guidance, guidance_snapshot


_ANALYSIS_RESULT = 'test.guidance.analysis.result'
_ANALYSIS_APPROVAL = 'test.guidance.analysis.approval'
_REPAIR_RESULT = 'test.guidance.repair.result'


@operation(
    op_id='test.guidance.analysis',
    inputs={},
    outputs={'result': scalar(_ANALYSIS_RESULT)},
    execution='cooperative',
)
async def _analysis_operation(ctx):
    return OperationResult({'result': {'status': 'completed'}})


@operation(
    op_id='test.guidance.repair',
    inputs={
        'analysis': one(_ANALYSIS_RESULT),
        'approval': one(_ANALYSIS_APPROVAL),
        'policy': one(A.REPAIR_POLICY),
    },
    outputs={'result': scalar(_REPAIR_RESULT)},
    execution='cooperative',
)
async def _repair_operation(ctx, analysis, approval, policy):
    return OperationResult({'result': {
        'guidance': list(active_guidance(policy)),
    }})


class _Flow:
    def __init__(self, version: int, policy: dict[str, object]) -> None:
        self.record = SimpleNamespace(
            ref=ArtifactRef(ArtifactKey.scalar(A.REPAIR_POLICY), version),
        )
        self.policy = policy
        self.updates = []

    async def head(self, _thread_id, _key):
        return self.record

    async def read(self, _thread_id, _ref):
        return self.policy

    async def update_artifacts(self, _thread_id, updates, *, request_id):
        self.updates.append((tuple(updates), request_id))
        return {'status': 'updated'}


def _executor(flow: _Flow) -> ActionExecutor:
    executor = object.__new__(ActionExecutor)
    executor.flow = flow
    executor.thread_id = 'thread-1'
    return executor


def _prepared(policy: dict[str, object], version: int = 1) -> PreparedAction:
    return PreparedAction(
        action=RepairGuidanceAction(
            kind='repair_guidance',
            message='只检查 memory.py',
            effect='replace',
        ),
        command_id='command-1',
        summary='替换 Repair 修复方向',
        needs_confirmation=False,
        source_message_id='message-1',
        expected_repair_policy_ref=ArtifactRef(
            ArtifactKey.scalar(A.REPAIR_POLICY), version,
        ),
        expected_guidance_revision_id=guidance_snapshot(policy)['revision_id'],
    )


def test_guidance_action_rejects_head_changed_after_planning() -> None:
    policy = {'user_guidance': ['检查 web.py']}
    flow = _Flow(2, policy)

    with pytest.raises(ValueError, match='changed after planning'):
        asyncio.run(_executor(flow).execute(_prepared(policy, version=1)))

    assert flow.updates == []


def test_guidance_action_commits_against_observed_unchanged_head() -> None:
    policy = {'user_guidance': ['检查 web.py']}
    flow = _Flow(1, policy)

    result = asyncio.run(_executor(flow).execute(_prepared(policy)))

    assert result == {'status': 'updated'}
    assert len(flow.updates) == 1
    updates, request_id = flow.updates[0]
    assert request_id == 'command-1'
    assert updates[0].target_ref == flow.record.ref
    assert updates[0].value['user_guidance'] == ['只检查 memory.py']


def test_real_flow_accepts_guidance_while_awaiting_and_after_completion(tmp_path) -> None:
    async def run() -> None:
        definition = FlowDefinition(
            (_analysis_operation, _repair_operation),
            (
                FlowStage(
                    'analysis',
                    ArtifactKey.scalar(_ANALYSIS_RESULT),
                    ArtifactKey.scalar(_ANALYSIS_APPROVAL),
                ),
                FlowStage('repair', ArtifactKey.scalar(_REPAIR_RESULT)),
            ),
        )
        flow = await ArtifactFlow.open(tmp_path / 'runtime', definition)
        try:
            policy_key = ArtifactKey.scalar(A.REPAIR_POLICY)
            await flow.create('thread-1', ArtifactCommit(
                'seed:thread-1',
                'user:create',
                (ArtifactDraft(policy_key, {'user_guidance': ['检查 web.py']}),),
                {policy_key: None},
            ))
            await flow.start('thread-1')
            waiting = await flow.wait_until_boundary('thread-1')
            assert waiting.status == 'awaiting_approval'

            executor = ActionExecutor(flow, 'thread-1')
            prepared = await executor.prepare(
                RepairGuidanceAction(
                    kind='repair_guidance',
                    message='优先检查 memory.py',
                    effect='replace',
                ),
                'message-awaiting',
            )
            updated = await executor.execute(prepared)
            assert updated.status == 'awaiting_approval'

            await flow.approve('thread-1', 'analysis')
            completed = await flow.wait_until_boundary('thread-1')
            assert completed.status == 'completed'
            result_record = await flow.head('thread-1', ArtifactKey.scalar(_REPAIR_RESULT))
            assert result_record is not None
            result = await flow.read('thread-1', result_record.ref)
            assert result['guidance'] == ['优先检查 memory.py']

            prepared = await executor.prepare(
                RepairGuidanceAction(
                    kind='repair_guidance',
                    message='同时核对 web.py',
                    effect='append',
                ),
                'message-completed',
            )
            reopened = await executor.execute(prepared)
            assert reopened.status == 'running'
            completed = await flow.wait_until_boundary('thread-1')
            assert completed.status == 'completed'
            result_record = await flow.head('thread-1', ArtifactKey.scalar(_REPAIR_RESULT))
            assert result_record is not None
            result = await flow.read('thread-1', result_record.ref)
            assert result['guidance'] == ['优先检查 memory.py', '同时核对 web.py']
        finally:
            await flow.close()

    asyncio.run(run())
