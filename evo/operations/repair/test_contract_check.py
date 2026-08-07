#!/usr/bin/env python3
"""Standalone collaboration contract check for the Repair agent loop.

Run directly for a concise self-check:

    ../.venv/bin/python tests/evo/test_repair_contract_check.py

The same checks are collected by pytest.  Values in the smoke run are temporary;
only boundary fields, value types, tool arguments, and workspace layout are frozen.
"""

from __future__ import annotations

import json
import stat
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evo.operations.repair.contracts import (  # noqa: E402
    ACTION_ARGUMENT_FIELDS,
    OBSERVATION_STATUSES,
    REPAIR_TOOLS,
    RESULT_STATUSES,
    TEST_LEVELS,
    FinishArguments,
    ObservationStatus,
    RepairAction,
    RepairAgentError,
    RepairCapabilityError,
    RepairContractError,
    RepairError,
    RepairInput,
    RepairObservation,
    RepairResult,
    RepairTool,
    RepairView,
    ResearchArguments,
    ResultStatus,
    ShellArguments,
    TestArguments as RepairTestArguments,
    TestLevel,
    WorkspaceArguments,
    contract_dict,
    repair_action,
)
from evo.operations.repair.dispatch import Capability, EXTERNAL_TOOLS  # noqa: E402
from evo.operations.repair.session import RepairSession  # noqa: E402
from evo.operations.repair.workspace import (  # noqa: E402
    DEFAULT_RUNTIME_ROOT,
    WorkspacePaths,
    artifact_path,
    workspace_hash,
    write_json,
)


FROZEN_CONTRACTS: dict[type[Any], dict[str, Any]] = {
    RepairInput: {
        'run_id': str,
        'objective': str,
        'guidance': str,
        'source_ref': str,
        'case_scope': str,
        'constraints': dict[str, Any],
        'budget': dict[str, Any],
    },
    RepairView: {
        'objective': str,
        'guidance': str,
        'workspace_hash': str,
        'diff_summary': str,
        'memory_brief': str,
        'recent_events': list[dict[str, Any]],
        'validation_evidence': list[dict[str, Any]],
        'remaining_budget': dict[str, Any],
    },
    RepairAction: {
        'call_id': str,
        'tool': RepairTool,
        'arguments': dict[str, Any],
    },
    RepairObservation: {
        'call_id': str,
        'status': ObservationStatus,
        'summary': str,
        'artifact_refs': list[str],
        'workspace_hash': str,
    },
    RepairResult: {
        'status': ResultStatus,
        'patch_ref': str,
        'evidence_refs': list[str],
        'summary': str,
        'unresolved': list[str],
    },
}

FROZEN_ACTION_ARGUMENTS = {
    'workspace': ({'operation', 'path', 'content'}, {'operation'}),
    'shell': ({'command', 'cwd', 'timeout_seconds'}, {'command'}),
    'test': ({'level'}, {'level'}),
    'research': ({'operation', 'query', 'urls'}, {'operation', 'query'}),
    'finish': ({'reason'}, {'reason'}),
}

FROZEN_ARGUMENT_TYPES: dict[type[Any], dict[str, Any]] = {
    WorkspaceArguments: {
        'operation': Literal['list', 'read', 'write', 'diff'],
        'path': str,
        'content': str,
    },
    ShellArguments: {
        'command': list[str],
        'cwd': Literal['source', 'work'],
        'timeout_seconds': int,
    },
    RepairTestArguments: {'level': TestLevel},
    ResearchArguments: {
        'operation': Literal['search', 'read'],
        'query': str,
        'urls': list[str],
    },
    FinishArguments: {'reason': str},
}

FROZEN_WORKSPACE_PATHS = {
    'root': '.',
    'control': 'control',
    'events': 'control/events.jsonl',
    'evidence': 'control/evidence',
    'logs': 'control/logs',
    'result': 'control/result.json',
    'sandbox': 'sandbox',
    'source': 'sandbox/source',
    'work': 'sandbox/work',
    'context': 'sandbox/context',
}

EVENT_FIELDS = {'sequence', 'timestamp', 'event', 'call_id', 'workspace_hash', 'payload'}


class ContractCheckFailure(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractCheckFailure(message)


def _field_names(contract: type[Any]) -> set[str]:
    return {field.name for field in fields(contract)}


def _runtime_type(annotation: Any) -> type[Any]:
    origin = get_origin(annotation)
    if origin is Literal:
        return type(get_args(annotation)[0])
    return origin or annotation


def _check_value(value: Any) -> dict[str, Any]:
    contract = type(value)
    schema = FROZEN_CONTRACTS.get(contract)
    _require(schema is not None, f'unknown contract value: {contract.__name__}')
    payload = contract_dict(value)
    actual_fields = set(payload)
    expected_fields = set(schema)
    _require(
        actual_fields == expected_fields,
        f'{contract.__name__} fields changed: expected {sorted(expected_fields)}, got {sorted(actual_fields)}',
    )
    for name, annotation in schema.items():
        expected_type = _runtime_type(annotation)
        _require(
            type(payload[name]) is expected_type,
            f'{contract.__name__}.{name} type changed: '
            f'expected {expected_type.__name__}, got {type(payload[name]).__name__}',
        )
    return payload


def _expect_contract_error(code: str, function: Any) -> None:
    try:
        function()
    except RepairContractError as exc:
        _require(exc.code == code, f'expected error {code}, got {exc.code}')
        return
    raise ContractCheckFailure(f'contract violation was accepted; expected {code}')


def test_repair_frozen_schemas() -> None:
    for contract, schema in FROZEN_CONTRACTS.items():
        _require(
            _field_names(contract) == set(schema),
            f'{contract.__name__} fields changed: '
            f'expected {sorted(schema)}, got {sorted(_field_names(contract))}',
        )
        _require(
            get_type_hints(contract) == schema,
            f'{contract.__name__} type annotations changed: '
            f'expected {schema!r}, got {get_type_hints(contract)!r}',
        )

    _require(REPAIR_TOOLS == ('workspace', 'shell', 'test', 'research', 'finish'), 'RepairTool values changed')
    _require(EXTERNAL_TOOLS == ('workspace', 'shell', 'test', 'research'), 'external capability set changed')
    _require(OBSERVATION_STATUSES == ('success', 'fail', 'error'), 'observation statuses changed')
    _require(RESULT_STATUSES == ('success', 'partial', 'failed'), 'result statuses changed')
    _require(TEST_LEVELS == ('L0', 'L1', 'L2'), 'test levels changed')
    _require(ACTION_ARGUMENT_FIELDS == FROZEN_ACTION_ARGUMENTS, 'tool argument fields changed')
    for arguments, schema in FROZEN_ARGUMENT_TYPES.items():
        _require(
            get_type_hints(arguments) == schema,
            f'{arguments.__name__} types changed: expected {schema!r}, got {get_type_hints(arguments)!r}',
        )
    _require(
        all(issubclass(error, RepairError) for error in (
            RepairContractError, RepairAgentError, RepairCapabilityError,
        )),
        'public Repair exceptions must inherit RepairError',
    )


def test_repair_tool_boundary() -> None:
    samples = [
        {'call_id': 'sample-workspace', 'tool': 'workspace', 'arguments': {
            'operation': 'write', 'path': 'work/probe.py', 'content': 'print("ok")\n',
        }},
        {'call_id': 'sample-shell', 'tool': 'shell', 'arguments': {
            'command': [sys.executable, 'probe.py'], 'cwd': 'work', 'timeout_seconds': 30,
        }},
        {'call_id': 'sample-test', 'tool': 'test', 'arguments': {'level': 'L1'}},
        {'call_id': 'sample-research', 'tool': 'research', 'arguments': {
            'operation': 'read', 'query': 'primary source', 'urls': ['https://example.com/reference'],
        }},
        {'call_id': 'sample-finish', 'tool': 'finish', 'arguments': {'reason': 'evidence complete'}},
    ]
    for sample in samples:
        action = repair_action(sample)
        _require(_check_value(action) == sample, f'{sample["tool"]} action did not round-trip')

    _expect_contract_error('action_fields_invalid', lambda: repair_action({
        'call_id': 'extra-top-level', 'tool': 'finish', 'arguments': {'reason': 'done'}, 'extra': True,
    }))
    _expect_contract_error('action_argument_fields_invalid', lambda: RepairAction(
        'extra-tool-argument', 'test', {'level': 'L0', 'fixture_only': True},
    ))
    _expect_contract_error('action_argument_fields_invalid', lambda: RepairAction(
        'missing-tool-argument', 'finish', {},
    ))


class _ScriptedAgent:
    def __init__(self, actions: list[RepairAction]) -> None:
        self.actions = list(actions)
        self.views: list[RepairView] = []

    def decide(self, view: RepairView) -> RepairAction:
        _check_value(view)
        self.views.append(view)
        _require(bool(self.actions), 'session requested more actions than the contract scenario provides')
        return self.actions.pop(0)

    def summarize(
        self,
        objective: str,
        guidance: str,
        previous_brief: str,
        events: list[dict[str, Any]],
    ) -> str:
        _require(all(type(value) is str for value in (objective, guidance, previous_brief)), 'summarize strings changed')
        _require(type(events) is list, 'summarize events must be a list')
        return f'{previous_brief}\ncompressed {len(events)} completed event(s)'.strip()

    def assess_finish(
        self,
        repair_input: RepairInput,
        view: RepairView,
        arguments: Mapping[str, Any],
    ) -> tuple[bool, str]:
        _check_value(repair_input)
        _check_value(view)
        _require(set(arguments) == {'reason'}, f'finish arguments changed: {sorted(arguments)}')
        return True, 'contract scenario satisfied'


class _ContractStubFactory:
    """Record boundary values without invoking any production capability."""

    def __init__(self) -> None:
        self.inputs: list[RepairInput] = []
        self.paths: list[WorkspacePaths] = []
        self.actions: list[RepairAction] = []
        self.observations: list[RepairObservation] = []

    def __call__(
        self,
        repair_input: RepairInput,
        paths: WorkspacePaths,
    ) -> Mapping[RepairTool, Capability]:
        _check_value(repair_input)
        self.inputs.append(repair_input)
        self.paths.append(paths)

        def stub(action: RepairAction) -> RepairObservation:
            _check_value(action)
            references: list[str] = []

            # These fixture side effects only satisfy the runtime completion gate.
            # No WorkspaceCapability or TestCapability implementation is exercised.
            if action.tool == 'workspace' and action.call_id == 'call-004':
                (paths.source / 'pkg/value.py').write_text('VALUE = 2\n', encoding='utf-8')
            current_hash = workspace_hash(paths.source)
            if action.tool == 'test':
                evidence = write_json(artifact_path(paths.evidence, action.call_id), {
                    'kind': 'test',
                    'call_id': action.call_id,
                    'level': action.arguments['level'],
                    'status': 'success',
                    'workspace_hash': current_hash,
                    'return_code': 0,
                    'outputs': [],
                })
                references.append(str(evidence))
            observation = RepairObservation(
                call_id=action.call_id,
                status='success',
                summary=f'{action.tool} contract stub accepted the action',
                artifact_refs=references,
                workspace_hash=current_hash,
            )
            _check_value(observation)
            self.actions.append(action)
            self.observations.append(observation)
            return observation

        return {tool: stub for tool in EXTERNAL_TOOLS}


def _scenario_actions() -> list[RepairAction]:
    return [
        RepairAction('call-001', 'research', {'operation': 'search', 'query': 'repair contract'}),
        RepairAction('call-002', 'workspace', {
            'operation': 'write', 'path': 'work/probe.py', 'content': 'print("probe-ok")\n',
        }),
        RepairAction('call-003', 'shell', {
            'command': ['python', 'probe.py'], 'cwd': 'work', 'timeout_seconds': 30,
        }),
        RepairAction('call-004', 'workspace', {
            'operation': 'write', 'path': 'source/pkg/value.py', 'content': 'VALUE = 2\n',
        }),
        RepairAction('call-005', 'test', {'level': 'L0'}),
        RepairAction('call-006', 'test', {'level': 'L1'}),
        RepairAction('call-007', 'test', {'level': 'L2'}),
        RepairAction('call-008', 'finish', {'reason': 'all current-hash evidence is present'}),
    ]


def _check_workspace_layout(paths: WorkspacePaths) -> None:
    _require(_field_names(WorkspacePaths) == set(FROZEN_WORKSPACE_PATHS), 'WorkspacePaths fields changed')
    for field, expected in FROZEN_WORKSPACE_PATHS.items():
        actual_path = getattr(paths, field)
        actual = '.' if actual_path == paths.root else actual_path.relative_to(paths.root).as_posix()
        _require(actual == expected, f'workspace {field} moved: expected {expected}, got {actual}')

    _require({path.name for path in paths.root.iterdir()} == {'control', 'sandbox'}, 'run root layout changed')
    _require(
        {path.name for path in paths.control.iterdir()} == {
            'events.jsonl', 'evidence', 'logs', 'result.json', 'result.patch',
        },
        'control directory layout changed',
    )
    _require(
        {path.name for path in paths.sandbox.iterdir()} == {'source', 'work', 'context'},
        'sandbox directory layout changed',
    )
    _require(not (paths.sandbox / 'control').exists(), 'control must not be exposed inside sandbox')
    _require(stat.S_IMODE(paths.control.stat().st_mode) & 0o077 == 0, 'control directory is not private')


def test_repair_end_to_end_contract() -> None:
    with tempfile.TemporaryDirectory(prefix='lazyrag-repair-contract-') as temporary:
        base = Path(temporary)
        source = base / 'original'
        (source / 'pkg').mkdir(parents=True)
        original = source / 'pkg/value.py'
        original.write_text('VALUE = 1\n', encoding='utf-8')
        runtime_root = base / 'runtime'
        repair_input = RepairInput(
            run_id='contract-check',
            objective='Change the generated candidate and verify the fixed boundaries.',
            guidance='Keep the public Repair contracts unchanged.',
            source_ref=str(source),
            case_scope='pkg',
            constraints={'collaboration_check': True},
            budget={'turns': 12, 'seconds': 120},
        )
        _check_value(repair_input)
        agent = _ScriptedAgent(_scenario_actions())
        factory = _ContractStubFactory()

        result = RepairSession(agent, factory, runtime_root=runtime_root).run(repair_input)
        result_payload = _check_value(result)

        _require(result.status == 'success', f'end-to-end session returned {result.status}: {result.summary}')
        _require(len(factory.inputs) == 1 and factory.inputs[0] == repair_input, 'factory input changed')
        _require(len(factory.paths) == 1, 'factory must receive one WorkspacePaths value per invocation')
        paths = factory.paths[0]
        _check_workspace_layout(paths)
        _require(DEFAULT_RUNTIME_ROOT == Path('/tmp/lazyrag-repair'), 'default /tmp root changed')

        _require(
            [action.tool for action in factory.actions]
            == ['research', 'workspace', 'shell', 'workspace', 'test', 'test', 'test'],
            'dispatcher/capability action flow changed',
        )
        _require(
            [observation.call_id for observation in factory.observations]
            == [action.call_id for action in factory.actions],
            'RepairAction -> RepairObservation call_id correlation changed',
        )
        _require(agent.views and all(view.objective == repair_input.objective for view in agent.views),
                 'RepairInput.objective was not pinned into every RepairView')
        _require(agent.views and all(view.guidance == repair_input.guidance for view in agent.views),
                 'RepairInput.guidance was not pinned into every RepairView')
        _require(
            {item.get('level') for item in agent.views[-1].validation_evidence} == set(TEST_LEVELS),
            'final RepairView does not contain current L0/L1/L2 evidence',
        )
        _require(
            all(view.workspace_hash == workspace_hash(paths.source) for view in agent.views[-2:]),
            'latest RepairView workspace hash is not runtime-owned',
        )

        context_path = paths.context / 'repair_view.json'
        context_payload = json.loads(context_path.read_text(encoding='utf-8'))
        _require(set(context_payload) == set(FROZEN_CONTRACTS[RepairView]), 'repair_view.json fields changed')
        _require(stat.S_IMODE(context_path.stat().st_mode) & stat.S_IWUSR == 0,
                 'repair_view.json must be read-only for the Agent workspace')
        persisted_result = json.loads(paths.result.read_text(encoding='utf-8'))
        _require(persisted_result == result_payload, 'result.json differs from returned RepairResult')

        events = [json.loads(line) for line in paths.events.read_text(encoding='utf-8').splitlines()]
        _require(events, 'events.jsonl is empty')
        for event in events:
            _require(set(event) == EVENT_FIELDS, f'event envelope fields changed: {sorted(event)}')
            payload_fields = {
                'input.received': set(FROZEN_CONTRACTS[RepairInput]),
                'agent.action': set(FROZEN_CONTRACTS[RepairAction]),
                'capability.observation': set(FROZEN_CONTRACTS[RepairObservation]),
                'invocation.finished': set(FROZEN_CONTRACTS[RepairResult]),
            }.get(event['event'])
            if payload_fields is not None:
                _require(set(event['payload']) == payload_fields,
                         f'{event["event"]} payload fields changed: {sorted(event["payload"])}')

        patch = Path(result.patch_ref).read_text(encoding='utf-8')
        _require(bool(patch), 'successful RepairResult.patch_ref must reference a non-empty patch')
        _require(original.read_text(encoding='utf-8') == 'VALUE = 1\n', 'source_ref was modified in place')


CHECKS = (
    ('固定数据结构', test_repair_frozen_schemas),
    ('组件调用参数', test_repair_tool_boundary),
    ('主干流转与 /tmp 布局', test_repair_end_to_end_contract),
)


def main() -> int:
    print('Repair 协作契约自检')
    for name, check in CHECKS:
        try:
            check()
        except Exception as exc:
            print(f'[FAIL] {name}: {exc}', file=sys.stderr)
            return 1
        print(f'[PASS] {name}')
    print(f'PASS: {len(CHECKS)} checks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
