from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, TypeAlias, TypedDict, cast, get_args


RepairTool: TypeAlias = Literal['workspace', 'shell', 'test', 'research', 'finish']
ObservationStatus: TypeAlias = Literal['success', 'fail', 'error']
ResultStatus: TypeAlias = Literal['success', 'partial', 'failed']
TestLevel: TypeAlias = Literal['L0', 'L1', 'L2']

REPAIR_TOOLS = cast(tuple[RepairTool, ...], get_args(RepairTool))
OBSERVATION_STATUSES = cast(tuple[ObservationStatus, ...], get_args(ObservationStatus))
RESULT_STATUSES = cast(tuple[ResultStatus, ...], get_args(ResultStatus))
TEST_LEVELS = cast(tuple[TestLevel, ...], get_args(TestLevel))
ACTION_ARGUMENT_FIELDS = {
    'workspace': ({'operation', 'path', 'content'}, {'operation'}),
    'shell': ({'command', 'cwd', 'timeout_seconds'}, {'command'}),
    'test': ({'level'}, {'level'}),
    'research': ({'operation', 'query', 'urls'}, {'operation', 'query'}),
    'finish': ({'reason'}, {'reason'}),
}


class WorkspaceArguments(TypedDict, total=False):
    operation: Literal['list', 'read', 'write', 'diff']
    path: str
    content: str


class ShellArguments(TypedDict, total=False):
    command: list[str]
    cwd: Literal['source', 'work']
    timeout_seconds: int


class TestArguments(TypedDict):
    level: TestLevel


class ResearchArguments(TypedDict, total=False):
    operation: Literal['search', 'read']
    query: str
    urls: list[str]


class FinishArguments(TypedDict):
    reason: str


class RepairError(RuntimeError):
    """Base exception for stable Repair skeleton failures."""

    def __init__(self, code: str, detail: str = '') -> None:
        self.code = _text(code, 'error code')
        self.detail = str(detail).strip()
        super().__init__(self.code, self.detail)

    def __str__(self) -> str:
        return self.code if not self.detail else f'{self.code}: {self.detail}'


class RepairContractError(RepairError):
    """A caller or component violated a frozen Repair contract."""


class RepairAgentError(RepairError):
    """The sole decision Agent failed to produce a valid model result."""


class RepairCapabilityError(RepairError):
    """A capability could not execute its mechanically valid request."""


@dataclass(frozen=True, slots=True)
class RepairInput:
    run_id: str
    objective: str
    guidance: str
    source_ref: str
    case_scope: str
    constraints: dict[str, Any]
    budget: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'run_id', _segment(self.run_id, 'run_id'))
        object.__setattr__(self, 'objective', _text(self.objective, 'objective'))
        object.__setattr__(self, 'guidance', str(self.guidance).strip())
        object.__setattr__(self, 'source_ref', _text(self.source_ref, 'source_ref'))
        object.__setattr__(self, 'case_scope', _text(self.case_scope, 'case_scope'))
        object.__setattr__(self, 'constraints', _dict(self.constraints, 'constraints'))
        object.__setattr__(self, 'budget', _dict(self.budget, 'budget'))


@dataclass(frozen=True, slots=True)
class RepairView:
    objective: str
    guidance: str
    workspace_hash: str
    diff_summary: str
    memory_brief: str
    recent_events: list[dict[str, Any]]
    validation_evidence: list[dict[str, Any]]
    remaining_budget: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'objective', _text(self.objective, 'objective'))
        object.__setattr__(self, 'guidance', str(self.guidance).strip())
        object.__setattr__(self, 'workspace_hash', _text(self.workspace_hash, 'workspace_hash'))
        object.__setattr__(self, 'diff_summary', str(self.diff_summary))
        object.__setattr__(self, 'memory_brief', str(self.memory_brief))
        object.__setattr__(self, 'recent_events', _dict_list(self.recent_events, 'recent_events'))
        object.__setattr__(
            self, 'validation_evidence', _dict_list(self.validation_evidence, 'validation_evidence'),
        )
        object.__setattr__(self, 'remaining_budget', _dict(self.remaining_budget, 'remaining_budget'))


@dataclass(frozen=True, slots=True)
class RepairAction:
    call_id: str
    tool: RepairTool
    arguments: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'call_id', _segment(self.call_id, 'call_id'))
        if self.tool not in REPAIR_TOOLS:
            raise RepairContractError('action_tool_invalid', str(self.tool))
        object.__setattr__(self, 'arguments', _arguments(self.tool, self.arguments))


@dataclass(frozen=True, slots=True)
class RepairObservation:
    call_id: str
    status: ObservationStatus
    summary: str
    artifact_refs: list[str]
    workspace_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, 'call_id', _text(self.call_id, 'call_id'))
        if self.status not in OBSERVATION_STATUSES:
            raise RepairContractError('observation_status_invalid', str(self.status))
        object.__setattr__(self, 'summary', _text(self.summary, 'summary'))
        object.__setattr__(self, 'artifact_refs', _strings(self.artifact_refs, 'artifact_refs'))
        object.__setattr__(self, 'workspace_hash', _text(self.workspace_hash, 'workspace_hash'))


@dataclass(frozen=True, slots=True)
class RepairResult:
    status: ResultStatus
    patch_ref: str
    evidence_refs: list[str]
    summary: str
    unresolved: list[str]

    def __post_init__(self) -> None:
        if self.status not in RESULT_STATUSES:
            raise RepairContractError('result_status_invalid', str(self.status))
        object.__setattr__(self, 'patch_ref', str(self.patch_ref).strip())
        object.__setattr__(self, 'evidence_refs', _strings(self.evidence_refs, 'evidence_refs'))
        object.__setattr__(self, 'summary', _text(self.summary, 'summary'))
        object.__setattr__(self, 'unresolved', _strings(self.unresolved, 'unresolved'))


def contract_dict(value: RepairInput | RepairView | RepairAction | RepairObservation | RepairResult
                  ) -> dict[str, Any]:
    return asdict(value)


def repair_action(value: Mapping[str, Any]) -> RepairAction:
    expected = {'call_id', 'tool', 'arguments'}
    if set(value) != expected:
        raise RepairContractError('action_fields_invalid', ','.join(sorted(set(value) ^ expected)))
    return RepairAction(
        call_id=value.get('call_id'),
        tool=value.get('tool'),
        arguments=value.get('arguments'),
    )


def _text(value: object, field: str) -> str:
    result = str(value or '').strip()
    if not result:
        raise RepairContractError('field_required', field)
    return result


def _segment(value: object, field: str) -> str:
    result = _text(value, field)
    if result in {'.', '..'} or any(character not in '-_.' and not character.isalnum() for character in result):
        raise RepairContractError(f'{field}_invalid', result)
    return result


def _dict(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RepairContractError('field_type_invalid', field)
    return dict(value)


def _arguments(tool: RepairTool, value: object) -> dict[str, Any]:
    result = _dict(value, 'arguments')
    allowed, required = ACTION_ARGUMENT_FIELDS[tool]
    unexpected = set(result) - allowed
    missing = required - set(result)
    if unexpected or missing:
        detail = f'unexpected={sorted(unexpected)}, missing={sorted(missing)}'
        raise RepairContractError('action_argument_fields_invalid', detail)
    if tool == 'finish':
        result['reason'] = _text(result['reason'], 'finish reason')
    return result


def _dict_list(value: object, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise RepairContractError('field_type_invalid', field)
    return [dict(item) for item in value]


def _strings(value: object, field: str) -> list[str]:
    if not isinstance(value, list):
        raise RepairContractError('field_type_invalid', field)
    return [str(item) for item in value]


__all__ = [
    'FinishArguments', 'ObservationStatus', 'RepairAction', 'RepairAgentError',
    'RepairCapabilityError', 'RepairContractError', 'RepairError', 'RepairInput',
    'RepairObservation', 'RepairResult', 'RepairTool', 'RepairView', 'ResearchArguments',
    'ShellArguments', 'TestArguments', 'TestLevel', 'WorkspaceArguments', 'contract_dict',
    'repair_action',
]
