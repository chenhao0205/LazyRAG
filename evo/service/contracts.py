from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ServiceError(RuntimeError):
    def __init__(self, status_code: int, detail: object) -> None:
        super().__init__(status_code, detail)
        self.status_code = status_code
        self.detail = detail


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ThreadInputs(StrictModel):
    kb_id: list[str] = Field(default_factory=list)
    knowledge_base_names: dict[str, str] = Field(default_factory=dict)
    csv_data: list[dict[str, str]] = Field(default_factory=list)
    imported_cases: list[dict[str, Any]] = Field(default_factory=list)
    supplement_existing_eval_set: bool = False
    router_chat_url: str = Field(min_length=1)
    router_admin_url: str = Field(min_length=1)
    algorithm_id: str = Field(min_length=1)
    num_case: int = Field(gt=0)
    case_deadline_seconds: float = Field(default=300.0, gt=0)

    @model_validator(mode='after')
    def validate_sources(self) -> Self:
        self.kb_id = [item.strip() for item in self.kb_id if item.strip()]
        self.knowledge_base_names = {
            key.strip(): value.strip()
            for key, value in self.knowledge_base_names.items()
            if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip()
        }
        rows: list[dict[str, str]] = []
        for row in self.csv_data:
            if len(row) != 1:
                raise ValueError('each csv_data item must contain one kb_id and csv_path')
            kb_id, path = next(iter(row.items()))
            if not kb_id.strip() or not path.strip():
                raise ValueError('csv_data kb_id and csv_path must be non-empty')
            rows.append({kb_id.strip(): path.strip()})
        self.csv_data = rows
        if not self.kb_id and not self.csv_data:
            raise ValueError('inputs.kb_id or inputs.csv_data is required')
        return self


class ThreadCreate(StrictModel):
    mode: Literal['interactive'] = 'interactive'
    automatic: bool = False
    title: str = ''
    inputs: ThreadInputs
    llm_config: dict[str, Any]

    @model_validator(mode='after')
    def validate_models(self) -> Self:
        required = ('llm', 'evo_llm', 'embed_main')
        missing = [name for name in required if not isinstance(self.llm_config.get(name), dict)]
        if missing:
            raise ValueError(f'llm_config requires model roles: {", ".join(missing)}')
        forbidden = {
            'eval_policy', 'repair_policy', 'candidate_config',
            'abtest_candidate_config',
        } & self.llm_config.keys()
        if forbidden:
            raise ValueError('llm_config cannot contain stage policy keys')
        return self


class CommandRequest(StrictModel):
    command_id: str = ''
    until_step: str = ''


class RetryRequest(StrictModel):
    command_id: str = ''
    stage: str = ''


class CaseRerunBody(StrictModel):
    command_id: str = ''
    stage: str = ''
    artifact_id: str = ''

    @model_validator(mode='after')
    def validate_start(self) -> Self:
        if bool(self.stage) == bool(self.artifact_id):
            raise ValueError('case rerun requires exactly one of stage or artifact_id')
        return self


class ControlRequest(StrictModel):
    command_id: str = ''


class AutomaticUpdateBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    base_version: int = Field(ge=1)
    enabled: bool


class ExternalResultBody(StrictModel):
    values: dict[str, Any]


class ArtifactValue(StrictModel):
    artifact_id: str = Field(min_length=1)
    partition_key: str = ''
    base_version: int = Field(ge=1)
    value: Any


class ArtifactUpdateBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    updates: list[ArtifactValue] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_targets(self) -> Self:
        targets = [(item.artifact_id, item.partition_key) for item in self.updates]
        if len(set(targets)) != len(targets):
            raise ValueError('updates cannot contain the same artifact twice')
        return self


class DatasetApplyBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    expected_revision: str = Field(min_length=1)
    changes: dict[str, Any]


class TopicApplyBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    expected_revision: str = Field(min_length=1)
    changes: list[dict[str, Any]] = Field(min_length=1)


class GenerationPlanApplyBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    expected_revision: str = Field(min_length=1)
    distribution: dict[str, dict[str, int]]

    @model_validator(mode='after')
    def validate_distribution(self) -> Self:
        expected_types = {'precision', 'reasoning'}
        expected_difficulties = {'easy', 'medium', 'hard'}
        if set(self.distribution) != expected_types:
            raise ValueError('distribution must contain precision and reasoning')
        for question_type, counts in self.distribution.items():
            if set(counts) != expected_difficulties:
                raise ValueError(f'distribution.{question_type} must contain easy, medium and hard')
            if any(isinstance(count, bool) or count < 0 for count in counts.values()):
                raise ValueError('distribution values must be non-negative integers')
        return self


class CasePatchBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    expected_revision: str = Field(min_length=1)
    changes: dict[str, Any]

    @model_validator(mode='after')
    def validate_changes(self) -> Self:
        if not self.changes or set(self.changes) - {'plan', 'generate', 'grading'}:
            raise ValueError('changes must contain plan, generate or grading')
        return self


_QAPLAN_LANES = frozenset({
    'precision_easy', 'precision_medium', 'precision_hard',
    'reasoning_easy', 'reasoning_medium', 'reasoning_hard',
})


class ConfigurationUpdateBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    target: Literal[
        'run_config', 'source_config', 'qaplan_plan_params', 'target_config', 'eval_policy',
        'repair_policy', 'candidate_config',
    ]
    base_version: int = Field(ge=1)
    value: dict[str, Any]

    @model_validator(mode='after')
    def validate_qaplan_plan_params(self) -> Self:
        if self.target != 'qaplan_plan_params':
            return self
        if 'lane_ratios' in self.value:
            raise ValueError('qaplan_plan_params uses lane_case_counts, not lane_ratios')
        if set(self.value) - {'lane_case_counts'}:
            raise ValueError('qaplan_plan_params only supports lane_case_counts')
        counts = self.value.get('lane_case_counts')
        if counts is None:
            return self
        if not isinstance(counts, dict) or set(counts) != _QAPLAN_LANES:
            raise ValueError('lane_case_counts must contain exactly the six lanes')
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts.values()):
            raise ValueError('lane_case_counts values must be non-negative integers')
        return self


class CaseSeed(StrictModel):
    artifact_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    value: Any


class CaseStructureBody(StrictModel):
    request_id: str = Field(min_length=1, max_length=160)
    partition_set_id: str = Field(min_length=1)
    base_version: int = Field(ge=1)
    case_ids: list[str]
    seeds: list[CaseSeed] = Field(default_factory=list)

    @model_validator(mode='after')
    def validate_cases(self) -> Self:
        if any(not case_id.strip() for case_id in self.case_ids):
            raise ValueError('case_ids must contain non-empty values')
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError('case_ids must be unique')
        targets = [(seed.artifact_id, seed.case_id) for seed in self.seeds]
        if len(set(targets)) != len(targets):
            raise ValueError('seeds cannot contain the same artifact twice')
        return self


class MessageBody(StrictModel):
    message_id: str = Field(default='', max_length=160)
    text: str = Field(default='', max_length=20000)
    content: str = Field(default='', max_length=20000)

    def message_text(self) -> str:
        return self.text if self.text.strip() else self.content


class AlgorithmOwner(StrictModel):
    thread_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')
    run_id: str = ''
    candidate_ref: str = ''

    @model_validator(mode='after')
    def validate_run(self) -> Self:
        if self.run_id and self.run_id != self.thread_id:
            raise ValueError('owner.run_id must match owner.thread_id')
        return self


class RegisterAlgorithmBody(StrictModel):
    algorithm_id: str = Field(min_length=1)
    name: str = ''
    code_path: str = Field(min_length=1)
    instance_count: int = Field(default=1, ge=1, le=4)
    config: dict[str, Any] = Field(default_factory=dict)
    owner: AlgorithmOwner
    wait_ready_seconds: float = Field(default=180.0, gt=0, le=900)
    cleanup_policy: Literal['thread_delete', 'manual'] = 'thread_delete'


class AlgorithmActionBody(StrictModel):
    action: Literal['healthcheck', 'start', 'restart', 'stop']
    wait_ready_seconds: float = Field(default=180.0, gt=0, le=900)


class AbStrategyBody(StrictModel):
    weights: dict[str, int] | None = None
    reason: str = ''
    owner: AlgorithmOwner | None = None


__all__ = [
    'AbStrategyBody', 'AlgorithmActionBody', 'AlgorithmOwner', 'ArtifactUpdateBody', 'ArtifactValue',
    'AutomaticUpdateBody', 'CasePatchBody', 'CaseRerunBody', 'CaseSeed', 'CaseStructureBody', 'CommandRequest',
    'ConfigurationUpdateBody', 'ControlRequest', 'DatasetApplyBody', 'ExternalResultBody', 'GenerationPlanApplyBody', 'MessageBody', 'RegisterAlgorithmBody',
    'RetryRequest',
    'ServiceError', 'StrictModel', 'ThreadCreate', 'ThreadInputs', 'TopicApplyBody',
]
