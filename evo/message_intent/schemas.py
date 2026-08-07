from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)


class MessageContentRef(StrictModel):
    uri: str = Field(max_length=512)
    sha256: str = Field(min_length=64, max_length=64)
    byte_size: int = Field(ge=0)


class MessageRequest(StrictModel):
    message_id: str = Field(default='', max_length=160)
    text: str = Field(min_length=1, max_length=20000)


class PendingConfirmation(StrictModel):
    confirmation_token: str = Field(max_length=80)
    expires_at: float
    origin_message_id: str = Field(max_length=160)
    base_observation_hash: str = Field(min_length=64, max_length=64)
    intent_ref: MessageContentRef


class FlowAction(StrictModel):
    kind: Literal['flow']
    command: Literal['start', 'approve', 'pause', 'resume', 'rerun', 'retry', 'cancel']
    stage: str = ''

    @model_validator(mode='after')
    def validate_stage(self) -> FlowAction:
        if self.command in {'approve', 'rerun'} and not self.stage:
            raise ValueError(f'{self.command} requires stage')
        if self.command not in {'approve', 'rerun', 'retry'} and self.stage:
            raise ValueError('stage is only valid for approve, rerun or retry')
        return self


class QueryAction(StrictModel):
    kind: Literal['query']
    query: Literal[
        'progress', 'run_history', 'stage_snapshot', 'case_snapshot',
        'operation_events', 'stage_result', 'artifact', 'artifact_history',
    ]
    stage: str = ''
    case_id: str = ''
    event_type: str = ''
    level: Literal['debug', 'info', 'warning', 'error'] | None = None
    limit: int = Field(default=50, ge=1, le=200)
    artifact_id: str = ''
    partition_key: str = ''
    version: int | None = Field(default=None, ge=1)

    @model_validator(mode='after')
    def validate_query(self) -> QueryAction:
        event_target = self.case_id or self.event_type or self.level is not None or self.limit != 50
        if self.query in {'progress', 'run_history'}:
            if self.stage or event_target or self.artifact_id or self.partition_key or self.version is not None:
                raise ValueError(f'{self.query} does not accept a query target')
            return self
        if self.query == 'operation_events':
            if self.artifact_id or self.partition_key or self.version is not None:
                raise ValueError('operation_events only accepts event filters')
            return self
        if self.query in {'stage_result', 'stage_snapshot'}:
            if not self.stage:
                raise ValueError(f'{self.query} requires stage')
            if event_target or self.artifact_id or self.partition_key or self.version is not None:
                raise ValueError(f'{self.query} only accepts stage')
            return self
        if self.query == 'case_snapshot':
            if not self.case_id:
                raise ValueError('case_snapshot requires case_id')
            if self.stage or self.event_type or self.level is not None or self.limit != 50:
                raise ValueError('case_snapshot only accepts case_id')
            if self.artifact_id or self.partition_key or self.version is not None:
                raise ValueError('case_snapshot only accepts case_id')
            return self
        if not self.artifact_id:
            raise ValueError(f'{self.query} requires artifact_id')
        if self.stage or event_target:
            raise ValueError(f'{self.query} only accepts an artifact target')
        if self.version is not None and self.query != 'artifact':
            raise ValueError('version is only valid for artifact')
        return self


class CaseAction(StrictModel):
    kind: Literal['case']
    command: Literal['rerun', 'retry']
    case_id: str = Field(min_length=1)
    stage: str = ''

    @model_validator(mode='after')
    def validate_command(self) -> CaseAction:
        if self.command == 'rerun' and not self.stage:
            raise ValueError('case rerun requires stage')
        if self.command != 'rerun' and self.stage:
            raise ValueError('stage is only valid for case rerun')
        return self


class RepairGuidanceAction(StrictModel):
    kind: Literal['repair_guidance']
    message: str = Field(min_length=1, max_length=4000)


class ConfirmationAction(StrictModel):
    kind: Literal['confirmation']
    decision: Literal['confirm', 'reject', 'amend', 'replace', 'unclear']
    confirmation_token: str = ''
    message: str = ''

    @model_validator(mode='after')
    def validate_confirmation(self) -> ConfirmationAction:
        if self.decision == 'confirm' and not self.confirmation_token:
            raise ValueError('confirm requires confirmation_token')
        return self


class ClarifyAction(StrictModel):
    kind: Literal['clarify']
    message: str = ''


class FinalAction(StrictModel):
    kind: Literal['final']
    message: str = ''


PlannedAction = Annotated[
    FlowAction | QueryAction | CaseAction | RepairGuidanceAction | ConfirmationAction | ClarifyAction | FinalAction,
    Field(discriminator='kind'),
]
PlannedActionAdapter = TypeAdapter(PlannedAction)


def parse_planned_action(value: Any) -> PlannedAction:
    return PlannedActionAdapter.validate_python(value)


class TurnPlan(StrictModel):
    turn_decision: Literal['next_action', 'needs_input', 'final']
    active_agenda: list[str] = Field(default_factory=list)
    next_action: PlannedAction | None = None
    user_message_effect: Literal['append', 'amend', 'replace', 'cancel', 'none'] = 'none'
    assistant_text: str = Field(default='', max_length=1000)

    @model_validator(mode='after')
    def validate_decision(self) -> TurnPlan:
        action = self.next_action
        if self.turn_decision == 'next_action':
            if action is None or action.kind in {'clarify', 'final'}:
                raise ValueError('next_action requires an executable action')
            return self
        if self.turn_decision == 'needs_input':
            if action is None:
                self.next_action = ClarifyAction(message=self.assistant_text)
            elif action.kind != 'clarify':
                raise ValueError('needs_input requires clarify')
            return self
        if action is None:
            self.next_action = FinalAction(message=self.assistant_text)
        elif action.kind != 'final':
            raise ValueError('final requires final')
        return self


class MessageTurnResult(StrictModel):
    thread_id: str
    turn_id: str
    message_id: str
    command_id: str = ''
    turn_decision: Literal[
        'needs_input', 'needs_confirmation', 'action_executed',
        'query_answered', 'final', 'rejected',
    ]
    assistant_text: str = ''
    observation_ref: MessageContentRef | None = None
    pending_confirmation_ref: MessageContentRef | None = None
    action_receipt_ref: MessageContentRef | None = None


class MessageHistoryItem(StrictModel):
    turn_id: str
    message_id: str
    command_id: str = ''
    status: str
    user_text: str = ''
    assistant_text: str = ''
    turn_decision: str = ''
    observation_ref: MessageContentRef | None = None
    pending_confirmation_ref: MessageContentRef | None = None
    action_receipt_ref: MessageContentRef | None = None


class MessageHistoryResponse(StrictModel):
    thread_id: str
    items: list[MessageHistoryItem]
    next_page_token: str = ''


__all__ = [
    'CaseAction', 'ClarifyAction', 'ConfirmationAction', 'FinalAction', 'FlowAction',
    'MessageContentRef', 'MessageHistoryItem', 'MessageHistoryResponse',
    'MessageRequest', 'MessageTurnResult', 'PendingConfirmation', 'PlannedAction',
    'QueryAction', 'RepairGuidanceAction', 'TurnPlan', 'parse_planned_action',
]
