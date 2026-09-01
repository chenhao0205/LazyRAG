from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping
from typing import Any

from evo import artifacts as A
from evo.artifact_flow import FlowDefinition, FlowSnapshot
from evo.artifact_runtime import (
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    AttemptSnapshot,
    PartitionSet,
    RecordedOperationEvent,
)
from evo.operations.eval.summary import (
    build_eval_detail_summary,
    build_eval_frontend_view,
    is_scored_badcase,
)

from .contracts import ServiceError
from .public import public_thread_state, public_value


class ProjectionService:
    def __init__(self, flow: Any, definition: FlowDefinition) -> None:
        self.flow = flow
        self.definition = definition

    async def gates(self, thread_id: str) -> dict[str, Any]:
        history = await self.flow.run_history(thread_id)
        snapshot = history.snapshot
        gates = []
        for stage in self.definition.stages:
            records = tuple(record for record in history.runtime.artifacts if record.ref.key == stage.result_key)
            effective = snapshot.runtime.completed_artifacts.get(stage.result_key)
            versions = [record.ref.version for record in records]
            gates.append({
                'step': stage.name,
                'artifact_id': stage.result_key.artifact_id,
                'versions': versions,
                'effective_version': None if effective is None else effective.version,
                'latest_version': max(versions, default=None),
            })
        return {'thread_id': thread_id, 'gates': gates}

    async def stage_snapshot(self, thread_id: str, stage: str) -> dict[str, Any]:
        if stage not in A.STEPS:
            raise ServiceError(422, f'stage must be one of: {", ".join(A.STEPS)}')
        return {'thread_id': thread_id, 'stage': stage, 'snapshot': public_value(
            await self.flow.stage_snapshot(thread_id, stage),
        )}

    async def case_snapshot(self, thread_id: str, case_id: str) -> dict[str, Any]:
        return {'thread_id': thread_id, 'case_id': case_id, 'snapshot': public_value(
            await self.flow.case_snapshot(thread_id, case_id),
        )}

    async def artifact(self, thread_id: str, artifact_id: str, partition_key: str = '',
                       version: int | None = None) -> dict[str, Any]:
        key = ArtifactKey(artifact_id, partition_key)
        record = (
            await self.flow.head(thread_id, key)
            if version is None
            else await self.flow.record(thread_id, ArtifactRef(key, version))
        )
        if record is None:
            raise ServiceError(404, 'artifact version not found')
        return {
            'thread_id': thread_id,
            'record': public_value(record),
            'value': public_value(await self.flow.read(thread_id, record.ref)),
        }

    async def artifact_history(self, thread_id: str, artifact_id: str, partition_key: str = '') -> dict[str, Any]:
        records = await self.flow.history(thread_id, ArtifactKey(artifact_id, partition_key))
        return {
            'thread_id': thread_id,
            'items': public_value(records),
            'total_size': len(records),
        }

    async def gate_content(self, thread_id: str, stage: str, version: int) -> dict[str, Any]:
        ref = await self._gate_ref(thread_id, stage, version)
        return {
            'thread_id': thread_id,
            'step': stage,
            'version': version,
            'content': public_value(await self.flow.read(thread_id, ref)),
        }

    async def gate_download(self, thread_id: str, stage: str, version: int) -> tuple[str, bytes]:
        content = (await self.gate_content(thread_id, stage, version))['content']
        return (
            f'{stage}-v{version}.json',
            json.dumps(
                content,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode(),
        )

    async def steps(self, thread_id: str) -> dict[str, Any]:
        _, _, pages = await _execution_projection(
            self.flow,
            self.definition,
            thread_id,
        )
        return {
            'thread_id': thread_id,
            'active_step_id': next(
                (item['step_id'] for item in pages if item['active']),
                '',
            ),
            'items': [_public_step(item) for item in pages],
            'total_size': len(pages),
        }

    async def events(self, thread_id: str, step_id: str = '', after_event_id: str = '') -> dict[str, Any]:
        return await execution_events(
            self.flow,
            self.definition,
            thread_id,
            step_id=step_id,
            after_event_id=after_event_id,
        )

    async def abtest_case_details(self, thread_id: str, version: int, page_size: int, page_token: str,
                                  keyword: str = '', outcome: str = '') -> dict[str, Any]:
        value = (await self.gate_content(thread_id, 'abtest', version))['content']
        data = value if isinstance(value, Mapping) else {}
        summary = data.get('summary') if isinstance(data.get('summary'), Mapping) else {}
        rows = _rows(data.get('case_deltas') or summary.get('case_deltas'))
        rows = rows or _comparison_rows(data)
        rows = _filter(rows, keyword, ('case_id', 'query', 'outcome'))
        if outcome:
            rows = [row for row in rows if row.get('outcome') == outcome]
        return _page(rows, page_size, page_token)

    async def trace_detail(self, thread_id: str, trace_id: str) -> dict[str, Any]:
        await self.flow.snapshot(thread_id)
        from evo.traces import build_trace_detail_view

        value = await asyncio.to_thread(build_trace_detail_view, trace_id)
        return public_value(value)

    async def trace_compare(self, thread_id: str, left: str, right: str) -> dict[str, Any]:
        await self.flow.snapshot(thread_id)
        from evo.traces import build_trace_compare_view

        value = await asyncio.to_thread(build_trace_compare_view, left, right)
        return public_value(value)

    async def candidates(self, thread_id: str, status: str, page_size: int, page_token: str) -> dict[str, Any]:
        if thread_id and not await self.flow.has_run(thread_id):
            raise ServiceError(404, f'thread not found: {thread_id}')
        run_ids = (thread_id,) if thread_id else await self.flow.run_ids()
        items = []
        for run_id in run_ids:
            for stage in ('repair', 'abtest'):
                key = ArtifactKey.scalar(A.ROOTS[stage])
                for record in await self.flow.history(run_id, key):
                    value = await self.flow.read(run_id, record.ref)
                    items.append(_candidate(run_id, stage, record.ref, value))
        if status:
            items = [item for item in items if item['status'] == status]
        items.sort(key=lambda item: item['candidate_id'])
        if page_token:
            items = [item for item in items if item['candidate_id'] > page_token]
        page = items[:page_size]
        return {
            'items': page,
            'next_page_token': (
                page[-1]['candidate_id'] if len(page) == page_size else ''
            ),
        }

    async def candidate(self, candidate_id: str) -> dict[str, Any]:
        thread_id, artifact, version = _parse_candidate(candidate_id)
        stage = next(
            (
                stage for stage in ('repair', 'abtest')
                if A.ROOTS[stage] == artifact
            ),
            '',
        )
        if not stage:
            raise ServiceError(404, 'candidate not found')
        ref = await self._gate_ref(thread_id, stage, version)
        value = await self.flow.read(thread_id, ref)
        return _candidate(thread_id, stage, ref, value, detail=True)

    async def eval_bad_cases(
        self,
        thread_id: str,
        version: int,
        page_size: int,
        page_token: str,
        keyword: str = '',
        failure_type: str = '',
    ) -> dict[str, Any]:
        record = await self._gate_record(thread_id, 'eval', version)
        summary = await _eval_detail_summary_from_record(self.flow, thread_id, record)
        rows = [
            _bad_case_item(row)
            for row in _rows(summary.get('bad_cases'))
            if is_scored_badcase(row)
        ]
        rows = _filter(rows, keyword, ('case_id', 'question', 'defect', 'reason', 'failure_type'))
        if failure_type:
            rows = [row for row in rows if row.get('failure_type') == failure_type]
        return _page(rows, page_size, page_token)

    async def eval_overview(self, thread_id: str, version: int) -> dict[str, Any]:
        record = await self._gate_record(thread_id, 'eval', version)
        config = await _run_config(self.flow, thread_id)
        summary = await _eval_detail_summary_from_record(
            self.flow,
            thread_id,
            record,
            live_progress=_eval_live_progress(record, config),
        )
        return {
            'thread_id': thread_id,
            'step': 'eval',
            'version': version,
            'content': public_value({
                'overview': summary.get('overview') or {},
                'case_overviews': summary.get('case_overviews') or [],
                'case_details': summary.get('case_details') or [],
                'guides': summary.get('guides') or {},
                'metrics': summary.get('metrics') or {},
                'quality_counts': summary.get('quality_counts') or {},
                'failure_type_counts': summary.get('failure_type_counts') or {},
                'retrieval_failure_type_counts': summary.get('retrieval_failure_type_counts') or {},
            }),
        }

    async def _gate_record(self, thread_id: str, stage: str, version: int) -> ArtifactRecord:
        ref = await self._gate_ref(thread_id, stage, version)
        record = await self.flow.record(thread_id, ref)
        if record is None:
            raise ServiceError(404, 'gate artifact version not found')
        return record

    async def _gate_ref(self, thread_id: str, stage: str, version: int) -> ArtifactRef:
        if stage not in A.ROOTS:
            raise ServiceError(422, f'step must be one of: {", ".join(A.STEPS)}')
        if version < 1:
            raise ServiceError(422, 'version must be positive')
        ref = ArtifactRef(ArtifactKey.scalar(A.ROOTS[stage]), version)
        if await self.flow.record(thread_id, ref) is None:
            raise ServiceError(404, 'gate artifact version not found')
        return ref


def _stage_status(stage_status: str, flow_status: str) -> str:
    if stage_status == 'awaiting_approval':
        return 'completed'
    return 'idle' if stage_status == 'pending' and flow_status == 'idle' else stage_status


def _comparison_rows(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    origin = value.get('origin') if isinstance(value.get('origin'), Mapping) else {}
    candidate = value.get('candidate') if isinstance(value.get('candidate'), Mapping) else {}
    before = {str(row.get('case_id') or ''): row for row in _rows(origin.get('cases'))}
    after = {str(row.get('case_id') or ''): row for row in _rows(candidate.get('cases'))}
    result = []
    for case_id in dict.fromkeys((*before, *after)):
        left, right = before.get(case_id, {}), after.get(case_id, {})
        old = float(left.get('overall') or left.get('overall_score') or 0)
        new = float(right.get('overall') or right.get('overall_score') or 0)
        result.append({
            'case_id': case_id,
            'outcome': (
                'improved' if new > old
                else 'regressed' if new < old
                else 'unchanged'
            ),
            'before': dict(left),
            'after': dict(right),
            'delta': {'overall_score': round(new - old, 4)},
        })
    return result


def _candidate(thread_id: str, stage: str, ref: ArtifactRef, value: object, *, detail: bool = False) -> dict[str, Any]:
    data = value if isinstance(value, Mapping) else {}
    row = {
        'candidate_id': f'{thread_id}:{ref.key.artifact_id}@v{ref.version}',
        'thread_id': thread_id,
        'source_step': stage,
        'source_ref': f'{ref.key.artifact_id}@v{ref.version}',
        'status': str(data.get('status') or ''),
        'summary': public_value({
            key: data[key]
            for key in ('status', 'verdict', 'algo_id', 'candidate_algo_id')
            if key in data
        }),
    }
    if detail:
        diff = data.get('diff') if isinstance(data.get('diff'), Mapping) else {}
        row['files'] = [str(path) for path in diff if '/' not in str(path)]
    return row


def _parse_candidate(value: str) -> tuple[str, str, int]:
    try:
        thread_id, ref = value.split(':', 1)
        artifact, version = ref.rsplit('@v', 1)
        return thread_id, artifact, int(version)
    except ValueError as exc:
        raise ServiceError(404, 'candidate not found') from exc


def _rows(value: object) -> list[dict[str, Any]]:
    return [dict(row) for row in value] if isinstance(value, list) else []


def _filter(rows: list[dict[str, Any]], keyword: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    keyword = keyword.strip().lower()
    return rows if not keyword else [
        row for row in rows
        if any(keyword in str(row.get(field) or '').lower() for field in fields)
    ]


def _page(rows: list[dict[str, Any]], size: int, token: str) -> dict[str, Any]:
    if not str(token or '0').isdigit():
        raise ServiceError(422, 'page_token must be a non-negative integer offset')
    offset = int(token or 0)
    page = rows[offset:offset + size]
    return {
        'items': public_value(page),
        'next_page_token': str(offset + size) if offset + size < len(rows) else '',
        'total_size': len(rows),
    }


async def _eval_detail_summary_from_record(
    flow: Any,
    thread_id: str,
    record: ArtifactRecord,
    *,
    live_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = await flow.read(thread_id, record.ref)
    value = value if isinstance(value, Mapping) else {}
    if 'bad_cases' in value or 'rows' in value:
        detail = dict(value)
        detail.setdefault(
            'bad_cases',
            [row for row in _rows(value.get('rows')) if row.get('quality_label') != 'good'],
        )
        detail.update(build_eval_frontend_view(detail, live_progress=live_progress))
        return detail

    judge_refs = [
        ref for ref in record.input_refs
        if ref.key.artifact_id == A.EVAL_JUDGE_RESULT
    ]
    if not judge_refs:
        raise ServiceError(409, f'{_ref_text(record.ref)} has no {A.EVAL_JUDGE_RESULT} provenance')
    values = await flow.read_many(thread_id, judge_refs)
    judges = []
    for ref in judge_refs:
        value = values.get(ref)
        if isinstance(value, Mapping):
            judges.append(value)
    summary = build_eval_detail_summary(tuple(judges))
    summary.update(build_eval_frontend_view(summary, live_progress=live_progress))
    return summary


def _eval_live_progress(record: ArtifactRecord, config: Mapping[str, Any]) -> dict[str, Any]:
    answer_ids = [
        ref.key.partition_key
        for ref in record.input_refs
        if ref.key.artifact_id == A.EVAL_RAG_ANSWER and ref.key.partition_key
    ]
    judged_ids = {
        ref.key.partition_key
        for ref in record.input_refs
        if ref.key.artifact_id == A.EVAL_JUDGE_RESULT and ref.key.partition_key
    }
    if not answer_ids and not judged_ids:
        judged_ids = {
            ref.key.partition_key
            for ref in record.input_refs
            if ref.key.partition_key
        }
    running_ids = [case_id for case_id in answer_ids if case_id not in judged_ids]
    completed_cases = len(judged_ids)
    answered_cases = len({case_id for case_id in answer_ids if case_id}) or completed_cases
    total_cases = max(_num_case(config), answered_cases, completed_cases)
    running_cases = len(running_ids)
    pending_cases = max(0, total_cases - answered_cases)
    current_case = {'case_id': '', 'stage': '', 'stage_label': ''}
    if running_ids:
        current_case = {
            'case_id': running_ids[-1],
            'stage': 'multi_dimension_judge',
            'stage_label': '多维评测',
        }
    elif pending_cases > 0 and answered_cases < total_cases:
        current_case = {
            'case_id': '',
            'stage': 'answer_generation',
            'stage_label': '生成回答',
        }
        running_cases = max(running_cases, 1)
        pending_cases = max(0, total_cases - completed_cases - running_cases)
    return {
        'total_cases': total_cases,
        'completed_cases': completed_cases,
        'running_cases': running_cases,
        'pending_cases': pending_cases,
        'current_case': current_case,
    }


async def _run_config(flow: Any, thread_id: str) -> Mapping[str, Any]:
    history = await flow.history(thread_id, ArtifactKey.scalar(A.RUN_CONFIG))
    if not history:
        return {}
    value = await flow.read(thread_id, history[-1].ref)
    return value if isinstance(value, Mapping) else {}


def _bad_case_item(row: Mapping[str, Any]) -> dict[str, Any]:
    score_source = (
        row.get('answer_correctness')
        if row.get('answer_correctness') is not None
        else row.get('overall_score')
    )
    try:
        score = round(float(score_source or 0.0), 4)
    except (TypeError, ValueError):
        score = 0.0
    payload = dict(row)
    payload.update({
        'query': row.get('query') or row.get('question'),
        'reference': row.get('reference') or row.get('ground_truth'),
        'answer': row.get('answer') or row.get('rag_answer'),
        'score': score,
        'Defect': row.get('Defect') or row.get('defect'),
        'Reason': row.get('Reason') or row.get('reason'),
        'trace_status': row.get('trace_status') or ('linked' if row.get('trace_id') else ''),
        'failure_detail': (
            row.get('failure_detail')
            or row.get('chat_error_message')
            or row.get('reason')
            or row.get('failure_type')
        ),
    })
    return payload


def _num_case(config: Mapping[str, Any]) -> int:
    inputs = config.get('inputs') if isinstance(config.get('inputs'), Mapping) else {}
    return int(config.get('num_case') or inputs.get('num_case') or 0)


def _ref_text(ref: ArtifactRef) -> str:
    key = (
        ref.key.artifact_id
        if not ref.key.partition_key
        else f'{ref.key.artifact_id}[{ref.key.partition_key}]'
    )
    return f'{key}@v{ref.version}'


_TERMINAL_ATTEMPTS = frozenset({
    'cancelled', 'succeeded', 'failed', 'interrupted', 'discarded',
})
_TERMINAL_STEPS = frozenset({
    'completed', 'paused', 'cancelled', 'canceled', 'failed',
})
_STEP_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    'lazyrag:evo:step-events:v1',
)


async def execution_events(flow: Any, definition: FlowDefinition, run_id: str, *, step_id: str = '',
                           after_event_id: str = '') -> dict[str, Any]:
    snapshot, items, pages = await _execution_projection(
        flow,
        definition,
        run_id,
    )
    page = None
    if step_id:
        page = _resolve_step(snapshot, pages, step_id)
        items = [item for item in items if item['step_id'] == step_id]
    if after_event_id:
        items = events_after(items, after_event_id)
    terminal = _terminal(snapshot) if page is None else _step_terminal(page)
    return {
        'thread_id': run_id,
        'step_id': step_id or None,
        'items': items,
        'terminal': terminal,
        'reason': _stream_end_reason(snapshot, page) if terminal else '',
        **public_thread_state(snapshot),
    }


async def _execution_projection(
    flow: Any, definition: FlowDefinition, run_id: str,
) -> tuple[FlowSnapshot, list[dict[str, Any]], list[dict[str, Any]],]:
    history = await flow.run_history(run_id)
    snapshot = history.snapshot
    attempts = history.runtime.attempts
    operation_events = history.runtime.operation_events
    results = {
        stage.name: tuple(record for record in history.runtime.artifacts if record.ref.key == stage.result_key)
        for stage in definition.stages
    }
    approvals = {
        stage.name: tuple(record for record in history.runtime.artifacts if record.ref.key == stage.approval_key)
        for stage in definition.stages
    }

    items = flow_events(
        snapshot,
        attempts,
        operation_events,
        results,
        approvals,
        await _historical_partition_sets(flow, run_id, attempts),
        definition,
    )
    pages = _step_pages(snapshot, items)
    _link_execution_pages(snapshot, items, pages)
    return snapshot, items, pages


def flow_events(snapshot: FlowSnapshot, attempts: tuple[AttemptSnapshot, ...],
                operation_events: tuple[RecordedOperationEvent, ...], results: Mapping[str, tuple[ArtifactRecord, ...]],
                approvals: Mapping[str, tuple[ArtifactRecord, ...]], partition_sets: Mapping[ArtifactRef, PartitionSet],
                definition: FlowDefinition) -> list[dict[str, Any]]:
    rows: list[tuple[float, int, str, dict[str, Any]]] = []
    events_by_attempt: dict[str, list[RecordedOperationEvent]] = {}
    for event in operation_events:
        events_by_attempt.setdefault(event.attempt_id, []).append(event)
    completions: dict[ArtifactKey, list[AttemptSnapshot]] = {}

    for attempt in sorted(attempts, key=lambda item: (item.created_at, item.attempt_id)):
        stage = _operation_stage(definition, attempt.operation_id)
        if not stage:
            continue
        case = attempt_case(snapshot, attempt, partition_sets)
        base = {
            'thread_id': snapshot.run_id,
            'step_id': f'{snapshot.run_id}:{stage}',
            'stage': stage,
            'next_step_id': '',
            'next_step_run_id': '',
            'operation_id': attempt.operation_id,
            'attempt_id': attempt.attempt_id,
            **({'case': case} if case is not None else {}),
        }
        rows.append((
            attempt.created_at,
            0,
            attempt.attempt_id,
            {
                **base,
                'event_id': f'{snapshot.run_id}:{attempt.attempt_id}:start',
                'event_type': attempt.operation_id,
                'status': 'running',
                'timestamp': attempt.created_at,
            },
        ))
        for event in sorted(
            events_by_attempt.get(attempt.attempt_id, ()),
            key=lambda item: item.sequence,
        ):
            rows.append((
                event.created_at,
                1,
                f'{attempt.attempt_id}:{event.sequence}',
                {
                    **base,
                    'event_id': f'{snapshot.run_id}:operation-event:{event.sequence}',
                    'event_type': event.event.event_type,
                    'level': event.event.level,
                    'timestamp': event.created_at,
                    'message': event.event.message,
                    'data': public_value(event.event.data),
                    **({'status': event.event.status} if event.event.status is not None else {}),
                    **(
                        {'progress': {'current': event.event.current, 'total': event.event.total}}
                        if event.event.current is not None or event.event.total is not None
                        else {}
                    ),
                },
            ))
        if attempt.status not in _TERMINAL_ATTEMPTS:
            continue

        finished = attempt.finished_at or attempt.started_at or attempt.created_at
        rows.append((
            finished,
            2,
            attempt.attempt_id,
            {
                **base,
                'event_id': f'{snapshot.run_id}:{attempt.attempt_id}:terminal',
                'event_type': attempt.operation_id,
                'status': _attempt_status(attempt.status),
                'timestamp': finished,
                **(
                    {'message': attempt.error.message}
                    if attempt.error is not None
                    else {}
                ),
            },
        ))
        if attempt.status == 'succeeded':
            for key in attempt.output_keys:
                completions.setdefault(key, []).append(attempt)

    result_attempts: dict[ArtifactRef, AttemptSnapshot | None] = {}
    result_counts: dict[ArtifactKey, int] = {}
    for stage, records in results.items():
        for index, record in enumerate(records):
            attempt = None
            if record.producer.startswith('operation:'):
                result_index = result_counts.get(record.ref.key, 0)
                attempt = _matching_attempt(completions, record, result_index)
                result_counts[record.ref.key] = result_index + 1
            result_attempts[record.ref] = attempt
            timestamp = _attempt_time(attempt, index)
            rows.append((
                timestamp,
                3,
                f'{stage}:{record.ref.version}',
                {
                    'thread_id': snapshot.run_id,
                    'step_id': f'{snapshot.run_id}:{stage}',
                    'stage': stage,
                    'next_step_id': '',
                    'next_step_run_id': '',
                    'event_id': f'{snapshot.run_id}:{stage}:v{record.ref.version}',
                    'event_type': 'step.finish',
                    'status': 'completed',
                    'timestamp': timestamp,
                    'artifact': _artifact(record),
                },
            ))

    for stage, records in approvals.items():
        stage_results = {record.ref: record for record in results.get(stage, ())}
        for index, record in enumerate(records):
            result = next(
                (stage_results[ref] for ref in record.input_refs if ref in stage_results),
                None,
            )
            attempt = None if result is None else result_attempts.get(result.ref)
            timestamp = _attempt_time(attempt, index)
            rows.append((
                timestamp,
                4,
                f'{stage}:approval:{record.ref.version}',
                {
                    'thread_id': snapshot.run_id,
                    'step_id': f'{snapshot.run_id}:{stage}',
                    'stage': stage,
                    'next_step_id': '',
                    'next_step_run_id': '',
                    'event_id': (
                        f'{snapshot.run_id}:{stage}:'
                        f'approval:v{record.ref.version}'
                    ),
                    'event_type': 'checkpoint.continue',
                    'status': 'completed',
                    'timestamp': timestamp,
                    'artifact': _artifact(record),
                },
            ))

    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    items = [row[3] for row in rows]
    _assign_step_ids(snapshot.run_id, items)
    return items


def _assign_step_ids(run_id: str, items: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    last_by_stage: dict[str, str] = {}
    current_stage = ''
    current_step_id = ''
    closed = True

    for item in items:
        stage = item['stage']
        if item['event_type'] == 'checkpoint.continue' and stage in last_by_stage:
            item['step_id'] = last_by_stage[stage]
            continue
        if stage != current_stage or closed:
            counts[stage] = counts.get(stage, 0) + 1
            current_stage = stage
            current_step_id = _step_id(run_id, stage, counts[stage])
            last_by_stage[stage] = current_step_id
            closed = False
        item['step_id'] = current_step_id
        if item['event_type'] == 'step.finish':
            closed = True


def _step_pages(snapshot: FlowSnapshot, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        step_id = item['step_id']
        page = by_id.get(step_id)
        if page is None:
            page = {
                'thread_id': snapshot.run_id,
                'step_id': step_id,
                'stage': item['stage'],
                'title': item['stage'],
                'order_index': len(pages),
                'event_count': 0,
                'next_step_id': '',
                'next_step_run_id': '',
                'version': None,
                'status': 'running',
                'continues_previous': bool(
                    pages and pages[-1]['stage'] == item['stage']
                ),
                'active': False,
                '_closed': False,
                '_started_at': item['timestamp'],
                '_ended_at': item['timestamp'],
                '_next_started_at': None,
                '_next_stage': '',
            }
            by_id[step_id] = page
            pages.append(page)

        page['event_count'] += 1
        page['_started_at'] = min(page['_started_at'], item['timestamp'])
        page['_ended_at'] = max(page['_ended_at'], item['timestamp'])
        if item['event_type'] == 'step.finish':
            page['status'] = 'completed'
            page['version'] = item['artifact']['version']
            page['_closed'] = True
        elif item.get('status') in {'failed', 'canceled'}:
            page['status'] = item['status']
        elif item.get('status') == 'running':
            page['status'] = 'running'

    _append_current_page(snapshot, pages)
    current = next(
        (page for page in reversed(pages) if page['stage'] == snapshot.current_stage),
        None,
    )
    if current is not None and not current['_closed']:
        progress = next(
            stage for stage in snapshot.stages
            if stage.stage == snapshot.current_stage
        )
        current['status'] = _stage_status(progress.status, snapshot.status)

    for index, page in enumerate(pages):
        page['order_index'] = index
        page['continues_previous'] = bool(
            index and pages[index - 1]['stage'] == page['stage']
        )
        page['active'] = (
            index == len(pages) - 1
            and page['status'] in {'running', 'pausing', 'cancelling'}
        )
    return pages


def _append_current_page(snapshot: FlowSnapshot, pages: list[dict[str, Any]]) -> None:
    if snapshot.status == 'completed':
        return
    progress = next(
        stage for stage in snapshot.stages
        if stage.stage == snapshot.current_stage
    )
    latest = next(
        (page for page in reversed(pages) if page['stage'] == progress.stage),
        None,
    )
    settled = progress.status in {'awaiting_approval', 'completed'}
    if settled and latest is not None:
        return
    if latest is not None and latest is pages[-1] and not latest['_closed']:
        return

    count = sum(page['stage'] == progress.stage for page in pages) + 1
    pages.append({
        'thread_id': snapshot.run_id,
        'step_id': _step_id(snapshot.run_id, progress.stage, count),
        'stage': progress.stage,
        'title': progress.stage,
        'order_index': len(pages),
        'event_count': 0,
        'next_step_id': '',
        'next_step_run_id': '',
        'version': (
            None if progress.result_ref is None else progress.result_ref.version
        ),
        'status': _stage_status(progress.status, snapshot.status),
        'continues_previous': bool(
            pages and pages[-1]['stage'] == progress.stage
        ),
        'active': False,
        '_closed': False,
        '_started_at': None,
        '_ended_at': None,
        '_next_started_at': None,
        '_next_stage': '',
    })


def _link_execution_pages(snapshot: FlowSnapshot, items: list[dict[str, Any]], pages: list[dict[str, Any]]) -> None:
    for index, page in enumerate(pages):
        next_page = pages[index + 1] if index + 1 < len(pages) else None
        if next_page is not None:
            page['next_step_id'] = next_page['step_id']
            page['next_step_run_id'] = next_page['step_id']
            page['_next_started_at'] = next_page['_started_at']
            page['_next_stage'] = next_page['stage']
            continue

        next_stage = _next_stage(snapshot, page['stage']) if page['_closed'] else ''
        page['_next_stage'] = next_stage
        if next_stage:
            count = sum(item['stage'] == next_stage for item in pages) + 1
            page['next_step_run_id'] = _step_id(
                snapshot.run_id,
                next_stage,
                count,
            )

    by_id = {page['step_id']: page for page in pages}
    for item in items:
        page = by_id[item['step_id']]
        item['next_step_id'] = page['next_step_id']
        item['next_step_run_id'] = page['next_step_run_id']


def _next_stage(snapshot: FlowSnapshot, stage: str) -> str:
    index = next(
        index for index, progress in enumerate(snapshot.stages)
        if progress.stage == stage
    )
    return (
        snapshot.stages[index + 1].stage
        if index + 1 < len(snapshot.stages)
        else ''
    )


def _step_id(run_id: str, stage: str, ordinal: int) -> str:
    return str(uuid.uuid5(
        _STEP_ID_NAMESPACE,
        f'{run_id}:{stage}:{ordinal}',
    ))


def _resolve_step(snapshot: FlowSnapshot, pages: list[dict[str, Any]], step_id: str) -> dict[str, Any]:
    page = next((item for item in pages if item['step_id'] == step_id), None)
    if page is not None:
        return page
    if pages and pages[-1]['next_step_run_id'] == step_id:
        source = pages[-1]
        return {
            'thread_id': snapshot.run_id,
            'step_id': step_id,
            'stage': source['_next_stage'],
            'title': source['_next_stage'],
            'order_index': len(pages),
            'event_count': 0,
            'next_step_id': '',
            'next_step_run_id': '',
            'version': None,
            'status': 'pending',
            'continues_previous': source['stage'] == source['_next_stage'],
            'active': False,
            '_closed': False,
            '_started_at': None,
            '_ended_at': None,
            '_next_started_at': None,
            '_next_stage': '',
        }
    raise ServiceError(422, 'unknown step_id for thread')


def _public_step(page: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in page.items()
        if not key.startswith('_') and key != 'thread_id'
    }


def _step_terminal(page: Mapping[str, Any]) -> bool:
    return bool(page['next_step_id']) or page['status'] in _TERMINAL_STEPS


def _stream_end_reason(snapshot: FlowSnapshot, page: Mapping[str, Any] | None) -> str:
    if page is not None:
        page_status = str(page['status'])
        if page_status == 'completed' or page['next_step_id']:
            pending = snapshot.pending_approval
            if pending is not None and pending.stage == page['stage']:
                return 'checkpoint_wait'
            if (
                snapshot.status == 'completed'
                and snapshot.current_stage == page['stage']
            ):
                return 'flow_completed'
            return 'step_completed'
        if page_status in {'cancelled', 'canceled'}:
            return 'cancelled'
        if page_status == 'failed':
            return 'failed'
        if page_status == 'paused':
            return 'user_paused'

    if snapshot.pending_approval is not None:
        return 'checkpoint_wait'
    return {
        'paused': 'user_paused',
        'cancelled': 'cancelled',
        'failed': 'failed',
        'completed': 'flow_completed',
    }.get(snapshot.status, 'step_completed')


def events_after(items: list[dict[str, Any]], event_id: str) -> list[dict[str, Any]]:
    for index, item in enumerate(items):
        if item['event_id'] == event_id:
            return items[index + 1:]
    raise ServiceError(422, 'unknown event_id for event scope')


def attempt_case(snapshot: FlowSnapshot, attempt: AttemptSnapshot,
                 historical: Mapping[ArtifactRef, PartitionSet] | None = None) -> dict[str, Any] | None:
    partition_key = attempt.partition_key
    if not partition_key:
        return None
    output = next((key for key in attempt.output_keys if key.partition_key), None)
    if output is None:
        return {'id': partition_key}
    partition_set_id = A.PARTITION_SET_BY_ARTIFACT.get(output.artifact_id)
    if partition_set_id is None:
        return {'id': partition_key}

    partitions = next(
        (
            (historical or {}).get(ref)
            for ref in attempt.input_refs
            if ref.key == ArtifactKey.scalar(partition_set_id)
        ),
        None,
    )
    if partitions is None:
        partitions = snapshot.runtime.partition_sets.get(
            ArtifactKey.scalar(partition_set_id)
        )
    if partitions is None or partition_key not in partitions:
        return {'id': partition_key}
    return {
        'id': partition_key,
        'index': partitions.keys.index(partition_key) + 1,
        'total': len(partitions.keys),
    }


async def _historical_partition_sets(flow: Any, run_id: str, attempts: tuple[AttemptSnapshot, ...]
                                     ) -> Mapping[ArtifactRef, PartitionSet]:
    partition_ids = frozenset(A.PARTITION_SET_BY_ARTIFACT.values())
    refs = tuple(dict.fromkeys(
        ref
        for attempt in attempts
        for ref in attempt.input_refs
        if ref.key.artifact_id in partition_ids
    ))
    if not refs:
        return {}
    values = await flow.read_many(run_id, refs)
    return {
        ref: value
        for ref, value in values.items()
        if isinstance(value, PartitionSet)
    }


def _operation_stage(definition: FlowDefinition, operation_id: str) -> str:
    index = definition.stage_index_for_operation(operation_id)
    return '' if index is None else definition.stages[index].name


def _terminal(snapshot: FlowSnapshot) -> bool:
    if snapshot.status in {
        'paused', 'awaiting_approval', 'cancelled', 'failed', 'completed',
    }:
        return True
    return False


def _matching_attempt(completions: Mapping[ArtifactKey, list[AttemptSnapshot]], record: ArtifactRecord, index: int
                      ) -> AttemptSnapshot | None:
    attempts = completions.get(record.ref.key, ())
    return attempts[index] if index < len(attempts) else None


def _attempt_time(attempt: AttemptSnapshot | None, fallback: int) -> float:
    if attempt is None:
        return float(fallback)
    return attempt.finished_at or attempt.started_at or attempt.created_at


def _attempt_status(status: str) -> str:
    return {
        'succeeded': 'completed',
        'cancelled': 'canceled',
        'interrupted': 'canceled',
        'discarded': 'canceled',
    }.get(status, status)


def _artifact(record: ArtifactRecord) -> dict[str, Any]:
    return {
        'artifact_id': record.ref.key.artifact_id,
        'partition_key': record.ref.key.partition_key,
        'version': record.ref.version,
        'ref': f'{record.ref.key.artifact_id}@v{record.ref.version}',
    }


__all__ = [
    'ProjectionService', 'attempt_case', 'events_after', 'flow_events',
]
