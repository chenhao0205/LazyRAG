from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from evo.artifact_runtime import ArtifactRuntimeError, DefinitionError
from evo.message_intent import (
    MessageConflictError,
    MessageInProgressError,
    MessageRequest,
)

from .contracts import (
    AbStrategyBody,
    AlgorithmActionBody,
    ArtifactUpdateBody,
    AutomaticUpdateBody,
    CasePatchBody,
    CaseRerunBody,
    CaseStructureBody,
    CommandRequest,
    ConfigurationUpdateBody,
    ControlRequest,
    DatasetApplyBody,
    ExternalResultBody,
    GenerationPlanApplyBody,
    MessageBody,
    RegisterAlgorithmBody,
    RetryRequest,
    ServiceError,
    ThreadCreate,
    TopicApplyBody,
)
from .core import EvoService
from .events import execution_stream, message_stream
from .public import public_message_result


def create_app(root: str | Path | None = None) -> FastAPI:
    service_root = Path(root) if root is not None else _service_root()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = await EvoService.open(service_root)
        app.state.service = service
        try:
            yield
        finally:
            await service.close()

    app = FastAPI(
        title='evo service',
        version='v1.0.0',
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def service_error(_: Request, error: ServiceError) -> JSONResponse:
        return JSONResponse({'detail': error.detail}, status_code=error.status_code)

    @app.exception_handler(RequestValidationError)
    async def request_error(_: Request, error: RequestValidationError) -> JSONResponse:
        detail = '; '.join(item['msg'] for item in error.errors())
        return JSONResponse({'detail': detail}, status_code=422)

    @app.exception_handler(DefinitionError)
    async def definition_error(_: Request, error: DefinitionError) -> JSONResponse:
        detail = str(error)
        status = 404 if detail.startswith('run not found:') else 409
        return JSONResponse({'detail': detail}, status_code=status)

    @app.exception_handler(ArtifactRuntimeError)
    async def runtime_error(_: Request, error: ArtifactRuntimeError) -> JSONResponse:
        return JSONResponse({'detail': str(error)}, status_code=409)

    @app.get('/healthz')
    async def healthz() -> dict[str, bool]:
        return {'ok': True}

    @app.post('/threads')
    async def create_thread(payload: ThreadCreate) -> dict[str, Any]:
        return await _service(app).create_thread(payload)

    @app.get('/threads')
    async def list_threads(page_size: Annotated[int, Query(ge=1, le=200)] = 10, page_token: str = '', status: str = ''
                           ) -> dict[str, Any]:
        return await _service(app).list_threads(page_size, page_token, status)

    @app.get('/threads/{thread_id}')
    async def get_thread(thread_id: str) -> dict[str, Any]:
        return await _service(app).public_thread(thread_id)

    @app.delete('/threads/{thread_id}')
    async def delete_thread(thread_id: str) -> dict[str, Any]:
        return await _service(app).delete_thread(thread_id)

    @app.post('/threads/{thread_id}/start')
    async def start_thread(thread_id: str, payload: CommandRequest) -> dict[str, str]:
        return await _service(app).start(thread_id, payload)

    @app.post('/threads/{thread_id}/continue')
    async def continue_thread(thread_id: str, payload: CommandRequest) -> dict[str, str]:
        return await _service(app).continue_thread(thread_id, payload)

    @app.post('/threads/{thread_id}/retry')
    async def retry_thread(thread_id: str, payload: RetryRequest) -> dict[str, str]:
        return await _service(app).retry(thread_id, payload)

    @app.post('/threads/{thread_id}/stages/{stage}/rerun')
    async def rerun_stage(thread_id: str, stage: str, payload: ControlRequest) -> dict[str, str]:
        return await _service(app).rerun_stage(thread_id, stage, payload)

    @app.post('/threads/{thread_id}/cases/{case_id}/rerun')
    async def rerun_case(thread_id: str, case_id: str, payload: CaseRerunBody) -> dict[str, str]:
        return await _service(app).rerun_case(thread_id, case_id, payload)

    @app.post('/threads/{thread_id}/cases/{case_id}/retry')
    async def retry_case(thread_id: str, case_id: str, payload: ControlRequest) -> dict[str, str]:
        return await _service(app).retry_case(thread_id, case_id, payload)

    @app.post('/threads/{thread_id}/attempts/{attempt_id}/result')
    async def submit_external_result(thread_id: str, attempt_id: str, payload: ExternalResultBody) -> dict[str, str]:
        return await _service(app).submit_external_result(thread_id, attempt_id, payload)

    @app.post('/threads/{thread_id}/pause')
    async def pause_thread(thread_id: str, payload: ControlRequest) -> dict[str, str]:
        return await _service(app).pause(thread_id, payload)

    @app.post('/threads/{thread_id}/resume')
    async def resume_thread(thread_id: str, payload: ControlRequest) -> dict[str, str]:
        return await _service(app).resume(thread_id, payload)

    @app.post('/threads/{thread_id}/cancel')
    async def cancel_thread(thread_id: str, payload: ControlRequest) -> dict[str, str]:
        return await _service(app).cancel(thread_id, payload)

    @app.put('/threads/{thread_id}/artifacts')
    async def update_artifacts(thread_id: str, payload: ArtifactUpdateBody) -> dict[str, Any]:
        return await _service(app).update_artifacts(thread_id, payload)

    @app.get('/threads/{thread_id}/artifacts/{artifact_id}')
    async def get_artifact(thread_id: str, artifact_id: str, partition_key: str = '',
                           version: Annotated[int | None, Query(ge=1)] = None) -> dict[str, Any]:
        return await _service(app).projections.artifact(thread_id, artifact_id, partition_key, version)

    @app.get('/threads/{thread_id}/artifacts/{artifact_id}/history')
    async def get_artifact_history(thread_id: str, artifact_id: str, partition_key: str = '') -> dict[str, Any]:
        return await _service(app).projections.artifact_history(thread_id, artifact_id, partition_key)

    @app.put('/threads/{thread_id}/configuration')
    async def update_configuration(thread_id: str, payload: ConfigurationUpdateBody) -> dict[str, Any]:
        return await _service(app).update_configuration(thread_id, payload)

    @app.put('/threads/{thread_id}/automatic')
    async def set_automatic(thread_id: str, payload: AutomaticUpdateBody) -> dict[str, Any]:
        return await _service(app).set_automatic(thread_id, payload)

    @app.put('/threads/{thread_id}/cases')
    async def update_cases(thread_id: str, payload: CaseStructureBody) -> dict[str, Any]:
        return await _service(app).update_cases(thread_id, payload)

    @app.get('/threads/{thread_id}/stages/{stage}')
    async def stage_snapshot(thread_id: str, stage: str) -> dict[str, Any]:
        return await _service(app).projections.stage_snapshot(thread_id, stage)

    @app.get('/threads/{thread_id}/cases/{case_id}')
    async def case_snapshot(thread_id: str, case_id: str) -> dict[str, Any]:
        return await _service(app).projections.case_snapshot(thread_id, case_id)

    @app.get('/threads/{thread_id}/gates')
    async def gates(thread_id: str) -> dict[str, Any]:
        return await _service(app).projections.gates(thread_id)

    @app.get('/threads/{thread_id}/steps')
    async def steps(thread_id: str) -> dict[str, Any]:
        return await _service(app).projections.steps(thread_id)

    @app.get('/threads/{thread_id}/dataset/topics')
    async def dataset_topics(thread_id: str, question_type: str = '', min_chunk_count: str = '',
                             max_chunk_count: str = '', page_size: str = '',
                             page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.topics(
            thread_id,
            question_type=question_type,
            min_chunk_count=_optional_query_int(min_chunk_count, 'min_chunk_count'),
            max_chunk_count=_optional_query_int(max_chunk_count, 'max_chunk_count'),
            page_size=_optional_query_int(page_size, 'page_size'),
            page_token=page_token,
        )

    @app.get('/threads/{thread_id}/dataset/materials/overview')
    async def dataset_materials_overview(thread_id: str, request: Request) -> dict[str, Any]:
        _query_keys(request, set())
        return await _service(app).projections.materials_overview(thread_id)

    @app.get('/threads/{thread_id}/dataset/topics/overview')
    async def dataset_topics_overview(thread_id: str, request: Request) -> dict[str, Any]:
        _query_keys(request, set())
        return await _service(app).projections.topics_overview(thread_id)

    @app.get('/threads/{thread_id}/dataset/cases/overview')
    async def dataset_cases_overview(thread_id: str, request: Request) -> dict[str, Any]:
        _query_keys(request, set())
        return await _service(app).projections.cases_overview(thread_id)

    @app.get('/threads/{thread_id}/dataset/result')
    async def dataset_result(thread_id: str, page_size: str = '', page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.dataset_result(
            thread_id,
            page_size=_optional_query_int(page_size, 'page_size'),
            page_token=page_token,
        )

    @app.get('/threads/{thread_id}/dataset/result:download')
    async def dataset_result_download(thread_id: str, format: str, revision: str) -> Response:  # noqa: A002
        if format != 'csv':
            raise ServiceError(400, 'format must be csv')
        filename, payload = await _service(app).projections.dataset_result_download(thread_id, revision)
        return Response(
            payload,
            media_type='text/csv; charset=utf-8',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'},
        )

    @app.get('/threads/{thread_id}/dataset/cases')
    async def dataset_cases(thread_id: str, plan_status: str = '', generate_status: str = '',
                            grading_status: str = '', source: str = '', question_type: str = '',
                            difficulty: str = '', page_size: str = '', page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.cases(
            thread_id,
            plan_status=plan_status,
            generate_status=generate_status,
            grading_status=grading_status,
            source=source,
            question_type=question_type,
            difficulty=difficulty,
            page_size=_optional_query_int(page_size, 'page_size'),
            page_token=page_token,
        )

    @app.get('/threads/{thread_id}/dataset/cases/{case_id}')
    async def dataset_case_detail(thread_id: str, case_id: str, request: Request) -> dict[str, Any]:
        _query_keys(request, set())
        return await _service(app).projections.case_detail(thread_id, case_id)

    @app.patch('/threads/{thread_id}/dataset/cases/{case_id}')
    async def patch_dataset_case(thread_id: str, case_id: str, payload: CasePatchBody) -> dict[str, Any]:
        return await _service(app).patch_case(thread_id, case_id, payload)

    @app.get('/threads/{thread_id}/dataset/materials/documents')
    async def dataset_materials_documents(thread_id: str, included: str = '', knowledge_base_id: str = '',
                                          page_size: str = '', page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.materials_documents(
            thread_id,
            included=_optional_query_bool(included, 'included'),
            knowledge_base_id=knowledge_base_id,
            page_size=_optional_query_int(page_size, 'page_size'),
            page_token=page_token,
        )

    @app.get('/threads/{thread_id}/dataset/materials/adjustment-options')
    async def dataset_material_adjustment_options(thread_id: str, request: Request) -> dict[str, Any]:
        _query_keys(request, set())
        return await _service(app).projections.material_adjustment_options(thread_id)

    @app.post('/threads/{thread_id}/dataset/materials:apply')
    async def apply_dataset_materials(thread_id: str, payload: DatasetApplyBody) -> dict[str, Any]:
        return await _service(app).apply_materials(thread_id, payload)

    @app.get('/threads/{thread_id}/dataset/materials/knowledge-bases/{knowledge_base_id}/documents/{document_id}')
    async def dataset_material_document_detail(thread_id: str, knowledge_base_id: str, document_id: str,
                                               selected: str = '', split_rule: str = '', page_size: str = '',
                                               page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.material_document_detail(
            thread_id,
            knowledge_base_id,
            document_id,
            selected=_optional_query_bool(selected, 'selected'),
            split_rule=split_rule,
            page_size=_optional_query_int(page_size, 'page_size'),
            page_token=page_token,
        )

    @app.get('/threads/{thread_id}/dataset/topics/{topic_id}')
    async def dataset_topic_detail(thread_id: str, topic_id: str, page_size: str = '',
                                   page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.topic_detail(
            thread_id,
            topic_id,
            page_size=_optional_query_int(page_size, 'page_size'),
            page_token=page_token,
        )

    @app.post('/threads/{thread_id}/dataset/topics:apply')
    async def apply_dataset_topic_names(thread_id: str, payload: TopicApplyBody) -> dict[str, Any]:
        return await _service(app).apply_topic_names(thread_id, payload)

    @app.post('/threads/{thread_id}/dataset/generation-plan:apply')
    async def apply_dataset_generation_plan(thread_id: str, payload: GenerationPlanApplyBody) -> dict[str, Any]:
        return await _service(app).apply_generation_plan(thread_id, payload)

    @app.get('/threads/{thread_id}/dataset/cases/{case_id}/topic-options')
    async def dataset_case_topic_options(thread_id: str, case_id: str, page_size: str = '',
                                         page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.case_topic_options(
            thread_id, case_id, page_size=_optional_query_int(page_size, 'page_size'), page_token=page_token,
        )

    @app.get('/threads/{thread_id}/gates/{step}/versions/{version}:download')
    async def gate_download(thread_id: str, step: str, version: int,
                            format: str = 'json'  # noqa: A002
                            ) -> Response:
        if format != 'json':
            raise ServiceError(422, 'format must be json')
        filename, payload = await _service(app).projections.gate_download(
            thread_id,
            step,
            version,
        )
        return Response(
            payload,
            media_type='application/octet-stream',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'},
        )

    @app.get('/threads/{thread_id}/gates/{step}/versions/{version}')
    async def gate_content(thread_id: str, step: str, version: int) -> dict[str, Any]:
        return await _service(app).projections.gate_content(thread_id, step, version)

    @app.get('/threads/{thread_id}/events:stream')
    async def event_stream(thread_id: str, request: Request, step_id: str = '') -> Response:
        _query_keys(request, {'step_id'})
        return await execution_stream(
            _service(app).projections,
            thread_id,
            step_id,
            request,
        )

    @app.get('/threads/{thread_id}/event-trace:stream')
    async def event_trace_stream(thread_id: str, request: Request, step_id: str = '') -> Response:
        _query_keys(request, {'step_id'})
        if not step_id:
            raise ServiceError(422, 'step_id is required')
        return await execution_stream(
            _service(app).projections,
            thread_id,
            step_id,
            request,
        )

    @app.get('/threads/{thread_id}/gates/abtest/versions/{version}/case-details')
    async def abtest_case_details(thread_id: str, version: int, page_size: Annotated[int, Query(ge=1, le=200)] = 50,
                                  page_token: str = '', keyword: str = '', outcome: str = '') -> dict[str, Any]:
        return await _service(app).projections.abtest_case_details(
            thread_id, version, page_size, page_token, keyword, outcome,
        )

    @app.get('/threads/{thread_id}/results/traces/{trace_id}')
    async def trace_detail(thread_id: str, trace_id: str) -> dict[str, Any]:
        return await _service(app).projections.trace_detail(thread_id, trace_id)

    @app.get('/threads/{thread_id}/results/traces:compare')
    async def trace_compare(thread_id: str, a: Annotated[str, Query(min_length=1)],
                            b: Annotated[str, Query(min_length=1)]) -> dict[str, Any]:
        return await _service(app).projections.trace_compare(thread_id, a, b)

    @app.get('/threads/{thread_id}/messages')
    async def message_history(thread_id: str, page_size: Annotated[int, Query(ge=1, le=200)] = 50, page_token: str = ''
                              ) -> dict[str, Any]:
        return await _service(app).message_history(
            thread_id,
            page_size,
            page_token,
        )

    @app.post('/threads/{thread_id}/messages')
    async def messages(thread_id: str, payload: MessageBody, request: Request) -> Any:
        text = payload.message_text()
        if not text.strip():
            raise ServiceError(422, 'text or content is required')
        try:
            result = await _service(app).message(
                thread_id,
                MessageRequest(message_id=payload.message_id, text=text),
            )
        except (MessageConflictError, MessageInProgressError) as exc:
            raise ServiceError(409, str(exc)) from exc
        except sqlite3.OperationalError as exc:
            if 'locked' in str(exc).lower():
                raise ServiceError(
                    409,
                    'message store is busy; retry the same message_id',
                ) from exc
            raise
        public = public_message_result(result)
        if 'text/event-stream' in request.headers.get('accept', ''):
            return message_stream(public)
        return public

    @app.get('/candidates')
    async def candidates(thread_id: str = '', status: str = '', page_size: Annotated[int, Query(ge=1, le=200)] = 20,
                         page_token: str = '') -> dict[str, Any]:
        return await _service(app).projections.candidates(
            thread_id,
            status,
            page_size,
            page_token,
        )

    @app.get('/candidates/{candidate_id:path}')
    async def candidate(candidate_id: str) -> dict[str, Any]:
        return await _service(app).projections.candidate(candidate_id)

    @app.get('/router/status')
    async def router_status() -> dict[str, Any]:
        return await _service(app).router.status()

    @app.get('/router/algorithms')
    async def router_algorithms(thread_id: str = '', algorithm_id: str = '',
                                status: Literal['all', 'starting', 'active', 'disabled', 'missing'] = 'all'
                                ) -> dict[str, Any]:
        return await _service(app).router.algorithms(
            thread_id=thread_id,
            algorithm_id=algorithm_id,
            status=status,
        )

    @app.post('/router/algorithms')
    async def register_algorithm(payload: RegisterAlgorithmBody) -> dict[str, Any]:
        return await _service(app).router.register(payload)

    @app.post('/router/algorithms/{algorithm_id}/action')
    async def algorithm_action(algorithm_id: str, payload: AlgorithmActionBody) -> dict[str, Any]:
        return await _service(app).router.action(algorithm_id, payload)

    @app.delete('/router/algorithms/{algorithm_id}')
    async def delete_algorithm(algorithm_id: str) -> dict[str, Any]:
        return await _service(app).router.delete(algorithm_id)

    @app.get('/router/ab-strategy')
    async def get_ab_strategy() -> dict[str, Any]:
        return await _service(app).router.get_ab_strategy()

    @app.put('/router/ab-strategy')
    async def put_ab_strategy(payload: AbStrategyBody) -> dict[str, Any]:
        return await _service(app).router.put_ab_strategy(payload)

    return app


def _service(app: FastAPI) -> EvoService:
    return app.state.service


def _query_keys(request: Request, allowed: set[str]) -> None:
    unsupported = set(request.query_params) - allowed
    if unsupported:
        raise ServiceError(422, f'unsupported query param: {min(unsupported)}')


def _optional_query_int(value: str, field: str) -> int | None:
    if value == '':
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ServiceError(400, f'{field} must be an integer') from None


def _optional_query_bool(value: str, field: str) -> bool | None:
    if value == '':
        return None
    if value == 'true':
        return True
    if value == 'false':
        return False
    raise ServiceError(400, f'{field} must be a boolean')


def _service_root() -> Path:
    configured = os.getenv('LAZYMIND_EVO_BASE_DIR')
    return Path(configured) if configured else Path('/var/lib/lazymind/evo')


__all__ = ['create_app']
