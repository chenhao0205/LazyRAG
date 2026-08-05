from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evo.llm import LazyLLMClient
from evo.traces.detail import build_trace_detail_view

from .agent import ModelCallError, ModelCallTimeout, _bounded_json, next_action
from .contracts import build_supported_plan, select_category, validate_analysis
from .demo import request_http, run_command
from .memory import WorkMemory, content_ref, write_json
from .opencode import EvoModelConfigError, OpenCodeSession, build_opencode_settings
from .validation import inside_repair_scope, repair_scope
from .web import read_web_pages, search_web


DEFAULT_BUDGET = {
    'turns': 20,
    'web_searches': 6,
    'page_reads': 12,
    'opencode_calls': 10,
    'command_runs': 10,
    'http_requests': 6,
    'seconds': 1800,
}
DEFAULT_MODEL_TIMEOUT_SECONDS = 120
MAX_CONSECUTIVE_MODEL_FAILURES = 3
TRACE_ERROR_MARKERS = (
    'connection refused', 'failed to establish', 'error', 'exception', 'timeout',
    'timed out', 'unavailable', '503',
)


def build_repair_plan(
    run_id: str,
    analysis_value: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        analysis = validate_analysis(analysis_value)
    except (TypeError, ValueError):
        return _failure('blocked', 'unverified_root_cause')
    category_id, category = select_category(analysis['categories'])
    if not inside_repair_scope(
        category['code_span'], policy.get('allowed_roots'), policy.get('blocked_roots'),
    ):
        return _failure('blocked', 'target_outside_repair_scope')
    result = run_phase1(
        str(run_id),
        {
            'category_id': category_id,
            'source_hash': analysis['source_hash'],
            'category': category,
        },
        policy,
    )
    if result.get('status') != 'supported':
        status = str(result.get('status') or 'failed')
        return _failure(
            status if status in {'blocked', 'exhausted', 'failed'} else 'failed',
            str(result.get('reason') or 'phase1_failed'),
        )
    guidance = [
        str(item).strip()
        for item in policy.get('user_guidance') or ()
        if str(item).strip()
    ]
    try:
        plan = build_supported_plan(category_id, category, result, guidance)
    except (TypeError, ValueError):
        return _failure('failed', 'phase1_invalid_result')
    return plan


def run_phase1(
    run_id: str,
    target: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    memory: WorkMemory | None = None
    session: OpenCodeSession | None = None
    try:
        guidance = policy.get('user_guidance') or []
        if not isinstance(guidance, (list, tuple)):
            raise ValueError('phase1_user_guidance_invalid')
        trace_evidence = _load_trace_evidence(target)
        if not trace_evidence:
            return _terminal('blocked', 'trace_evidence_unavailable')
        source_value = (
            policy.get('candidate_source_dir')
            or os.getenv('LAZYMIND_EVO_CHAT_SOURCE')
            or '/app/algorithm'
        )
        scope = repair_scope(policy.get('allowed_roots'), policy.get('blocked_roots'))
        memory = WorkMemory.create(
            run_id,
            target,
            policy,
            Path(str(source_value)).resolve(),
            str(target.get('source_hash') or ''),
            scope,
        )
        budget = _limits(policy.get('phase1_budget'))
        try:
            config = build_opencode_settings(_llm_config(policy).get('evo_llm'))
        except EvoModelConfigError as exc:
            return _terminal('blocked', exc.reason)
        restored = memory.restored_session
        session = OpenCodeSession(
            category_id=str(target.get('category_id') or ''),
            workdir=memory.work_root,
            artifact_root=memory.artifact_root,
            config=config,
            # Reserve enough of the Phase-1 wall-clock budget for the Agent to
            # observe a stalled executor and choose another action.
            timeout_s=min(
                int(policy.get('opencode_timeout_s') or 900),
                max(60, min(600, budget['seconds'] // 6)),
            ),
            session_id=str(restored.get('session_id') or ''),
            calls=int(restored.get('calls') or 0),
        )
        client = LazyLLMClient(llm_config=_llm_config(policy), model='evo_llm')
        model_timeout = _positive_seconds(
            policy.get('phase1_llm_timeout_s'), DEFAULT_MODEL_TIMEOUT_SECONDS,
        )
        counters = {key: 0 for key in DEFAULT_BUDGET if key != 'seconds'}
        deadline = time.monotonic() + budget['seconds']
        memory.record(
            'phase1.started',
            f"started category {target.get('category_id')}",
            {
                'trace_evidence': trace_evidence,
                'resumed_session': bool(session.session_id),
                'guidance': memory.guidance,
            },
        )
        consecutive_model_failures = 0
        for turn_no in range(1, budget['turns'] + 1):
            counters['turns'] = turn_no
            if time.monotonic() >= deadline:
                return _terminal('exhausted', 'phase1_time_budget_exhausted')
            try:
                action = next_action(
                    client,
                    memory.context(counters, budget),
                    min(model_timeout, deadline - time.monotonic()),
                )
            except (ModelCallTimeout, ModelCallError) as exc:
                consecutive_model_failures += 1
                reason = (
                    'phase1_model_timeout'
                    if isinstance(exc, ModelCallTimeout)
                    else f'phase1_model_error:{exc}'
                )
                memory.record('model.failed', reason, {'turn': turn_no, 'reason': reason})
                if consecutive_model_failures < MAX_CONSECUTIVE_MODEL_FAILURES:
                    continue
                return _terminal('failed', reason)
            consecutive_model_failures = 0
            memory.record(
                'agent.decision',
                f'{action.action}: {action.reason}',
                {'turn': turn_no, 'action': action.action, 'request': action.request},
            )
            try:
                result = _execute_action(
                    action.action,
                    action.reason,
                    action.request,
                    memory,
                    session,
                    counters,
                    budget,
                    policy,
                    deadline,
                )
            except (TypeError, ValueError) as exc:
                memory.record(
                    'action.rejected',
                    str(exc),
                    {'action': action.action, 'error_type': type(exc).__name__},
                )
                continue
            if result is not None:
                return result
        return _terminal('exhausted', 'phase1_turn_budget_exhausted')
    except Exception as exc:
        if memory is not None:
            memory.record(
                'phase1.failed',
                f'{type(exc).__name__}: {exc}',
                {'error_type': type(exc).__name__, 'reason': str(exc)},
            )
        return _terminal('failed', f'phase1_error:{type(exc).__name__}')
    finally:
        if memory is not None:
            if session is not None:
                try:
                    memory.checkpoint(session.session_id, session.calls)
                except Exception as exc:
                    memory.record(
                        'checkpoint.failed',
                        f'{type(exc).__name__}: {exc}',
                        {'error_type': type(exc).__name__, 'reason': str(exc)},
                    )
            memory.close()


def _execute_action(
    action: str,
    reason: str,
    request: Mapping[str, Any],
    memory: WorkMemory,
    session: OpenCodeSession,
    counters: dict[str, int],
    budget: Mapping[str, int],
    policy: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any] | None:
    if action == 'search_web':
        _consume(counters, budget, 'web_searches')
        query = _text(request, 'query')
        result = search_web(query, memory.artifact_root)
        memory.record(
            'web.search',
            f"{result.get('status')}: {query}; {len(result.get('results') or ())} results",
            result,
        )
        return None
    if action == 'read_web':
        urls = _strings(request.get('urls'), maximum=3)
        if not urls:
            raise ValueError('read_web_urls_missing')
        fresh = [url for url in urls if url not in memory.read_urls()]
        if not fresh:
            raise ValueError('read_web_urls_already_read')
        if any(url not in memory.known_urls() for url in fresh):
            raise ValueError('read_web_url_not_discovered')
        _consume(counters, budget, 'page_reads', len(fresh))
        result = read_web_pages(
            _text(request, 'question'),
            fresh,
            memory.work_root,
            memory.artifact_root,
            seen_urls=memory.read_urls(),
        )
        memory.record(
            'web.read',
            f"read {len(result.get('pages') or ())} pages",
            {'content_trust': 'external_untrusted', **result},
        )
        return None
    if action == 'opencode':
        if memory.consecutive_failures('opencode.result') >= 2:
            memory.record(
                'phase1.stopped',
                'OpenCode failed twice without a successful observation',
                {'status': 'blocked', 'reason': 'opencode_repeated_failure'},
            )
            return _terminal('blocked', 'opencode_repeated_failure')
        _consume(counters, budget, 'opencode_calls')
        instruction = _text(request, 'instruction')
        expected = str(request.get('expected_result') or '').strip()
        memory.write_context(counters, budget)
        result = session.run(
            instruction,
            expected,
            max(0.1, deadline - time.monotonic()),
        )
        summary = str(result.get('report', {}).get('summary') or result.get('reason') or 'OpenCode finished')
        memory.record('opencode.result', summary, result)
        if result.get('invalid_changes'):
            memory.restore_source()
            raise ValueError('phase1_workspace_tainted')
        memory.checkpoint(session.session_id, session.calls)
        return None
    if action == 'run_command':
        _consume(counters, budget, 'command_runs')
        command = request.get('command')
        if not isinstance(command, list):
            raise ValueError('run_command_argv_required')
        expected = str(request.get('expected_result') or '').strip()
        result = run_command(
            memory.work_root,
            memory.artifact_root,
            command,
            attempt=counters['command_runs'],
            timeout_seconds=min(180.0, max(0.1, deadline - time.monotonic())),
            output_limit=256 * 1024,
            expected_source_hash=memory.source_digest,
        )
        memory.record(
            'command.result',
            f"{result['status']}: {' '.join(result['command'][:6])}; expected={expected}",
            {**result, 'expected_result': expected},
        )
        memory.checkpoint(session.session_id, session.calls)
        return None
    if action == 'http_request':
        _consume(counters, budget, 'http_requests')
        result = request_http(
            _text(request, 'url'),
            str(request.get('method') or 'GET'),
            _allowed_origins(policy),
            memory.artifact_root,
            attempt=counters['http_requests'],
            timeout_seconds=min(15.0, max(0.1, deadline - time.monotonic())),
        )
        memory.record(
            'http.result',
            f"{result['method']} {result['url']} -> {result.get('status_code') or result['status']}",
            result,
        )
        return None
    if action == 'finish':
        proposal = {
            'target': _text(request, 'target'),
            'change': _text(request, 'change'),
            'expected_result': _text(request, 'expected_result'),
        }
        if gaps := memory.completion_gaps(proposal):
            memory.record(
                'finish.rejected',
                f"explicit user requirements remain: {', '.join(gaps)}",
                {'gaps': gaps, 'proposal': proposal},
            )
            return None
        evidence = _selected_evidence(request.get('evidence_uris'), memory.evidence_refs())
        memory.record('phase1.finished', reason, {'proposal': proposal})
        workspace_ref = memory.checkpoint(session.session_id, session.calls)
        result_path = memory.artifact_root / 'result.json'
        write_json(
            result_path,
            {'proposal': proposal, 'evidence_refs': evidence, 'reason': reason},
        )
        return {
            'status': 'supported',
            'proposal': proposal,
            'validation': {
                'verdict': 'supports',
                'reason': reason,
                'evidence_refs': evidence,
                'result_ref': content_ref(result_path, memory.artifact_root),
                'workspace_ref': workspace_ref,
                'journal_ref': memory.journal_ref(),
            },
        }
    if action == 'stop':
        status = str(request.get('status') or 'blocked').strip()
        if status not in {'blocked', 'exhausted', 'failed'}:
            raise ValueError('stop_status_invalid')
        stop_reason = str(request.get('reason') or reason).strip()
        memory.record('phase1.stopped', stop_reason, {'status': status})
        return _terminal(status, stop_reason)
    raise ValueError(f'unknown_agent_action:{action}')


def _load_trace_evidence(target: Mapping[str, Any]) -> list[dict[str, Any]]:
    category = target.get('category') if isinstance(target.get('category'), Mapping) else {}
    result = []
    for case_id, trace_id in list((category.get('cases') or {}).items())[:2]:
        detail = build_trace_detail_view(str(trace_id))
        if detail.get('trace_status') != 'success':
            continue
        nodes = []
        trace = detail.get('trace') if isinstance(detail.get('trace'), Mapping) else {}
        root = trace.get('root') if isinstance(trace.get('root'), Mapping) else None
        stack = [root] if root else []
        while stack and len(nodes) < 40:
            node = stack.pop()
            raw_node = {
                'name': node.get('name'),
                'type': node.get('type'),
                'status': node.get('status'),
            }
            detail_text = _bounded_json(
                {
                    'input': node.get('input'),
                    'output': node.get('output'),
                    'metadata': node.get('metadata'),
                },
                1600,
            )
            if str(node.get('status') or '').casefold() not in {'ok', 'success', 'completed'} or any(
                marker in detail_text.casefold() for marker in TRACE_ERROR_MARKERS
            ):
                raw_node['error_evidence'] = detail_text
                nodes.append(raw_node)
            stack.extend(reversed([
                child for child in node.get('children') or () if isinstance(child, Mapping)
            ]))
        result.append({
            'case_id': str(case_id),
            'trace_id': str(trace_id),
            'query': detail.get('query'),
            'summary': detail.get('summary'),
            'nodes': nodes,
        })
    return result


def _failure(status: str, reason: str) -> dict[str, str]:
    return {'id': 'repair.plan', 'status': status, 'reason': reason}


def _terminal(status: str, reason: str) -> dict[str, Any]:
    return {'status': status, 'reason': str(reason or status)}


def _limits(value: object) -> dict[str, int]:
    raw = value if isinstance(value, Mapping) else {}
    result = {}
    for key, default in DEFAULT_BUDGET.items():
        candidate = raw.get(key, default)
        if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0:
            raise ValueError(f'phase1_budget_invalid:{key}')
        result[key] = min(candidate, 7200 if key == 'seconds' else 100)
    return result


def _consume(counters: dict[str, int], budget: Mapping[str, int], key: str, amount: int = 1) -> None:
    if counters[key] + amount > budget[key]:
        raise ValueError(f'{key}_budget_exhausted')
    counters[key] += amount


def _text(value: Mapping[str, Any], key: str) -> str:
    text = str(value.get(key) or '').strip()
    if not text:
        raise ValueError(f'{key}_missing')
    return text


def _strings(value: object, maximum: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:maximum]


def _selected_evidence(value: object, available: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    requested = _strings(value, maximum=8)
    by_uri = {str(item.get('uri') or ''): dict(item) for item in available}
    if not requested or any(uri not in by_uri for uri in requested):
        raise ValueError('finish_requires_decisive_evidence_uris')
    return [by_uri[uri] for uri in dict.fromkeys(requested)]


def _positive_seconds(value: object, default: float) -> float:
    candidate = default if value is None else value
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)) or candidate <= 0:
        raise ValueError('phase1_llm_timeout_invalid')
    return min(float(candidate), 300.0)


def _allowed_origins(policy: Mapping[str, Any]) -> list[str]:
    configured = policy.get('phase1_demo_allowed_origins') or []
    if not isinstance(configured, (list, tuple)):
        raise ValueError('phase1_demo_allowed_origins_invalid')
    values = [str(item).strip() for item in configured if str(item).strip()]
    for name in (
        'LAZYMIND_DOCUMENT_PROCESSOR_URL',
        'LAZYMIND_EVO_TARGET_CHAT_URL',
        'LAZYMIND_EVO_KB_BASE_URL',
        'LAZYMIND_EVO_CHUNK_BASE_URL',
    ):
        if value := os.getenv(name, '').strip():
            values.append(value)
    result = []
    for value in values:
        parsed = urlsplit(value)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname:
            raise ValueError(f'phase1_demo_allowed_origin_invalid:{value}')
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        origin = f'{parsed.scheme}://{parsed.hostname}:{port}'
        if origin not in result:
            result.append(origin)
    return result


def _llm_config(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    value = policy.get('llm_config')
    return value if isinstance(value, Mapping) else {}
