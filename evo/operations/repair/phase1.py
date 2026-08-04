from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from evo.llm import LazyLLMClient
from evo.repair_model import EvoModelConfigError, opencode_settings
from evo.traces.detail import build_trace_detail_view

from .agent import ModelCallError, ModelCallTimeout, _bounded_json, next_action
from .contracts import build_supported_plan, select_category, validate_analysis
from .demo import _command as normalize_demo_command
from .demo import _origin as http_origin
from .demo import request_http, run_command
from .memory import WorkMemory, content_ref, write_json
from .opencode import OpenCodeSession
from .validation import inside_repair_scope, repair_scope
from .web import normalize_http_url, read_web_pages, search_web


DEFAULT_BUDGET = {
    'turns': 20,
    'web_searches': 6,
    'page_reads': 12,
    'opencode_calls': 10,
    'command_runs': 10,
    'http_requests': 6,
    'artifact_reads': 8,
    'seconds': 1800,
}
DEFAULT_MODEL_TIMEOUT_SECONDS = 120
MAX_CONSECUTIVE_MODEL_FAILURES = 3
TRACE_ERROR_MARKERS = (
    'connection refused', 'failed to establish', 'error', 'exception', 'timeout',
    'timed out', 'unavailable', '503',
)
FORCE_RERUN_REASONS = frozenset({
    'explicit_user_request',
    'independent_revalidation',
    'prior_result_inconclusive',
    'stale_external_data',
    'suspected_nondeterminism',
})


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
            config = opencode_settings(_llm_config(policy).get('evo_llm'))
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
                **_phase1_lineage(memory),
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
        query = _text(request, 'query')
        reused, rerun_reason = _reuse_completed_investigation(
            memory,
            'web.search',
            {'status': 'completed', 'query': query},
            request,
            allow_cross_revision=True,
        )
        if reused:
            return None
        if not rerun_reason and memory.has_searched_query(query):
            raise ValueError('web_search_query_already_searched')
        _consume(counters, budget, 'web_searches')
        result = {
            **search_web(
                query,
                memory.artifact_root,
                seen_urls=set() if rerun_reason else memory.known_urls(),
            ),
            **_rerun_metadata(rerun_reason),
        }
        memory.record(
            'web.search',
            f"{result.get('status')}: {query}; {len(result.get('results') or ())} results",
            result,
        )
        return None
    if action == 'read_web':
        question = _text(request, 'question')
        urls = _strings(request.get('urls'), maximum=3)
        if not urls:
            raise ValueError('read_web_urls_missing')
        normalized = []
        for value in urls:
            url = normalize_http_url(value)
            if not url:
                raise ValueError('read_web_url_invalid')
            if url not in normalized:
                normalized.append(url)
        known_urls = {
            url
            for value in memory.known_urls()
            if (url := normalize_http_url(value))
        }
        if any(url not in known_urls for url in normalized):
            raise ValueError('read_web_url_not_discovered')
        probe = {
            'status': 'completed',
            'question': question,
            'pages': [
                {'requested_url': url, 'url': url}
                for url in normalized
            ],
        }
        reused, rerun_reason = _reuse_completed_investigation(
            memory,
            'web.read',
            probe,
            request,
            allow_cross_revision=True,
        )
        if reused:
            return None
        read_urls = {
            url
            for value in memory.read_urls()
            if (url := normalize_http_url(value))
        }
        fresh = normalized if rerun_reason else [url for url in normalized if url not in read_urls]
        if not fresh:
            raise ValueError('read_web_urls_already_read')
        _consume(counters, budget, 'page_reads', len(fresh))
        result = read_web_pages(
            question,
            fresh,
            memory.work_root,
            memory.artifact_root,
            seen_urls=set() if rerun_reason else read_urls,
            seen_pages=() if rerun_reason else memory.read_page_fingerprints(),
        )
        pages = result.get('pages') if isinstance(result.get('pages'), list) else []
        successful_pages = [
            page
            for page in pages
            if isinstance(page, Mapping)
            and page.get('status') in {'readable', 'duplicate'}
        ]
        # A forced revalidation must be judged only by this fetch. Historical
        # successes are useful for an incremental retry, but they cannot turn a
        # failed freshness check into a completed investigation.
        satisfied_urls = set() if rerun_reason else set(read_urls)
        for page in successful_pages:
            for name in ('requested_url', 'canonical_url', 'url'):
                if url := normalize_http_url(page.get(name)):
                    satisfied_urls.add(url)
        read_status = (
            'completed' if all(url in satisfied_urls for url in normalized)
            else 'partial' if successful_pages or (
                not rerun_reason and any(url in read_urls for url in normalized)
            )
            else 'failed'
        )
        memory.record(
            'web.read',
            f"read {len(result.get('pages') or ())} pages",
            {
                'status': read_status,
                'requested_urls': normalized,
                'content_trust': 'external_untrusted',
                **result,
                **_rerun_metadata(rerun_reason),
            },
        )
        return None
    if action == 'opencode':
        instruction = _text(request, 'instruction')
        expected = str(request.get('expected_result') or '').strip()
        workspace_before = memory.workspace_digest()
        reused, rerun_reason = _reuse_completed_investigation(
            memory,
            'opencode.result',
            {
                'status': 'completed',
                'instruction': instruction,
                'expected_result': expected,
                'workspace_before_sha256': workspace_before,
                'guidance_revision_id': memory.guidance_revision_id,
            },
            request,
            allow_cross_revision=True,
        )
        if reused:
            return None
        if memory.consecutive_failures('opencode.result') >= 2:
            memory.record(
                'phase1.stopped',
                'OpenCode failed twice without a successful observation',
                {'status': 'blocked', 'reason': 'opencode_repeated_failure'},
            )
            return _terminal('blocked', 'opencode_repeated_failure')
        _consume(counters, budget, 'opencode_calls')
        memory.write_context(counters, budget)
        result = session.run(
            instruction,
            expected,
            max(0.1, deadline - time.monotonic()),
        )
        workspace_after = memory.workspace_digest(refresh=True)
        result = {
            **result,
            'instruction': instruction,
            'expected_result': expected,
            'workspace_before_sha256': workspace_before,
            'workspace_after_sha256': workspace_after,
            'guidance_revision_id': memory.guidance_revision_id,
            **_rerun_metadata(rerun_reason),
        }
        result['investigation_key'] = memory.investigation_key('opencode.result', result)
        summary = str(result.get('report', {}).get('summary') or result.get('reason') or 'OpenCode finished')
        memory.record('opencode.result', summary, result)
        if result.get('invalid_changes'):
            memory.restore_source()
            raise ValueError('phase1_workspace_tainted')
        memory.checkpoint(session.session_id, session.calls)
        return None
    if action == 'run_command':
        command = request.get('command')
        if not isinstance(command, list):
            raise ValueError('run_command_argv_required')
        normalized_command = normalize_demo_command(command, memory.work_root)
        expected = str(request.get('expected_result') or '').strip()
        workspace_before = memory.workspace_digest()
        reused, rerun_reason = _reuse_completed_investigation(
            memory,
            'command.result',
            {
                'status': 'completed',
                'command': normalized_command,
                'expected_result': expected,
                'workspace_before_sha256': workspace_before,
            },
            request,
            allow_cross_revision=True,
        )
        if reused:
            return None
        _consume(counters, budget, 'command_runs')
        result = run_command(
            memory.work_root,
            memory.artifact_root,
            command,
            attempt=counters['command_runs'],
            timeout_seconds=min(180.0, max(0.1, deadline - time.monotonic())),
            output_limit=256 * 1024,
            expected_source_hash=memory.source_digest,
        )
        workspace_after = memory.workspace_digest(refresh=True)
        result = {
            **result,
            'expected_result': expected,
            'workspace_before_sha256': workspace_before,
            'workspace_after_sha256': workspace_after,
            **_rerun_metadata(rerun_reason),
        }
        result['investigation_key'] = memory.investigation_key('command.result', result)
        memory.record(
            'command.result',
            f"{result['status']}: {' '.join(result['command'][:6])}; expected={expected}",
            result,
        )
        memory.checkpoint(session.session_id, session.calls)
        return None
    if action == 'http_request':
        url = _text(request, 'url')
        method = str(request.get('method') or 'GET').strip().upper()
        if method not in {'GET', 'HEAD'}:
            raise ValueError('http_method_not_allowed')
        allowed_origins = _allowed_origins(policy)
        if http_origin(url) not in set(allowed_origins):
            raise ValueError(f'http_origin_not_allowed:{url}')
        workspace_sha256 = memory.workspace_digest()
        reused, rerun_reason = _reuse_completed_investigation(
            memory,
            'http.result',
            {
                'status': 'completed',
                'url': url,
                'method': method,
                'workspace_sha256': workspace_sha256,
            },
            request,
            allow_cross_revision=False,
        )
        if reused:
            return None
        _consume(counters, budget, 'http_requests')
        result = request_http(
            url,
            method,
            allowed_origins,
            memory.artifact_root,
            attempt=counters['http_requests'],
            timeout_seconds=min(15.0, max(0.1, deadline - time.monotonic())),
        )
        result = {
            **result,
            'workspace_sha256': workspace_sha256,
            **_rerun_metadata(rerun_reason),
        }
        memory.record(
            'http.result',
            f"{result['method']} {result['url']} -> {result.get('status_code') or result['status']}",
            result,
        )
        return None
    if action == 'read_artifact':
        uri = _text(request, 'uri')
        offset_bytes = _request_integer(request, 'offset_bytes', default=0)
        max_bytes = _request_integer(request, 'max_bytes', default=4096)
        probe = {
            'status': 'completed',
            'uri': uri,
            'offset_bytes': offset_bytes,
            'max_bytes': max_bytes,
        }
        reused, rerun_reason = _reuse_completed_investigation(
            memory,
            'artifact.read',
            probe,
            request,
            allow_cross_revision=True,
        )
        if reused:
            return None
        _consume(counters, budget, 'artifact_reads')
        result = memory.read_artifact(
            uri,
            offset_bytes=offset_bytes,
            max_bytes=max_bytes,
        )
        result.update(_rerun_metadata(rerun_reason))
        result['investigation_key'] = memory.investigation_key('artifact.read', result)
        memory.record(
            'artifact.read',
            f"read {result['returned_bytes']} bytes from registered artifact",
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
        lineage = _phase1_lineage(memory)
        memory.record(
            'phase1.finished',
            reason,
            {'proposal': proposal, 'evidence_refs': evidence, **lineage},
        )
        workspace_ref = memory.checkpoint(session.session_id, session.calls)
        result_path = memory.artifact_root / 'result.json'
        write_json(
            result_path,
            {
                'proposal': proposal,
                'evidence_refs': evidence,
                'reason': reason,
                **lineage,
            },
        )
        return {
            'status': 'supported',
            'proposal': proposal,
            **lineage,
            'validation': {
                'verdict': 'supports',
                'reason': reason,
                'evidence_refs': evidence,
                'result_ref': content_ref(result_path, memory.artifact_root),
                'workspace_ref': workspace_ref,
                'journal_ref': memory.journal_ref(),
                **lineage,
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


def _request_integer(value: Mapping[str, Any], key: str, *, default: int) -> int:
    candidate = value.get(key, default)
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError(f'{key}_invalid')
    return candidate


def _selected_evidence(value: object, available: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    requested = _strings(value, maximum=8)
    by_uri = {str(item.get('uri') or ''): dict(item) for item in available}
    if not requested or any(uri not in by_uri for uri in requested):
        raise ValueError('finish_requires_decisive_evidence_uris')
    return [by_uri[uri] for uri in dict.fromkeys(requested)]


def _reuse_completed_investigation(
    memory: WorkMemory,
    event: str,
    probe: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    allow_cross_revision: bool,
) -> tuple[bool, str]:
    """Reuse a matching observation before consuming budget or invoking a tool."""
    rerun_reason = _force_rerun_reason(request)
    if rerun_reason:
        return False, rerun_reason
    source = memory.completed_investigation(
        event,
        probe,
        allow_cross_revision=allow_cross_revision,
    )
    if source is None:
        return False, ''
    memory.record_investigation_reuse(event, source)
    return True, ''


def _force_rerun_reason(request: Mapping[str, Any]) -> str:
    force = request.get('force_rerun', False)
    if not isinstance(force, bool):
        raise ValueError('force_rerun_invalid')
    reason = str(request.get('rerun_reason') or '').strip()
    if not force:
        if reason:
            raise ValueError('rerun_reason_without_force_rerun')
        return ''
    if reason not in FORCE_RERUN_REASONS:
        raise ValueError('force_rerun_reason_invalid')
    return reason


def _rerun_metadata(reason: str) -> dict[str, Any]:
    return {'force_rerun': True, 'rerun_reason': reason} if reason else {}


def _phase1_lineage(memory: WorkMemory) -> dict[str, Any]:
    provenance = memory.guidance_provenance()
    return {
        'guidance_revision_id': memory.guidance_revision_id,
        'guidance_provenance': provenance,
        'workspace_sha256': memory.workspace_digest(),
        'recovery': dict(memory.recovery),
    }


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
