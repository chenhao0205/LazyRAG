from __future__ import annotations

import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest


def _load_phase1() -> types.ModuleType:
    """Load Phase-1 without importing the application-only LazyLLM runtime."""
    traces = types.ModuleType('evo.traces')
    traces.__path__ = []
    detail = types.ModuleType('evo.traces.detail')
    detail.build_trace_detail_view = lambda *_args, **_kwargs: {}
    opencode = types.ModuleType('evo.operations.repair.opencode')
    opencode.OpenCodeSession = type('OpenCodeSession', (), {})
    web = types.ModuleType('evo.operations.repair.web')
    web.normalize_http_url = lambda value: value
    web.read_web_pages = lambda *_args, **_kwargs: {}
    web.search_web = lambda *_args, **_kwargs: {}
    validation = types.ModuleType('evo.operations.repair.validation')
    validation.inside_repair_scope = lambda *_args, **_kwargs: True
    validation.repair_scope = lambda *_args, **_kwargs: {}
    replacements = {
        'evo.traces': traces,
        'evo.traces.detail': detail,
        'evo.operations.repair.opencode': opencode,
        'evo.operations.repair.web': web,
        'evo.operations.repair.validation': validation,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        path = Path(__file__).parents[2] / 'evo' / 'operations' / 'repair' / 'phase1.py'
        name = 'evo.operations.repair._phase1_reuse_test'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture(scope='module')
def phase1() -> types.ModuleType:
    return _load_phase1()


class _Session:
    session_id = 'session-current'
    calls = 0


class _Memory:
    def __init__(self, root: Path, *, completed: bool = True) -> None:
        self.artifact_root = root / 'run' / 'category' / 'target' / 'attempt-1'
        self.work_root = root / 'work-root'
        self.artifact_root.mkdir(parents=True)
        (self.work_root / 'work').mkdir(parents=True)
        (self.work_root / 'work' / 'demo.py').write_text('print("ok")\n', encoding='utf-8')
        self.source_digest = 'a' * 64
        self.completed = completed
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.read_artifact_calls = 0
        self.checkpoint_calls = 0
        self.guidance_revision_id = 'gr_current'
        self.recovery = {'mode': 'same_revision_resume'}
        self._known_urls = {'https://example.com/docs'}
        self._read_urls = set()
        self._page_fingerprints = []
        self._evidence = [{
            'uri': 'phase1://current/command.json',
            'sha256': 'c' * 64,
        }]

    def completed_investigation(self, event, probe, *, allow_cross_revision=True):
        if not self.completed:
            return None
        return {
            'event': event,
            'summary': 'prior completed observation',
            'data': dict(probe),
            '_event_ref': {'uri': 'phase1://prior/event.json', 'sha256': 'e' * 64},
            'provenance': {'guidance_revision_id': 'gr_current'},
        }

    def record_investigation_reuse(self, event, source, *, reason='same_input_and_dependency'):
        self.records.append(('investigation.reused', {
            'source_event': event,
            'reason': reason,
            'source': dict(source),
        }))
        return {'uri': 'phase1://current/reuse.json', 'sha256': 'f' * 64}

    def workspace_digest(self, *, refresh=False):
        return 'a' * 64

    def investigation_key(self, event, data):
        return f'{event}:stable'

    def read_artifact(self, *_args, **_kwargs):
        self.read_artifact_calls += 1
        raise AssertionError('artifact reader must not run on a reuse hit')

    def known_urls(self):
        return set(self._known_urls)

    def read_urls(self):
        return set(self._read_urls)

    def read_page_fingerprints(self):
        return list(self._page_fingerprints)

    def record(self, event, _summary, data):
        self.records.append((event, dict(data)))
        return {'uri': f'phase1://current/{event}.json', 'sha256': 'd' * 64}

    def checkpoint(self, _session_id, _calls):
        self.checkpoint_calls += 1
        return {'uri': 'phase1://current/workspace', 'sha256': 'b' * 64}

    def completion_gaps(self, _proposal):
        return []

    def evidence_refs(self):
        return list(self._evidence)

    def guidance_provenance(self):
        return {'guidance_revision_id': self.guidance_revision_id}

    def journal_ref(self):
        return {'uri': 'phase1://current/journal.jsonl', 'sha256': 'a' * 64}


def _counters(phase1: types.ModuleType) -> dict[str, int]:
    return {key: 0 for key in phase1.DEFAULT_BUDGET if key != 'seconds'}


def _execute(phase1, action, request, memory, counters, policy=None):
    return phase1._execute_action(
        action,
        'test reason',
        request,
        memory,
        _Session(),
        counters,
        phase1.DEFAULT_BUDGET,
        policy or {},
        time.monotonic() + 30,
    )


def test_reuse_hits_skip_command_http_and_artifact_tools_and_budget(
    phase1, tmp_path, monkeypatch,
) -> None:
    memory = _Memory(tmp_path)
    counters = _counters(phase1)

    def unexpected(*_args, **_kwargs):
        raise AssertionError('real tool must not run on a reuse hit')

    monkeypatch.setattr(phase1, 'run_command', unexpected)
    monkeypatch.setattr(phase1, 'request_http', unexpected)
    _execute(phase1, 'run_command', {'command': ['python', 'work/demo.py']}, memory, counters)
    _execute(
        phase1,
        'http_request',
        {'url': 'http://127.0.0.1:8765/health', 'method': 'GET'},
        memory,
        counters,
        {'phase1_demo_allowed_origins': ['http://127.0.0.1:8765']},
    )
    _execute(
        phase1,
        'read_artifact',
        {'uri': 'phase1://current/evidence.json', 'offset_bytes': 0, 'max_bytes': 4096},
        memory,
        counters,
    )

    assert counters['command_runs'] == 0
    assert counters['http_requests'] == 0
    assert counters['artifact_reads'] == 0
    assert memory.read_artifact_calls == 0
    assert [data['source_event'] for event, data in memory.records if event == 'investigation.reused'] == [
        'command.result', 'http.result', 'artifact.read',
    ]


@pytest.mark.parametrize(
    ('extra', 'error'),
    [
        ({'force_rerun': True}, 'force_rerun_reason_invalid'),
        ({'force_rerun': True, 'rerun_reason': 'try_for_a_better_answer'}, 'force_rerun_reason_invalid'),
        ({'rerun_reason': 'independent_revalidation'}, 'rerun_reason_without_force_rerun'),
    ],
)
def test_force_rerun_rejects_missing_or_invalid_reason(
    phase1, tmp_path, monkeypatch, extra, error,
) -> None:
    memory = _Memory(tmp_path)
    counters = _counters(phase1)
    monkeypatch.setattr(
        phase1, 'run_command',
        lambda *_args, **_kwargs: pytest.fail('invalid rerun must not invoke command'),
    )

    with pytest.raises(ValueError, match=error):
        _execute(
            phase1,
            'run_command',
            {'command': ['python', 'work/demo.py'], **extra},
            memory,
            counters,
        )

    assert counters['command_runs'] == 0


def test_valid_force_rerun_executes_and_consumes_budget(phase1, tmp_path, monkeypatch) -> None:
    memory = _Memory(tmp_path)
    counters = _counters(phase1)
    calls = []

    def command(*_args, **_kwargs):
        calls.append(True)
        return {
            'status': 'completed',
            'command': ['python', 'work/demo.py'],
            'result_ref': {'uri': 'phase1://current/command.json', 'sha256': 'c' * 64},
        }

    monkeypatch.setattr(phase1, 'run_command', command)
    _execute(
        phase1,
        'run_command',
        {
            'command': ['python', 'work/demo.py'],
            'force_rerun': True,
            'rerun_reason': 'independent_revalidation',
        },
        memory,
        counters,
    )

    assert calls == [True]
    assert counters['command_runs'] == 1
    assert memory.checkpoint_calls == 1
    result = next(data for event, data in memory.records if event == 'command.result')
    assert result['force_rerun'] is True
    assert result['rerun_reason'] == 'independent_revalidation'
    assert not any(event == 'investigation.reused' for event, _data in memory.records)


def test_failed_web_read_remains_retryable(phase1, tmp_path, monkeypatch) -> None:
    memory = _Memory(tmp_path, completed=False)
    counters = _counters(phase1)
    monkeypatch.setattr(phase1, 'read_web_pages', lambda *_args, **_kwargs: {
        'question': 'read docs',
        'pages': [{
            'status': 'failed',
            'url': 'https://example.com/docs',
            'content_ref': None,
        }],
    })

    _execute(
        phase1,
        'read_web',
        {'question': 'read docs', 'urls': ['https://example.com/docs']},
        memory,
        counters,
    )

    result = next(data for event, data in memory.records if event == 'web.read')
    assert result['status'] == 'failed'
    assert counters['page_reads'] == 1


def test_partial_web_batch_retries_only_failed_urls_then_becomes_reusable(
    phase1, tmp_path, monkeypatch,
) -> None:
    memory = _Memory(tmp_path, completed=False)
    first_url = 'https://example.com/first'
    second_url = 'https://example.com/second'
    memory._known_urls = {first_url, second_url}
    memory._read_urls = {first_url}
    captured = []

    def read_pages(_question, urls, *_args, **_kwargs):
        captured.append(list(urls))
        return {
            'question': 'read docs',
            'pages': [{
                'status': 'readable',
                'requested_url': second_url,
                'url': second_url,
                'canonical_url': second_url,
                'content_ref': {'uri': 'phase1://current/second.txt', 'sha256': 'a' * 64},
            }],
        }

    monkeypatch.setattr(phase1, 'read_web_pages', read_pages)
    counters = _counters(phase1)
    _execute(
        phase1,
        'read_web',
        {'question': 'read docs', 'urls': [first_url, second_url]},
        memory,
        counters,
    )

    result = next(data for event, data in memory.records if event == 'web.read')
    assert captured == [[second_url]]
    assert result['status'] == 'completed'
    assert result['requested_urls'] == [first_url, second_url]
    assert counters['page_reads'] == 1


def test_forced_web_revalidation_ignores_historical_fingerprints(
    phase1, tmp_path, monkeypatch,
) -> None:
    memory = _Memory(tmp_path)
    memory._page_fingerprints = [{'content_sha256': 'a' * 64}]
    captured = []

    def read_pages(_question, urls, *_args, **kwargs):
        captured.append(kwargs['seen_pages'])
        return {
            'question': 'read docs',
            'pages': [{
                'status': 'readable',
                'requested_url': urls[0],
                'url': urls[0],
                'canonical_url': urls[0],
                'content_ref': {'uri': 'phase1://current/docs.txt', 'sha256': 'a' * 64},
            }],
        }

    monkeypatch.setattr(phase1, 'read_web_pages', read_pages)
    counters = _counters(phase1)
    _execute(
        phase1,
        'read_web',
        {
            'question': 'read docs',
            'urls': ['https://example.com/docs'],
            'force_rerun': True,
            'rerun_reason': 'stale_external_data',
        },
        memory,
        counters,
    )

    assert captured == [()]


def test_failed_forced_web_revalidation_is_not_hidden_by_historical_success(
    phase1, tmp_path, monkeypatch,
) -> None:
    url = 'https://example.com/docs'
    memory = _Memory(tmp_path)
    memory._read_urls = {url}
    monkeypatch.setattr(phase1, 'read_web_pages', lambda *_args, **_kwargs: {
        'question': 'read docs',
        'pages': [{
            'status': 'failed',
            'requested_url': url,
            'url': url,
            'content_ref': None,
        }],
    })

    _execute(
        phase1,
        'read_web',
        {
            'question': 'read docs',
            'urls': [url],
            'force_rerun': True,
            'rerun_reason': 'stale_external_data',
        },
        memory,
        _counters(phase1),
    )

    result = next(data for event, data in memory.records if event == 'web.read')
    assert result['status'] == 'failed'


def test_partial_forced_web_revalidation_requires_every_current_fetch(
    phase1, tmp_path, monkeypatch,
) -> None:
    first_url = 'https://example.com/first'
    second_url = 'https://example.com/second'
    memory = _Memory(tmp_path)
    memory._known_urls = {first_url, second_url}
    memory._read_urls = {first_url, second_url}
    monkeypatch.setattr(phase1, 'read_web_pages', lambda *_args, **_kwargs: {
        'question': 'read docs',
        'pages': [
            {
                'status': 'readable',
                'requested_url': first_url,
                'url': first_url,
                'canonical_url': first_url,
                'content_ref': {'uri': 'phase1://current/first.txt', 'sha256': 'a' * 64},
            },
            {
                'status': 'failed',
                'requested_url': second_url,
                'url': second_url,
                'content_ref': None,
            },
        ],
    })

    _execute(
        phase1,
        'read_web',
        {
            'question': 'read docs',
            'urls': [first_url, second_url],
            'force_rerun': True,
            'rerun_reason': 'independent_revalidation',
        },
        memory,
        _counters(phase1),
    )

    result = next(data for event, data in memory.records if event == 'web.read')
    assert result['status'] == 'partial'


def test_finish_accepts_only_current_memory_evidence(phase1, tmp_path) -> None:
    memory = _Memory(tmp_path)
    counters = _counters(phase1)
    proposal = {
        'target': 'algorithm/retry.py:retry',
        'change': 'preserve the retry result',
        'expected_result': 'the repaired answer is returned',
    }

    with pytest.raises(ValueError, match='finish_requires_decisive_evidence_uris'):
        _execute(
            phase1,
            'finish',
            {**proposal, 'evidence_uris': ['phase1://old/stale-command.json']},
            memory,
            counters,
        )

    result = _execute(
        phase1,
        'finish',
        {**proposal, 'evidence_uris': [memory.evidence_refs()[0]['uri']]},
        memory,
        counters,
    )

    assert result['status'] == 'supported'
    assert result['validation']['evidence_refs'] == memory.evidence_refs()
    finished = next(data for event, data in memory.records if event == 'phase1.finished')
    assert finished['evidence_refs'] == memory.evidence_refs()
