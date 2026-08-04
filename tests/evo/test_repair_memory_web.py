from __future__ import annotations

from evo.operations.repair.memory import WorkMemory


def _memory(tmp_path, name, *, previous=(), history=None, guidance=()):
    artifact_root = tmp_path / name
    work_root = tmp_path / f'{name}-work'
    artifact_root.mkdir()
    (work_root / 'work').mkdir(parents=True)
    return WorkMemory(
        target={'category_id': 'category-1'},
        guidance=list(guidance),
        scope={},
        work_root=work_root,
        artifact_root=artifact_root,
        previous_attempts=tuple(previous),
        source_digest='source-hash',
        history_attempts=None if history is None else tuple(history),
    )


def test_work_memory_restores_cross_attempt_web_index(tmp_path) -> None:
    first = _memory(tmp_path, 'attempt-1')
    first.record('web.search', 'searched docs', {
        'query': '  Retry   behavior ',
        'status': 'completed',
        'results': [{
            'title': 'Guide',
            'url': 'https://example.com/guide',
            'canonical_url': 'https://example.com/guide',
            'snippet': 'Reference',
        }],
    })
    first.record('web.read', 'read docs', {
        'question': 'How does retry work?',
        'pages': [{
            'status': 'readable',
            'url': 'https://example.com/guide',
            'canonical_url': 'https://example.com/guide',
            'content_sha256': 'a' * 64,
            'content_simhash': '0123456789abcdef',
            'character_count': 500,
            'similarity_token_count': 80,
            'content_ref': {'uri': 'phase1://page', 'sha256': 'a' * 64},
        }],
    })

    resumed = _memory(tmp_path, 'attempt-2', previous=(first.artifact_root,))

    assert resumed.has_searched_query('retry behavior')
    assert resumed.known_urls() == {'https://example.com/guide'}
    assert resumed.read_urls() == {'https://example.com/guide'}
    assert resumed.read_page_fingerprints()[0]['content_simhash'] == '0123456789abcdef'
    assert resumed.context({}, {})['web_investigation'] == {
        'searched_query_count': 1,
        'searched_queries': ['Retry behavior'],
        'read_page_count': 1,
        'read_urls': ['https://example.com/guide'],
    }


def test_work_memory_deduplicates_query_index_without_dropping_raw_events(tmp_path) -> None:
    memory = _memory(tmp_path, 'attempt')
    for query in ('Retry behavior', ' retry   BEHAVIOR '):
        memory.record('web.search', query, {
            'query': query,
            'status': 'completed',
            'results': [],
        })

    assert memory.searched_queries() == ['Retry behavior']
    assert len((memory.artifact_root / 'journal.jsonl').read_text().splitlines()) == 2


def test_failed_web_actions_remain_retryable(tmp_path) -> None:
    memory = _memory(tmp_path, 'attempt')
    memory.record('web.search', 'search unavailable', {
        'query': 'retry behavior',
        'status': 'unavailable',
        'results': [],
    })
    memory.record('web.read', 'fetch failed', {
        'question': 'How does retry work?',
        'pages': [{
            'status': 'failed',
            'url': 'https://example.com/guide',
            'content_ref': None,
        }],
    })

    assert not memory.has_searched_query('retry behavior')
    assert memory.read_urls() == set()


def test_web_index_survives_attempt_without_workspace_checkpoint(tmp_path) -> None:
    interrupted = _memory(tmp_path, 'interrupted-attempt')
    interrupted.record('web.search', 'searched before interruption', {
        'query': 'repair evidence',
        'status': 'completed',
        'results': [],
    })

    resumed = _memory(
        tmp_path,
        'resumed-attempt',
        previous=(),
        history=(interrupted.artifact_root,),
    )

    assert resumed.has_searched_query('repair evidence')
    assert resumed.restored_session == {}


def test_reused_empty_search_does_not_satisfy_required_search_gate(tmp_path) -> None:
    first = _memory(tmp_path, 'attempt-1', guidance=('请联网搜索官方文档',))
    first.record('web.search', 'search returned no results', {
        'query': 'repair evidence',
        'status': 'completed',
        'results': [],
    })
    resumed = _memory(
        tmp_path,
        'attempt-2',
        previous=(first.artifact_root,),
        guidance=('请联网搜索官方文档',),
    )
    source = resumed.completed_investigation(
        'web.search', {'query': 'repair evidence', 'status': 'completed'},
    )
    assert source is not None

    resumed.record_investigation_reuse('web.search', source)

    assert resumed.completion_gaps({'change': 'use the observed behavior'}) == [
        'required_web_search_missing',
    ]


def test_materialized_duplicate_page_satisfies_required_read_gate(tmp_path) -> None:
    memory = _memory(tmp_path, 'attempt', guidance=('请读取网页并参考官方文档',))
    memory.record('web.read', 'read duplicate docs', {
        'status': 'completed',
        'question': 'How does retry work?',
        'pages': [{
            'status': 'duplicate',
            'duplicate_kind': 'near',
            'url': 'https://example.com/retry',
            'content_ref': {'uri': 'phase1://page', 'sha256': 'a' * 64},
        }],
    })

    assert memory.completion_gaps({'change': 'use the observed behavior'}) == []


def test_url_only_duplicate_does_not_satisfy_required_read_gate(tmp_path) -> None:
    memory = _memory(tmp_path, 'attempt', guidance=('请读取网页并参考官方文档',))
    memory.record('web.read', 'redirected to known URL', {
        'status': 'completed',
        'question': 'How does retry work?',
        'pages': [{
            'status': 'duplicate',
            'duplicate_kind': 'url',
            'url': 'https://example.com/retry',
            'content_ref': None,
        }],
    })

    assert memory.completion_gaps({'change': 'use the observed behavior'}) == [
        'required_web_page_read_missing',
    ]
