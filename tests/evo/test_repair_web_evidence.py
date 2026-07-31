from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any

import pytest

from evo.operations.repair.web_evidence import (
    build_web_evidence,
    clean_page_content,
    clean_search_results,
)
from evo.operations.repair.web_evidence_contracts import (
    PageSelection,
    SearchResultSet,
)


def _raw_search_results() -> list[dict[str, Any]]:
    return [
        {
            'title': '  Python API reference  ',
            'url': 'https://docs.example.com/python',
            'snippet': '  Public   API behavior.  ',
            'source': 'tavily',
            'extra': {'score': 0.99, 'raw_content': 'must not leak'},
        },
        {
            'title': 'Issue discussion',
            'url': 'https://issues.example.com/42',
            'snippet': 'A maintainer explains the failure mode.',
            'source': 'tavily',
        },
        {
            'title': 'Migration guide',
            'url': 'https://docs.example.com/migration',
            'snippet': 'The option changed in version 2.',
            'source': 'tavily',
        },
        {
            'title': 'Unrelated result',
            'url': 'https://blog.example.com/unrelated',
            'snippet': 'Not selected.',
            'source': 'tavily',
        },
    ]


def _candidates(
    raw_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return clean_search_results(
        '  What is the documented retry behavior?  ',
        raw_results if raw_results is not None else _raw_search_results(),
    )


def _fetched_page(
    url: str,
    content: str,
    *,
    title: str = 'Fetched title',
    final_url: str | None = None,
) -> dict[str, Any]:
    return {
        'url': url,
        'success': True,
        'result': {
            'status': 'ok',
            'url': url,
            'final_url': final_url or url,
            'status_code': 200,
            'content_type': 'text/html; charset=utf-8',
            'title': title,
            'description': 'Fetched description',
            'content': content,
        },
    }


def _failed_page(url: str, error: str = 'RuntimeError: unavailable') -> dict[str, Any]:
    return {
        'url': url,
        'success': False,
        'error': error,
    }


def _batch_fetch_result(*items: dict[str, Any]) -> dict[str, Any]:
    succeeded = sum(item.get('success') is True for item in items)
    return {
        'success': True,
        'tool': 'url_fetch',
        'result': {
            'total': len(items),
            'succeeded': succeeded,
            'failed': len(items) - succeeded,
            'results': list(items),
        },
    }


def _pages(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    pages = evidence.get('pages')
    assert isinstance(pages, list)
    return pages


def _warnings(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = evidence.get('warnings')
    assert isinstance(warnings, list)
    assert all(isinstance(item, dict) for item in warnings)
    return warnings


def test_clean_search_results_preserves_provider_order_and_projects_strict_fields() -> None:
    result = _candidates()

    assert result['query'] == 'What is the documented retry behavior?'
    assert [item['title'] for item in result['results']] == [
        'Python API reference',
        'Issue discussion',
        'Migration guide',
        'Unrelated result',
    ]
    assert result['results'][0] == {
        'title': 'Python API reference',
        'url': 'https://docs.example.com/python',
        'snippet': 'Public API behavior.',
    }
    assert all(
        set(item) == {'title', 'url', 'snippet'}
        for item in result['results']
    )
    assert 'source' not in result['results'][0]
    assert 'extra' not in result['results'][0]
    assert 'raw_content' not in str(result)


def test_clean_search_results_accepts_actual_named_tool_result_event() -> None:
    result = clean_search_results(
        'What is the documented retry behavior?',
        {
            'tag': 'tool_results',
            'tool_results': [{
                'id': 'call-1',
                'name': 'TavilySearch_search',
                'arguments': {
                    'query': 'What is the documented retry behavior?',
                },
                'result': _raw_search_results(),
            }],
        },
    )

    assert len(result['results']) == 4
    assert result['results'][0]['title'] == 'Python API reference'


def test_clean_search_results_checks_query_from_actual_tool_arguments() -> None:
    with pytest.raises(ValueError, match='does not match'):
        clean_search_results(
            'What is the documented retry behavior?',
            {
                'name': 'TavilySearch_search',
                'arguments': {
                    'query': 'What is the documented timeout behavior?',
                },
                'result': _raw_search_results(),
            },
        )


def test_clean_search_results_accepts_future_provider_search_envelope() -> None:
    result = clean_search_results(
        'What is the documented retry behavior?',
        {
            'id': 'call-1',
            'name': 'FutureProviderSearch_search',
            'arguments': (
                '{"query": "What is the documented retry behavior?"}'
            ),
            'result': _raw_search_results(),
        },
        expected_tool_name='FutureProviderSearch_search',
        expected_call_id='call-1',
    )

    assert len(result['results']) == 4


@pytest.mark.parametrize('tool_name', [
    'ArxivSearch_search',
    'CompletelyUnrelatedSearch_search',
])
def test_clean_search_results_rejects_unconfigured_search_like_tool(
    tool_name: str,
) -> None:
    with pytest.raises(ValueError, match='web_search'):
        clean_search_results(
            'What is the documented retry behavior?',
            {
                'id': 'call-1',
                'name': tool_name,
                'arguments': {
                    'query': 'What is the documented retry behavior?',
                },
                'result': _raw_search_results(),
            },
        )


def test_clean_search_results_requires_query_on_named_search_event() -> None:
    with pytest.raises(ValueError, match='arguments.query'):
        clean_search_results(
            'What is the documented retry behavior?',
            {
                'id': 'call-1',
                'name': 'TavilySearch_search',
                'result': _raw_search_results(),
            },
        )


def test_clean_search_results_rejects_provider_content_method() -> None:
    with pytest.raises(ValueError, match='web_search|get_content'):
        clean_search_results(
            'What is the documented retry behavior?',
            {
                'name': 'TavilySearch_get_content',
                'result': _raw_search_results(),
            },
        )


def test_clean_search_results_rejects_an_envelope_from_another_tool() -> None:
    with pytest.raises(ValueError, match='web_search|url_fetch'):
        clean_search_results(
            'What is the documented retry behavior?',
            {
                'name': 'url_fetch',
                'result': _raw_search_results(),
            },
        )


def test_clean_search_results_deduplicates_canonical_urls_and_filters_invalid_urls() -> None:
    result = clean_search_results(
        'one question',
        [
            {
                'title': 'First',
                'url': ' HTTPS://Example.COM:443/docs#overview ',
                'snippet': 'first',
            },
            {
                'title': 'Duplicate fragment',
                'url': 'https://example.com/docs#examples',
                'snippet': 'duplicate',
            },
            {'title': 'FTP', 'url': 'ftp://example.com/file', 'snippet': 'bad'},
            {'title': 'Relative', 'url': '/docs/local', 'snippet': 'bad'},
            {'title': 'Credentials', 'url': 'https://u:p@example.com/', 'snippet': 'bad'},
            {'title': 'Loopback', 'url': 'http://127.0.0.1/admin', 'snippet': 'bad'},
            {'title': 'Localhost', 'url': 'http://localhost/admin', 'snippet': 'bad'},
            {'url': 'https://example.net/no-title', 'snippet': 'missing title'},
            {'title': 'Missing URL', 'snippet': 'bad'},
            {
                'title': 'Second valid result',
                'url': 'http://example.org/reference',
                'snippet': 'second',
            },
        ],
    )

    assert result['results'] == [
        {
            'title': 'First',
            'url': 'https://example.com/docs',
            'snippet': 'first',
        },
        {
            'title': 'Second valid result',
            'url': 'http://example.org/reference',
            'snippet': 'second',
        },
    ]


def test_clean_search_results_applies_limit_after_filtering_and_deduplication() -> None:
    raw = [
        {'title': 'invalid', 'url': 'javascript:alert(1)', 'snippet': ''},
        *[
            {
                'title': f'Result {index}',
                'url': f'https://example.com/{index}',
                'snippet': f'Summary {index}',
            }
            for index in range(10)
        ],
    ]

    result = clean_search_results('one question', raw, max_results=3)

    assert [item['url'] for item in result['results']] == [
        'https://example.com/0',
        'https://example.com/1',
        'https://example.com/2',
    ]


def test_clean_search_results_accepts_single_tool_manager_package() -> None:
    result = clean_search_results(
        'one question',
        ({'ok': True, 'value': _raw_search_results()},),
    )

    assert [item['url'] for item in result['results']] == [
        item['url']
        for item in _raw_search_results()
    ]


def test_clean_search_results_bounds_warning_count() -> None:
    result = clean_search_results(
        'one question',
        [
            {'title': f'Invalid {index}', 'url': f'file:///tmp/{index}', 'snippet': ''}
            for index in range(150)
        ],
    )

    assert len(result['warnings']) == 100
    assert any(
        warning['code'] == 'warning_limit_reached'
        for warning in result['warnings']
    )


@pytest.mark.parametrize('max_results', [True, 0, -1, 1.5])
def test_clean_search_results_rejects_invalid_result_limits(max_results: object) -> None:
    with pytest.raises((TypeError, ValueError), match='max_results'):
        clean_search_results(
            'one question',
            _raw_search_results(),
            max_results=max_results,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize('query', [
    '',
    '   ',
    '- first question\n- second question',
    'What is Python, and what is the weather in Tokyo?',
    'What is Python; what is the weather in Tokyo?',
    'Python 是什么，以及东京天气如何？',
    'Python 是什么，东京天气如何？',
])
def test_clean_search_results_requires_one_non_list_query(query: str) -> None:
    with pytest.raises(ValueError, match='query'):
        clean_search_results(query, _raw_search_results())


def test_build_web_evidence_uses_selection_order_not_fetch_or_candidate_order() -> None:
    candidates = _candidates()
    selected = [
        'https://docs.example.com/migration',
        'https://docs.example.com/python',
    ]
    fetched = _batch_fetch_result(
        _fetched_page(
            'https://docs.example.com/python',
            'Python contract\nThe retry option accepts an integer.',
        ),
        _fetched_page(
            'https://docs.example.com/migration',
            'Migration guide\nVersion 2 changes the retry default.',
        ),
    )

    evidence = build_web_evidence(candidates, selected, fetched)
    pages = _pages(evidence)

    assert [page['url'] for page in pages] == selected
    assert all(
        page['queries'] == ['What is the documented retry behavior?']
        for page in pages
    )
    assert [page['title'] for page in pages] == [
        'Migration guide',
        'Python API reference',
    ]
    assert all({'title', 'url', 'content', 'content_sha256'} <= set(page) for page in pages)


def test_build_web_evidence_rejects_urls_outside_candidates() -> None:
    with pytest.raises(ValueError, match='candidate|selection|URL|url'):
        build_web_evidence(
            _candidates(),
            ['https://attacker.example/not-in-search-results'],
            _batch_fetch_result(),
        )


def test_build_web_evidence_rejects_more_than_three_selected_pages() -> None:
    candidates = _candidates()
    selected = [item['url'] for item in candidates['results']]

    with pytest.raises(ValueError, match='3|three|selection'):
        build_web_evidence(
            candidates,
            selected,
            _batch_fetch_result(),
        )


@pytest.mark.parametrize('raw_fetch_result', [
    [],
    {'success': True, 'tool': 'url_fetch', 'result': []},
])
def test_build_web_evidence_degrades_malformed_fetch_results_to_unavailable(
    raw_fetch_result: object,
) -> None:
    evidence = build_web_evidence(
        _candidates(),
        ['https://docs.example.com/python'],
        raw_fetch_result,
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['warnings'][-1]['code'] == 'fetch_failed'


def test_build_web_evidence_rejects_an_envelope_from_another_tool() -> None:
    with pytest.raises(ValueError, match='url_fetch|other_tool'):
        build_web_evidence(
            _candidates(),
            ['https://docs.example.com/python'],
            {
                'success': True,
                'tool': 'other_tool',
                'result': {
                    'total': 0,
                    'succeeded': 0,
                    'failed': 0,
                    'results': [],
                },
            },
        )


def test_build_web_evidence_accepts_url_fetch_single_page_envelope() -> None:
    url = 'https://docs.example.com/python'
    evidence = build_web_evidence(
        _candidates(),
        [url],
        {
            'success': True,
            'tool': 'url_fetch',
            'result': {
                'status': 'ok',
                'url': url,
                'final_url': url,
                'status_code': 200,
                'content_type': 'text/html',
                'content': 'Reference\nThe retry count must be non-negative.',
            },
        },
    )

    assert evidence['status'] == 'ready'
    assert [page['url'] for page in _pages(evidence)] == [url]


def test_build_web_evidence_accepts_single_tool_manager_package() -> None:
    url = 'https://docs.example.com/python'
    wrapped = ({
        'ok': True,
        'value': _batch_fetch_result(
            _fetched_page(
                url,
                'Reference\nThe retry count must be non-negative.',
            ),
        ),
    },)

    evidence = build_web_evidence(_candidates(), [url], wrapped)

    assert evidence['status'] == 'ready'
    assert [page['url'] for page in _pages(evidence)] == [url]


def test_build_web_evidence_accepts_actual_named_tool_result_event() -> None:
    url = 'https://docs.example.com/python'
    event = {
        'id': 'call-1',
        'name': 'url_fetch',
        'arguments': {'urls': [url]},
        'result': _batch_fetch_result(
            _fetched_page(
                url,
                'Reference\nThe retry count must be non-negative.',
            ),
        ),
    }

    evidence = build_web_evidence(_candidates(), [url], event)

    assert evidence['status'] == 'ready'
    assert [page['url'] for page in _pages(evidence)] == [url]


def test_build_web_evidence_turns_named_tool_failure_into_terminal_result() -> None:
    url = 'https://docs.example.com/python'
    event = {
        'id': 'call-1',
        'name': 'url_fetch',
        'arguments': {'urls': [url]},
        'result': 'TimeoutError: https://private.example?api_key=SECRET',
    }

    evidence = build_web_evidence(_candidates(), [url], event)

    assert evidence['status'] == 'unavailable'
    assert evidence['warnings'][-1]['code'] == 'fetch_failed'
    assert evidence['warnings'][-1]['message'] == 'External request timed out.'
    assert 'SECRET' not in str(evidence)


def test_build_web_evidence_accepts_contract_models_as_inputs() -> None:
    url = 'https://docs.example.com/python'
    candidates = SearchResultSet.model_validate(_candidates())
    selection = PageSelection.model_validate({
        'selected_pages': [{'url': url}],
    })

    evidence = build_web_evidence(
        candidates,
        selection,
        _batch_fetch_result(
            _fetched_page(
                url,
                'Reference\nThe retry count must be non-negative.',
            ),
        ),
    )

    assert evidence['status'] == 'ready'
    assert [page['url'] for page in _pages(evidence)] == [url]


def test_build_web_evidence_preserves_successes_when_one_fetch_fails() -> None:
    selected = [
        'https://docs.example.com/python',
        'https://issues.example.com/42',
        'https://docs.example.com/migration',
    ]
    fetched = _batch_fetch_result(
        _fetched_page(
            selected[0],
            'Reference\nRetries are configured with retry_count.',
        ),
        _failed_page(
            selected[1],
            'TimeoutError: timed out requesting https://provider.test/?api_key=secret',
        ),
        _fetched_page(
            selected[2],
            'Migration\nThe default retry count changed in version 2.',
        ),
    )

    evidence = build_web_evidence(_candidates(), selected, fetched)

    assert [page['url'] for page in _pages(evidence)] == [
        selected[0],
        selected[2],
    ]
    assert any(
        warning.get('url') == selected[1]
        and 'timed out' in str(warning.get('message') or warning.get('error') or '')
        for warning in _warnings(evidence)
    )
    assert 'secret' not in str(evidence)


def test_build_web_evidence_rejects_non_text_content_types() -> None:
    url = 'https://docs.example.com/python'
    fetched = _fetched_page(
        url,
        'This must not be retained as readable evidence.',
    )
    fetched['result']['content_type'] = 'application/pdf'

    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(fetched),
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['pages'] == []
    assert any(
        warning['code'] == 'unsupported_content_type'
        for warning in _warnings(evidence)
    )


def test_build_web_evidence_rejects_unsuccessful_page_status() -> None:
    url = 'https://docs.example.com/python'
    fetched = _fetched_page(
        url,
        'This failed response must not become evidence.',
    )
    fetched['result']['status'] = 'error'
    fetched['result']['status_code'] = 500

    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(fetched),
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['pages'] == []
    assert any(
        warning['code'] == 'fetch_failed'
        for warning in _warnings(evidence)
    )


@pytest.mark.parametrize('status_code', ['500', None, True])
def test_build_web_evidence_requires_a_strict_success_status_code(
    status_code: object,
) -> None:
    url = 'https://docs.example.com/python'
    fetched = _fetched_page(
        url,
        'This response must not become evidence.',
    )
    fetched['result']['status_code'] = status_code

    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(fetched),
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['warnings'][-1]['code'] == 'fetch_failed'


@pytest.mark.parametrize('content_type', [
    'text/css',
    'text/javascript',
    'text/event-stream',
    'image/svg+xml',
    'evil+json',
])
def test_build_web_evidence_rejects_non_document_text_mime_types(
    content_type: str,
) -> None:
    url = 'https://docs.example.com/python'
    fetched = _fetched_page(url, 'body { display: none; }')
    fetched['result']['content_type'] = content_type

    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(fetched),
    )

    assert evidence['status'] == 'unavailable'
    assert evidence['warnings'][-1]['code'] == 'unsupported_content_type'


def test_build_web_evidence_accepts_application_json_suffix_mime_type() -> None:
    url = 'https://docs.example.com/python'
    fetched = _fetched_page(url, '{"retry_count": 3}')
    fetched['result']['content_type'] = 'application/vnd.api+json'

    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(fetched),
    )

    assert evidence['status'] == 'ready'
    assert evidence['pages'][0]['content'] == '{"retry_count": 3}'


def test_build_web_evidence_keeps_only_first_copy_of_duplicate_page_content() -> None:
    selected = [
        'https://docs.example.com/python',
        'https://issues.example.com/42',
    ]
    duplicate_content = 'Reference\nThe retry count must be a non-negative integer.'
    evidence = build_web_evidence(
        _candidates(),
        selected,
        _batch_fetch_result(
            _fetched_page(selected[0], duplicate_content),
            _fetched_page(selected[1], duplicate_content),
        ),
    )

    assert evidence['status'] == 'partial'
    assert [page['url'] for page in _pages(evidence)] == [selected[0]]
    assert any(
        warning['code'] == 'duplicate_page_content'
        and warning.get('url') == selected[1]
        for warning in _warnings(evidence)
    )


def test_build_web_evidence_deduplicates_the_retained_truncated_content() -> None:
    selected = [
        'https://docs.example.com/python',
        'https://issues.example.com/42',
    ]
    common_prefix = 'Reference\n' + ('same documented prefix ' * 30)
    evidence = build_web_evidence(
        _candidates(),
        selected,
        _batch_fetch_result(
            _fetched_page(selected[0], common_prefix + 'first tail'),
            _fetched_page(selected[1], common_prefix + 'second tail'),
        ),
        max_page_chars=240,
        max_total_chars=500,
    )

    assert [page['url'] for page in _pages(evidence)] == [selected[0]]
    assert any(
        warning['code'] == 'duplicate_page_content'
        and warning.get('url') == selected[1]
        for warning in _warnings(evidence)
    )


def test_build_web_evidence_propagates_url_fetch_truncation_metadata() -> None:
    url = 'https://docs.example.com/python'
    fetched = _fetched_page(
        url,
        'Reference\nThe retained excerpt fits the Repair page budget.',
    )
    fetched['result']['content_truncated'] = True

    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(fetched),
        max_page_chars=5000,
        max_total_chars=5000,
    )

    page = _pages(evidence)[0]
    assert evidence['status'] == 'partial'
    assert page['truncated'] is True
    assert any(
        warning['code'] == 'content_truncated'
        and warning.get('url') == url
        for warning in _warnings(evidence)
    )


def test_build_web_evidence_drops_empty_pages_and_cleans_repeated_boilerplate() -> None:
    python_url = 'https://docs.example.com/python'
    issues_url = 'https://issues.example.com/42'
    fetched = _batch_fetch_result(
        _fetched_page(
            python_url,
            '''
            Skip to content

            Python API reference
            The retry option accepts an integer.
            The   retry option accepts an integer.

            Accept all cookies
            Privacy policy
            Terms of service
            © 2026 Example Inc.
            ''',
        ),
        _fetched_page(
            issues_url,
            ' \n\t\n Accept all cookies \n Privacy policy \n © 2026 Example Inc. ',
        ),
    )

    evidence = build_web_evidence(
        _candidates(),
        [python_url, issues_url],
        fetched,
    )
    pages = _pages(evidence)

    assert [page['url'] for page in pages] == [python_url]
    assert pages[0]['content'] == (
        'Python API reference\n'
        'The retry option accepts an integer.\n'
        'The   retry option accepts an integer.'
    )
    assert any(warning.get('url') == issues_url for warning in _warnings(evidence))


def test_clean_page_content_preserves_code_structure_and_repeated_values() -> None:
    content = '''
        {
          "values": [
            "same",
            "same"
          ]
        }
    '''

    assert clean_page_content(content) == (
        '{\n'
        '  "values": [\n'
        '    "same",\n'
        '    "same"\n'
        '  ]\n'
        '}'
    )


def test_clean_page_content_preserves_repeated_top_level_code_lines() -> None:
    content = (
        'print("connection established")\n'
        'print("connection established")'
    )

    assert clean_page_content(content) == content


def test_clean_page_content_preserves_top_level_code_literal_whitespace() -> None:
    content = 'pattern = "a   b"'

    assert clean_page_content(content) == content


def test_clean_page_content_preserves_technical_single_word_lines() -> None:
    content = (
        'register\n'
        'subscribe\n'
        'next\n'
        '注册\n'
        'copyright = metadata.get("copyright")'
    )

    assert clean_page_content(content) == content


def test_build_web_evidence_sanitizes_inherited_warning_messages() -> None:
    candidates = _candidates()
    candidates['warnings'] = [{
        'code': 'invalid_search_result',
        'message': 'request failed api_key=SECRET',
        'url': 'https://docs.example.com/python?api_key=SECRET',
        'queries': ['api_key=SECRET'],
    }]
    url = 'https://docs.example.com/python'

    evidence = build_web_evidence(
        candidates,
        [url],
        _batch_fetch_result(
            _fetched_page(
                url,
                'Reference\nThe retry count must be non-negative.',
            ),
        ),
    )

    assert 'SECRET' not in str(evidence)
    assert evidence['warnings'][0]['message'] == (
        'web_search returned an invalid result that was discarded.'
    )


def test_build_web_evidence_applies_per_page_and_total_character_budgets() -> None:
    selected = [
        'https://docs.example.com/python',
        'https://issues.example.com/42',
        'https://docs.example.com/migration',
    ]
    fetched = _batch_fetch_result(*[
        _fetched_page(
            url,
            f'Page {index}\n' + chr(ord('A') + index) * 500,
        )
        for index, url in enumerate(selected)
    ])

    evidence = build_web_evidence(
        _candidates(),
        selected,
        fetched,
        max_page_chars=240,
        max_total_chars=500,
    )
    pages = _pages(evidence)

    assert evidence['status'] == 'partial'
    assert 1 <= len(pages) <= 3
    assert all(0 < len(page['content']) <= 240 for page in pages)
    assert sum(len(page['content']) for page in pages) <= 500
    assert len(pages[0]['content']) == 240
    if len(pages) >= 2:
        assert len(pages[1]['content']) <= 240
    assert [page['url'] for page in pages] == selected[:len(pages)]


@pytest.mark.parametrize(
    ('max_page_chars', 'max_total_chars'),
    [
        (True, 100),
        (0, 100),
        (100, True),
        (100, 0),
    ],
)
def test_build_web_evidence_rejects_invalid_character_budgets(
    max_page_chars: object,
    max_total_chars: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match='max_.*chars|budget'):
        build_web_evidence(
            _candidates(),
            ['https://docs.example.com/python'],
            _batch_fetch_result(),
            max_page_chars=max_page_chars,  # type: ignore[arg-type]
            max_total_chars=max_total_chars,  # type: ignore[arg-type]
        )


def test_page_hashes_cover_the_exact_retained_utf8_content() -> None:
    url = 'https://docs.example.com/python'
    evidence = build_web_evidence(
        _candidates(),
        [url],
        _batch_fetch_result(
            _fetched_page(url, '接口约定\n重试次数必须是非负整数。'),
        ),
    )
    page = _pages(evidence)[0]

    assert page['content_sha256'] == hashlib.sha256(
        page['content'].encode('utf-8'),
    ).hexdigest()


def test_build_web_evidence_is_deterministic_and_does_not_mutate_inputs() -> None:
    candidates = _candidates()
    selected = [
        'https://docs.example.com/migration',
        'https://docs.example.com/python',
    ]
    first_fetch = _batch_fetch_result(
        _fetched_page(selected[1], 'Reference\nThe retry count is configurable.'),
        _fetched_page(selected[0], 'Migration\nThe retry default changed.'),
    )
    reordered_fetch = _batch_fetch_result(
        deepcopy(first_fetch['result']['results'][1]),
        deepcopy(first_fetch['result']['results'][0]),
    )
    candidates_before = deepcopy(candidates)
    selected_before = deepcopy(selected)
    fetch_before = deepcopy(first_fetch)

    first = build_web_evidence(candidates, selected, first_fetch)
    second = build_web_evidence(
        deepcopy(candidates),
        deepcopy(selected),
        reordered_fetch,
    )

    assert first == second
    assert candidates == candidates_before
    assert selected == selected_before
    assert first_fetch == fetch_before
