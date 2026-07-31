from __future__ import annotations

import json
import re
import textwrap
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from ipaddress import ip_address
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

from pydantic import ValidationError

from .web_evidence_contracts import (
    PageSelection,
    RepairWebEvidence,
    SearchResultSet,
)

DEFAULT_MAX_SEARCH_RESULTS = 8
DEFAULT_MAX_PAGE_CHARS = 3000
DEFAULT_MAX_TOTAL_CHARS = 7000
MAX_SELECTED_PAGES = 3

_MAX_QUERY_CHARS = 500
_MAX_TITLE_CHARS = 300
_MAX_SNIPPET_CHARS = 1000
_MAX_URL_CHARS = 2048
_MIN_USEFUL_CONTENT_CHARS = 8
_MIN_BUDGET_CHARS = 200
_MAX_BUDGET_CHARS = 50_000
_MAX_WARNINGS = 100

_INLINE_WHITESPACE = re.compile(r'\s+')
_LIST_ITEM_QUERY = re.compile(r'(?:^|\s)(?:[-*•]|\d{1,2}[.)、])\s+')
_QUESTION_CLAUSE_MARKER = re.compile(
    r'(?:'
    r'\b(?:what|who|when|where|why|how|which)\b'
    r'|(?:^|\s)(?:is|are|do|does|can|could|should|would)\s+\w+'
    r'|(?:什么|谁|何时|哪里|为何|为什么|如何|怎么|是否|能否|可否)'
    r')',
    re.IGNORECASE,
)
_SECOND_QUESTION_CLAUSE = re.compile(
    r'(?:'
    r'\b(?:and|also)\s+'
    r'(?:what|who|when|where|why|how|which|is|are|do|does|can|could|should|would)\b'
    r'|(?:以及|并且|还有|另外)[^，；?？]{0,40}'
    r'(?:什么|谁|何时|哪里|为何|为什么|如何|怎么|是否|能否|可否)'
    r')',
    re.IGNORECASE,
)
# These are the provider leaves currently registered under the existing
# web_search capability. A trusted assembler may pass an exact future leaf to
# ``clean_search_results``; arbitrary Search-like suffixes are not provenance.
DEFAULT_WEB_SEARCH_TOOL_NAMES = frozenset({
    'web_search',
    'GoogleSearch_search',
    'BingSearch_search',
    'BochaSearch_search',
    'TavilySearch_search',
})
_BOILERPLATE_LINES = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r'(?:skip to (?:main )?content|back to top)',
        r'(?:table of contents|on this page)',
        r'(?:sign in|log in|register now|create an account|subscribe now)',
        r'(?:subscribe to (?:our )?(?:newsletter|updates))',
        r'(?:previous\s+next|previous page|next page)',
        r'(?:privacy policy|terms of (?:use|service)|cookie policy)',
        r'(?:accept(?: all)? cookies?|reject(?: all)? cookies?|manage cookie settings)',
        r'(?:all rights reserved|copyright\s+(?:(?:©|\(c\))\s*)?\d{4}(?:\s+.{0,160})?)',
        r'(?:©\s*\d{4}(?:\s+.{0,160})?)',
        r'(?:跳到正文|返回顶部|目录|本页内容)',
        r'(?:登录账号|注册账号|立即订阅)',
        r'(?:隐私政策|使用条款|服务条款|Cookie\s*政策)',
        r'(?:接受(?:全部)? Cookie|拒绝(?:全部)? Cookie|管理 Cookie 设置)',
        r'(?:版权所有|保留所有权利)',
    )
)
_UNAVAILABLE_MESSAGES = {
    'search_unavailable': 'No web_search capability was available for this Repair target.',
    'no_search_results': 'web_search returned no usable page candidates.',
    'no_page_selected': 'Repair did not select a page for url_fetch.',
    'fetch_failed': 'url_fetch did not return a usable result for the selected pages.',
}
_SAFE_WARNING_MESSAGES = {
    'invalid_search_result': 'web_search returned an invalid result that was discarded.',
    'duplicate_search_result': 'A duplicate web_search result was discarded.',
    'search_result_limit_reached': 'The configured web_search result limit was reached.',
    'search_unavailable': 'web_search did not return a usable result.',
    'no_search_results': 'web_search returned no usable page candidates.',
    'no_page_selected': 'Repair did not select a page for url_fetch.',
    'warning_limit_reached': 'Additional web-evidence warnings were omitted.',
    'fetch_failed': 'url_fetch did not return a usable page.',
    'fetch_result_missing': 'url_fetch returned no result for a selected page.',
    'unsupported_content_type': 'Fetched content was not a supported readable text type.',
    'empty_page_content': 'Fetched content did not contain enough readable body text.',
    'duplicate_page_content': 'Duplicate fetched page content was discarded.',
    'content_truncated': (
        'Fetched page content was truncated by url_fetch or the evidence budget.'
    ),
    'total_content_budget_exhausted': 'The external-evidence content budget was exhausted.',
}


@dataclass(frozen=True)
class _Candidate:
    title: str
    url: str
    snippet: str
    queries: tuple[str, ...]


@dataclass(frozen=True)
class _FetchEntry:
    requested_url: str
    success: bool
    page: Mapping[str, Any] | None = None
    error: str = ''


@dataclass(frozen=True)
class _SearchEventMetadata:
    tool_name: str
    call_id: str
    query: object = None
    has_arguments: bool = False


def clean_search_results(
    query: str,
    raw_results: object,
    *,
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    expected_tool_name: str | None = None,
    expected_call_id: str | None = None,
) -> dict[str, Any]:
    """Project one web-search response to ordered title, URL, and snippet items.

    Provider ranking is intentionally preserved. The function performs no
    semantic scoring or reranking.
    """
    normalized_query = validate_single_search_question(query)
    expected_tool, expected_id = _expected_search_event(
        expected_tool_name,
        expected_call_id,
    )
    event = _extract_search_event_metadata(raw_results)
    if event is None:
        if expected_tool is not None:
            raise ValueError(
                'web_search result must include named tool-call metadata'
            )
    else:
        allowed_tool_names = set(DEFAULT_WEB_SEARCH_TOOL_NAMES)
        if expected_tool is not None:
            allowed_tool_names.add(expected_tool)
        if event.tool_name not in allowed_tool_names:
            raise ValueError(
                'tool result must come from web_search, '
                f'got {event.tool_name}'
            )
        if not event.has_arguments:
            raise ValueError(
                'named web_search result must include arguments.query'
            )
        event_query = validate_single_search_question(event.query)
        if event_query != normalized_query:
            raise ValueError(
                'web_search query does not match the tool call arguments'
            )
        if expected_tool is not None and event.tool_name != expected_tool:
            raise ValueError(
                'web_search tool name does not match the prepared call'
            )
        if expected_id is not None and event.call_id != expected_id:
            raise ValueError(
                'web_search call ID does not match the prepared call'
            )
    limit = _bounded_int(max_results, 'max_results', minimum=1, maximum=50)
    raw_items, envelope_warning_code, envelope_warning = _extract_search_items(
        raw_results,
        expected_tool_name=expected_tool,
    )
    warnings: list[dict[str, Any]] = []
    if envelope_warning:
        warnings.append(_warning(envelope_warning_code, envelope_warning))

    cleaned: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    reached_limit = False
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            warnings.append(_warning(
                'invalid_search_result',
                f'Search result at index {index} is not an object.',
            ))
            continue
        url = _normalize_http_url(item.get('url'))
        if not url:
            warnings.append(_warning(
                'invalid_search_result',
                f'Search result at index {index} has no valid public HTTP(S) URL.',
            ))
            continue
        if url in seen_urls:
            warnings.append(_warning(
                'duplicate_search_result',
                'Duplicate search result URL was discarded.',
                url=url,
            ))
            continue

        title = _clip_text(_clean_inline_text(item.get('title')), _MAX_TITLE_CHARS)
        snippet = _clip_text(_clean_inline_text(item.get('snippet')), _MAX_SNIPPET_CHARS)
        if not title:
            warnings.append(_warning(
                'invalid_search_result',
                'Search result without a title was discarded.',
                url=url,
            ))
            continue

        if len(cleaned) >= limit:
            reached_limit = True
            break
        seen_urls.add(url)
        cleaned.append({
            'title': title,
            'url': url,
            'snippet': snippet,
        })

    if reached_limit:
        warnings.append(_warning(
            'search_result_limit_reached',
            f'Search results were limited to the first {limit} valid unique URLs.',
        ))

    return _dump_search_result_set({
        'query': normalized_query,
        'results': cleaned,
        'warnings': warnings,
    })


def validate_single_search_question(query: str) -> str:
    """Reject clearly combined search requests without guessing semantic intent."""
    if not isinstance(query, str):
        raise TypeError('web search query must be a string')
    stripped = query.strip()
    if not stripped:
        raise ValueError('web search query must contain non-whitespace text')
    if len(stripped) > _MAX_QUERY_CHARS:
        raise ValueError(f'web search query must not exceed {_MAX_QUERY_CHARS} characters')
    if '\n' in stripped or '\r' in stripped:
        raise ValueError('web search query must contain one question on one line')
    if len(re.findall(r'[?？]', stripped)) > 1:
        raise ValueError('web search query must not contain multiple questions')
    if _LIST_ITEM_QUERY.search(stripped):
        raise ValueError('web search query must not contain a list of questions')
    normalized = _clean_inline_text(stripped)
    if (
        _SECOND_QUESTION_CLAUSE.search(normalized)
        or _has_separated_question_clauses(normalized)
    ):
        raise ValueError('web search query must contain one explicit question')
    return normalized


def _has_separated_question_clauses(query: str) -> bool:
    for boundary in re.finditer(r'[,，;；]', query):
        left = query[:boundary.start()]
        right = query[boundary.end():]
        if (
            _QUESTION_CLAUSE_MARKER.search(left)
            and _QUESTION_CLAUSE_MARKER.search(right)
        ):
            return True
    return False


def build_web_evidence(
    candidates: SearchResultSet | Mapping[str, Any] | Sequence[SearchResultSet | Mapping[str, Any]],
    selection: PageSelection | Mapping[str, Any] | Sequence[str],
    raw_fetch_result: object,
    *,
    max_page_chars: int = DEFAULT_MAX_PAGE_CHARS,
    max_total_chars: int = DEFAULT_MAX_TOTAL_CHARS,
) -> dict[str, Any]:
    """Clean fetched pages selected from prior web-search candidates.

    Selection order is preserved. Every selected URL must have appeared in the
    supplied search candidates; this prevents a model from inventing fetch
    targets after search.
    """
    page_budget = _bounded_int(
        max_page_chars,
        'max_page_chars',
        minimum=_MIN_BUDGET_CHARS,
        maximum=_MAX_BUDGET_CHARS,
    )
    total_budget = _bounded_int(
        max_total_chars,
        'max_total_chars',
        minimum=_MIN_BUDGET_CHARS,
        maximum=_MAX_BUDGET_CHARS,
    )
    candidate_index, inherited_warnings = _candidate_index(candidates)
    selected_urls = _selected_urls(selection)
    selected: list[_Candidate] = []
    for url in selected_urls:
        candidate = candidate_index.get(url)
        if candidate is None:
            raise ValueError(
                'selected URL was not returned by web_search: '
                f'{url}'
            )
        selected.append(candidate)

    entries, global_fetch_error = _extract_fetch_entries(raw_fetch_result)
    entry_index = {
        normalized: entry
        for entry in entries
        for normalized in (_normalize_http_url(entry.requested_url),)
        if normalized
    }
    warnings = list(inherited_warnings)
    pages: list[dict[str, Any]] = []
    remaining = total_budget
    seen_retained_content: set[str] = set()

    for candidate in selected:
        entry = entry_index.get(candidate.url)
        if entry is None:
            warnings.append(_warning(
                'fetch_failed' if global_fetch_error else 'fetch_result_missing',
                global_fetch_error or 'url_fetch returned no result for the selected URL.',
                url=candidate.url,
            ))
            continue
        if not entry.success or entry.page is None:
            warnings.append(_warning(
                'fetch_failed',
                entry.error or 'url_fetch failed for the selected URL.',
                url=candidate.url,
            ))
            continue

        page_failure = _page_failure(entry.page)
        if page_failure:
            warnings.append(_warning(
                'fetch_failed',
                page_failure,
                url=candidate.url,
            ))
            continue
        content_type = _clean_inline_text(entry.page.get('content_type')).lower()
        if not _is_readable_content_type(content_type):
            warnings.append(_warning(
                'unsupported_content_type',
                'Fetched page content type is not supported as readable text.',
                url=candidate.url,
            ))
            continue
        content = clean_page_content(entry.page.get('content'))
        if len(content) < _MIN_USEFUL_CONTENT_CHARS:
            warnings.append(_warning(
                'empty_page_content',
                'Fetched page did not contain enough readable body text.',
                url=candidate.url,
            ))
            continue
        if remaining <= 0:
            warnings.append(_warning(
                'total_content_budget_exhausted',
                'The total external-evidence content budget was exhausted.',
                url=candidate.url,
            ))
            continue

        retained, budget_truncated = _truncate_content(
            content,
            min(page_budget, remaining),
        )
        upstream_truncated = entry.page.get('content_truncated') is True
        truncated = upstream_truncated or budget_truncated
        if len(retained) < _MIN_USEFUL_CONTENT_CHARS:
            warnings.append(_warning(
                'empty_page_content',
                'Fetched page had no useful text within the remaining content budget.',
                url=candidate.url,
            ))
            continue
        content_sha256 = sha256(retained.encode('utf-8')).hexdigest()
        if content_sha256 in seen_retained_content:
            warnings.append(_warning(
                'duplicate_page_content',
                'Fetched page duplicated retained body text from another URL.',
                url=candidate.url,
            ))
            continue
        if truncated:
            message = (
                'Fetched page content was truncated by url_fetch or the '
                'configured evidence budget.'
                if upstream_truncated
                else 'Fetched page content was truncated to the configured '
                'evidence budget.'
            )
            warnings.append(_warning(
                'content_truncated',
                message,
                url=candidate.url,
            ))

        seen_retained_content.add(content_sha256)
        final_url = _normalize_http_url(entry.page.get('final_url')) or candidate.url
        pages.append({
            'evidence_id': _evidence_id(candidate.url, content_sha256),
            'queries': list(candidate.queries),
            'title': candidate.title,
            'url': candidate.url,
            'final_url': final_url,
            'snippet': candidate.snippet,
            'content': retained,
            'content_sha256': content_sha256,
            'character_count': len(retained),
            'truncated': truncated,
        })
        remaining -= len(retained)

    status = (
        'unavailable'
        if not pages
        else 'partial'
        if (
            len(pages) != len(selected)
            or any(page['truncated'] for page in pages)
        )
        else 'ready'
    )
    return _dump_web_evidence({
        'status': status,
        'content_trust': 'external_untrusted',
        'pages': pages,
        'total_character_count': sum(page['character_count'] for page in pages),
        'warnings': warnings,
    })


def build_unavailable_web_evidence(
    candidates: (
        SearchResultSet
        | Mapping[str, Any]
        | Sequence[SearchResultSet | Mapping[str, Any]]
        | None
    ) = None,
    *,
    code: str,
) -> dict[str, Any]:
    """Finish evidence collection when searching or fetching cannot proceed."""
    if code not in _UNAVAILABLE_MESSAGES:
        raise ValueError(
            'unavailable evidence code must be one of: '
            f'{", ".join(sorted(_UNAVAILABLE_MESSAGES))}'
        )
    inherited_warnings: list[dict[str, Any]] = []
    if candidates is not None:
        _, inherited_warnings = _candidate_index(candidates)
    warnings = [
        *inherited_warnings,
        _warning(code, _UNAVAILABLE_MESSAGES[code]),
    ]
    return _dump_web_evidence({
        'status': 'unavailable',
        'content_trust': 'external_untrusted',
        'pages': [],
        'total_character_count': 0,
        'warnings': warnings,
    })


def prepare_url_fetch(
    candidates: SearchResultSet | Mapping[str, Any] | Sequence[SearchResultSet | Mapping[str, Any]],
    selection: PageSelection | Mapping[str, Any] | Sequence[str],
) -> dict[str, list[str]]:
    """Validate model-selected URLs and return arguments for the existing tool."""
    candidate_index, _ = _candidate_index(candidates)
    selected_urls = _selected_urls(selection)
    for url in selected_urls:
        if url not in candidate_index:
            raise ValueError(
                'selected URL was not returned by web_search: '
                f'{url}'
            )
    return {'urls': selected_urls}


def clean_page_content(value: object) -> str:
    """Normalize readable page text while preserving paragraph order and wording."""
    if not isinstance(value, str):
        return ''
    normalized = unicodedata.normalize('NFC', value.replace('\u00a0', ' '))
    normalized = ''.join(
        char if char in '\n\t' or unicodedata.category(char) != 'Cc' else ' '
        for char in normalized
    )
    normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
    normalized = textwrap.dedent(normalized)

    lines: list[str] = []
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if not stripped or _is_boilerplate_line(stripped):
            continue
        is_indented = raw_line[:1].isspace()
        line = (
            raw_line.rstrip()
            if is_indented
            else stripped
        )
        lines.append(line)
    return '\n'.join(lines)


def _extract_search_items(
    raw: object,
    *,
    expected_tool_name: str | None = None,
) -> tuple[list[object], str, str]:
    value = _unwrap_tool_observation(
        raw,
        expected_tool='web_search',
        expected_search_tool_name=expected_tool_name,
    )
    if isinstance(value, Mapping) and 'ok' in value:
        if value.get('ok') is not True:
            return (
                [],
                'search_unavailable',
                _clean_error(value.get('msg')) or 'web_search tool call failed.',
            )
        value = value.get('value')
    if isinstance(value, Mapping) and value.get('success') is False:
        return (
            [],
            'search_unavailable',
            _clean_error(value.get('error')) or 'web_search returned an error.',
        )
    if isinstance(value, Mapping) and 'result' in value and not _looks_like_search_item(value):
        value = value.get('result')
    if isinstance(value, Mapping) and 'results' in value:
        nested = value.get('results')
        if not isinstance(nested, Sequence) or isinstance(
            nested,
            (str, bytes, bytearray),
        ):
            return (
                [],
                'invalid_search_result',
                'web_search returned an unsupported results field.',
            )
        value = nested
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value), 'invalid_search_result', ''
    if value in (None, ''):
        return [], 'invalid_search_result', ''
    if isinstance(value, str):
        return [], 'search_unavailable', _clean_error(value)
    return (
        [],
        'invalid_search_result',
        'web_search returned an unsupported result shape.',
    )


def _extract_search_event_metadata(
    raw: object,
) -> _SearchEventMetadata | None:
    """Read provenance from one streamed/named search tool envelope."""
    value = raw
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        items = list(value)
        if len(items) == 1 and isinstance(items[0], Mapping):
            value = items[0]
    if isinstance(value, Mapping) and 'tool_results' in value:
        tool_results = value.get('tool_results')
        if (
            not isinstance(tool_results, Sequence)
            or isinstance(tool_results, (str, bytes, bytearray))
            or len(tool_results) != 1
            or not isinstance(tool_results[0], Mapping)
        ):
            raise ValueError(
                'tool_results event must contain exactly one tool result'
            )
        value = tool_results[0]
    if not isinstance(value, Mapping):
        return None

    declared_tool = _clean_inline_text(value.get('name') or value.get('tool'))
    if not declared_tool:
        return None
    call_id = _clean_inline_text(value.get('id'))
    if 'arguments' not in value:
        return _SearchEventMetadata(
            tool_name=declared_tool,
            call_id=call_id,
        )

    arguments = value.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                'web_search tool arguments must be a JSON object'
            ) from exc
    if not isinstance(arguments, Mapping):
        raise ValueError('web_search tool arguments must be an object')
    if 'query' not in arguments:
        raise ValueError('web_search tool arguments must contain query')
    return _SearchEventMetadata(
        tool_name=declared_tool,
        call_id=call_id,
        query=arguments.get('query'),
        has_arguments=True,
    )


def _expected_search_event(
    tool_name: object,
    call_id: object,
) -> tuple[str | None, str | None]:
    if (tool_name is None) != (call_id is None):
        raise ValueError(
            'expected web_search tool name and call ID must be provided together'
        )
    if tool_name is None:
        return None, None
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError('expected web_search tool name must be non-empty')
    if not isinstance(call_id, str) or not call_id.strip():
        raise ValueError('expected web_search call ID must be non-empty')
    normalized_tool_name = tool_name.strip()
    normalized_call_id = call_id.strip()
    if (
        normalized_tool_name != tool_name
        or normalized_call_id != call_id
        or len(normalized_tool_name) > 256
        or len(normalized_call_id) > 256
    ):
        raise ValueError(
            'expected web_search tool name and call ID must be normalized '
            'and at most 256 characters'
        )
    if (
        normalized_tool_name != 'web_search'
        and not normalized_tool_name.endswith('Search_search')
    ):
        raise ValueError(
            'expected web_search tool must be web_search or *Search_search'
        )
    return normalized_tool_name, normalized_call_id


def _looks_like_search_item(value: Mapping[str, Any]) -> bool:
    return 'url' in value and ('title' in value or 'snippet' in value)


def _candidate_index(
    candidates: (
        SearchResultSet
        | Mapping[str, Any]
        | Sequence[SearchResultSet | Mapping[str, Any]]
    ),
) -> tuple[dict[str, _Candidate], list[dict[str, Any]]]:
    raw_sets: Sequence[SearchResultSet | Mapping[str, Any]]
    if isinstance(candidates, (SearchResultSet, Mapping)):
        raw_sets = [candidates]
    elif isinstance(candidates, Sequence) and not isinstance(
        candidates,
        (str, bytes, bytearray),
    ):
        raw_sets = candidates
    else:
        raise TypeError('search candidates must be a result set or a sequence of result sets')

    index: dict[str, _Candidate] = {}
    warnings: list[dict[str, Any]] = []
    for raw_set in raw_sets:
        result_set = (
            raw_set
            if isinstance(raw_set, SearchResultSet)
            else SearchResultSet.model_validate(raw_set)
        )
        warnings.extend(
            _sanitize_inherited_warning(warning.model_dump(mode='json'))
            for warning in result_set.warnings
        )
        for result in result_set.results:
            url = _normalize_http_url(result.url)
            if not url:
                continue
            existing = index.get(url)
            if existing is None:
                index[url] = _Candidate(
                    title=result.title,
                    url=url,
                    snippet=result.snippet,
                    queries=(result_set.query,),
                )
            elif result_set.query not in existing.queries:
                index[url] = _Candidate(
                    title=existing.title,
                    url=existing.url,
                    snippet=existing.snippet,
                    queries=(*existing.queries, result_set.query),
                )
    return index, warnings


def _selected_urls(
    selection: PageSelection | Mapping[str, Any] | Sequence[str],
) -> list[str]:
    raw_urls: object
    if isinstance(selection, PageSelection):
        raw_urls = [item.url for item in selection.selected_pages]
    elif isinstance(selection, Mapping):
        if 'selected_urls' in selection:
            raw_urls = selection.get('selected_urls')
        elif 'urls' in selection:
            raw_urls = selection.get('urls')
        else:
            raw_urls = selection.get('selected_pages')
    else:
        raw_urls = selection
    if not isinstance(raw_urls, Sequence) or isinstance(raw_urls, (str, bytes, bytearray)):
        raise TypeError('page selection must contain a sequence of URLs')

    result: list[str] = []
    seen: set[str] = set()
    for item in raw_urls:
        raw_url = item.get('url') if isinstance(item, Mapping) else item
        url = _normalize_http_url(raw_url)
        if not url:
            raise ValueError('selected page URL must be an absolute HTTP(S) URL')
        if url not in seen:
            seen.add(url)
            result.append(url)
    if not result:
        raise ValueError('at least one page must be selected')
    if len(result) > MAX_SELECTED_PAGES:
        raise ValueError(f'at most {MAX_SELECTED_PAGES} pages may be selected')
    try:
        validated = PageSelection.model_validate({
            'selected_pages': [{'url': url} for url in result],
        })
    except ValidationError as exc:
        raise ValueError(f'repair web-page selection contract failed: {exc}') from exc
    return [item.url for item in validated.selected_pages]


def _extract_fetch_entries(raw: object) -> tuple[list[_FetchEntry], str]:
    value = _unwrap_tool_observation(raw, expected_tool='url_fetch')
    if isinstance(value, Mapping) and 'ok' in value:
        if value.get('ok') is not True:
            return [], _clean_error(value.get('msg')) or 'url_fetch tool call failed.'
        value = value.get('value')
    if isinstance(value, Mapping) and value.get('success') is False:
        return [], _clean_error(value.get('error')) or 'url_fetch returned an error.'
    if (
        isinstance(value, Mapping)
        and value.get('tool') not in (None, '', 'url_fetch')
    ):
        raise ValueError('fetch result must come from the url_fetch tool')
    if (
        isinstance(value, Mapping)
        and value.get('success') is True
        and 'result' in value
    ):
        value = value.get('result')

    if isinstance(value, Mapping) and isinstance(value.get('results'), Sequence):
        entries = [
            _fetch_entry(item)
            for item in value.get('results', [])
            if isinstance(item, Mapping)
        ]
        if not entries:
            return [], 'url_fetch returned no page results.'
        return entries, ''
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [], 'url_fetch returned an unsupported result shape.'
    if isinstance(value, Mapping) and 'content' in value:
        url = _clean_inline_text(value.get('url') or value.get('final_url'))
        return [_FetchEntry(requested_url=url, success=True, page=value)], ''
    if value in (None, ''):
        return [], 'url_fetch returned no result.'
    if isinstance(value, str):
        return [], _clean_error(value)
    return [], 'url_fetch returned an unsupported result shape.'


def _unwrap_tool_observation(
    value: object,
    *,
    expected_tool: str,
    expected_search_tool_name: str | None = None,
) -> object:
    """Unwrap one existing ToolManager or streamed tool-result envelope."""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        if (
            len(items) == 1
            and isinstance(items[0], Mapping)
            and ('ok' in items[0] or 'name' in items[0])
        ):
            value = items[0]
    if isinstance(value, Mapping) and 'tool_results' in value:
        tool_results = value.get('tool_results')
        if (
            not isinstance(tool_results, Sequence)
            or isinstance(tool_results, (str, bytes, bytearray))
            or len(tool_results) != 1
            or not isinstance(tool_results[0], Mapping)
        ):
            raise ValueError('tool_results event must contain exactly one tool result')
        value = tool_results[0]
    if isinstance(value, Mapping):
        declared_tool = _clean_inline_text(value.get('name') or value.get('tool'))
        accepted_search_tools = set(DEFAULT_WEB_SEARCH_TOOL_NAMES)
        if expected_search_tool_name is not None:
            accepted_search_tools.add(expected_search_tool_name)
        declared_tool_is_valid = (
            declared_tool in accepted_search_tools
            if expected_tool == 'web_search'
            else declared_tool == expected_tool
        )
        if declared_tool and not declared_tool_is_valid:
            raise ValueError(
                f'tool result must come from {expected_tool}, got {declared_tool}'
            )
        if 'name' in value and 'result' in value:
            return value.get('result')
    return value


def _fetch_entry(value: Mapping[str, Any]) -> _FetchEntry:
    requested_url = _clean_inline_text(value.get('url'))
    success = value.get('success') is True
    page = value.get('result') if isinstance(value.get('result'), Mapping) else None
    if page is not None and not requested_url:
        requested_url = _clean_inline_text(page.get('url') or page.get('final_url'))
    return _FetchEntry(
        requested_url=requested_url,
        success=success and page is not None,
        page=page,
        error=_clean_error(value.get('error')),
    )


def _normalize_http_url(value: object) -> str:
    if not isinstance(value, str):
        return ''
    text = value.strip()
    if (
        not text
        or len(text) > _MAX_URL_CHARS
        or any(char.isspace() for char in text)
    ):
        return ''
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return ''
    scheme = parsed.scheme.lower()
    if scheme not in {'http', 'https'} or not parsed.hostname:
        return ''
    if parsed.username or parsed.password:
        return ''
    try:
        hostname = parsed.hostname.rstrip('.').encode('idna').decode('ascii').lower()
    except UnicodeError:
        return ''
    if not hostname or any(char.isspace() for char in hostname):
        return ''
    if hostname == 'localhost' or hostname.endswith(('.localhost', '.local')):
        return ''
    try:
        if not ip_address(hostname).is_global:
            return ''
    except ValueError:
        pass
    host = f'[{hostname}]' if ':' in hostname and not hostname.startswith('[') else hostname
    default_port = (scheme == 'http' and port == 80) or (scheme == 'https' and port == 443)
    netloc = host if port is None or default_port else f'{host}:{port}'
    normalized = SplitResult(
        scheme=scheme,
        netloc=netloc,
        path=parsed.path or '/',
        query=parsed.query,
        fragment='',
    )
    return urlunsplit(normalized)


def _clean_inline_text(value: object) -> str:
    if not isinstance(value, str):
        return ''
    normalized = unicodedata.normalize('NFC', value.replace('\u00a0', ' '))
    normalized = ''.join(
        char if unicodedata.category(char) != 'Cc' else ' '
        for char in normalized
    )
    return _INLINE_WHITESPACE.sub(' ', normalized).strip()


def _clean_error(value: object) -> str:
    if isinstance(value, Mapping):
        value = value.get('reason') or value.get('detail') or value.get('message')
    text = _clean_inline_text(value)
    lowered = text.lower()
    if 'timed out' in lowered or 'timeout' in lowered:
        return 'External request timed out.'
    if 'rate limit' in lowered or 'too many requests' in lowered or '429' in lowered:
        return 'External service rate limit was reached.'
    if 'unauthorized' in lowered or 'authentication' in lowered or re.search(r'\b401\b', lowered):
        return 'External service authentication failed.'
    if 'forbidden' in lowered or re.search(r'\b403\b', lowered):
        return 'External service denied the request.'
    if (
        'connection' in lowered
        or 'could not resolve' in lowered
        or 'dns' in lowered
    ):
        return 'External service could not be reached.'
    exception = re.match(r'([A-Za-z_][A-Za-z0-9_.]{0,79}(?:Error|Exception))\b', text)
    if exception:
        return f'External tool failed with {exception.group(1)}.'
    return 'External tool call failed.'


def _sanitize_inherited_warning(value: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild warnings so direct public-API callers cannot inject raw errors."""
    code = _clean_inline_text(value.get('code'))
    message = _SAFE_WARNING_MESSAGES.get(code, 'External web-evidence warning.')
    return _warning(code, message)


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit - 1].rstrip() + '…'


def _is_boilerplate_line(line: str) -> bool:
    if len(line) > 240:
        return False
    stripped = line.strip(' \t|·•—–-:：.。!！')
    return any(pattern.fullmatch(stripped) for pattern in _BOILERPLATE_LINES)


def _is_readable_content_type(value: str) -> bool:
    media_type = value.partition(';')[0].strip()
    return (
        media_type in {
            'text/html',
            'text/plain',
            'text/markdown',
            'text/x-markdown',
            'text/xml',
            'application/json',
            'application/xml',
            'application/xhtml+xml',
        }
        or (
            media_type.startswith('application/')
            and media_type.endswith(('+json', '+xml'))
        )
    )


def _page_failure(page: Mapping[str, Any]) -> str:
    status = _clean_inline_text(page.get('status')).lower()
    if status != 'ok':
        return 'Fetched page did not report a successful status.'
    status_code = page.get('status_code')
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or not 200 <= status_code < 300
    ):
        return 'Fetched page did not report a successful HTTP status.'
    return ''


def _truncate_content(content: str, limit: int) -> tuple[str, bool]:
    if len(content) <= limit:
        return content, False
    prefix = content[:limit]
    minimum_boundary = max(_MIN_USEFUL_CONTENT_CHARS, int(limit * 0.65))
    newline = prefix.rfind('\n')
    if newline >= minimum_boundary:
        prefix = prefix[:newline]
    return prefix.rstrip(), True


def _evidence_id(url: str, content_sha256: str) -> str:
    digest = sha256(f'{url}\0{content_sha256}'.encode()).hexdigest()
    return f'web_ev_{digest[:16]}'


def _warning(code: str, message: str, *, url: str | None = None) -> dict[str, Any]:
    warning: dict[str, Any] = {
        'code': code,
        'message': message,
    }
    if url:
        warning['url'] = url
    return warning


def _bounded_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f'{label} must be an integer')
    if value < minimum or value > maximum:
        raise ValueError(f'{label} must be between {minimum} and {maximum}')
    return value


def _dump_search_result_set(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload['warnings'] = _limit_warnings(payload.get('warnings'))
    try:
        return SearchResultSet.model_validate(payload).model_dump(mode='json')
    except ValidationError as exc:
        raise ValueError(f'repair web-search result contract failed: {exc}') from exc


def _dump_web_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload['warnings'] = _limit_warnings(payload.get('warnings'))
    try:
        return RepairWebEvidence.model_validate(payload).model_dump(mode='json')
    except ValidationError as exc:
        raise ValueError(f'repair web-evidence contract failed: {exc}') from exc


def _limit_warnings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in value:
        if hasattr(item, 'model_dump'):
            item = item.model_dump(mode='json')
        if not isinstance(item, Mapping):
            continue
        warning = dict(item)
        queries = warning.get('queries')
        key = (
            warning.get('code'),
            warning.get('message'),
            warning.get('url'),
            tuple(queries) if isinstance(queries, list) else (),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(warning)
    if len(unique) <= _MAX_WARNINGS:
        return unique
    tail_count = 10
    head_count = _MAX_WARNINGS - tail_count - 1
    omitted = len(unique) - head_count - tail_count
    return [
        *unique[:head_count],
        _warning(
            'warning_limit_reached',
            f'{omitted} additional web-evidence warnings were omitted.',
        ),
        *unique[-tail_count:],
    ]


__all__ = [
    'DEFAULT_MAX_PAGE_CHARS',
    'DEFAULT_MAX_SEARCH_RESULTS',
    'DEFAULT_MAX_TOTAL_CHARS',
    'DEFAULT_WEB_SEARCH_TOOL_NAMES',
    'MAX_SELECTED_PAGES',
    'build_unavailable_web_evidence',
    'build_web_evidence',
    'clean_page_content',
    'clean_search_results',
    'prepare_url_fetch',
    'validate_single_search_question',
]
