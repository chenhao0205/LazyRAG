from __future__ import annotations

import hashlib
import importlib
import json
import re
import textwrap
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import (
    parse_qs,
    parse_qsl,
    quote_plus,
    unquote,
    urlencode,
    urljoin,
    urlparse,
    urlsplit,
    urlunsplit,
)

import requests
from bs4 import BeautifulSoup

from .memory import content_ref, write_json


_MAX_URL_CHARS = 2048
_MIN_NEAR_DUPLICATE_CHARS = 200
_NEAR_DUPLICATE_DISTANCE = 3
_NEAR_DUPLICATE_LENGTH_RATIO = 0.85
_TRACKING_QUERY_KEYS = frozenset({
    'dclid', 'fbclid', 'gclid', 'mc_cid', 'mc_eid', 'msclkid',
    'oly_anon_id', 'oly_enc_id', 'vero_conv', 'vero_id',
})
_TRACKING_QUERY_PREFIXES = ('utm_',)
_INLINE_WHITESPACE = re.compile(r'\s+')
_NOISE_ATTRIBUTE = re.compile(
    r'(?:^|[-_\s])(?:advert|breadcrumb|consent|cookie|footer|modal|nav|newsletter|'
    r'promo|share|sidebar|subscribe)(?:$|[-_\s])',
    re.IGNORECASE,
)
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


def search_web(query: str, artifact_root: Path, limit: int = 5, *,
               seen_urls: set[str] | None = None) -> dict[str, Any]:
    """Use the registered web-search providers; snippets are discovery data, not evidence."""
    question = _clean_inline_text(query)
    if not question:
        raise ValueError('web_search_query_empty')
    result_limit = max(1, min(int(limit), 20))
    failures = []
    try:
        providers = _web_search_providers()
    except Exception as exc:
        providers = []
        failures.append(type(exc).__name__)
    raw: object = []
    for provider in providers:
        try:
            candidate = provider.search(question)
        except Exception as exc:
            failures.append(type(exc).__name__)
            continue
        items = candidate if isinstance(candidate, list) else (
            candidate.get('results') if isinstance(candidate, Mapping) else []
        )
        if items:
            raw = candidate
            break
        failures.append(f'{type(provider).__name__}:empty')
    if not raw:
        try:
            raw = _open_search(question, result_limit)
        except Exception as exc:
            failures.append(f'OpenSearch:{type(exc).__name__}')
    if not raw:
        result = {
            'query': question,
            'results': [],
            'status': 'unavailable',
            'failures': failures,
            'duplicate_count': 0,
        }
        _write_web_result(artifact_root / 'web' / 'searches', question, result)
        return result
    items = raw if isinstance(raw, list) else raw.get('results') if isinstance(raw, Mapping) else []
    results = []
    seen = {
        normalized
        for url in seen_urls or ()
        if (normalized := normalize_http_url(url))
    }
    duplicate_count = 0
    for item in items or ():
        if not isinstance(item, Mapping):
            continue
        url = normalize_http_url(item.get('url') or item.get('link'))
        if not url:
            continue
        if url in seen:
            duplicate_count += 1
            continue
        title = _clip_text(_clean_inline_text(item.get('title') or item.get('name')), 300)
        if not title:
            continue
        seen.add(url)
        results.append({
            'title': title,
            'url': url,
            'canonical_url': url,
            'snippet': _clip_text(_clean_inline_text(
                item.get('snippet') or item.get('description') or item.get('content'),
            ), 1000),
        })
        if len(results) >= result_limit:
            break
    result = {
        'query': question,
        'results': results,
        'status': 'completed',
        'duplicate_count': duplicate_count,
    }
    _write_web_result(artifact_root / 'web' / 'searches', question, result)
    return result


def read_web_pages(question: str, urls: Sequence[str], work_root: Path, artifact_root: Path, *,
                   seen_urls: set[str] | None = None,
                   seen_pages: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Fetch selected pages through the existing url_fetch tool and persist readable bodies."""
    prompt = _clean_inline_text(question)
    selected = []
    for value in urls:
        url = normalize_http_url(value)
        if not url:
            raise ValueError('web_read_url_invalid')
        if url not in selected:
            selected.append(url)
        if len(selected) >= 3:
            break
    if not prompt or not selected:
        raise ValueError('web_read_requires_question_and_urls')

    try:
        response = _fetch_pages(selected)
    except Exception as exc:
        pages = [{'status': 'failed', 'title': '', 'url': url, 'excerpt': '', 'content_ref': None,
                  'reason': type(exc).__name__} for url in selected]
        result = {'question': prompt, 'pages': pages, 'duplicate_count': 0}
        _write_web_result(artifact_root / 'web' / 'reads', prompt + json.dumps(selected), result)
        return result
    pages = []
    seen_final = {
        normalized
        for url in seen_urls or ()
        if (normalized := normalize_http_url(url))
    }
    fingerprints = [dict(item) for item in seen_pages if isinstance(item, Mapping)]
    duplicate_count = 0
    for requested, page, error in _fetched_pages(selected, response):
        requested_url = normalize_http_url(requested)
        final_url = normalize_http_url(page.get('final_url') or page.get('url') or requested)
        if not final_url:
            final_url = requested_url
        if final_url in seen_final:
            duplicate_count += 1
            pages.append(_url_duplicate_page(requested_url, final_url, page, error))
            continue
        page = _enhance_page(final_url, page)
        final_url = normalize_http_url(page.get('final_url') or page.get('url') or final_url) or final_url
        if final_url in seen_final:
            duplicate_count += 1
            pages.append(_url_duplicate_page(requested_url, final_url, page, error))
            continue
        seen_final.add(final_url)
        content = clean_page_content(page.get('content'))
        content_type = str(page.get('content_type') or '').casefold()
        status = (
            'failed' if error else
            'unsupported'
            if content_type and not any(
                token in content_type for token in ('html', 'text', 'json', 'xml')
            ) else
            'empty' if not content else
            'readable'
        )
        page_ref = None
        content_sha256 = ''
        content_simhash = ''
        similarity_token_count = 0
        duplicate_of = ''
        duplicate_kind = ''
        if status == 'readable':
            fingerprint = content_fingerprint(content)
            content_sha256 = fingerprint['content_sha256']
            content_simhash = fingerprint['content_simhash']
            similarity_token_count = fingerprint['similarity_token_count']
            duplicate = _find_duplicate_page(fingerprint, fingerprints)
            if duplicate is not None:
                status = 'duplicate'
                duplicate_count += 1
                duplicate_of = str(duplicate.get('url') or '')
                duplicate_kind = str(duplicate.get('kind') or '')
            else:
                fingerprints.append({**fingerprint, 'url': final_url})
        if status in {'readable', 'duplicate'} and content_sha256:
            # Content-addressed names keep every forced revalidation immutable.
            # The same URL may legitimately produce a different body later.
            name = f'page-{_digest(final_url)[:12]}-{content_sha256}.txt'
            work_path = work_root / 'web' / 'pages' / name
            artifact_path = artifact_root / 'web' / 'pages' / name
            work_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            if not work_path.exists():
                work_path.write_text(content, encoding='utf-8')
            if not artifact_path.exists():
                artifact_path.write_text(content, encoding='utf-8')
            page_ref = content_ref(artifact_path, artifact_root)
        pages.append({
            'status': status,
            'title': _clip_text(_clean_inline_text(page.get('title')), 300),
            'requested_url': requested_url,
            'url': final_url,
            'canonical_url': final_url,
            'excerpt': (
                _relevant_excerpt(prompt, content)
                if status == 'readable' or duplicate_kind == 'near' else ''
            ),
            'content_ref': page_ref,
            'content_sha256': content_sha256,
            'content_simhash': content_simhash,
            'character_count': len(content),
            'similarity_token_count': similarity_token_count,
            'duplicate_of': duplicate_of,
            'duplicate_kind': duplicate_kind,
            'reason': error,
        })
    result = {'question': prompt, 'pages': pages, 'duplicate_count': duplicate_count}
    _write_web_result(artifact_root / 'web' / 'reads', prompt + json.dumps(selected), result)
    return result


def _url_duplicate_page(
    requested_url: str,
    final_url: str,
    page: Mapping[str, Any],
    error: str,
) -> dict[str, Any]:
    return {
        'status': 'duplicate',
        'title': _clip_text(_clean_inline_text(page.get('title')), 300),
        'requested_url': requested_url,
        'url': final_url,
        'canonical_url': final_url,
        'excerpt': '',
        'content_ref': None,
        'content_sha256': '',
        'content_simhash': '',
        'character_count': 0,
        'similarity_token_count': 0,
        'duplicate_of': final_url,
        'duplicate_kind': 'url',
        'reason': error,
    }


def _write_web_result(directory: Path, request_key: str, result: Mapping[str, Any]) -> None:
    serialized = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    name = f'{_digest(request_key)[:20]}-{_digest(serialized)[:20]}.json'
    path = directory / name
    if not path.exists():
        write_json(path, result)


def _fetch_pages(urls: list[str]) -> object:
    from lazymind.chat.engine.tools.web_search import url_fetch

    return url_fetch(urls=urls)


def _enhance_page(url: str, page: Mapping[str, Any]) -> Mapping[str, Any]:
    """Replace navigation-heavy HTML extraction with Repair's focused body.

    The shared url_fetch remains the network/security boundary. Repair performs
    a second safe read only for HTML so documentation pages can keep tables and
    code examples that the shared 4k generic extraction often truncates.
    """
    content_type = str(page.get('content_type') or '').casefold()
    if content_type and 'html' not in content_type and 'xhtml' not in content_type:
        return page
    try:
        enhanced = _fetch_repair_page(url)
    except Exception:
        return page
    return enhanced if str(enhanced.get('content') or '').strip() else page


def _fetch_repair_page(url: str) -> dict[str, Any]:
    from lazymind.chat.engine.tools.infra.web_search_support import fetch_public_url

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        ),
    }
    with requests.Session() as session:
        response = fetch_public_url(session, url, timeout=15, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        if redirect := _html_redirect(response.url, soup):
            response = fetch_public_url(session, redirect, timeout=15, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
    return {
        'url': url,
        'final_url': response.url,
        'content_type': str(response.headers.get('Content-Type') or 'text/html'),
        'title': str(soup.title.string if soup.title and soup.title.string else '').strip(),
        'content': _extract_repair_text(soup),
    }


def _html_redirect(base_url: str, soup: BeautifulSoup) -> str:
    refresh = soup.find('meta', attrs={'http-equiv': re.compile(r'^refresh$', re.I)})
    if refresh:
        match = re.search(r'url\s*=\s*[\"\']?([^\"\';]+)', str(refresh.get('content') or ''), re.I)
        if match:
            return urljoin(base_url, match.group(1).strip())
    canonical = soup.find('link', attrs={'rel': lambda value: value and 'canonical' in value})
    href = str(canonical.get('href') or '').strip() if canonical else ''
    return urljoin(base_url, href) if href else ''


def _extract_repair_text(soup: BeautifulSoup, limit: int = 60_000) -> str:
    root = (
        soup.select_one('#main-content')
        or soup.select_one('[role="main"] article')
        or soup.find('article')
        or soup.select_one('.td-content')
        or soup.find('main')
        or soup.body
        or soup
    )
    for tag in root.select('script, style, noscript, template, svg, nav, aside, footer, form, dialog'):
        tag.decompose()
    for tag in tuple(root.find_all(['div', 'section', 'header'])):
        if _is_noise_container(tag):
            tag.decompose()
    blocks = []
    seen_text = set()
    for node in root.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li', 'pre', 'tr']):
        if node.find_parent(['p', 'li', 'pre', 'tr']) is not None:
            continue
        is_code = node.name == 'pre'
        if is_code:
            for br in node.find_all('br'):
                br.replace_with('\n')
            text = clean_page_content(node.get_text('', strip=False))
        else:
            text = _clean_inline_text(node.get_text(' ', strip=True))
            if _is_boilerplate_line(text):
                text = ''
        if not text or (not is_code and text in seen_text):
            continue
        if not is_code:
            seen_text.add(text)
        blocks.append(text)
    if not blocks:
        blocks = [clean_page_content(root.get_text('\n', strip=True))]
    return clean_page_content('\n'.join(blocks))[:limit]


def _is_noise_container(tag: Any) -> bool:
    values = [tag.get('id'), tag.get('role'), tag.get('aria-label')]
    classes = tag.get('class') or ()
    values.extend(classes if isinstance(classes, (list, tuple)) else (classes,))
    identity = ' '.join(str(value) for value in values if value)
    return bool(_NOISE_ATTRIBUTE.search(identity))


def _web_search_providers() -> list[Any]:
    """Load only providers present in this LazyLLM build.

    Repair owns this compatibility boundary because provider availability differs
    between the Evo image and the chat-service image.
    """
    search = importlib.import_module('lazyllm.tools.tools.search')
    providers = []
    for name in ('GoogleSearch', 'BingSearch', 'BochaSearch', 'TavilySearch'):
        provider_type = getattr(search, name, None)
        if provider_type is None:
            continue
        try:
            providers.append(provider_type())
        except Exception:
            continue
    return providers


def _open_search(question: str, limit: int) -> list[dict[str, str]]:
    """Credential-free Repair fallback for deployments where HTML search is blocked."""
    from lazymind.chat.engine.tools.infra.web_search_support import fetch_public_url

    headers = {
        'User-Agent': (
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        ),
    }
    for query in _search_query_variants(question):
        url = f'https://lite.duckduckgo.com/lite/?q={quote_plus(query)}'
        with requests.Session() as session:
            response = fetch_public_url(session, url, timeout=15, headers=headers)
            response.raise_for_status()
        if results := _parse_open_results(response.text, limit):
            return results
    return []


def _search_query_variants(question: str) -> list[str]:
    """Normalize Agent queries locally without extending the shared WebSearch tool."""
    normalized = ' '.join(str(question or '').split())
    if not normalized:
        return []
    concise = ' '.join(normalized.split()[:10])
    return list(dict.fromkeys((concise, normalized)))


def _parse_open_results(html: str, limit: int) -> list[dict[str, str]]:
    results = []
    for link in BeautifulSoup(html, 'html.parser').select('a.result-link'):
        href = str(link.get('href') or '').strip()
        if href.startswith('//'):
            href = f'https:{href}'
        parsed = urlparse(href)
        redirect = parse_qs(parsed.query).get('uddg') if parsed.netloc.endswith('duckduckgo.com') else None
        target = unquote(redirect[0]) if redirect else href
        if urlparse(target).scheme not in {'http', 'https'}:
            continue
        results.append({
            'title': link.get_text(' ', strip=True),
            'url': target,
            'snippet': '',
        })
        if len(results) >= max(1, min(int(limit), 20)):
            break
    return results


def _fetched_pages(urls: list[str], response: object) -> list[tuple[str, Mapping[str, Any], str]]:
    payload = response.get('result') if isinstance(response, Mapping) else None
    if not isinstance(payload, Mapping):
        return [(url, {}, 'url_fetch_invalid_response') for url in urls]
    rows = payload.get('results')
    if not isinstance(rows, list):
        return [
            (url, payload, '') if index == 0 else (url, {}, 'url_fetch_missing_result')
            for index, url in enumerate(urls)
        ]
    result = []
    returned = set()
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        requested = str(item.get('url') or '').strip()
        if normalized := normalize_http_url(requested):
            returned.add(normalized)
        page = item.get('result') if isinstance(item.get('result'), Mapping) else {}
        error = '' if item.get('success') is True else str(item.get('error') or 'url_fetch_failed')
        result.append((requested, page, error))
    result.extend(
        (url, {}, 'url_fetch_missing_result')
        for url in urls
        if normalize_http_url(url) not in returned
    )
    return result


def normalize_http_url(value: object) -> str:
    """Return a stable, public HTTP(S) URL for deduplication and fetching."""
    if not isinstance(value, str):
        return ''
    text = value.strip()
    if not text or len(text) > _MAX_URL_CHARS or any(char.isspace() for char in text):
        return ''
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError:
        return ''
    scheme = parsed.scheme.lower()
    if scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        return ''
    try:
        hostname = parsed.hostname.rstrip('.').encode('idna').decode('ascii').lower()
    except UnicodeError:
        return ''
    if not hostname or hostname == 'localhost' or hostname.endswith(('.localhost', '.local')):
        return ''
    try:
        if not ip_address(hostname).is_global:
            return ''
    except ValueError:
        pass
    host = f'[{hostname}]' if ':' in hostname and not hostname.startswith('[') else hostname
    default_port = (scheme == 'http' and port == 80) or (scheme == 'https' and port == 443)
    netloc = host if port is None or default_port else f'{host}:{port}'
    query = urlencode([
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_query_key(key)
    ], doseq=True)
    return urlunsplit((scheme, netloc, parsed.path or '/', query, ''))


def clean_page_content(value: object) -> str:
    """Remove text-level boilerplate without rewriting code or paragraph content."""
    if not isinstance(value, str):
        return ''
    normalized = unicodedata.normalize('NFC', value.replace('\u00a0', ' '))
    normalized = ''.join(
        char if char in '\n\t' or unicodedata.category(char) != 'Cc' else ' '
        for char in normalized
    )
    normalized = textwrap.dedent(normalized.replace('\r\n', '\n').replace('\r', '\n'))
    lines = []
    for raw_line in normalized.splitlines():
        stripped = raw_line.strip()
        if not stripped or _is_boilerplate_line(stripped):
            continue
        lines.append(raw_line.rstrip() if raw_line[:1].isspace() else stripped)
    return '\n'.join(lines)


def content_fingerprint(content: str) -> dict[str, Any]:
    """Build compact exact and near-duplicate fingerprints using only stdlib."""
    clean = clean_page_content(content)
    similarity_text = unicodedata.normalize('NFKC', clean).casefold()
    tokens = re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', similarity_text)
    shingles = (
        ['\0'.join(tokens[index:index + 3]) for index in range(len(tokens) - 2)]
        if len(tokens) >= 3 else tokens
    )
    vector = [0] * 64
    for shingle, weight in Counter(shingles).items():
        digest = int.from_bytes(hashlib.blake2b(shingle.encode('utf-8'), digest_size=8).digest(), 'big')
        for bit in range(64):
            vector[bit] += weight if digest & (1 << bit) else -weight
    simhash = sum(1 << bit for bit, score in enumerate(vector) if score >= 0)
    return {
        'content_sha256': hashlib.sha256(clean.encode('utf-8')).hexdigest(),
        'content_simhash': f'{simhash:016x}',
        'character_count': len(clean),
        'similarity_token_count': len(tokens),
    }


def _find_duplicate_page(current: Mapping[str, Any],
                         previous: Sequence[Mapping[str, Any]]) -> dict[str, str] | None:
    for item in previous:
        if current.get('content_sha256') == item.get('content_sha256'):
            return {'kind': 'exact', 'url': str(item.get('url') or '')}
    current_size = int(current.get('character_count') or 0)
    current_tokens = int(current.get('similarity_token_count') or 0)
    current_simhash = _simhash_value(current.get('content_simhash'))
    if current_size < _MIN_NEAR_DUPLICATE_CHARS or current_tokens < 20 or current_simhash is None:
        return None
    for item in previous:
        other_size = int(item.get('character_count') or 0)
        other_tokens = int(item.get('similarity_token_count') or 0)
        other_simhash = _simhash_value(item.get('content_simhash'))
        if other_size < _MIN_NEAR_DUPLICATE_CHARS or other_tokens < 20 or other_simhash is None:
            continue
        length_ratio = min(current_size, other_size) / max(current_size, other_size)
        if length_ratio < _NEAR_DUPLICATE_LENGTH_RATIO:
            continue
        if (current_simhash ^ other_simhash).bit_count() <= _NEAR_DUPLICATE_DISTANCE:
            return {'kind': 'near', 'url': str(item.get('url') or '')}
    return None


def _simhash_value(value: object) -> int | None:
    try:
        return int(str(value), 16)
    except (TypeError, ValueError):
        return None


def _is_tracking_query_key(value: str) -> bool:
    key = str(value).casefold()
    return key in _TRACKING_QUERY_KEYS or key.startswith(_TRACKING_QUERY_PREFIXES)


def _clean_inline_text(value: object) -> str:
    if not isinstance(value, str):
        return ''
    normalized = unicodedata.normalize('NFC', value.replace('\u00a0', ' '))
    normalized = ''.join(
        char if unicodedata.category(char) != 'Cc' else ' '
        for char in normalized
    )
    return _INLINE_WHITESPACE.sub(' ', normalized).strip()


def _clip_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit - 1].rstrip() + '…'


def _is_boilerplate_line(line: str) -> bool:
    if len(line) > 240:
        return False
    stripped = line.strip(' \t|·•—–-:：.。!！')
    return any(pattern.fullmatch(stripped) for pattern in _BOILERPLATE_LINES)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _relevant_excerpt(question: str, content: str, limit: int = 1200) -> str:
    paragraphs = [line.strip() for line in content.splitlines() if line.strip()]
    if not paragraphs:
        return ''
    english = set(re.findall(r'[a-z0-9_]{2,}', question.casefold()))
    chinese = set(re.findall(r'[\u4e00-\u9fff]', question))
    ranked = sorted(
        enumerate(paragraphs),
        key=lambda item: (
            -sum(item[1].casefold().count(term) for term in english)
            - sum(char in item[1] for char in chinese),
            item[0],
        ),
    )
    selected = sorted(ranked[:3])
    return '\n'.join(paragraph for _, paragraph in selected)[:limit]
