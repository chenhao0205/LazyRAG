from __future__ import annotations

import json

from bs4 import BeautifulSoup

from evo.operations.repair import web


def test_normalize_http_url_removes_tracking_and_canonicalizes_origin() -> None:
    assert web.normalize_http_url(
        ' HTTPS://Example.COM:443/docs?id=7&utm_source=test&fbclid=secret#overview '
    ) == 'https://example.com/docs?id=7'
    assert web.normalize_http_url('http://127.0.0.1/private') == ''
    assert web.normalize_http_url('https://user:password@example.com/') == ''


def test_search_web_deduplicates_current_and_previous_results(tmp_path, monkeypatch) -> None:
    class Provider:
        def search(self, query):
            assert query == 'repair retry behavior'
            return [
                {
                    'title': 'Already seen',
                    'url': 'https://example.com/docs?utm_source=new#part',
                    'snippet': 'duplicate from a prior search',
                },
                {
                    'title': 'Fresh result',
                    'url': 'HTTPS://EXAMPLE.ORG:443/guide#top',
                    'snippet': '  useful   result  ',
                },
                {
                    'title': 'Same fresh result',
                    'url': 'https://example.org/guide#other',
                    'snippet': 'duplicate in this response',
                },
            ]

    monkeypatch.setattr(web, '_web_search_providers', lambda: [Provider()])
    result = web.search_web(
        ' repair   retry behavior ',
        tmp_path,
        seen_urls={'https://example.com/docs'},
    )

    assert result['results'] == [{
        'title': 'Fresh result',
        'url': 'https://example.org/guide',
        'canonical_url': 'https://example.org/guide',
        'snippet': 'useful result',
    }]
    assert result['duplicate_count'] == 2
    persisted = json.loads(next((tmp_path / 'web' / 'searches').iterdir()).read_text())
    assert persisted == result


def test_extract_repair_text_combines_dom_and_text_noise_filtering() -> None:
    soup = BeautifulSoup(
        '''
        <html><body>
          <nav>Navigation menu</nav>
          <main>
            <div class="cookie-banner">Accept all cookies</div>
            <header class="article-header"><h1>Retry guide</h1></header>
            <p>The retry count must be non-negative.</p>
            <section id="related-work"><h2>Related Work</h2><p>Prior repair research.</p></section>
            <p>Privacy Policy</p>
            <pre>print("same")\nprint("same")</pre>
            <section id="newsletter-promo">Unrelated recommendation</section>
          </main>
        </body></html>
        ''',
        'html.parser',
    )

    content = web._extract_repair_text(soup)

    assert content == (
        'Retry guide\n'
        'The retry count must be non-negative.\n'
        'Related Work\n'
        'Prior repair research.\n'
        'print("same")\n'
        'print("same")'
    )


def test_extract_repair_text_keeps_legitimate_login_documentation() -> None:
    soup = BeautifulSoup(
        '''
        <main>
          <section id="login">
            <h2>Login API</h2>
            <p>Send the access token in the Authorization header.</p>
          </section>
        </main>
        ''',
        'html.parser',
    )

    assert web._extract_repair_text(soup) == (
        'Login API\nSend the access token in the Authorization header.'
    )


def test_read_web_pages_keeps_near_duplicate_as_cold_artifact(tmp_path, monkeypatch) -> None:
    base = (
        'The repair must preserve the verified source hash, keep tenant filters, '
        'and return deterministic evidence for every validation command. '
    ) * 10
    changed = base.replace('every validation command', 'each validation command', 1)
    prior = web.content_fingerprint(base)

    def fetch_pages(urls):
        return {
            'result': {
                'results': [{
                    'url': urls[0],
                    'success': True,
                    'result': {
                        'url': urls[0],
                        'final_url': urls[0],
                        'content_type': 'text/plain',
                        'title': 'Copied guide',
                        'content': changed,
                    },
                }],
            },
        }

    monkeypatch.setattr(web, '_fetch_pages', fetch_pages)
    result = web.read_web_pages(
        'How should validation behave?',
        ['https://mirror.example.org/guide'],
        tmp_path / 'work',
        tmp_path / 'artifacts',
        seen_pages=[{**prior, 'url': 'https://docs.example.com/guide'}],
    )

    assert result['duplicate_count'] == 1
    assert result['pages'][0]['status'] == 'duplicate'
    assert result['pages'][0]['duplicate_kind'] == 'near'
    assert result['pages'][0]['duplicate_of'] == 'https://docs.example.com/guide'
    ref = result['pages'][0]['content_ref']
    assert ref['sha256'] == result['pages'][0]['content_sha256']
    assert result['pages'][0]['excerpt']


def test_lightweight_near_duplicate_keeps_distinct_documents() -> None:
    first = web.content_fingerprint(
        ('Repair validation checks source hashes, tenant scope, and command output. ' * 10)
    )
    second = web.content_fingerprint(
        ('Web search documentation explains provider credentials and HTTP redirects. ' * 10)
    )

    assert web._find_duplicate_page(
        second,
        [{**first, 'url': 'https://example.com/first'}],
    ) is None


def test_read_web_pages_persists_distinct_clean_content(tmp_path, monkeypatch) -> None:
    def fetch_pages(urls):
        return {
            'result': {
                'results': [{
                    'url': urls[0],
                    'success': True,
                    'result': {
                        'url': urls[0],
                        'final_url': f'{urls[0]}?utm_campaign=test#result',
                        'content_type': 'text/plain',
                        'title': 'Guide',
                        'content': 'On this page\nThe command must return the expected result.',
                    },
                }],
            },
        }

    monkeypatch.setattr(web, '_fetch_pages', fetch_pages)
    result = web.read_web_pages(
        'What must the command return?',
        ['https://docs.example.com/guide'],
        tmp_path / 'work',
        tmp_path / 'artifacts',
    )

    page = result['pages'][0]
    assert page['status'] == 'readable'
    assert page['url'] == 'https://docs.example.com/guide'
    assert page['character_count'] == len('The command must return the expected result.')
    assert page['content_sha256']
    assert page['content_simhash']
    assert page['content_ref']['sha256'] == page['content_sha256']


def test_force_read_same_url_keeps_both_artifact_versions(tmp_path, monkeypatch) -> None:
    bodies = iter(('The documented value is one.', 'The documented value is two.'))

    def fetch_pages(urls):
        return {
            'result': {
                'results': [{
                    'url': urls[0],
                    'success': True,
                    'result': {
                        'url': urls[0],
                        'final_url': urls[0],
                        'content_type': 'text/plain',
                        'title': 'Versioned guide',
                        'content': next(bodies),
                    },
                }],
            },
        }

    monkeypatch.setattr(web, '_fetch_pages', fetch_pages)
    arguments = (
        'What value is documented?',
        ['https://docs.example.com/versioned'],
        tmp_path / 'work',
        tmp_path / 'artifacts',
    )

    first = web.read_web_pages(*arguments)
    second = web.read_web_pages(*arguments)

    first_ref = first['pages'][0]['content_ref']
    second_ref = second['pages'][0]['content_ref']
    assert first_ref['uri'] != second_ref['uri']
    assert first_ref['sha256'] != second_ref['sha256']
    assert {path.read_text() for path in (tmp_path / 'artifacts' / 'web' / 'pages').iterdir()} == {
        'The documented value is one.',
        'The documented value is two.',
    }


def test_redirect_to_seen_url_records_duplicate_alias(tmp_path, monkeypatch) -> None:
    def fetch_pages(urls):
        return {
            'result': {
                'results': [{
                    'url': urls[0],
                    'success': True,
                    'result': {
                        'url': urls[0],
                        'final_url': 'https://docs.example.com/canonical',
                        'content_type': 'text/plain',
                        'title': 'Canonical guide',
                        'content': 'Already known.',
                    },
                }],
            },
        }

    monkeypatch.setattr(web, '_fetch_pages', fetch_pages)
    result = web.read_web_pages(
        'Read the guide',
        ['https://mirror.example.com/guide'],
        tmp_path / 'work',
        tmp_path / 'artifacts',
        seen_urls={'https://docs.example.com/canonical'},
    )

    assert result['duplicate_count'] == 1
    assert result['pages'][0]['status'] == 'duplicate'
    assert result['pages'][0]['requested_url'] == 'https://mirror.example.com/guide'
    assert result['pages'][0]['canonical_url'] == 'https://docs.example.com/canonical'
    assert result['pages'][0]['duplicate_kind'] == 'url'
