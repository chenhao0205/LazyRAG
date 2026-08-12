from fastapi import FastAPI
from fastapi.testclient import TestClient

from lazymind.chat.api import knowledge_search_routes as routes
from lazymind.chat.api.knowledge_search_routes import router
from lazymind.chat.service import knowledge_search_service as svc


class Node:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_search_passes_trusted_user_to_lazyllm_payload(monkeypatch):
    captured = {}
    monkeypatch.setattr(svc, '_ensure_kb_search_runtime', lambda: (['retriever'], None, None))

    def fake_search(payload, **kwargs):
        captured['payload'] = payload
        captured['kwargs'] = kwargs
        return [Node(uid='chunk-1', text='fixture text', relevance_score=0.5,
                     global_metadata={'kb_id': 'kb-1', 'docid': 'lazy-doc-1', 'file_name': 'fixture.txt'})]

    monkeypatch.setattr(svc, 'search_kb', fake_search)
    hits = svc.search('user-1', ' query ', ['kb-1'], 3)

    assert captured['payload'] == {'query': 'query', 'filters': {'kb_id': ['kb-1']}, 'user_id': 'user-1'}
    assert captured['kwargs']['k_max'] == 3
    assert hits[0].doc_id == 'lazy-doc-1'


def test_search_requires_user_query_and_kb_ids():
    for args in (('', 'q', ['kb'], 1), ('user', '', ['kb'], 1), ('user', 'q', [], 1)):
        try:
            svc.search(*args)
        except svc.KnowledgeSearchError as error:
            assert error.code == 'INVALID_ARGUMENT'
        else:
            raise AssertionError('expected INVALID_ARGUMENT')


def test_internal_route_enforces_token_and_required_user_id(monkeypatch):
    monkeypatch.setattr(routes, 'expected_internal_token', lambda: 'test-token')
    calls = {}

    def fake_search(**kwargs):
        calls.update(kwargs)
        return [svc.KnowledgeSearchHit(kb_id='kb-1', doc_id='lazy-doc-1', chunk_id='chunk-1', text='text', score=0.7)]

    monkeypatch.setattr(svc, 'search', fake_search)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    assert client.post('/internal/knowledge:search', json={'user_id': 'user-1', 'query': 'q', 'kb_ids': ['kb-1']}).status_code == 401
    assert client.post('/internal/knowledge:search', headers={routes.INTERNAL_TOKEN_HEADER: 'test-token'},
                       json={'query': 'q', 'kb_ids': ['kb-1']}).status_code == 422
    response = client.post('/internal/knowledge:search', headers={routes.INTERNAL_TOKEN_HEADER: 'test-token'},
                           json={'user_id': 'user-1', 'query': 'q', 'kb_ids': ['kb-1'], 'top_k': 2})
    assert response.status_code == 200
    assert calls == {'user_id': 'user-1', 'query': 'q', 'kb_ids': ['kb-1'], 'top_k': 2}
    assert response.json()['hits'][0]['doc_id'] == 'lazy-doc-1'


def test_internal_route_maps_backend_error(monkeypatch):
    monkeypatch.setattr(routes, 'expected_internal_token', lambda: 'test-token')
    monkeypatch.setattr(svc, 'search', lambda **kwargs: (_ for _ in ()).throw(svc.KnowledgeSearchError('BACKEND_UNAVAILABLE', 'down')))
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).post('/internal/knowledge:search', headers={routes.INTERNAL_TOKEN_HEADER: 'test-token'},
                                    json={'user_id': 'user-1', 'query': 'q', 'kb_ids': ['kb-1']})
    assert response.status_code == 503
    assert response.json()['detail']['code'] == 'BACKEND_UNAVAILABLE'
