import json
import os
import hashlib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

_DOCUMENTS: dict[tuple[str, ...], Any] = {}
DOCS_PAGE_SIZE = 100
CHUNK_PAGE_SIZE = 200


class KnowledgeBaseClient:
    """Read document indexes and DocNode batches from a knowledge base."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        http_get_json: Callable[[str], dict[str, Any]] | None = None,
        document: Any | None = None,
        document_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self._http_get_json = http_get_json
        self._document = document
        self._document_factory = document_factory

    def list_documents(self, kb_id: str) -> list[dict[str, Any]]:
        return self._list_documents_from_doc_server(kb_id)

    def count_valid_chunks(
        self,
        kb_id: str,
        doc_ids: list[str],
        groups: list[str],
        allowed_types: list[str],
        max_scan_chunks: int,
    ) -> dict[str, Any]:
        capacities = {group: {doc_id: 0 for doc_id in doc_ids} for group in groups}
        filtered: Counter[str] = Counter()
        invalid: Counter[str] = Counter()
        scanned = 0

        for doc_id, group, batch in self._iter_raw_chunks(kb_id, doc_ids, groups):
            scanned += len(batch)
            if scanned > max_scan_chunks:
                raise ValueError(f'max_scan_chunks exceeded: {scanned} > {max_scan_chunks}')
            embedding_candidates = []
            for node in batch:
                reason = _content_ineligible_reason(node, allowed_types)
                if reason.startswith('filtered_type:'):
                    filtered[reason.partition(':')[2]] += 1
                elif reason:
                    invalid[reason] += 1
                else:
                    embedding_candidates.append(node)
            self._try_attach_stored_embeddings(
                self._get_document(), embedding_candidates, kb_id, doc_id, group,
            )
            for node in embedding_candidates:
                reason = _embedding_ineligible_reason(node)
                if reason:
                    invalid[reason] += 1
                else:
                    capacities[group][doc_id] += 1

        effective = sum(sum(items.values()) for items in capacities.values())
        return {
            'scanned_count': scanned,
            'effective_count': effective,
            'capacities': capacities,
            'filtered_count_by_type': dict(filtered),
            'invalid_count_by_reason': dict(invalid),
        }

    def fetch_valid_chunks(
        self,
        kb_id: str,
        doc_id: str,
        group: str,
        allowed_types: list[str],
        limit: int,
        *,
        order_by: str,
    ) -> list[Any]:
        if order_by != 'stable_chunk_id_hash':
            raise ValueError('order_by must be stable_chunk_id_hash')
        if limit <= 0:
            return []

        nodes = []
        document = self._get_document()
        for _, _, batch in self._iter_raw_chunks(kb_id, [doc_id], [group]):
            candidates = [node for node in batch if not _content_ineligible_reason(node, allowed_types)]
            self._try_attach_stored_embeddings(document, candidates, kb_id, doc_id, group)
            nodes.extend(node for node in candidates if not _embedding_ineligible_reason(node))
        nodes.sort(key=lambda node: hashlib.sha256(_node_uid(node).encode()).hexdigest())
        return nodes[:limit]

    def iter_chunks(
        self,
        kb_id: str,
        doc_ids: list[str] | None,
        groups: list[str],
        page_size: int,
        *,
        require_embeddings: bool = True,
    ) -> Iterator[list[Any]]:
        if not groups:
            raise ValueError('groups is required')
        if page_size <= 0:
            raise ValueError('page_size must be positive')

        resolved_doc_ids = [doc['doc_id'] for doc in self.list_documents(kb_id)] if doc_ids is None else doc_ids
        if not resolved_doc_ids:
            return

        document = self._get_document()
        for doc_id in resolved_doc_ids:
            for group in groups:
                offset = 0
                while True:
                    try:
                        nodes, total = document.get_nodes(
                            doc_ids=[doc_id],
                            group=group,
                            kb_id=kb_id,
                            limit=page_size,
                            offset=offset,
                            return_total=True,
                            sort_by_number=True,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f'failed to read chunks: kb_id={kb_id} doc_id={doc_id} group={group}'
                        ) from exc
                    batch = list(nodes or [])
                    if not batch:
                        break
                    if require_embeddings:
                        self._attach_stored_embeddings(
                            document, batch, kb_id=kb_id, doc_id=doc_id, group=group,
                        )
                        self._require_embeddings(batch, kb_id=kb_id, doc_id=doc_id, group=group)
                    yield batch
                    offset += len(batch)
                    if offset >= int(total or offset):
                        break

    def _iter_raw_chunks(
        self,
        kb_id: str,
        doc_ids: list[str],
        groups: list[str],
    ) -> Iterator[tuple[str, str, list[Any]]]:
        document = self._get_document()
        for doc_id in doc_ids:
            for group in groups:
                offset = 0
                while True:
                    try:
                        nodes, total = document.get_nodes(
                            doc_ids=[doc_id],
                            group=group,
                            kb_id=kb_id,
                            limit=CHUNK_PAGE_SIZE,
                            offset=offset,
                            return_total=True,
                            sort_by_number=True,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f'failed to read chunks: kb_id={kb_id} doc_id={doc_id} group={group}'
                        ) from exc
                    batch = list(nodes or [])
                    if not batch:
                        break
                    yield doc_id, group, batch
                    offset += len(batch)
                    if offset >= int(total or offset):
                        break

    def _get_document(self) -> Any:
        if self._document is not None:
            return self._document
        if self._document_factory is not None:
            self._document = self._document_factory()
            return self._document

        self._document = _build_document()
        return self._document

    @staticmethod
    def _require_embeddings(nodes: list[Any], *, kb_id: str, doc_id: str, group: str) -> None:
        missing = [
            str(getattr(node, 'uid', '') or getattr(node, '_uid', '') or '')
            for node in nodes
            if not _has_embedding(node)
        ]
        if missing:
            raise RuntimeError(
                f'chunk embeddings are unavailable in Milvus: kb_id={kb_id} doc_id={doc_id} '
                f'group={group} chunk_ids={missing}'
            )

    @staticmethod
    def _attach_stored_embeddings(
        document: Any,
        nodes: list[Any],
        *,
        kb_id: str,
        doc_id: str,
        group: str,
    ) -> None:
        '''Read vectors explicitly because the LazyLLM UID lookup omits vector output fields.

        The installed LazyLLM version passes `output_fields=None` for UID lookups.
        Milvus then returns only the UID even though `embedding_embed_main` exists.
        '''
        missing = [node for node in nodes if not _has_embedding(node)]
        if not missing:
            return

        try:
            store, vector_store = _milvus_store(document)
            uids = [str(getattr(node, 'uid', '') or getattr(node, '_uid', '') or '') for node in missing]
            uids = [uid for uid in uids if uid]
            if not uids:
                return
            collection = store._gen_collection_name(group)
            with vector_store._client_context() as client:
                if not client.has_collection(collection):
                    return
                client.load_collection(collection)
                fields = [
                    field.get('name')
                    for field in client.describe_collection(collection_name=collection).get('fields', [])
                    if str(field.get('name') or '').startswith('embedding_')
                ]
                if not fields:
                    return
                rows = client.query(
                    collection_name=collection,
                    filter=f'uid in {uids!r}',
                    output_fields=['uid', *fields],
                )
            embeddings = {
                str(row.get('uid') or ''): {
                    key.removeprefix('embedding_'): list(value)
                    for key, value in row.items()
                    if key.startswith('embedding_') and value is not None
                }
                for row in rows
            }
            for node in missing:
                uid = str(getattr(node, 'uid', '') or getattr(node, '_uid', '') or '')
                if embedding := embeddings.get(uid):
                    node.embedding = embedding
        except Exception as exc:
            raise RuntimeError(
                f'failed to read stored embeddings: kb_id={kb_id} doc_id={doc_id} group={group}'
            ) from exc

    @classmethod
    def _try_attach_stored_embeddings(
        cls,
        document: Any,
        nodes: list[Any],
        kb_id: str,
        doc_id: str,
        group: str,
    ) -> None:
        try:
            cls._attach_stored_embeddings(document, nodes, kb_id=kb_id, doc_id=doc_id, group=group)
        except RuntimeError:
            return

    def _list_documents_from_doc_server(self, kb_id: str) -> list[dict[str, Any]]:
        docs: list[dict[str, Any]] = []
        page = 1
        while True:
            data = self._get_docs_page(kb_id, page)
            items = data.get('items') or []
            docs.extend(doc for item in items if (doc := self._document_row(item)))

            page_size = _int(data.get('page_size')) or DOCS_PAGE_SIZE
            total = _int(data.get('total'))
            if not items or len(docs) >= total or len(items) < page_size:
                break
            page += 1
        return docs

    def _get_docs_page(self, kb_id: str, page: int) -> dict[str, Any]:
        query = urlencode({
            'kb_id': kb_id,
            'include_deleted_or_canceled': 'false',
            'page': page,
            'page_size': DOCS_PAGE_SIZE,
        })
        payload = self._get_json(f'{self._doc_server_base_url()}/v1/docs?{query}')
        if _int(payload.get('code')) != 200:
            raise RuntimeError(f'doc server /v1/docs failed: {payload.get("msg") or payload}')
        return payload['data']

    def _get_json(self, url: str) -> dict[str, Any]:
        if self._http_get_json is not None:
            return self._http_get_json(url)
        request = Request(url, headers={'Accept': 'application/json'})
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode('utf-8'))

    def _doc_server_base_url(self) -> str:
        if self.base_url:
            return self.base_url.rstrip('/')
        value = os.getenv('LAZYMIND_EVO_KB_BASE_URL', '').strip()
        if not value:
            raise ValueError('LAZYMIND_EVO_KB_BASE_URL is required')
        return value.rstrip('/')

    def _document_row(self, item: dict[str, Any]) -> dict[str, Any] | None:
        doc = item.get('doc') or {}
        relation = item.get('relation') or {}
        snapshot = item.get('snapshot') or {}
        get = doc.get
        doc_id = str(get('doc_id') or '')
        if not doc_id:
            return None
        return {
            'doc_id': doc_id,
            'filename': str(get('filename') or get('display_name') or doc_id),
            'file_type': str(get('file_type') or ''),
            'path': str(get('path') or ''),
            'upload_status': get('upload_status', ''),
            'status': str(snapshot.get('status') or get('status') or ''),
            'row': {'doc': dict(doc), 'relation': dict(relation), 'snapshot': dict(snapshot)},
        }


def _build_document() -> Any:
    from lazymind.config import config
    from lazymind.parsing.service.build_document import build_document

    algo_id = str(config['algo_id'] or config['agentic_kb_name'] or '').strip()
    if not algo_id:
        raise ValueError('algo_id is required')
    key = ('local', algo_id)
    if key not in _DOCUMENTS:
        _DOCUMENTS[key] = build_document(algo_id, serve=False)
    return _DOCUMENTS[key]


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _has_embedding(node: Any) -> bool:
    value = getattr(node, 'embedding', None)
    return isinstance(value, Mapping) and any(bool(vector) for vector in value.values())


def _node_uid(node: Any) -> str:
    return str(getattr(node, 'uid', '') or getattr(node, '_uid', '') or '')


def _content_ineligible_reason(node: Any, allowed_types: list[str]) -> str:
    metadata = getattr(node, 'metadata', {}) or {}
    node_type = str(metadata.get('type') or metadata.get('node_type') or 'unknown').strip().lower()
    if node_type not in allowed_types:
        return f'filtered_type:{node_type}'
    if not str(getattr(node, 'text', '') or '').strip():
        return 'empty_text'
    return ''


def _embedding_ineligible_reason(node: Any) -> str:
    if not _has_embedding(node):
        return 'missing_embedding'
    try:
        from .models import normalize_embedding
        normalize_embedding(getattr(node, 'embedding', None))
    except ValueError:
        return 'invalid_embedding'
    return ''


def _milvus_store(document: Any) -> tuple[Any, Any]:
    impl = getattr(document, '_impl', None)
    store = getattr(impl, 'store', None)
    store_impl = getattr(store, 'vector_initialized_impl', None)
    vector_store = getattr(store_impl, 'vector_store', None)
    if store is None or vector_store is None:
        raise RuntimeError('Milvus vector store is unavailable')
    return store, vector_store
