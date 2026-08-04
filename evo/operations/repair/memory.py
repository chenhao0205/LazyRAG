from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import deque
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evo.repair_guidance import (
    active_guidance,
    guidance_snapshot,
    is_append_guidance_successor,
)

from .memory_projection import build_working_memory, investigation_key
from .source import ALGORITHM_APP, copy_source, source_hash, source_root


_INDEXED_EVENTS = frozenset({
    'opencode.result',
    'command.result',
    'web.search',
    'web.read',
    'http.result',
    'artifact.read',
    'investigation.reused',
    'phase1.finished',
})
_MAX_HISTORY_ATTEMPTS = 64


class WorkMemory:
    """Append-only Phase-1 history backed by a disposable working directory."""

    def __init__(
        self,
        *,
        target: Mapping[str, Any],
        guidance: Sequence[str],
        scope: Mapping[str, Any],
        work_root: Path,
        artifact_root: Path,
        previous_attempts: tuple[Path, ...],
        source_digest: str,
        history_attempts: tuple[Path, ...] | None = None,
        guidance_state: Mapping[str, Any] | None = None,
        recovery: Mapping[str, Any] | None = None,
    ) -> None:
        self.target = dict(target)
        policy_view = {
            'user_guidance': [str(item).strip() for item in guidance if str(item).strip()],
            **({'guidance_state': dict(guidance_state)} if guidance_state else {}),
        }
        self.guidance_state = guidance_snapshot(policy_view)
        self.guidance = list(active_guidance(policy_view))
        self.scope = dict(scope)
        self.work_root = work_root
        self.artifact_root = artifact_root
        self.previous_attempts = previous_attempts
        self.history_attempts = history_attempts if history_attempts is not None else previous_attempts
        self.source_digest = source_digest
        self.recovery = {
            'mode': 'same_revision_resume' if previous_attempts else 'fresh',
            'source_attempt': previous_attempts[-1].name if previous_attempts else '',
            'restore_workspace': bool(previous_attempts),
            'session_resume_allowed': bool(previous_attempts),
            **dict(recovery or {}),
        }
        self._records, self._indexed_records, self._web_records = _load_memory_records(
            (*previous_attempts, artifact_root),
            (*self.history_attempts, *previous_attempts, artifact_root),
        )
        self._sequence = len(_load_journal(artifact_root))
        self._workspace_digest: str | None = None
        self._core_refs = self._write_core_artifacts()

    @classmethod
    def create(
        cls,
        run_id: str,
        target: Mapping[str, Any],
        policy: Mapping[str, Any],
        source_dir: Path,
        expected_source_hash: str,
        scope: Mapping[str, Any],
    ) -> WorkMemory:
        source = source_root(source_dir)
        if not (source / ALGORITHM_APP).is_file():
            raise ValueError('candidate_source_invalid')
        if source_hash(source) != expected_source_hash:
            raise ValueError('source_hash_mismatch')

        base = Path(
            str(policy.get('phase1_artifact_dir') or '')
            or Path(os.getenv('LAZYMIND_EVO_BASE_DIR') or '/var/lib/lazymind/evo')
            / 'artifacts' / 'repair' / 'phase1'
        ).resolve()
        category_id = _safe_segment(str(target.get('category_id') or ''))
        parent = base / _safe_segment(run_id) / category_id / _target_hash(target)
        parent.mkdir(parents=True, exist_ok=True)
        attempts = tuple(sorted(
            parent.glob('attempt-*'),
            key=lambda item: (item.stat().st_mtime_ns, item.name),
        ))
        restorable = tuple(path for path in attempts if (path / 'checkpoint.complete').is_file())
        current_guidance = guidance_snapshot(policy)
        recovery = _select_recovery(restorable, current_guidance, expected_source_hash)
        source_attempt = recovery.get('source_attempt_path')
        previous = (source_attempt,) if isinstance(source_attempt, Path) else ()
        artifact_root = Path(tempfile.mkdtemp(prefix='attempt-', dir=parent)).resolve()
        revision_id = str(current_guidance.get('revision_id') or 'legacy')
        workspace_key = hashlib.sha256(
            f'{parent}:{revision_id}'.encode('utf-8')
        ).hexdigest()[:20]
        work_root = (Path('/tmp') / f'lazyrag-repair-phase1-{workspace_key}').resolve()
        try:
            # An OpenCode session retains its cwd. Recreate the same path for
            # every attempt of one run/category/target, then restore checkpoint.
            shutil.rmtree(work_root, ignore_errors=True)
            for name in ('source', 'work', 'memory', 'logs'):
                (work_root / name).mkdir(parents=True, exist_ok=True)
            for name in ('events', 'runs', 'web', 'opencode/calls', 'opencode/reports', 'checkpoint'):
                (artifact_root / name).mkdir(parents=True, exist_ok=True)
            copy_source(source, work_root / 'source')
            if previous and recovery.get('restore_workspace'):
                saved_work = previous[-1] / 'checkpoint' / 'work'
                if saved_work.is_dir():
                    shutil.copytree(saved_work, work_root / 'work', dirs_exist_ok=True)
            metadata = {
                'source_dir': str(source),
                'source_hash': expected_source_hash,
                'category_id': category_id,
                'target_hash': _target_hash(target),
                'resumed_from': previous[-1].name if previous else '',
                'guidance_state': current_guidance,
                'recovery': _public_recovery(recovery),
            }
            write_json(artifact_root / 'metadata.json', metadata)
            write_json(work_root / 'memory' / 'metadata.json', metadata)
            return cls(
                target=target,
                guidance=active_guidance(policy),
                guidance_state=current_guidance,
                scope=scope,
                work_root=work_root,
                artifact_root=artifact_root,
                previous_attempts=previous,
                history_attempts=attempts[-_MAX_HISTORY_ATTEMPTS:],
                source_digest=expected_source_hash,
                recovery=_public_recovery(recovery),
            )
        except Exception:
            cleanup_workdir(work_root)
            shutil.rmtree(artifact_root, ignore_errors=True)
            raise

    @property
    def restored_session(self) -> dict[str, Any]:
        if not self.previous_attempts or not self.recovery.get('session_resume_allowed'):
            return {}
        attempt = self.previous_attempts[-1]
        return _verified_checkpoint_session(
            attempt,
            _read_json(attempt / 'checkpoint.complete'),
            guidance_revision_id=self.guidance_revision_id,
            active_content_hash=self.active_content_hash,
            workspace_sha256=self.workspace_digest(),
        )

    def record(self, event: str, summary: str, data: Mapping[str, Any]) -> dict[str, str]:
        self._sequence += 1
        event_path = self.artifact_root / 'events' / f'{self._sequence:04d}-{_safe_segment(event)}.json'
        payload = {
            'time': datetime.now(timezone.utc).isoformat(),
            'event': event,
            'summary': str(summary).strip()[:2000],
            'data': dict(data),
            'provenance': self.guidance_provenance(),
        }
        write_json(event_path, payload)
        ref = content_ref(event_path, self.artifact_root)
        journal_record = {
            'time': payload['time'],
            'event': event,
            'summary': payload['summary'],
            'file': event_path.relative_to(self.artifact_root).as_posix(),
            'ref': ref,
        }
        with (self.artifact_root / 'journal.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(journal_record, ensure_ascii=False, sort_keys=True) + '\n')
        memory_record = {**payload, '_event_ref': ref}
        self._records.append(memory_record)
        if event in _INDEXED_EVENTS:
            self._indexed_records.append(memory_record)
        if event in {'web.search', 'web.read'}:
            self._web_records.append(memory_record)
        return ref

    def context(self, counters: Mapping[str, int], budget: Mapping[str, int]) -> dict[str, Any]:
        work_files = [
            path.relative_to(self.work_root).as_posix()
            for path in sorted((self.work_root / 'work').rglob('*'))
            if path.is_file()
        ][-80:]
        return build_working_memory(
            target=self.target,
            guidance=self.guidance,
            scope=self.scope,
            indexed_records=self._indexed_records,
            recent_records=self._records,
            workspace={
                'source': 'source/ (read-only projection)',
                'work': 'work/ (all experiments and Demo files)',
                'workspace_sha256': self.workspace_digest(),
                'recovery': dict(self.recovery),
            },
            work_files=work_files,
            counters=counters,
            budget=budget,
            core_refs=self._core_refs,
            web_investigation=self.web_investigation(),
            guidance_state=self.guidance_state,
        )

    def write_context(self, counters: Mapping[str, int], budget: Mapping[str, int]) -> Path:
        path = self.work_root / 'memory' / 'context.json'
        write_json(path, self.context(counters, budget))
        return path

    def checkpoint(self, session_id: str, calls: int) -> dict[str, str]:
        destination = self.artifact_root / 'checkpoint' / 'work'
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(self.work_root / 'work', destination)
        workspace_sha256 = self.workspace_digest()
        session_path = self.artifact_root / 'checkpoint' / 'session.json'
        write_json(
            session_path,
            {
                'session_id': str(session_id),
                'calls': int(calls),
                'guidance_revision_id': self.guidance_revision_id,
                'active_content_hash': self.active_content_hash,
                'workspace_sha256': workspace_sha256,
            },
        )
        write_json(self.artifact_root / 'checkpoint.complete', {
            'completed': True,
            'guidance_revision_id': self.guidance_revision_id,
            'active_content_hash': self.active_content_hash,
            'workspace_sha256': workspace_sha256,
            'session_sha256': _file_sha256(session_path),
        })
        return directory_ref(destination, self.artifact_root)

    def evidence_refs(self) -> list[dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        workspace_sha256 = self.workspace_digest(refresh=True)
        latest: dict[tuple[str, str], Mapping[str, Any]] = {}
        for record in self._indexed_records:
            event = str(record.get('event') or '')
            if event not in {'command.result', 'http.result'}:
                continue
            if self.record_applicability(record) != 'current':
                continue
            data = record.get('data') if isinstance(record.get('data'), Mapping) else {}
            latest[(event, investigation_key(event, data))] = record
        for record in latest.values():
            event = str(record.get('event') or '')
            data = record.get('data') if isinstance(record.get('data'), Mapping) else {}
            if data.get('status') != 'completed':
                continue
            if event == 'command.result' and str(
                data.get('workspace_after_sha256') or ''
            ) != workspace_sha256:
                continue
            if event == 'http.result' and str(
                data.get('workspace_sha256') or ''
            ) != workspace_sha256:
                continue
            ref = data.get('result_ref') if isinstance(data.get('result_ref'), Mapping) else None
            uri = str((ref or {}).get('uri') or '')
            if uri:
                result[uri] = dict(ref)
        return list(result.values())

    @property
    def guidance_revision_id(self) -> str:
        return str(self.guidance_state.get('revision_id') or '')

    @property
    def active_content_hash(self) -> str:
        return str(self.guidance_state.get('active_content_hash') or '')

    @property
    def active_directive_ids(self) -> list[str]:
        return [
            str(item.get('directive_id') or '')
            for item in self.guidance_state.get('active_directives') or ()
            if isinstance(item, Mapping) and str(item.get('directive_id') or '')
        ]

    def guidance_provenance(self) -> dict[str, Any]:
        return {
            'guidance_revision_id': self.guidance_revision_id,
            'active_directive_ids': self.active_directive_ids,
            'active_content_hash': self.active_content_hash,
        }

    def record_applicability(self, record: Mapping[str, Any]) -> str:
        provenance = record.get('provenance') if isinstance(record.get('provenance'), Mapping) else {}
        revision_id = str(provenance.get('guidance_revision_id') or '')
        if not self.guidance_revision_id or revision_id == self.guidance_revision_id:
            return 'current'
        record_directive_ids = {
            str(item).strip()
            for item in provenance.get('active_directive_ids') or ()
            if str(item).strip()
        }
        if record_directive_ids and not record_directive_ids.issubset(
            set(self.active_directive_ids)
        ):
            return 'superseded'
        event = str(record.get('event') or '')
        if event in {'web.search', 'web.read', 'artifact.read'}:
            return 'reusable'
        if event == 'phase1.finished':
            return 'superseded'
        return 'needs_revalidation'

    def completed_investigation(
        self,
        event: str,
        data: Mapping[str, Any],
        *,
        allow_cross_revision: bool = True,
    ) -> Mapping[str, Any] | None:
        key = investigation_key(event, data)
        for record in reversed(self._indexed_records):
            if record.get('event') != event or not isinstance(record.get('data'), Mapping):
                continue
            if investigation_key(event, record['data']) != key:
                continue
            applicability = self.record_applicability(record)
            reusable = applicability == 'current' or (
                allow_cross_revision and applicability in {'reusable', 'needs_revalidation'}
            )
            if not reusable:
                continue
            # The latest applicable outcome is authoritative. A failed forced
            # revalidation must not expose an older success as reusable evidence.
            if record['data'].get('status') != 'completed':
                return None
            if event in {'opencode.result', 'command.result'}:
                current_workspace = str(data.get('workspace_before_sha256') or '')
                after_workspace = str(record['data'].get('workspace_after_sha256') or '')
                if not current_workspace or after_workspace != current_workspace:
                    # Reusing a state-changing action without replaying its
                    # filesystem effects would leave the Workspace inconsistent.
                    return None
            return record
        return None

    def record_investigation_reuse(
        self,
        event: str,
        source: Mapping[str, Any],
        *,
        reason: str = 'same_input_and_dependency',
    ) -> dict[str, str]:
        data = source.get('data') if isinstance(source.get('data'), Mapping) else {}
        key = investigation_key(event, data)
        for record in reversed(self._indexed_records):
            reuse_data = record.get('data') if isinstance(record.get('data'), Mapping) else {}
            if (
                record.get('event') == 'investigation.reused'
                and reuse_data.get('source_event') == event
                and reuse_data.get('investigation_key') == key
                and self.record_applicability(record) == 'current'
            ):
                existing_ref = record.get('_event_ref')
                return dict(existing_ref) if isinstance(existing_ref, Mapping) else {}
        source_ref = source.get('_event_ref') if isinstance(source.get('_event_ref'), Mapping) else {}
        source_provenance = (
            source.get('provenance') if isinstance(source.get('provenance'), Mapping) else {}
        )
        return self.record(
            'investigation.reused',
            f'reused {event}: {source.get("summary") or "completed observation"}',
            {
                'status': 'completed',
                'source_event': event,
                'investigation_key': key,
                'source_event_ref': dict(source_ref),
                'source_guidance_revision_id': str(
                    source_provenance.get('guidance_revision_id') or ''
                ),
                'satisfies_completion_gate': _satisfies_completion_gate(event, data),
                'reason': reason,
            },
        )

    def workspace_digest(self, *, refresh: bool = False) -> str:
        """Return a content-based revision without rescanning on every context call."""
        if self._workspace_digest is None or refresh:
            self._workspace_digest = _workspace_tree_digest(
                self.work_root / 'work', self.source_digest,
            )
        return self._workspace_digest

    def investigation_key(self, event: str, data: Mapping[str, Any]) -> str:
        return investigation_key(event, data)

    def has_completed_investigation(self, event: str, data: Mapping[str, Any]) -> bool:
        return self.completed_investigation(event, data) is not None

    def read_artifact(
        self,
        uri: str,
        *,
        offset_bytes: int = 0,
        max_bytes: int = 4096,
    ) -> dict[str, Any]:
        """Read a registered text artifact through an integrity-checked byte window."""
        if isinstance(offset_bytes, bool) or not isinstance(offset_bytes, int) or offset_bytes < 0:
            raise ValueError('artifact_offset_invalid')
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or not 1 <= max_bytes <= 8192
        ):
            raise ValueError('artifact_max_bytes_invalid')
        clean_uri = str(uri or '').strip()
        registered = self.registered_artifacts().get(clean_uri)
        if registered is None:
            raise ValueError('artifact_ref_not_registered')
        path = self._artifact_path(clean_uri)
        total_bytes = path.stat().st_size
        if _file_sha256(path) != registered.get('sha256'):
            raise ValueError('artifact_integrity_mismatch')
        if offset_bytes > total_bytes:
            raise ValueError('artifact_offset_out_of_range')
        content, returned_bytes = _read_utf8_window(path, offset_bytes, max_bytes)
        end = offset_bytes + returned_bytes
        if not returned_bytes and offset_bytes < total_bytes:
            raise ValueError('artifact_window_too_small')
        return {
            'status': 'completed',
            'uri': clean_uri,
            'sha256': registered['sha256'],
            'offset_bytes': offset_bytes,
            'max_bytes': max_bytes,
            'returned_bytes': returned_bytes,
            'next_offset_bytes': end,
            'total_bytes': total_bytes,
            'truncated': end < total_bytes,
            'content_trust': 'untrusted_artifact',
            'content': content,
            'artifact_ref': dict(registered),
        }

    def registered_artifacts(self) -> dict[str, dict[str, str]]:
        """Return only refs emitted by trusted Repair event fields."""
        refs: dict[str, dict[str, str]] = {}

        def add(value: object) -> None:
            if not isinstance(value, Mapping):
                return
            uri = str(value.get('uri') or '')
            digest = str(value.get('sha256') or '')
            if uri and len(digest) == 64:
                refs.setdefault(uri, {'uri': uri, 'sha256': digest})

        for ref in self._core_refs.values():
            add(ref)
        seen_records: set[str] = set()
        for record in (*self._indexed_records, *self._records):
            event_ref = record.get('_event_ref')
            event_uri = str(event_ref.get('uri') or '') if isinstance(event_ref, Mapping) else ''
            if event_uri and event_uri in seen_records:
                continue
            if event_uri:
                seen_records.add(event_uri)
            add(event_ref)
            data = record.get('data') if isinstance(record.get('data'), Mapping) else {}
            event = record.get('event')
            if event == 'command.result':
                for name in ('result_ref', 'stdout_ref', 'stderr_ref'):
                    add(data.get(name))
            elif event == 'http.result':
                add(data.get('result_ref'))
            elif event == 'web.read':
                for page in data.get('pages') or ():
                    if isinstance(page, Mapping):
                        add(page.get('content_ref'))
            elif event == 'opencode.result':
                artifacts = data.get('artifacts')
                if isinstance(artifacts, Mapping):
                    for name in ('prompt', 'stdout', 'events', 'report'):
                        add(artifacts.get(name))
        return refs

    def _write_core_artifacts(self) -> dict[str, dict[str, str]]:
        root_path = self.artifact_root / 'memory' / 'root_cause.json'
        guidance_path = self.artifact_root / 'memory' / 'guidance.json'
        write_json(root_path, {'schema_version': 1, 'target': self.target})
        write_json(guidance_path, {
            'schema_version': 2,
            'user_guidance': self.guidance,
            'guidance_state': self.guidance_state,
        })
        return {
            'root_cause': content_ref(root_path, self.artifact_root),
            'guidance': content_ref(guidance_path, self.artifact_root),
        }

    def _artifact_path(self, uri: str) -> Path:
        if (
            not uri.startswith('phase1://')
            or any(token in uri for token in ('\\', '\0', '?', '#', '%'))
        ):
            raise ValueError('artifact_uri_invalid')
        for root in self._attempt_roots():
            prefix = f"phase1://{'/'.join(root.parts[-4:])}/"
            if not uri.startswith(prefix):
                continue
            relative = uri.removeprefix(prefix)
            parts = relative.split('/')
            if not relative or any(part in {'', '.', '..'} for part in parts):
                raise ValueError('artifact_path_invalid')
            unresolved = root.joinpath(*parts)
            cursor = root
            for part in parts:
                cursor = cursor / part
                if cursor.is_symlink():
                    raise ValueError('artifact_symlink_forbidden')
            try:
                path = unresolved.resolve(strict=True)
            except OSError as exc:
                raise ValueError('artifact_missing') from exc
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError('artifact_path_invalid')
            return path
        raise ValueError('artifact_scope_invalid')

    def _attempt_roots(self) -> tuple[Path, ...]:
        parent = self.artifact_root.resolve().parent
        roots = []
        for value in (*self.history_attempts, *self.previous_attempts, self.artifact_root):
            root = value.resolve()
            if root.parent == parent and root not in roots:
                roots.append(root)
        return tuple(roots)

    def completion_gaps(self, proposal: Mapping[str, Any]) -> list[str]:
        """Return explicit user tool commitments that have no recorded observation.

        The Agent remains free to design its investigation. This only prevents a
        finish action from claiming completion after the user explicitly required
        a web search/page read that never happened.
        """
        guidance = '\n'.join(self.guidance).casefold()
        proposal_text = json.dumps(proposal, ensure_ascii=False, default=str).casefold()
        if '如果采用 opensearch' in guidance and 'opensearch' not in proposal_text:
            return []
        gaps = []
        if _explicitly_requires(guidance, ('联网搜索', 'web search', 'search the web')):
            searched = any(
                record.get('event') == 'web.search'
                and isinstance(record.get('data'), Mapping)
                and record['data'].get('status') == 'completed'
                and bool(record['data'].get('results'))
                and self.record_applicability(record) == 'current'
                for record in self._web_records
            ) or self._has_current_reuse('web.search')
            if not searched:
                gaps.append('required_web_search_missing')
        if _explicitly_requires(
            guidance,
            ('读取网页', '网页读取', '读取一份', '阅读官方', 'read web', 'read page'),
        ):
            page_read = any(
                record.get('event') == 'web.read'
                and isinstance(record.get('data'), Mapping)
                and any(
                    _is_materialized_web_page(page)
                    for page in record['data'].get('pages') or ()
                )
                and self.record_applicability(record) == 'current'
                for record in self._web_records
            ) or self._has_current_reuse('web.read')
            if not page_read:
                gaps.append('required_web_page_read_missing')
        return gaps

    def _has_current_reuse(self, source_event: str) -> bool:
        return any(
            record.get('event') == 'investigation.reused'
            and isinstance(record.get('data'), Mapping)
            and record['data'].get('status') == 'completed'
            and record['data'].get('source_event') == source_event
            and record['data'].get('satisfies_completion_gate') is True
            and self.record_applicability(record) == 'current'
            for record in self._indexed_records
        )

    def consecutive_failures(self, event: str) -> int:
        count = 0
        for record in reversed(self._records):
            current = record.get('event')
            if current in {'agent.decision', 'action.rejected'}:
                continue
            if current != event:
                break
            data = record.get('data') if isinstance(record.get('data'), Mapping) else {}
            if data.get('status') != 'failed':
                break
            count += 1
        return count

    def known_urls(self) -> set[str]:
        urls = {
            str(item.get('canonical_url') or item.get('url') or '').strip()
            for record in self._web_records
            if record.get('event') == 'web.search'
            and self.record_applicability(record) != 'superseded'
            for item in (
                (record.get('data') or {}).get('results')
                if isinstance(record.get('data'), Mapping) else ()
            ) or ()
            if isinstance(item, Mapping) and str(item.get('url') or '').strip()
        }
        for guidance in self.guidance:
            urls.update(
                token.rstrip('.,;，。；')
                for token in guidance.split()
                if token.startswith(('http://', 'https://'))
            )
        return urls

    def read_urls(self) -> set[str]:
        urls = set()
        for record in self._web_records:
            if (
                record.get('event') != 'web.read'
                or self.record_applicability(record) == 'superseded'
            ):
                continue
            data = record.get('data') if isinstance(record.get('data'), Mapping) else {}
            for page in data.get('pages') or ():
                if not isinstance(page, Mapping) or page.get('status') not in {'readable', 'duplicate'}:
                    continue
                for name in ('requested_url', 'canonical_url', 'url'):
                    if value := str(page.get(name) or '').strip():
                        urls.add(value)
        return urls

    def has_searched_query(self, query: str) -> bool:
        normalized = _normalized_query(query)
        return any(
            _normalized_query(str((record.get('data') or {}).get('query') or '')) == normalized
            for record in self._web_records
            if (
                record.get('event') == 'web.search'
                and isinstance(record.get('data'), Mapping)
                and record['data'].get('status') == 'completed'
                and self.record_applicability(record) != 'superseded'
            )
        )

    def searched_queries(self) -> list[str]:
        queries: dict[str, str] = {}
        for record in self._web_records:
            if (
                record.get('event') != 'web.search'
                or not isinstance(record.get('data'), Mapping)
                or record['data'].get('status') != 'completed'
                or self.record_applicability(record) == 'superseded'
            ):
                continue
            query = ' '.join(str(record['data'].get('query') or '').split())
            if query:
                queries.setdefault(_normalized_query(query), query)
        return list(queries.values())

    def read_page_fingerprints(self) -> list[dict[str, Any]]:
        fingerprints = []
        for record in self._web_records:
            if (
                record.get('event') != 'web.read'
                or not isinstance(record.get('data'), Mapping)
                or self.record_applicability(record) == 'superseded'
            ):
                continue
            for page in record['data'].get('pages') or ():
                if not isinstance(page, Mapping) or page.get('status') != 'readable':
                    continue
                content_ref_value = page.get('content_ref') if isinstance(page.get('content_ref'), Mapping) else {}
                content_sha256 = str(page.get('content_sha256') or content_ref_value.get('sha256') or '')
                if not content_sha256:
                    continue
                fingerprints.append({
                    'url': str(page.get('canonical_url') or page.get('url') or ''),
                    'content_sha256': content_sha256,
                    'content_simhash': str(page.get('content_simhash') or ''),
                    'character_count': _integer_or_zero(page.get('character_count')),
                    'similarity_token_count': _integer_or_zero(page.get('similarity_token_count')),
                    'content_ref': dict(content_ref_value),
                })
        return fingerprints

    def web_investigation(self) -> dict[str, Any]:
        queries = self.searched_queries()
        urls = sorted(self.read_urls())
        return {
            'searched_query_count': len(queries),
            'searched_queries': queries[-20:],
            'read_page_count': len(self.read_page_fingerprints()),
            'read_urls': urls[-20:],
        }

    def journal_ref(self) -> dict[str, str]:
        path = self.artifact_root / 'journal.jsonl'
        if not path.is_file():
            path.write_text('', encoding='utf-8')
        return content_ref(path, self.artifact_root)

    def close(self) -> None:
        cleanup_workdir(self.work_root)

    def restore_source(self) -> None:
        metadata = _read_json(self.artifact_root / 'metadata.json')
        origin = source_root(metadata.get('source_dir'))
        shutil.rmtree(self.work_root / 'source', ignore_errors=True)
        copy_source(origin, self.work_root / 'source')
        if source_hash(self.work_root / 'source') != self.source_digest:
            raise ValueError('source_restore_failed')
        self.workspace_digest(refresh=True)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n',
        encoding='utf-8',
    )


def content_ref(path: Path, artifact_root: Path) -> dict[str, str]:
    return {
        'uri': content_uri(path, artifact_root),
        'sha256': _file_sha256(path),
    }


def directory_ref(path: Path, artifact_root: Path) -> dict[str, str]:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob('*') if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(b'\0')
        digest.update(_file_digest(item))
    return {'uri': content_uri(path, artifact_root), 'sha256': digest.hexdigest()}


def content_uri(path: Path, artifact_root: Path) -> str:
    relative = path.relative_to(artifact_root).as_posix()
    identity = '/'.join(artifact_root.parts[-4:])
    return f'phase1://{identity}/{relative}'


def cleanup_workdir(work_root: Path | None) -> None:
    if work_root is not None and work_root.name.startswith('lazyrag-repair-phase1-'):
        shutil.rmtree(work_root, ignore_errors=True)


def _select_recovery(
    attempts: Sequence[Path],
    current_state: Mapping[str, Any],
    source_digest: str,
) -> dict[str, Any]:
    current_revision = str(current_state.get('revision_id') or '')
    current_hash = str(current_state.get('active_content_hash') or '')
    effect = str(current_state.get('effect') or 'append')
    parent_revision = str(current_state.get('parent_revision_id') or '')
    candidates = []
    for path in attempts:
        metadata = _read_json(path / 'metadata.json')
        if str(metadata.get('source_hash') or '') != source_digest:
            continue
        checkpoint = _read_json(path / 'checkpoint.complete')
        saved_work = path / 'checkpoint' / 'work'
        if checkpoint.get('completed') is not True or not saved_work.is_dir():
            continue
        expected_workspace = str(checkpoint.get('workspace_sha256') or '')
        checkpoint_revision = str(checkpoint.get('guidance_revision_id') or '')
        checkpoint_hash = str(checkpoint.get('active_content_hash') or '')
        # New recovery is fail-closed. Older attempts without these integrity
        # fields remain available to the audit index but cannot seed Workspace.
        if not expected_workspace or not checkpoint_revision or not checkpoint_hash:
            continue
        if _workspace_tree_digest(saved_work, source_digest) != expected_workspace:
            continue
        try:
            state, explicit = _attempt_guidance_state(path)
        except (TypeError, ValueError):
            continue
        if checkpoint_revision != str(state.get('revision_id') or ''):
            continue
        if checkpoint_hash != str(state.get('active_content_hash') or ''):
            continue
        session = _verified_checkpoint_session(
            path,
            checkpoint,
            guidance_revision_id=checkpoint_revision,
            active_content_hash=checkpoint_hash,
            workspace_sha256=expected_workspace,
        )
        candidates.append((path, state, explicit, bool(session)))

    for path, state, explicit, session_valid in reversed(candidates):
        if (
            current_revision
            and str(state.get('revision_id') or '') == current_revision
            and str(state.get('active_content_hash') or '') == current_hash
        ):
            return {
                'mode': 'same_revision_resume' if explicit else 'legacy_compatible',
                'source_attempt': path.name,
                'source_attempt_path': path,
                'restore_workspace': True,
                'session_resume_allowed': bool(explicit and session_valid),
            }

    if effect == 'append' and parent_revision:
        for path, state, _explicit, _session_valid in reversed(candidates):
            if (
                str(state.get('revision_id') or '') == parent_revision
                and is_append_guidance_successor(state, current_state)
            ):
                return {
                    'mode': 'fork_compatible',
                    'source_attempt': path.name,
                    'source_attempt_path': path,
                    'restore_workspace': True,
                    'session_resume_allowed': False,
                }

    if effect == 'withdraw' and current_hash:
        for path, state, _explicit, _session_valid in reversed(candidates):
            if str(state.get('active_content_hash') or '') == current_hash:
                return {
                    'mode': 'restore_ancestor',
                    'source_attempt': path.name,
                    'source_attempt_path': path,
                    'restore_workspace': True,
                    'session_resume_allowed': False,
                }

    return {
        'mode': 'reset_for_new_direction' if attempts else 'fresh',
        'source_attempt': '',
        'restore_workspace': False,
        'session_resume_allowed': False,
    }


def _public_recovery(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key != 'source_attempt_path'
    }


def _attempt_guidance_state(attempt: Path) -> tuple[dict[str, Any], bool]:
    metadata = _read_json(attempt / 'metadata.json')
    state = metadata.get('guidance_state')
    if isinstance(state, Mapping):
        return guidance_snapshot({
            'guidance_state': state,
            'user_guidance': [
                str(item.get('text') or '')
                for item in state.get('active_directives') or ()
                if isinstance(item, Mapping) and str(item.get('text') or '').strip()
            ],
        }), True
    guidance = _read_json(attempt / 'memory' / 'guidance.json')
    return guidance_snapshot(guidance), False


def _verified_checkpoint_session(
    attempt: Path,
    checkpoint: Mapping[str, Any],
    *,
    guidance_revision_id: str,
    active_content_hash: str,
    workspace_sha256: str,
) -> dict[str, Any]:
    session_path = attempt / 'checkpoint' / 'session.json'
    expected_sha256 = str(checkpoint.get('session_sha256') or '')
    if (
        len(expected_sha256) != 64
        or not session_path.is_file()
        or _file_sha256(session_path) != expected_sha256
    ):
        return {}
    session = _read_json(session_path)
    session_id = session.get('session_id')
    calls = session.get('calls')
    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > 500
        or isinstance(calls, bool)
        or not isinstance(calls, int)
        or not 0 <= calls <= 1_000_000
    ):
        return {}
    if (
        str(session.get('guidance_revision_id') or '') != guidance_revision_id
        or str(session.get('active_content_hash') or '') != active_content_hash
        or str(session.get('workspace_sha256') or '') != workspace_sha256
    ):
        return {}
    return dict(session)


def _guidance_provenance(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'guidance_revision_id': str(state.get('revision_id') or ''),
        'active_content_hash': str(state.get('active_content_hash') or ''),
        'active_directive_ids': [
            str(item.get('directive_id') or '')
            for item in state.get('active_directives') or ()
            if isinstance(item, Mapping) and str(item.get('directive_id') or '')
        ],
    }


def _target_hash(target: Mapping[str, Any]) -> str:
    stable = {
        'category_id': target.get('category_id'),
        'source_hash': target.get('source_hash'),
        'category': target.get('category'),
    }
    return hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()[:24]


def _safe_segment(value: str) -> str:
    text = ''.join(char if char.isalnum() or char in '._-' else '_' for char in value.strip())
    text = text.strip('._-')
    if not text or text in {'.', '..'}:
        raise ValueError('unsafe_artifact_segment')
    return text[:160]


def _load_memory_records(
    recent_attempts: Sequence[Path],
    history_attempts: Sequence[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    provenance_by_attempt: dict[Path, dict[str, Any]] = {}

    def provenance(attempt: Path) -> dict[str, Any]:
        resolved = attempt.resolve()
        if resolved not in provenance_by_attempt:
            try:
                state, _ = _attempt_guidance_state(resolved)
                value = _guidance_provenance(state)
            except (TypeError, ValueError):
                # Keep corrupt/partial attempts available for audit without
                # allowing their observations to influence the active direction.
                value = {
                    'guidance_revision_id': 'invalid-guidance-state',
                    'active_content_hash': '',
                    'active_directive_ids': ['invalid-guidance-state'],
                }
            provenance_by_attempt[resolved] = value
        return provenance_by_attempt[resolved]

    recent: deque[dict[str, Any]] = deque(maxlen=80)
    for attempt in _unique_paths(recent_attempts):
        for item in _load_journal(attempt):
            value = _load_event(attempt, item, provenance(attempt))
            if value:
                recent.append(value)
    indexed = []
    web_records = []
    seen = set()
    for attempt in _unique_paths(history_attempts):
        for item in _load_journal(attempt):
            event = item.get('event')
            if event not in _INDEXED_EVENTS:
                continue
            value = _load_event(attempt, item, provenance(attempt))
            if value:
                uri = str((value.get('_event_ref') or {}).get('uri') or '')
                identity = uri or f'{attempt}:{item.get("file")}'
                if identity in seen:
                    continue
                seen.add(identity)
                indexed.append(value)
                if event in {'web.search', 'web.read'}:
                    web_records.append(value)
    return list(recent), indexed, web_records


def _load_event(
    attempt: Path,
    journal_item: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    relative = str(journal_item.get('file') or '')
    parts = Path(relative).parts
    if (
        not relative
        or Path(relative).is_absolute()
        or any(part in {'', '.', '..'} for part in parts)
    ):
        return {}
    root = attempt.resolve()
    unresolved = attempt.joinpath(*parts)
    cursor = attempt
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            return {}
    try:
        path = unresolved.resolve(strict=True)
    except OSError:
        return {}
    if not path.is_relative_to(root) or not path.is_file():
        return {}
    value = _read_json(path)
    if not value:
        return {}
    ref = journal_item.get('ref')
    expected_ref = content_ref(path, attempt)
    if (
        not isinstance(ref, Mapping)
        or str(ref.get('uri') or '') != expected_ref['uri']
        or str(ref.get('sha256') or '') != expected_ref['sha256']
        or str(journal_item.get('event') or '') != str(value.get('event') or '')
    ):
        return {}
    raw_provenance = value.get('provenance')
    event_provenance = (
        dict(raw_provenance)
        if isinstance(raw_provenance, Mapping)
        and str(raw_provenance.get('guidance_revision_id') or '')
        else dict(provenance or {})
    )
    return {
        **value,
        'provenance': event_provenance,
        '_event_ref': expected_ref,
    }


def _unique_paths(values: Sequence[Path]) -> tuple[Path, ...]:
    result = []
    for value in values:
        path = value.resolve()
        if path not in result:
            result.append(path)
    return tuple(result)


def _load_journal(attempt: Path) -> list[dict[str, Any]]:
    path = attempt / 'journal.jsonl'
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _explicitly_requires(text: str, phrases: Sequence[str]) -> bool:
    if not any(phrase in text for phrase in phrases):
        return False
    return not any(
        marker in text
        for marker in ('不要联网', '禁止联网', '无需联网', '不需要联网', 'do not search the web')
    )


def _normalized_query(value: str) -> str:
    return ' '.join(str(value or '').split()).casefold()


def _integer_or_zero(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _satisfies_completion_gate(event: str, data: Mapping[str, Any]) -> bool:
    if event == 'web.search':
        return bool(data.get('results'))
    if event == 'web.read':
        return any(
            _is_materialized_web_page(page)
            for page in data.get('pages') or ()
        )
    return data.get('status') == 'completed'


def _is_materialized_web_page(page: object) -> bool:
    """Treat deduplicated page bodies as valid reads, but not URL-only aliases."""
    return (
        isinstance(page, Mapping)
        and page.get('status') in {'readable', 'duplicate'}
        and isinstance(page.get('content_ref'), Mapping)
    )


def _file_sha256(path: Path) -> str:
    return _file_digest(path).hex()


def _workspace_tree_digest(root: Path, source_digest: str) -> str:
    digest = hashlib.sha256()
    digest.update(source_digest.encode('utf-8'))
    digest.update(b'\0')
    for path in sorted(candidate for candidate in root.rglob('*') if candidate.is_file()):
        digest.update(path.relative_to(root).as_posix().encode('utf-8'))
        digest.update(b'\0')
        # A fixed-width content digest prevents path/content boundary
        # collisions even when file bytes contain NUL characters.
        digest.update(_file_digest(path))
    return digest.hexdigest()


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    _update_digest_from_file(digest, path)
    return digest.digest()


def _update_digest_from_file(digest: Any, path: Path) -> None:
    with path.open('rb') as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)


def _read_utf8_window(path: Path, offset: int, limit: int) -> tuple[str, int]:
    with path.open('rb') as stream:
        if offset:
            stream.seek(offset)
            marker = stream.read(1)
            if marker and marker[0] & 0xC0 == 0x80:
                raise ValueError('artifact_offset_not_utf8_boundary')
        stream.seek(offset)
        chunk = stream.read(limit)
    while chunk:
        try:
            return chunk.decode('utf-8'), len(chunk)
        except UnicodeDecodeError as exc:
            if exc.reason != 'unexpected end of data' or exc.end != len(chunk):
                raise ValueError('artifact_not_utf8_text') from exc
            chunk = chunk[:-1]
    return '', 0
