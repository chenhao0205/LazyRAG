from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import RepairAction, RepairInput, RepairObservation, RepairView, contract_dict
from .workspace import WorkspacePaths, diff_summary, write_context


MemorySummarizer = Callable[[str, str, str, list[dict[str, Any]]], str]
MAX_MEMORY_BRIEF_CHARS = 12_000
MAX_PINNED_CONTEXT_CHARS = 8_000
MAX_EVENT_STRING_CHARS = 4_000
MAX_COLLECTION_ITEMS = 40
HIDDEN_CONSTRAINT_KEYS = {'api_key', 'credential', 'llm_config', 'model_config', 'password', 'secret', 'token'}


@dataclass(frozen=True, slots=True)
class MemoryEvent:
    sequence: int
    timestamp: str
    event: str
    call_id: str
    workspace_hash: str
    payload: dict[str, Any]


class EventMemory:
    """Runtime-owned append-only facts and a model-authored bounded projection."""

    def __init__(self, paths: WorkspacePaths, repair_input: RepairInput, recent_limit: int = 12) -> None:
        self.paths = paths
        self.repair_input = repair_input
        self.recent_limit = max(1, int(recent_limit))
        self._lock = threading.RLock()
        self.paths.events.touch(mode=0o600, exist_ok=True)
        self.paths.events.chmod(0o600)
        if not self.read():
            self.append('input.received', contract_dict(repair_input))

    def append(
        self,
        event: str,
        payload: Mapping[str, Any],
        *,
        call_id: str = '',
        workspace_hash: str = '',
    ) -> MemoryEvent:
        with self._lock:
            record = MemoryEvent(
                sequence=self._last_sequence() + 1,
                timestamp=datetime.now(timezone.utc).isoformat(),
                event=str(event),
                call_id=str(call_id),
                workspace_hash=str(workspace_hash),
                payload=dict(payload),
            )
            line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True, default=str) + '\n'
            descriptor = os.open(self.paths.events, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                encoded = line.encode('utf-8')
                if os.write(descriptor, encoded) != len(encoded):
                    raise OSError('events_append_incomplete')
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return record

    def record_action(self, action: RepairAction, workspace_hash: str) -> MemoryEvent:
        return self.append(
            'agent.action',
            contract_dict(action),
            call_id=str(getattr(action, 'call_id', '')),
            workspace_hash=workspace_hash,
        )

    def record_observation(self, observation: RepairObservation) -> MemoryEvent:
        return self.append(
            'capability.observation',
            contract_dict(observation),
            call_id=observation.call_id,
            workspace_hash=observation.workspace_hash,
        )

    def project(
        self,
        workspace_hash: str,
        validation_evidence: list[dict[str, Any]],
        remaining_budget: dict[str, Any],
        summarize: MemorySummarizer,
    ) -> RepairView:
        records = self._valid_records(self.read())
        brief, through = self._latest_brief(records)
        uncompacted = [
            record for record in records
            if record.sequence > through and record.event != 'memory.compressed'
        ]
        if len(uncompacted) > self.recent_limit:
            compactable = uncompacted[:-self.recent_limit]
            brief = summarize(
                self.repair_input.objective,
                self.repair_input.guidance,
                brief,
                [self._project_event(record) for record in compactable],
            ).strip()[-MAX_MEMORY_BRIEF_CHARS:]
            through = compactable[-1].sequence
            self.append(
                'memory.compressed',
                {'memory_brief': brief, 'through_sequence': through},
                workspace_hash=workspace_hash,
            )
        recent = [
            self._project_event(record)
            for record in uncompacted
            if record.sequence > through
        ][-self.recent_limit:]
        view = RepairView(
            objective=self.repair_input.objective,
            guidance=self.repair_input.guidance,
            workspace_hash=workspace_hash,
            diff_summary=diff_summary(self.repair_input.source_ref, self.paths.source),
            memory_brief=self._memory_brief(brief),
            recent_events=recent,
            validation_evidence=validation_evidence,
            remaining_budget=remaining_budget,
        )
        write_context(self.paths, contract_dict(view))
        return view

    def read(self) -> list[MemoryEvent]:
        with self._lock:
            result: list[MemoryEvent] = []
            for line in self.paths.events.read_text(encoding='utf-8').splitlines():
                try:
                    value = json.loads(line)
                    result.append(MemoryEvent(
                        sequence=int(value['sequence']),
                        timestamp=str(value['timestamp']),
                        event=str(value['event']),
                        call_id=str(value.get('call_id') or ''),
                        workspace_hash=str(value.get('workspace_hash') or ''),
                        payload=dict(value.get('payload') or {}),
                    ))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            return result

    def observations(self) -> list[RepairObservation]:
        result = []
        for record in self.read():
            if record.event != 'capability.observation':
                continue
            try:
                result.append(RepairObservation(**record.payload))
            except (TypeError, ValueError):
                continue
        return result

    def artifact_refs(self) -> list[str]:
        roots = (self.paths.evidence.resolve(), self.paths.logs.resolve())
        references = []
        for observation in self.observations():
            for reference in observation.artifact_refs:
                path = Path(reference).resolve()
                if path.is_file() and any(path.is_relative_to(root) for root in roots):
                    references.append(str(path))
        return list(dict.fromkeys(references))

    def _last_sequence(self) -> int:
        records = self.read()
        return records[-1].sequence if records else 0

    @staticmethod
    def _latest_brief(records: Sequence[MemoryEvent]) -> tuple[str, int]:
        for record in reversed(records):
            if record.event != 'memory.compressed':
                continue
            return (
                str(record.payload.get('memory_brief') or ''),
                int(record.payload.get('through_sequence') or 0),
            )
        return '', 0

    @staticmethod
    def _valid_records(records: Sequence[MemoryEvent]) -> list[MemoryEvent]:
        observed = {
            record.call_id for record in records
            if record.event == 'capability.observation' and record.call_id
        }
        return [
            record for record in records
            if record.event != 'agent.action' or record.call_id in observed
        ]

    def _memory_brief(self, history: str) -> str:
        constraints = {
            str(key): _bounded_value(value)
            for key, value in self.repair_input.constraints.items()
            if str(key).lower() not in HIDDEN_CONSTRAINT_KEYS
        }
        test_commands = constraints.get('test_commands')
        if isinstance(test_commands, Mapping):
            constraints['test_commands'] = {
                str(level): 'configured' for level in test_commands
            }
        pinned = json.dumps({
            'case_scope': self.repair_input.case_scope,
            'constraints': constraints,
        }, ensure_ascii=False, sort_keys=True, default=str)
        if len(pinned) > MAX_PINNED_CONTEXT_CHARS:
            pinned = pinned[:MAX_PINNED_CONTEXT_CHARS] + '…'
        return f'Pinned runtime context:\n{pinned}\nHistorical memory:\n{history or "(empty)"}'

    @staticmethod
    def _project_event(record: MemoryEvent) -> dict[str, Any]:
        payload = record.payload
        projected = False
        if record.event == 'input.received':
            constraints = payload.get('constraints')
            payload = {
                'objective': _clip(str(payload.get('objective') or '')),
                'guidance': _clip(str(payload.get('guidance') or '')),
                'case_scope': _clip(str(payload.get('case_scope') or '')),
                'constraint_keys': (
                    sorted(str(key) for key in constraints)
                    if isinstance(constraints, Mapping) else []
                ),
            }
            projected = True
        elif record.event == 'capability.observation':
            payload = {
                'call_id': payload.get('call_id'),
                'status': payload.get('status'),
                'summary': _clip(str(payload.get('summary') or '')),
                'artifact_refs': _bounded_value(payload.get('artifact_refs') or []),
                'workspace_hash': payload.get('workspace_hash'),
            }
            projected = True
        return {
            'sequence': record.sequence,
            'event': record.event,
            'call_id': record.call_id,
            'workspace_hash': record.workspace_hash,
            'payload': payload if projected else _bounded_value(payload),
        }


def _bounded_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = list(value.items())[:MAX_COLLECTION_ITEMS]
        return {str(key): _bounded_item(item) for key, item in items}
    if isinstance(value, (list, tuple)):
        return [_bounded_item(item) for item in value[:MAX_COLLECTION_ITEMS]]
    if isinstance(value, str):
        return _clip(value)
    return value


def _bounded_item(value: Any) -> Any:
    if isinstance(value, str):
        return _clip(value)
    if isinstance(value, (Mapping, list, tuple)):
        return _clip(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    return value


def _clip(value: str) -> str:
    return value if len(value) <= MAX_EVENT_STRING_CHARS else value[:MAX_EVENT_STRING_CHARS] + '…'


__all__ = ['EventMemory', 'MemoryEvent', 'MemorySummarizer']
