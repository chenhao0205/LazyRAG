from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    TEST_LEVELS,
    RepairAction,
    RepairObservation,
)
from .workspace import WorkspacePaths, artifact_path, write_json


MAX_EVIDENCE_SUMMARY_CHARS = 2_000


def record_finish_evidence(
    paths: WorkspacePaths,
    action: RepairAction,
    current_hash: str,
    semantic_satisfied: bool,
    scope_satisfied: bool,
    assessment: str,
) -> RepairObservation:
    passed = semantic_satisfied and scope_satisfied
    evidence = {
        'kind': 'finish',
        'call_id': action.call_id,
        'status': 'success' if passed else 'fail',
        'workspace_hash': current_hash,
        'semantic_satisfied': semantic_satisfied,
        'scope_satisfied': scope_satisfied,
        'assessment': assessment,
    }
    path = write_json(artifact_path(paths.evidence, action.call_id), evidence)
    return RepairObservation(
        call_id=action.call_id,
        status='success' if passed else 'fail',
        summary=assessment or ('finish accepted' if passed else 'finish rejected'),
        artifact_refs=[str(path)],
        workspace_hash=current_hash,
    )


def validation_evidence(
    paths: WorkspacePaths,
    observations: Sequence[RepairObservation],
    current_hash: str,
) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for observation in observations:
        if observation.workspace_hash != current_hash:
            continue
        for reference in observation.artifact_refs:
            value = _read_evidence(paths, reference)
            level = str(value.get('level') or '')
            if (
                value.get('kind') != 'test'
                or level not in TEST_LEVELS
                or value.get('call_id') != observation.call_id
                or value.get('workspace_hash') != current_hash
            ):
                continue
            latest[level] = {
                'kind': 'test',
                'call_id': observation.call_id,
                'level': level,
                'status': value.get('status'),
                'workspace_hash': current_hash,
                'return_code': value.get('return_code'),
                'summary': _test_summary(value),
                'artifact_ref': reference,
            }
    return [latest[level] for level in TEST_LEVELS if level in latest]


def check_completion(
    paths: WorkspacePaths,
    observations: list[RepairObservation],
    current_hash: str,
    finish_call_id: str,
) -> bool:
    """Require current tests and the current Finish call for one workspace hash."""
    levels: dict[str, bool] = {}
    finish = False
    for observation in observations:
        if observation.workspace_hash != current_hash:
            continue
        for reference in observation.artifact_refs:
            value = _read_evidence(paths, reference)
            if value.get('workspace_hash') != current_hash or value.get('call_id') != observation.call_id:
                continue
            if value.get('kind') == 'test' and value.get('level') in TEST_LEVELS:
                levels[str(value['level'])] = (
                    observation.status == 'success' and value.get('status') == 'success'
                )
            if value.get('kind') == 'finish' and observation.call_id == finish_call_id:
                finish = (
                    observation.status == 'success'
                    and value.get('status') == 'success'
                    and value.get('semantic_satisfied') is True
                    and value.get('scope_satisfied') is True
                )
    return levels == {level: True for level in TEST_LEVELS} and finish


def _read_evidence(paths: WorkspacePaths, reference: str) -> dict[str, Any]:
    try:
        path = Path(reference).resolve()
        root = paths.evidence.resolve()
        if not path.is_file() or not path.is_relative_to(root):
            return {}
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _test_summary(value: dict[str, Any]) -> str:
    parts = []
    for output in value.get('outputs') or ():
        if not isinstance(output, dict):
            continue
        command = output.get('command')
        stdout = str(output.get('stdout') or '').strip()
        stderr = str(output.get('stderr') or '').strip()
        parts.append(f'command={command!r}\nstdout={stdout}\nstderr={stderr}')
    return '\n\n'.join(parts)[-MAX_EVIDENCE_SUMMARY_CHARS:]


__all__ = [
    'check_completion', 'record_finish_evidence', 'validation_evidence',
]
