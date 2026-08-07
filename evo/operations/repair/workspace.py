from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .contracts import RepairContractError, RepairInput


IGNORED_NAMES = {'.git', '.mypy_cache', '.pytest_cache', '__pycache__'}
DEFAULT_RUNTIME_ROOT = Path('/tmp/lazyrag-repair')
MAX_DIFF_CHARS = 12_000


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    root: Path
    control: Path
    events: Path
    evidence: Path
    logs: Path
    result: Path
    sandbox: Path
    source: Path
    work: Path
    context: Path


def initialize_workspace(repair_input: RepairInput, base: Path = DEFAULT_RUNTIME_ROOT) -> WorkspacePaths:
    source_ref = Path(repair_input.source_ref).expanduser().resolve()
    if not source_ref.is_dir():
        raise RepairContractError('source_ref_invalid', repair_input.source_ref)
    root = (base / repair_input.run_id).resolve()
    paths = WorkspacePaths(
        root=root,
        control=root / 'control',
        events=root / 'control/events.jsonl',
        evidence=root / 'control/evidence',
        logs=root / 'control/logs',
        result=root / 'control/result.json',
        sandbox=root / 'sandbox',
        source=root / 'sandbox/source',
        work=root / 'sandbox/work',
        context=root / 'sandbox/context',
    )
    for directory in (
        paths.control, paths.evidence, paths.logs,
        paths.source, paths.work, paths.context,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    paths.control.chmod(0o700)
    if not any(paths.source.iterdir()):
        _copy_source(source_ref, paths.source)
    return paths


def workspace_hash(source: Path) -> str:
    digest = hashlib.sha256()
    for path in _files(source):
        relative = path.relative_to(source).as_posix()
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def changed_paths(source_ref: str, candidate: Path) -> list[str]:
    original = Path(source_ref).expanduser().resolve()
    left = {path.relative_to(original).as_posix(): path for path in _files(original)}
    right = {path.relative_to(candidate).as_posix(): path for path in _files(candidate)}
    changed = []
    for name in sorted(left.keys() | right.keys()):
        if name not in left or name not in right or left[name].read_bytes() != right[name].read_bytes():
            changed.append(name)
    return changed


def diff_summary(source_ref: str, candidate: Path) -> str:
    paths = changed_paths(source_ref, candidate)
    if not paths:
        return 'no changes'
    patch = _patch_text(source_ref, candidate)
    clipped = patch[:MAX_DIFF_CHARS]
    suffix = '' if len(patch) <= MAX_DIFF_CHARS else f'\n… diff truncated at {MAX_DIFF_CHARS} characters'
    return f"{len(paths)} changed file(s): {', '.join(paths[:20])}\n{clipped}{suffix}"


def write_patch(source_ref: str, candidate: Path, destination: Path) -> Path:
    text = _patch_text(source_ref, candidate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    os.replace(temporary, destination)
    return destination


def artifact_path(directory: Path, call_id: str) -> Path:
    """Return a runtime-owned filename without treating model output as a path."""
    digest = hashlib.sha256(str(call_id).encode('utf-8')).hexdigest()[:24]
    return directory / f'{digest}.json'


def _patch_text(source_ref: str, candidate: Path) -> str:
    original = Path(source_ref).expanduser().resolve()
    left = {path.relative_to(original).as_posix(): path for path in _files(original)}
    right = {path.relative_to(candidate).as_posix(): path for path in _files(candidate)}
    lines: list[str] = []
    for name in sorted(left.keys() | right.keys()):
        before = _text_lines(left.get(name))
        after = _text_lines(right.get(name))
        if before == after:
            continue
        lines.extend(difflib.unified_diff(
            before,
            after,
            fromfile=f'a/{name}' if name in left else '/dev/null',
            tofile=f'b/{name}' if name in right else '/dev/null',
        ))
    return ''.join(lines)


def path_in_scope(path: str, case_scope: str) -> bool:
    candidate = _relative(path)
    roots = [_relative(item) for item in case_scope.splitlines() if item.strip()]
    return any(candidate == root or candidate.startswith(f'{root}/') for root in roots)


def safe_path(paths: WorkspacePaths, area: str, relative: str = '') -> Path:
    roots = {'source': paths.source, 'work': paths.work, 'context': paths.context}
    if area not in roots:
        raise RepairContractError('workspace_area_invalid', area)
    root = roots[area].resolve()
    value = root / _relative(relative) if relative else root
    resolved = value.resolve()
    if resolved != root and root not in resolved.parents:
        raise RepairContractError('workspace_path_outside_sandbox', relative)
    return resolved


def write_context(paths: WorkspacePaths, value: dict[str, Any]) -> Path:
    destination = paths.context / 'repair_view.json'
    if destination.exists():
        destination.chmod(stat.S_IRUSR | stat.S_IWUSR)
    write_json(destination, value)
    destination.chmod(stat.S_IRUSR)
    return destination


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return path


def _copy_source(source: Path, destination: Path) -> None:
    for child in source.iterdir():
        if child.name in IGNORED_NAMES:
            continue
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target, ignore=shutil.ignore_patterns(*IGNORED_NAMES))
        elif child.is_file():
            shutil.copy2(child, target)


def _files(root: Path) -> Iterable[Path]:
    return (
        path for path in sorted(root.rglob('*'))
        if path.is_file() and not any(part in IGNORED_NAMES for part in path.relative_to(root).parts)
    )


def _relative(value: str) -> str:
    text = str(value).strip().replace('\\', '/')
    path = PurePosixPath(text)
    if not text or path.is_absolute() or any(part in {'', '.', '..'} for part in path.parts):
        raise RepairContractError('relative_path_invalid', text)
    return path.as_posix()


def _text_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        return path.read_text(encoding='utf-8').splitlines(keepends=True)
    except UnicodeDecodeError:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return [f'<binary sha256={digest}>\n']


__all__ = [
    'WorkspacePaths', 'artifact_path', 'changed_paths', 'diff_summary', 'initialize_workspace',
    'path_in_scope', 'safe_path', 'workspace_hash', 'write_context', 'write_json', 'write_patch',
]
