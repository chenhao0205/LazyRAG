from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .errors import (
    DefinitionError,
    _integer,
    _string,
    _text,
    _tuple_of,
    _unique,
)


_FAILURE_PREFIX = '__artifact_runtime_failure__.'
RUN_CONFIGURATION_ARTIFACT_ID = '__artifact_runtime__.configuration'


@dataclass(frozen=True, order=True)
class ArtifactKey:
    artifact_id: str
    partition_key: str = ''

    def __post_init__(self) -> None:
        _text(self.artifact_id, 'artifact_id')
        _string(self.partition_key, 'partition_key')

        if self.partition_key and not self.partition_key.strip():
            raise DefinitionError('partition_key must be non-empty when set')

    @classmethod
    def scalar(cls, artifact_id: str) -> ArtifactKey:
        return cls(artifact_id)

    @classmethod
    def partition(cls, artifact_id: str, partition_key: str) -> ArtifactKey:
        _text(partition_key, 'partition_key')
        return cls(artifact_id, partition_key)


@dataclass(frozen=True, order=True)
class ArtifactRef:
    key: ArtifactKey
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, ArtifactKey):
            raise TypeError('key must be ArtifactKey')
        _integer(self.version, 'version', minimum=1)


@dataclass(frozen=True)
class ArtifactRecord:
    ref: ArtifactRef
    producer: str
    input_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ArtifactRef):
            raise TypeError('ref must be ArtifactRef')

        _text(self.producer, 'producer')
        inputs = tuple(sorted(_tuple_of(
            self.input_refs, ArtifactRef, 'input_refs must contain ArtifactRef values'
        )))
        _unique(inputs, 'input_refs must contain at most one ref per artifact key',
                key=lambda ref: ref.key)

        object.__setattr__(self, 'input_refs', inputs)


@dataclass(frozen=True)
class PartitionSet:
    keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        keys = tuple(self.keys)
        for key in keys:
            _text(key, 'partition key')

        _unique(keys, 'partition keys must be unique')

        object.__setattr__(self, 'keys', keys)

    def __contains__(self, partition_key: object) -> bool:
        return partition_key in self.keys


@dataclass(frozen=True, order=True)
class PartitionGuard:
    partition_set_key: ArtifactKey
    partition_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.partition_set_key, ArtifactKey):
            raise TypeError('partition_set_key must be ArtifactKey')

        if self.partition_set_key.partition_key:
            raise DefinitionError('partition_set_key must identify a scalar artifact')
        _text(self.partition_key, 'partition_key')


@dataclass(frozen=True)
class ArtifactDraft:
    key: ArtifactKey
    value: object
    input_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, ArtifactKey):
            raise TypeError('artifact write key must be ArtifactKey')

        if isinstance(self.value, PartitionSet) and self.key.partition_key:
            raise DefinitionError('PartitionSet must be written as a scalar artifact')

        object.__setattr__(self, 'input_refs', merge_refs(self.input_refs))


@dataclass(frozen=True)
class ArtifactCommit:
    commit_id: str
    producer: str
    writes: tuple[ArtifactDraft, ...]
    expected_heads: Mapping[ArtifactKey, ArtifactRef | None] = field(default_factory=dict)
    partition_guards: tuple[PartitionGuard, ...] = ()

    def __post_init__(self) -> None:
        _text(self.commit_id, 'artifact commit id')
        _text(self.producer, 'artifact commit producer')
        writes = _tuple_of(self.writes, ArtifactDraft,
                           'artifact commit writes must contain ArtifactDraft values')
        if not writes:
            raise DefinitionError('artifact commit must contain at least one write')
        _unique(writes, 'artifact commit write keys must be unique',
                key=lambda write: write.key)

        expected_heads = dict(self.expected_heads)
        for key, ref in expected_heads.items():
            if not isinstance(key, ArtifactKey):
                raise TypeError('expected_heads keys must be ArtifactKey values')
            if ref is not None and not isinstance(ref, ArtifactRef):
                raise TypeError('expected_heads values must be ArtifactRef or None')
            if ref is not None and ref.key != key:
                raise DefinitionError('expected head must identify its artifact key')

        guards = _tuple_of(self.partition_guards, PartitionGuard,
                           'partition_guards must contain PartitionGuard values')
        _unique(guards, 'partition guards must be unique')

        object.__setattr__(self, 'writes', writes)
        object.__setattr__(self, 'expected_heads', MappingProxyType(expected_heads))
        object.__setattr__(self, 'partition_guards', guards)

    @property
    def output_keys(self) -> tuple[ArtifactKey, ...]:
        return tuple(write.key for write in self.writes)


@dataclass(frozen=True)
class ArtifactSnapshot:
    records: Mapping[ArtifactKey, ArtifactRecord] = field(default_factory=dict)
    partition_sets: Mapping[ArtifactKey, PartitionSet] = field(default_factory=dict)

    def __post_init__(self) -> None:
        records = dict(self.records)
        partition_sets = dict(self.partition_sets)
        for key, record in records.items():
            if not isinstance(key, ArtifactKey) or not isinstance(record, ArtifactRecord):
                raise TypeError('records must map ArtifactKey to ArtifactRecord')
            if record.ref.key != key:
                raise DefinitionError('artifact record key must match its ref')
        for key, partitions in partition_sets.items():
            if not isinstance(key, ArtifactKey) or key.partition_key:
                raise TypeError('partition_sets keys must be scalar ArtifactKey values')
            if not isinstance(partitions, PartitionSet):
                raise TypeError('partition_sets values must be PartitionSet')
            if key not in records:
                raise DefinitionError('partition set must reference a visible artifact record')

        object.__setattr__(self, 'records', MappingProxyType(records))
        object.__setattr__(self, 'partition_sets', MappingProxyType(partition_sets))

    def effective_records(self) -> Mapping[ArtifactKey, ArtifactRecord]:
        effective = dict(self.records)
        while _drop_stale(effective):
            pass
        return MappingProxyType(effective)


def _drop_stale(records: dict[ArtifactKey, ArtifactRecord]) -> bool:
    stale = tuple(
        key for key, record in records.items()
        if any(records.get(ref.key) is None or records[ref.key].ref != ref
               for ref in record.input_refs)
    )
    for key in stale:
        del records[key]
    return bool(stale)


def merge_refs(*groups: Iterable[ArtifactRef]) -> tuple[ArtifactRef, ...]:
    refs: dict[ArtifactKey, ArtifactRef] = {}

    for group in groups:
        for ref in group:
            if not isinstance(ref, ArtifactRef):
                raise TypeError('artifact refs must contain ArtifactRef values')
            previous = refs.get(ref.key)
            if previous is not None and previous != ref:
                raise DefinitionError(f'conflicting refs for artifact key {ref.key}')
            refs[ref.key] = ref

    return tuple(sorted(refs.values()))


def failure_key(output_key: ArtifactKey) -> ArtifactKey:
    if not isinstance(output_key, ArtifactKey):
        raise TypeError('output_key must be ArtifactKey')
    if is_failure_key(output_key):
        raise DefinitionError('cannot create a failure key for an internal failure artifact')
    return ArtifactKey(f'{_FAILURE_PREFIX}{output_key.artifact_id}', output_key.partition_key)


def is_failure_key(key: ArtifactKey) -> bool:
    if not isinstance(key, ArtifactKey):
        raise TypeError('key must be ArtifactKey')
    return key.artifact_id.startswith(_FAILURE_PREFIX)


__all__ = [
    'ArtifactCommit', 'ArtifactDraft', 'ArtifactKey', 'ArtifactRecord', 'ArtifactRef',
    'ArtifactSnapshot', 'PartitionGuard', 'PartitionSet',
    'failure_key', 'is_failure_key', 'merge_refs', 'RUN_CONFIGURATION_ARTIFACT_ID',
]
