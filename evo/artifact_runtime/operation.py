from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import logging
from types import MappingProxyType
from typing import Literal, Protocol, Self, TypeVar, cast, get_args

from .artifact import (
    ArtifactCommit,
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    ArtifactDraft,
    PartitionGuard,
    PartitionSet,
    is_failure_key,
    merge_refs,
)
from .errors import (
    DefinitionError,
    _integer,
    _known,
    _positive_number,
    _string,
    _text,
    _tuple_of,
    _unique,
)
from .state import CaseFailure, EventLevel, EventStatus, OperationEvent


BindingMode = Literal['one', 'each', 'keyed', 'all']
OutputMode = Literal['scalar', 'partitioned']
ExecutionMode = Literal['cooperative', 'isolated']
EventReporter = Callable[[OperationEvent], Awaitable[None]]
_RECORDER: ContextVar[Callable[[OperationEvent], None] | None] = ContextVar('operation_recorder', default=None)
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundAggregate:
    partition_set_ref: ArtifactRef
    member_refs: tuple[ArtifactRef, ...]
    failure_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.partition_set_ref, ArtifactRef):
            raise TypeError('partition_set_ref must be ArtifactRef')
        if self.partition_set_ref.key.partition_key:
            raise DefinitionError('partition_set_ref must identify a scalar artifact')

        member_refs = _tuple_of(self.member_refs, ArtifactRef,
                                'member_refs must contain ArtifactRef values')
        if any(not ref.key.partition_key for ref in member_refs):
            raise DefinitionError('all input refs must identify partitioned artifacts')
        _unique(member_refs, 'all input refs must have unique partition keys',
                key=lambda ref: ref.key.partition_key)

        failure_refs = merge_refs(self.failure_refs)
        if any(not is_failure_key(ref.key) for ref in failure_refs):
            raise DefinitionError('all failure refs must identify internal failure artifacts')
        member_cases = {ref.key.partition_key for ref in member_refs}
        if any(ref.key.partition_key in member_cases for ref in failure_refs):
            raise DefinitionError('all input cannot contain both a value and failure for one case')

        object.__setattr__(self, 'member_refs', member_refs)
        object.__setattr__(self, 'failure_refs', failure_refs)


@dataclass(frozen=True)
class AggregateValue(Mapping[str, object]):
    entries: tuple[tuple[str, object], ...]
    failure_items: tuple[tuple[str, CaseFailure], ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(self.entries)
        failures = tuple(self.failure_items)
        for case_id, _ in (*entries, *failures):
            _text(case_id, 'aggregate case id')
        _unique(entries, 'aggregate values must have unique case ids', key=lambda item: item[0])
        if not all(isinstance(failure, CaseFailure) for _, failure in failures):
            raise TypeError('aggregate failures must contain CaseFailure values')
        _unique(failures, 'aggregate failures must have unique case ids',
                key=lambda item: item[0])
        if {case_id for case_id, _ in entries} & {case_id for case_id, _ in failures}:
            raise DefinitionError('aggregate case cannot be both successful and failed')
        object.__setattr__(self, 'entries', entries)
        object.__setattr__(self, 'failure_items', failures)

    def __getitem__(self, case_id: str) -> object:
        for current, value in self.entries:
            if current == case_id:
                return value
        raise KeyError(case_id)

    def __iter__(self) -> Iterator[str]:
        return (case_id for case_id, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def failures(self) -> Mapping[str, CaseFailure]:
        return MappingProxyType(dict(self.failure_items))


BoundInput = ArtifactRef | BoundAggregate


@dataclass(frozen=True)
class InputSpec:
    artifact_id: str
    mode: BindingMode
    partition_set_id: str = ''

    def __post_init__(self) -> None:
        _text(self.artifact_id, 'input artifact_id')
        _known(self.mode, get_args(BindingMode), 'input binding mode')

        if self.mode in {'each', 'all'}:
            _text(self.partition_set_id, 'partition_set_id')
        elif self.partition_set_id:
            raise DefinitionError(f'{self.mode} input cannot declare partition_set_id')

    def validate_value(self, name: str, value: BoundInput, partition_key: str) -> None:
        if self.mode == 'one':
            if not isinstance(value, ArtifactRef):
                raise TypeError(f'{name} one binding must contain ArtifactRef')
            if value.key != ArtifactKey.scalar(self.artifact_id):
                raise DefinitionError(f'{name} ref does not match its one binding')
            return

        if self.mode == 'all':
            if not isinstance(value, BoundAggregate):
                raise TypeError(f'{name} all binding must contain BoundAggregate')
            if value.partition_set_ref.key != ArtifactKey.scalar(self.partition_set_id):
                raise DefinitionError(f'{name} partition set does not match its all binding')
            if any(ref.key.artifact_id != self.artifact_id for ref in value.member_refs):
                raise DefinitionError(f'{name} refs do not match their all binding')
            return

        if not isinstance(value, ArtifactRef):
            raise TypeError(f'{name} {self.mode} binding must contain ArtifactRef')
        if value.key != ArtifactKey.partition(self.artifact_id, partition_key):
            raise DefinitionError(f'{name} ref does not match its {self.mode} binding')


def one(artifact_id: str) -> InputSpec:
    return InputSpec(artifact_id, 'one')


def each(artifact_id: str, *, over: str) -> InputSpec:
    return InputSpec(artifact_id, 'each', over)


def keyed(artifact_id: str) -> InputSpec:
    return InputSpec(artifact_id, 'keyed')


def all_items(artifact_id: str, *, over: str) -> InputSpec:
    return InputSpec(artifact_id, 'all', over)


@dataclass(frozen=True)
class OutputSpec:
    artifact_id: str
    mode: OutputMode = 'scalar'
    partition_set_id: str = ''

    def __post_init__(self) -> None:
        _text(self.artifact_id, 'output artifact_id')
        _known(self.mode, get_args(OutputMode), 'output mode')

        if self.mode == 'scalar':
            if self.partition_set_id:
                raise DefinitionError('scalar output cannot declare partition_set_id')
        elif self.partition_set_id:
            _text(self.partition_set_id, 'partition_set_id')

    def key_for(self, partition_key: str) -> ArtifactKey:
        if self.mode == 'partitioned':
            return ArtifactKey.partition(self.artifact_id, partition_key)
        return ArtifactKey.scalar(self.artifact_id)


def scalar(artifact_id: str) -> OutputSpec:
    return OutputSpec(artifact_id)


def partitioned(artifact_id: str, *, over: str = '') -> OutputSpec:
    return OutputSpec(artifact_id, 'partitioned', over)


@dataclass(frozen=True)
class OperationSpec:
    op_id: str
    inputs: Mapping[str, InputSpec]
    outputs: Mapping[str, OutputSpec]
    execution: ExecutionMode = 'isolated'
    max_concurrency: int = 1
    timeout: float | None = 300.0
    driver_input: str | None = field(init=False)

    def __post_init__(self) -> None:
        _text(self.op_id, 'op_id')
        inputs = dict(self.inputs)
        outputs = dict(self.outputs)
        if not outputs:
            raise DefinitionError('operation must declare at least one output')

        for name, binding in inputs.items():
            _text(name, 'input name')
            if not isinstance(binding, InputSpec):
                raise TypeError('operation inputs must contain InputSpec values')
        _unique(tuple(inputs.values()), 'operation input bindings must be unique',
                key=lambda binding: (binding.artifact_id, binding.mode))

        for name, output in outputs.items():
            _text(name, 'output name')
            if not isinstance(output, OutputSpec):
                raise TypeError('operation outputs must contain OutputSpec values')
        _unique(tuple(outputs.values()), 'operation output artifact ids must be unique',
                key=lambda output: output.artifact_id)

        drivers = [name for name, binding in inputs.items() if binding.mode == 'each']
        if len(drivers) > 1:
            raise DefinitionError('operation supports one driving each input')
        driver_input = drivers[0] if drivers else None

        if any(binding.mode == 'keyed' for binding in inputs.values()) and driver_input is None:
            raise DefinitionError('keyed inputs require one driving each input')

        self._validate_outputs(inputs, outputs, driver_input)

        _known(self.execution, get_args(ExecutionMode), 'execution mode')
        _integer(self.max_concurrency, 'max_concurrency', minimum=1)
        if self.timeout is not None:
            _positive_number(self.timeout, 'timeout')

        object.__setattr__(self, 'inputs', MappingProxyType(inputs))
        object.__setattr__(self, 'outputs', MappingProxyType(outputs))
        object.__setattr__(self, 'driver_input', driver_input)

    @staticmethod
    def _validate_outputs(inputs: Mapping[str, InputSpec], outputs: Mapping[str, OutputSpec], driver_input: str | None
                          ) -> None:
        if driver_input is not None:
            if not all(output.mode == 'partitioned' for output in outputs.values()):
                raise DefinitionError('partitioned invocation must use only partitioned outputs')

            partition_set_id = inputs[driver_input].partition_set_id
            if any(
                output.partition_set_id and output.partition_set_id != partition_set_id
                for output in outputs.values()
            ):
                raise DefinitionError('partitioned outputs must use the driving partition set')
            return

        partition_sets = {
            output.partition_set_id
            for output in outputs.values()
            if output.mode == 'partitioned'
        }
        if '' in partition_sets:
            raise DefinitionError('batch partitioned outputs must declare partition_set_id')

        scalar_outputs = {
            output.artifact_id
            for output in outputs.values()
            if output.mode == 'scalar'
        }
        missing = sorted(partition_sets - scalar_outputs)
        if missing:
            joined = ', '.join(missing)
            raise DefinitionError(
                f'batch partitioned outputs must also output their PartitionSet: {joined}'
            )

    @property
    def partition_set_id(self) -> str:
        if self.driver_input is None:
            return ''
        return self.inputs[self.driver_input].partition_set_id


@dataclass(frozen=True)
class OperationContext:
    run_id: str
    invocation_id: str
    partition_key: str = ''
    _reporter: EventReporter | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _text(self.run_id, 'run_id')
        _text(self.invocation_id, 'invocation_id')
        _string(self.partition_key, 'partition_key')

    async def record(self, event_type: str, message: str = '', *, level: EventLevel = 'info',
                     status: EventStatus | None = None, data: Mapping[str, object] | None = None,
                     current: int | None = None, total: int | None = None, **fields: object) -> None:
        event = OperationEvent(event_type, level, status, message, {**(data or {}), **fields}, current, total)
        if self._reporter is not None:
            await self._reporter(event)


@dataclass(frozen=True)
class OperationResult:
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        values = dict(self.values)
        for name in values:
            _text(name, 'operation result name')
        object.__setattr__(self, 'values', MappingProxyType(values))

    def validate_for(self, spec: OperationSpec) -> Self:
        if not isinstance(spec, OperationSpec):
            raise TypeError('spec must be OperationSpec')
        if set(self.values) != set(spec.outputs):
            raise DefinitionError(f'{spec.op_id} result names must match declared outputs')
        return self


class Operation(Protocol):
    spec: OperationSpec
    __module__: str
    __qualname__: str

    async def __call__(self, ctx: OperationContext, **inputs: object) -> OperationResult:
        ...


OperationFunction = Callable[..., Awaitable[OperationResult]]
F = TypeVar('F', bound=OperationFunction)


def operation(*, op_id: str, inputs: Mapping[str, InputSpec], outputs: Mapping[str, OutputSpec],
              execution: ExecutionMode = 'isolated', max_concurrency: int = 1, timeout: float | None = 300.0
              ) -> Callable[[F], F]:
    spec = OperationSpec(op_id, inputs, outputs, execution, max_concurrency, timeout)

    def decorate(function: F) -> F:
        if not inspect.iscoroutinefunction(function):
            raise DefinitionError(f'{op_id} must be declared with async def')
        if spec.execution == 'isolated' and '<locals>' in function.__qualname__:
            raise DefinitionError(f'{op_id} isolated operation must be declared at module scope')
        if hasattr(function, 'spec'):
            raise DefinitionError(f'{op_id} function already declares an operation spec')

        _validate_signature(function, spec)
        function.spec = spec  # type: ignore[attr-defined]
        return cast(F, function)

    return decorate


def record_process(function: F) -> F:
    if not inspect.iscoroutinefunction(function):
        raise DefinitionError('record_process can only decorate async functions')

    @wraps(function)
    async def wrapped(ctx: OperationContext, *args: object, **kwargs: object) -> OperationResult:
        loop = asyncio.get_running_loop()
        pending = []

        def schedule(event: OperationEvent) -> None:
            if ctx._reporter is not None:
                pending.append(asyncio.run_coroutine_threadsafe(ctx._reporter(event), loop))

        token = _RECORDER.set(schedule)
        try:
            return await function(ctx, *args, **kwargs)
        finally:
            _RECORDER.reset(token)
            for result in await asyncio.gather(
                *(asyncio.wrap_future(future) for future in pending), return_exceptions=True,
            ):
                if isinstance(result, BaseException):
                    _LOG.warning('operation event could not be recorded: %s', result)

    return cast(F, wrapped)


def record_event(event_type: str, message: str = '', *, level: EventLevel = 'info', status: EventStatus | None = None,
                 data: Mapping[str, object] | None = None, current: int | None = None, total: int | None = None,
                 **fields: object) -> None:
    recorder = _RECORDER.get()
    if recorder is None:
        raise DefinitionError('record_event requires an active @record_process operation')
    recorder(OperationEvent(event_type, level, status, message, {**(data or {}), **fields}, current, total))


def _validate_signature(function: OperationFunction, spec: OperationSpec) -> None:
    parameters = tuple(inspect.signature(function).parameters.values())
    if not parameters or parameters[0].name != 'ctx':
        raise DefinitionError(f'{spec.op_id} first parameter must be named ctx')
    if parameters[0].kind not in {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }:
        raise DefinitionError(f'{spec.op_id} ctx parameter must be positional')

    arguments = parameters[1:]
    if any(
        parameter.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for parameter in arguments
    ):
        raise DefinitionError(f'{spec.op_id} must not use variadic input parameters')
    if {parameter.name for parameter in arguments} != set(spec.inputs):
        raise DefinitionError(f'{spec.op_id} parameters must match declared input names')
    if any(parameter.default is not inspect.Parameter.empty for parameter in arguments):
        raise DefinitionError(f'{spec.op_id} input parameters must not declare defaults')


@dataclass(frozen=True)
class OperationInvocation:
    operation: Operation
    inputs: Mapping[str, BoundInput]
    expected_heads: Mapping[ArtifactKey, ArtifactRef | None] = field(default_factory=dict)
    partition_key: str = ''
    retry_request_id: str = ''
    invocation_id: str = field(init=False)
    output_keys: Mapping[str, ArtifactKey | None] = field(init=False)
    partition_set_key: ArtifactKey | None = field(init=False)

    def __post_init__(self) -> None:
        if not callable(self.operation) or not isinstance(
            getattr(self.operation, 'spec', None), OperationSpec
        ):
            raise TypeError('operation must be a declared Operation')

        inputs = dict(self.inputs)
        if set(inputs) != set(self.operation.spec.inputs):
            raise DefinitionError('invocation inputs must match operation inputs')

        _string(self.partition_key, 'partition_key')
        _string(self.retry_request_id, 'retry_request_id')
        has_driver = self.operation.spec.driver_input is not None
        if has_driver != bool(self.partition_key):
            raise DefinitionError('partition_key must be set exactly for partitioned invocation')

        output_keys = {
            name: (
                None
                if output.mode == 'partitioned' and not self.partition_key
                else output.key_for(self.partition_key)
            )
            for name, output in self.operation.spec.outputs.items()
        }
        concrete_outputs = tuple(key for key in output_keys.values() if key is not None)
        if len(set(concrete_outputs)) != len(concrete_outputs):
            raise DefinitionError('invocation output artifact keys must be unique')

        expected_heads = self._validated_expected_heads(
            self.expected_heads,
            output_keys,
        )
        for name, binding in self.operation.spec.inputs.items():
            binding.validate_value(name, inputs[name], self.partition_key)

        partition_set_key = (
            ArtifactKey.scalar(self.operation.spec.partition_set_id)
            if has_driver
            else None
        )
        invocation_id = _invocation_id(
            self.operation.spec.op_id,
            inputs,
            output_keys,
            self.retry_request_id,
        )

        object.__setattr__(self, 'inputs', MappingProxyType(inputs))
        object.__setattr__(self, 'expected_heads', MappingProxyType(expected_heads))
        object.__setattr__(self, 'invocation_id', invocation_id)
        object.__setattr__(self, 'output_keys', MappingProxyType(output_keys))
        object.__setattr__(self, 'partition_set_key', partition_set_key)

    def _validated_expected_heads(self, values: Mapping[ArtifactKey, ArtifactRef | None],
                                  output_keys: Mapping[str, ArtifactKey | None]
                                  ) -> dict[ArtifactKey, ArtifactRef | None]:
        expected_heads = dict(values)
        concrete_outputs = {key for key in output_keys.values() if key is not None}
        dynamic_outputs = {
            output.artifact_id
            for name, output in self.operation.spec.outputs.items()
            if output_keys[name] is None
        }

        for key in concrete_outputs:
            expected_heads.setdefault(key, None)

        for key, ref in expected_heads.items():
            if not isinstance(key, ArtifactKey):
                raise TypeError('expected_heads keys must be ArtifactKey values')
            if key not in concrete_outputs and key.artifact_id not in dynamic_outputs:
                raise DefinitionError('expected_heads must describe invocation outputs')
            if key.artifact_id in dynamic_outputs and not key.partition_key:
                raise DefinitionError('dynamic output heads must identify partitioned artifacts')
            if ref is not None and not isinstance(ref, ArtifactRef):
                raise TypeError('expected_heads values must be ArtifactRef or None')
            if ref is not None and ref.key != key:
                raise DefinitionError('expected output head must identify its artifact key')

        return expected_heads

    def value_refs(self) -> tuple[ArtifactRef, ...]:
        return _bound_refs(self.inputs, lineage=False)

    def lineage_refs(self) -> tuple[ArtifactRef, ...]:
        return _bound_refs(self.inputs, lineage=True)

    def bind_values(self, values: Mapping[ArtifactRef, object]) -> Mapping[str, object]:
        bound: dict[str, object] = {}
        for name, value in self.inputs.items():
            if isinstance(value, ArtifactRef):
                bound[name] = values[value]
            else:
                successful = tuple(
                    (ref.key.partition_key, values[ref])
                    for ref in value.member_refs
                )
                failures: dict[str, CaseFailure] = {}
                for ref in value.failure_refs:
                    failure = values[ref]
                    if not isinstance(failure, CaseFailure):
                        raise DefinitionError('failure artifact must contain CaseFailure')
                    failures.setdefault(failure.case_id, failure)
                bound[name] = AggregateValue(successful, tuple(failures.items()))
        return MappingProxyType(bound)

    def is_current(self, records: Mapping[ArtifactKey, ArtifactRecord],
                   effective_records: Mapping[ArtifactKey, ArtifactRecord],
                   partition_sets: Mapping[ArtifactKey, PartitionSet]) -> bool:
        for ref in self.lineage_refs():
            current = effective_records.get(ref.key)
            if current is None or current.ref != ref:
                return False

        for key, expected in self.expected_heads.items():
            current = records.get(key)
            if (None if current is None else current.ref) != expected:
                return False

        for artifact_id in {
            self.operation.spec.outputs[name].artifact_id
            for name, key in self.output_keys.items()
            if key is None
        }:
            expected_keys = {
                key for key in self.expected_heads if key.artifact_id == artifact_id
            }
            current_keys = {
                key for key in records if key.artifact_id == artifact_id
            }
            if current_keys != expected_keys:
                return False

        if self.partition_set_key is not None:
            partitions = partition_sets.get(self.partition_set_key)
            if partitions is None or self.partition_key not in partitions:
                return False

        return True

    def artifact_commit(self, result: OperationResult) -> ArtifactCommit:
        result.validate_for(self.operation.spec)
        input_refs = self.lineage_refs()
        writes: list[ArtifactDraft] = []
        partition_values: dict[str, tuple[str, ...]] = {}
        partition_sets: dict[str, object] = {}

        for name, output in self.operation.spec.outputs.items():
            value = result.values[name]
            key = self.output_keys[name]
            if key is not None:
                writes.append(ArtifactDraft(key, value, input_refs))
                if output.mode == 'scalar':
                    partition_sets[output.artifact_id] = value
                continue

            if not isinstance(value, Mapping):
                raise DefinitionError(f'{name} batch partitioned output must be a mapping')

            partition_items = tuple(value.items())
            partition_keys = tuple(key for key, _ in partition_items)
            for partition_key, partition_value in partition_items:
                _text(partition_key, f'{name} partition key')
                writes.append(ArtifactDraft(
                    ArtifactKey.partition(output.artifact_id, partition_key),
                    partition_value,
                    input_refs,
                ))

            previous_keys = partition_values.setdefault(
                output.partition_set_id,
                partition_keys,
            )
            if previous_keys != partition_keys:
                raise DefinitionError(
                    f'partitioned outputs over {output.partition_set_id} must share keys'
                )

        for partition_set_id, partition_keys in partition_values.items():
            partitions = partition_sets[partition_set_id]
            if not isinstance(partitions, PartitionSet):
                raise DefinitionError(f'{partition_set_id} output must be PartitionSet')
            if partitions.keys != partition_keys:
                raise DefinitionError(
                    f'partitioned output keys must match {partition_set_id}'
                )

        guards = (
            ()
            if self.partition_set_key is None
            else (PartitionGuard(self.partition_set_key, self.partition_key),)
        )
        return ArtifactCommit(
            self.invocation_id,
            f'operation:{self.operation.spec.op_id}',
            tuple(writes),
            {write.key: self.expected_heads.get(write.key) for write in writes},
            guards,
        )


def _bound_refs(inputs: Mapping[str, BoundInput], *, lineage: bool) -> tuple[ArtifactRef, ...]:
    refs: list[ArtifactRef] = []
    for value in inputs.values():
        if isinstance(value, ArtifactRef):
            refs.append(value)
        else:
            if lineage:
                refs.append(value.partition_set_ref)
            refs.extend(value.member_refs)
            refs.extend(value.failure_refs)
    return merge_refs(refs)


def _invocation_id(op_id: str, inputs: Mapping[str, BoundInput], outputs: Mapping[str, ArtifactKey | None],
                   retry_request_id: str) -> str:
    payload = {
        'operation': op_id,
        'inputs': [
            [name, *_bound_identity(value)]
            for name, value in sorted(inputs.items())
        ],
        'outputs': [
            [
                name,
                None if key is None else key.artifact_id,
                None if key is None else key.partition_key,
            ]
            for name, key in sorted(outputs.items())
        ],
        'retry_request_id': retry_request_id,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()
    ).hexdigest()
    return f'{op_id}:{digest}'


def _bound_identity(value: BoundInput) -> list[object]:
    if isinstance(value, ArtifactRef):
        return [
            'ref',
            value.key.artifact_id,
            value.key.partition_key,
            value.version,
        ]
    return [
        'all',
        value.partition_set_ref.key.artifact_id,
        value.partition_set_ref.version,
        [
            [ref.key.artifact_id, ref.key.partition_key, ref.version]
            for ref in value.member_refs
        ],
        [
            [ref.key.artifact_id, ref.key.partition_key, ref.version]
            for ref in value.failure_refs
        ],
    ]


__all__ = [
    'AggregateValue', 'BindingMode', 'BoundAggregate', 'BoundInput', 'ExecutionMode',
    'InputSpec', 'Operation', 'OperationContext', 'OperationInvocation', 'OperationResult',
    'EventReporter', 'OperationSpec', 'OutputMode', 'OutputSpec', 'all_items', 'each',
    'keyed', 'one', 'operation', 'partitioned', 'record_event', 'record_process', 'scalar',
]
