from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

import networkx as nx

from .artifact import (
    RUN_CONFIGURATION_ARTIFACT_ID,
    ArtifactCommit,
    ArtifactKey,
    ArtifactRecord,
    ArtifactRef,
    ArtifactSnapshot,
    PartitionGuard,
    PartitionSet,
    _drop_stale,
    failure_key,
    is_failure_key,
    merge_refs,
)
from .errors import DefinitionError, PlanningError, _tuple_of
from .operation import (
    BoundAggregate,
    BoundInput,
    Operation,
    OperationInvocation,
    OperationSpec,
    _bound_refs,
)
from .state import (
    ArtifactRetryRequest,
    AttemptSnapshot,
    CaseFailure,
    InvocationSnapshot,
    RunStatus,
    RuntimeErrorInfo,
    RuntimeProgress,
    RuntimeSnapshot,
)


@dataclass(frozen=True)
class PlanReady:
    view: ArtifactSnapshot
    invocations: tuple[OperationInvocation, ...]
    failure_refs: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class PlanAwaiting:
    view: ArtifactSnapshot
    artifact_keys: tuple[ArtifactKey, ...]
    failure_refs: tuple[ArtifactRef, ...] = ()


@dataclass(frozen=True)
class PlanComplete:
    view: ArtifactSnapshot
    failure_refs: tuple[ArtifactRef, ...] = ()


PlanningResult: TypeAlias = PlanReady | PlanAwaiting | PlanComplete


@dataclass(frozen=True)
class RuntimeDefinition:
    operations: tuple[Operation, ...]
    artifact_modes: Mapping[str, str]
    partition_set_by_artifact: Mapping[str, str]
    producer_by_artifact: Mapping[str, Operation]
    terminal_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        terminals = tuple(self.terminal_artifact_ids)
        if not operations:
            raise DefinitionError('runtime definition requires at least one operation')
        if not terminals:
            raise DefinitionError('runtime definition requires at least one terminal artifact')
        object.__setattr__(self, 'operations', operations)
        for name in ('artifact_modes', 'partition_set_by_artifact', 'producer_by_artifact'):
            value = MappingProxyType(dict(getattr(self, name)))
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'terminal_artifact_ids', terminals)

    @property
    def partition_set_ids(self) -> frozenset[str]:
        return frozenset(self.partition_set_by_artifact.values())

    def validate_commit(self, commit: ArtifactCommit) -> None:
        if not isinstance(commit, ArtifactCommit):
            raise TypeError('commit must be ArtifactCommit')

        partition_sets = {
            write.key: write.value
            for write in commit.writes
            if isinstance(write.value, PartitionSet)
        }
        guards = set(commit.partition_guards)
        for guard in guards:
            if guard.partition_set_key.artifact_id not in self.partition_set_ids:
                raise DefinitionError(
                    f'unknown partition set: {guard.partition_set_key.artifact_id}'
                )

        for write in commit.writes:
            artifact_id = write.key.artifact_id
            mode = self.artifact_modes.get(artifact_id)
            if mode is None:
                raise DefinitionError(f'unknown artifact: {artifact_id}')
            if (mode == 'partitioned') != bool(write.key.partition_key):
                raise DefinitionError(f'{artifact_id} requires a {mode} artifact key')

            is_partition_set = artifact_id in self.partition_set_ids
            if is_partition_set != isinstance(write.value, PartitionSet):
                expected = 'PartitionSet' if is_partition_set else 'ordinary artifact value'
                raise DefinitionError(f'{artifact_id} requires {expected}')
            if mode != 'partitioned':
                continue

            set_key = ArtifactKey.scalar(self.partition_set_by_artifact[artifact_id])
            current_commit_set = partition_sets.get(set_key)
            if (current_commit_set is not None
                    and write.key.partition_key not in current_commit_set):
                raise DefinitionError(
                    f'{write.key} is not present in the committed PartitionSet'
                )
            if (current_commit_set is None
                    and PartitionGuard(set_key, write.key.partition_key) not in guards):
                raise DefinitionError(f'{write.key} requires partition membership protection')


def compile_operations(operations: Sequence[Operation]) -> RuntimeDefinition:
    declared = tuple(operations)
    if not declared:
        raise DefinitionError('at least one operation is required')

    by_id: dict[str, Operation] = {}
    artifact_modes: dict[str, str] = {RUN_CONFIGURATION_ARTIFACT_ID: 'scalar'}
    producer_by_artifact: dict[str, Operation] = {}
    partition_set_by_artifact: dict[str, str] = {}

    def assign(target: dict[str, str], key: str, value: str, label: str) -> None:
        previous = target.setdefault(key, value)
        if previous != value:
            raise DefinitionError(f'{label} uses both {previous} and {value}')

    for operation in declared:
        spec = getattr(operation, 'spec', None)
        if not callable(operation) or not isinstance(spec, OperationSpec):
            raise TypeError('operations must contain declared Operation callables')
        if spec.op_id in by_id:
            raise DefinitionError(f'duplicate operation id: {spec.op_id}')
        by_id[spec.op_id] = operation

        for binding in spec.inputs.values():
            mode = 'scalar' if binding.mode == 'one' else 'partitioned'
            assign(artifact_modes, binding.artifact_id, mode, f'artifact {binding.artifact_id}')
            if binding.mode in {'each', 'all'}:
                assign(artifact_modes, binding.partition_set_id, 'scalar',
                       f'artifact {binding.partition_set_id}')
                assign(partition_set_by_artifact, binding.artifact_id,
                       binding.partition_set_id, f'partitioned artifact {binding.artifact_id}')

        if spec.driver_input is not None:
            for binding in spec.inputs.values():
                if binding.mode == 'keyed':
                    assign(partition_set_by_artifact, binding.artifact_id,
                           spec.partition_set_id, f'partitioned artifact {binding.artifact_id}')

        for output in spec.outputs.values():
            if output.artifact_id == RUN_CONFIGURATION_ARTIFACT_ID:
                raise DefinitionError('run configuration is an external runtime input')
            assign(artifact_modes, output.artifact_id, output.mode,
                   f'artifact {output.artifact_id}')
            previous = producer_by_artifact.get(output.artifact_id)
            if previous is not None:
                raise DefinitionError(
                    f'artifact {output.artifact_id} has multiple writers: '
                    f'{previous.spec.op_id}, {spec.op_id}'
                )
            producer_by_artifact[output.artifact_id] = operation
            if output.mode == 'partitioned':
                partition_set_id = output.partition_set_id or spec.partition_set_id
                assign(artifact_modes, partition_set_id, 'scalar',
                       f'artifact {partition_set_id}')
                assign(partition_set_by_artifact, output.artifact_id,
                       partition_set_id, f'partitioned artifact {output.artifact_id}')

    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    for operation in declared:
        dependencies = {binding.artifact_id for binding in operation.spec.inputs.values()}
        dependencies.update(
            binding.partition_set_id
            for binding in operation.spec.inputs.values()
            if binding.mode in {'each', 'all'}
        )
        for artifact_id in dependencies:
            producer = producer_by_artifact.get(artifact_id)
            if producer is not None:
                graph.add_edge(producer.spec.op_id, operation.spec.op_id)

    try:
        order = tuple(nx.lexicographical_topological_sort(graph, key=str))
    except nx.NetworkXUnfeasible as exc:
        edges = nx.find_cycle(graph)
        cycle = ' -> '.join((edges[0][0], *(target for _, target in edges)))
        raise DefinitionError(f'operation dependencies must be acyclic: {cycle}') from exc

    ordered = tuple(by_id[op_id] for op_id in order)
    consumed = {
        binding.artifact_id
        for operation in ordered
        for binding in operation.spec.inputs.values()
    }
    consumed.update(
        binding.partition_set_id
        for operation in ordered
        for binding in operation.spec.inputs.values()
        if binding.mode in {'each', 'all'}
    )
    structural_sets = frozenset(partition_set_by_artifact.values())
    terminal_set = set(producer_by_artifact) - consumed - structural_sets
    terminals = tuple(
        output.artifact_id
        for operation in ordered
        for output in operation.spec.outputs.values()
        if output.artifact_id in terminal_set
    )
    return RuntimeDefinition(
        ordered,
        artifact_modes,
        partition_set_by_artifact,
        producer_by_artifact,
        terminals,
    )


def plan_next(definition: RuntimeDefinition, artifacts: ArtifactSnapshot, retries: Iterable[ArtifactRetryRequest] = ()
              ) -> PlanningResult:
    if not isinstance(definition, RuntimeDefinition):
        raise TypeError('definition must be RuntimeDefinition')
    if not isinstance(artifacts, ArtifactSnapshot):
        raise TypeError('artifacts must be ArtifactSnapshot')
    pending = _tuple_of(
        retries, ArtifactRetryRequest, 'retries must contain ArtifactRetryRequest values'
    )
    if any(request.status != 'pending' for request in pending):
        raise DefinitionError('planner retries must be pending')

    effective = dict(artifacts.effective_records())
    changed = True
    while changed:
        changed = _remove_inactive_partitions(definition, artifacts, effective)
        changed |= _drop_stale(effective)
        partition_sets = _effective_partition_sets(artifacts, effective)
        for operation in definition.operations:
            changed |= _validate_outputs(operation, effective, partition_sets)
    partition_sets = _effective_partition_sets(artifacts, effective)
    view = ArtifactSnapshot(effective, partition_sets)
    planner = _DemandPlanner(definition, artifacts, view)

    satisfied = True
    if pending:
        retry_invocations: set[tuple[str, str]] = set()
        for request in pending:
            operation = definition.producer_by_artifact.get(request.artifact_key.artifact_id)
            if operation is None:
                raise PlanningError(f'retry target has no producer: {request.artifact_key}')
            identity = (
                operation.spec.op_id,
                request.artifact_key.partition_key if operation.spec.driver_input else '',
            )
            if identity in retry_invocations:
                raise PlanningError('one invocation cannot satisfy multiple retry requests')
            retry_invocations.add(identity)
            satisfied &= planner.require_retry(request)
    else:
        for artifact_id in definition.terminal_artifact_ids:
            satisfied &= planner.require_family(artifact_id)

    invocations = planner.ready_invocations()
    failures = merge_refs(
        planner.failure_refs(),
        (
            record.ref
            for key, record in view.records.items()
            if is_failure_key(key) and record.producer.startswith('runtime:failure:')
        ),
    )
    if invocations:
        return PlanReady(view, invocations, failures)
    if planner.awaiting:
        return PlanAwaiting(view, tuple(sorted(planner.awaiting)), failures)
    if satisfied:
        return PlanComplete(view, failures)
    raise PlanningError('terminal artifact demand cannot be resolved')


def obsolete_retries(definition: RuntimeDefinition, artifacts: ArtifactSnapshot, retries: Iterable[ArtifactRetryRequest]
                     ) -> tuple[ArtifactRetryRequest, ...]:
    requests = tuple(retries)
    view = plan_next(definition, artifacts).view
    return tuple(
        request
        for request in requests
        if (
            view.records.get(request.artifact_key) is None
            or view.records[request.artifact_key].ref != request.base_ref
        )
    )


class _DemandPlanner:
    def __init__(self, definition: RuntimeDefinition, artifacts: ArtifactSnapshot, view: ArtifactSnapshot) -> None:
        self.definition = definition
        self.artifacts = artifacts
        self.view = view
        self.awaiting: set[ArtifactKey] = set()
        self.failed: dict[ArtifactKey, tuple[ArtifactRef, ...]] = {}
        self._visited: set[tuple[str, str]] = set()
        self._ready: dict[tuple[str, str], OperationInvocation] = {}
        self._operation_order = {
            operation.spec.op_id: index
            for index, operation in enumerate(definition.operations)
        }

    def require_retry(self, request: ArtifactRetryRequest) -> bool:
        current = self.view.records.get(request.artifact_key)
        if current is None or current.ref != request.base_ref:
            raise PlanningError(
                f'retry base is no longer effective: {request.artifact_key}'
            )
        return self.require_key(request.artifact_key, request.request_id)

    def require_family(self, artifact_id: str) -> bool:
        if self.definition.artifact_modes[artifact_id] == 'scalar':
            return self.require_key(ArtifactKey.scalar(artifact_id))

        set_key = ArtifactKey.scalar(self.definition.partition_set_by_artifact[artifact_id])
        if not self.require_key(set_key):
            return False
        partitions = self.view.partition_sets.get(set_key)
        if partitions is None:
            return False
        satisfied = True
        for partition_key in partitions.keys:
            satisfied &= self.require_key(ArtifactKey.partition(artifact_id, partition_key))
        return satisfied

    def require_key(self, artifact_key: ArtifactKey, retry_request_id: str = '') -> bool:
        if not retry_request_id and (
            artifact_key in self.view.records or artifact_key in self.failed
        ):
            return True
        operation = self.definition.producer_by_artifact.get(artifact_key.artifact_id)
        if operation is None:
            self.awaiting.add(artifact_key)
            return False
        partition_key = artifact_key.partition_key if operation.spec.driver_input else ''
        identity = (operation.spec.op_id, partition_key)
        previous = self._ready.get(identity)
        if previous is not None:
            if retry_request_id and previous.retry_request_id != retry_request_id:
                raise PlanningError('one invocation cannot satisfy multiple retry requests')
            return False
        if identity in self._visited:
            return artifact_key in self.failed
        self._visited.add(identity)

        output_keys = tuple(
            output.key_for(partition_key)
            for output in operation.spec.outputs.values()
            if output.mode == 'scalar' or partition_key
        )
        direct_failures = merge_refs(
            record.ref
            for output_key in output_keys
            if (record := self.view.records.get(failure_key(output_key))) is not None
            and record.producer.startswith('runtime:failure:')
        )
        if direct_failures:
            self.failed.update(dict.fromkeys(output_keys, direct_failures))
            return True

        if operation.spec.driver_input is not None:
            set_key = ArtifactKey.scalar(operation.spec.partition_set_id)
            if not self.require_key(set_key):
                return False
            partitions = self.view.partition_sets.get(set_key)
            if partitions is None or partition_key not in partitions:
                return False

        input_keys = _input_keys(
            operation,
            self.view.partition_sets,
            None if not partition_key else partition_key,
        )
        if input_keys is None:
            return False
        for input_artifact_keys in input_keys.values():
            for input_artifact_key in input_artifact_keys:
                self.require_key(input_artifact_key)
        blocking_failures = [
            ref
            for name, keys in input_keys.items()
            for key in (keys[:1] if operation.spec.inputs[name].mode == 'all' else keys)
            for ref in self.failed.get(key, ())
        ]
        if blocking_failures:
            failures = merge_refs(blocking_failures)
            self.failed.update(dict.fromkeys(output_keys, failures))
            return True
        inputs = _bind_inputs(
            operation,
            self.view.records,
            self.view.partition_sets,
            None if not partition_key else partition_key,
            self.failed,
        )
        if inputs is None:
            return False
        self._ready[identity] = OperationInvocation(
            operation,
            inputs,
            _expected_heads(operation, partition_key, self.artifacts),
            partition_key,
            retry_request_id,
        )
        return False

    def ready_invocations(self) -> tuple[OperationInvocation, ...]:
        def order(invocation: OperationInvocation) -> tuple[int, int, str]:
            operation_index = self._operation_order[invocation.operation.spec.op_id]
            partition_index = 0
            if invocation.partition_key:
                set_key = ArtifactKey.scalar(invocation.operation.spec.partition_set_id)
                partitions = self.view.partition_sets[set_key]
                partition_index = partitions.keys.index(invocation.partition_key)
            return operation_index, partition_index, invocation.partition_key

        return tuple(sorted(self._ready.values(), key=order))

    def failure_refs(self) -> tuple[ArtifactRef, ...]:
        return merge_refs(*self.failed.values())


def _remove_inactive_partitions(definition: RuntimeDefinition, artifacts: ArtifactSnapshot,
                                effective: dict[ArtifactKey, ArtifactRecord]) -> bool:
    changed = False
    partition_sets = _effective_partition_sets(artifacts, effective)
    for key in tuple(effective):
        if not key.partition_key:
            continue
        partition_set_id = definition.partition_set_by_artifact.get(key.artifact_id)
        if partition_set_id is None:
            continue
        partitions = partition_sets.get(ArtifactKey.scalar(partition_set_id))
        if partitions is None or key.partition_key not in partitions:
            del effective[key]
            changed = True
    return changed


def _validate_outputs(operation: Operation, effective: dict[ArtifactKey, ArtifactRecord],
                      partition_sets: Mapping[ArtifactKey, PartitionSet]) -> bool:
    partition_keys = _partition_keys(operation, partition_sets)
    changed = False
    if operation.spec.driver_input is not None:
        active_keys = set(partition_keys or ())
        output_ids = {output.artifact_id for output in operation.spec.outputs.values()}
        for key, record in tuple(effective.items()):
            if (key.artifact_id in output_ids and key.partition_key not in active_keys
                    and record.producer == f'operation:{operation.spec.op_id}'):
                del effective[key]
                changed = True
        if partition_keys is None:
            return changed
    invocation_keys: tuple[str | None, ...] = (
        tuple(partition_keys) if operation.spec.driver_input is not None else (None,)
    )
    for partition_key in invocation_keys:
        inputs = _bind_inputs(operation, effective, partition_sets, partition_key)
        expected_inputs = None if inputs is None else _bound_refs(inputs, lineage=True)
        for output in operation.spec.outputs.values():
            keys = (
                (ArtifactKey.partition(output.artifact_id, partition_key),)
                if partition_key is not None
                else (ArtifactKey.scalar(output.artifact_id),)
                if output.mode == 'scalar'
                else tuple(key for key in effective if key.artifact_id == output.artifact_id)
            )
            for key in keys:
                record = effective.get(key)
                if record is None or not record.producer.startswith('operation:'):
                    continue
                if expected_inputs is None and any(
                    is_failure_key(ref.key) for ref in record.input_refs
                ):
                    continue
                if (expected_inputs is None
                        or record.producer != f'operation:{operation.spec.op_id}'
                        or record.input_refs != expected_inputs):
                    del effective[key]
                    changed = True
    return changed


def _effective_partition_sets(artifacts: ArtifactSnapshot, effective: Mapping[ArtifactKey, ArtifactRecord]
                              ) -> dict[ArtifactKey, PartitionSet]:
    return {
        key: partitions
        for key, partitions in artifacts.partition_sets.items()
        if key in effective
    }


def _expected_heads(operation: Operation, partition_key: str, artifacts: ArtifactSnapshot
                    ) -> dict[ArtifactKey, ArtifactRef | None]:
    expected: dict[ArtifactKey, ArtifactRef | None] = {}
    for output in operation.spec.outputs.values():
        if output.mode == 'partitioned' and not partition_key:
            expected.update(
                (key, record.ref)
                for key, record in artifacts.records.items()
                if key.artifact_id == output.artifact_id
            )
            continue
        key = output.key_for(partition_key)
        record = artifacts.records.get(key)
        expected[key] = None if record is None else record.ref
    return expected


def _partition_keys(operation: Operation, partition_sets: Mapping[ArtifactKey, PartitionSet]) -> tuple[str, ...] | None:
    if operation.spec.driver_input is None:
        return ()
    partitions = partition_sets.get(ArtifactKey.scalar(operation.spec.partition_set_id))
    return None if partitions is None else partitions.keys


def _bind_inputs(operation: Operation, effective: Mapping[ArtifactKey, ArtifactRecord],
                 partition_sets: Mapping[ArtifactKey, PartitionSet], partition_key: str | None,
                 failures: Mapping[ArtifactKey, tuple[ArtifactRef, ...]] | None = None) -> dict[str, BoundInput] | None:
    keys_by_input = _input_keys(operation, partition_sets, partition_key)
    if keys_by_input is None:
        return None
    failed = {} if failures is None else failures
    inputs: dict[str, BoundInput] = {}
    for name, binding in operation.spec.inputs.items():
        keys = keys_by_input[name]
        if binding.mode == 'all':
            partition_set = effective.get(keys[0])
            if partition_set is None:
                return None
            members: list[ArtifactRef] = []
            failure_refs: list[ArtifactRef] = []
            for key in keys[1:]:
                record = effective.get(key)
                if record is not None:
                    members.append(record.ref)
                    continue
                current_failures = failed.get(key)
                if current_failures is None:
                    return None
                failure_refs.extend(current_failures)
            inputs[name] = BoundAggregate(
                partition_set.ref,
                tuple(members),
                merge_refs(failure_refs),
            )
            continue
        record = effective.get(keys[0])
        if record is None:
            return None
        inputs[name] = record.ref
    return inputs


def _input_keys(operation: Operation, partition_sets: Mapping[ArtifactKey, PartitionSet], partition_key: str | None
                ) -> dict[str, tuple[ArtifactKey, ...]] | None:
    keys: dict[str, tuple[ArtifactKey, ...]] = {}
    for name, binding in operation.spec.inputs.items():
        if binding.mode == 'one':
            keys[name] = (ArtifactKey.scalar(binding.artifact_id),)
        elif binding.mode in {'each', 'keyed'}:
            if partition_key is None:
                return None
            keys[name] = (ArtifactKey.partition(binding.artifact_id, partition_key),)
        else:
            set_key = ArtifactKey.scalar(binding.partition_set_id)
            partitions = partition_sets.get(set_key)
            members = () if partitions is None else tuple(
                ArtifactKey.partition(binding.artifact_id, current)
                for current in partitions.keys
            )
            keys[name] = (set_key, *members)
    return keys


def project_progress(definition: RuntimeDefinition, view: ArtifactSnapshot, attempts: Iterable[AttemptSnapshot],
                     failures: Iterable[CaseFailure]) -> RuntimeProgress:
    attempt_values = tuple(attempts)
    failure_values = tuple(failures)
    latest_attempts: dict[tuple[str, str], AttemptSnapshot] = {}
    for attempt in sorted(attempt_values, key=lambda item: item.created_at):
        latest_attempts[(attempt.operation_id, attempt.partition_key)] = attempt
    active = {
        (attempt.operation_id, attempt.partition_key)
        for attempt in attempt_values
        if attempt.status in {'scheduled', 'running', 'cancelling'}
    }
    failed = {
        (failure.operation_id, failure.case_id)
        for failure in failure_values
    }

    statuses: list[tuple[str, str, str]] = []
    case_operations: dict[str, list[str]] = {
        case_id: []
        for partitions in view.partition_sets.values()
        for case_id in partitions.keys
    }
    for operation in definition.operations:
        partitions = _partition_keys(operation, view.partition_sets)
        if partitions is None:
            continue
        invocation_partitions = ('',) if operation.spec.driver_input is None else partitions
        for partition_key in invocation_partitions:
            identity = (operation.spec.op_id, partition_key)
            if identity in active:
                status = 'running'
            elif _outputs_complete(operation, partition_key, view):
                status = 'completed'
            elif identity in failed or (
                (latest := latest_attempts.get(identity)) is not None
                and latest.status == 'failed'
            ):
                status = 'failed'
            else:
                status = 'pending'
            statuses.append((operation.spec.op_id, partition_key, status))
            if partition_key:
                case_operations.setdefault(partition_key, []).append(status)

    counts = Counter(status for _, _, status in statuses)
    case_counts: Counter[str] = Counter()
    failed_cases = {failure.case_id for failure in failure_values}
    for case_id, operation_statuses in case_operations.items():
        if case_id in failed_cases:
            case_status = 'failed'
        elif 'running' in operation_statuses:
            case_status = 'running'
        elif operation_statuses and all(
            status == 'completed' for status in operation_statuses
        ):
            case_status = 'completed'
        else:
            case_status = 'pending'
        case_counts[case_status] += 1

    total = len(statuses)
    terminal = counts['completed'] + counts['failed']
    return RuntimeProgress(
        total,
        counts['completed'],
        counts['running'],
        counts['failed'],
        counts['pending'],
        0.0 if not total else round(terminal * 100 / total, 2),
        len(case_operations),
        case_counts['completed'],
        case_counts['running'],
        case_counts['failed'],
        case_counts['pending'],
    )


def project_runtime_snapshot(run_id: str, status: RunStatus, error: RuntimeErrorInfo | None,
                             definition: RuntimeDefinition, decision: PlanningResult,
                             attempts: Iterable[AttemptSnapshot], failures: Iterable[CaseFailure], *,
                             active_attempts: Iterable[AttemptSnapshot] | None = None) -> RuntimeSnapshot:
    attempt_values = tuple(attempts)
    failure_values = tuple(failures)
    active = tuple(
        active_attempts
        if active_attempts is not None
        else (
            attempt
            for attempt in attempt_values
            if attempt.status in {'scheduled', 'running', 'cancelling'}
        )
    )
    terminal = status in {'cancelled', 'failed', 'completed'}
    active_ids = {attempt.invocation_id for attempt in active}
    ready_count = 0
    awaiting: tuple[ArtifactKey, ...] = ()
    if not terminal and isinstance(decision, PlanReady):
        ready_count = sum(
            invocation.invocation_id not in active_ids
            for invocation in decision.invocations
        )
    elif not terminal and isinstance(decision, PlanAwaiting):
        awaiting = decision.artifact_keys

    view = decision.view
    return RuntimeSnapshot(
        run_id,
        status,
        tuple(
            InvocationSnapshot(
                attempt.invocation_id,
                attempt.operation_id,
                attempt.partition_key,
            )
            for attempt in active
            if not terminal and attempt.status in {'scheduled', 'running'}
        ),
        ready_count,
        {
            key: record.ref
            for key, record in view.records.items()
            if not is_failure_key(key)
        },
        view.partition_sets,
        error,
        active,
        awaiting,
        failure_values,
        project_progress(definition, view, attempt_values, failure_values),
    )


def _outputs_complete(operation: Operation, partition_key: str, view: ArtifactSnapshot) -> bool:
    for output in operation.spec.outputs.values():
        if operation.spec.driver_input is not None:
            if output.key_for(partition_key) not in view.records:
                return False
            continue
        if output.mode == 'scalar':
            if ArtifactKey.scalar(output.artifact_id) not in view.records:
                return False
            continue
        partitions = view.partition_sets.get(
            ArtifactKey.scalar(output.partition_set_id)
        )
        if partitions is None or any(
            ArtifactKey.partition(output.artifact_id, case_id) not in view.records
            for case_id in partitions.keys
        ):
            return False
    return True


__all__ = [
    'PlanAwaiting', 'PlanComplete', 'PlanReady', 'PlanningResult', 'RuntimeDefinition',
    'compile_operations', 'obsolete_retries', 'plan_next', 'project_progress',
    'project_runtime_snapshot',
]
