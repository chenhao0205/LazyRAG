from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

from evo.artifact_flow.gate import SQLiteFlowGate
from evo.artifact_flow.query import FlowQueryService
from evo.artifact_flow.service import FlowService
from evo.artifact_flow.state import CheckpointPolicy, FlowRunState
from evo.artifact_runtime.evo import catalog as C
from evo.artifact_runtime.evo.adapter import build_evo_artifact_adapter
from evo.artifact_runtime.evo.flow import EvoFlowSpec
from evo.artifact_runtime.evo.flow_ops import default_evo_ops
from evo.artifact_runtime.kernel import ArtifactKey, ArtifactRef, ConcurrencyLimits, SQLiteArtifactStore
from evo.operations.abtest import abtest_materializers
from evo.operations.analysis import analysis_materializers
from evo.operations.dataset import dataset_materializers
from evo.operations.dataset.csv_loader import as_text, norm_text
from evo.operations.eval import eval_materializers
from evo.operations.repair import repair_materializers

CONFIG_ARTIFACTS = {
    'run_config': C.RUN_CONFIG,
    'source_config': C.CORPUS_SOURCE_CONFIG,
    'target_config': C.EVAL_TARGET_CONFIG,
    'eval_policy': C.EVAL_POLICY,
    'repair_policy': C.REPAIR_POLICY,
    'candidate_config': C.ABTEST_CANDIDATE_CONFIG,
}
EVO_MAX_IN_FLIGHT = 8
EVO_PARTITION_OP_LIMIT = 4
EVO_EVAL_ANSWER_OP_LIMIT = 2


class RuntimePort:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.store_root = root / 'artifact-store'
        self.store_root.mkdir(parents=True, exist_ok=True)

    def store(self) -> SQLiteArtifactStore:
        return SQLiteArtifactStore(self.store_root)

    def spec(self, num_case: int) -> EvoFlowSpec:
        return EvoFlowSpec(EvoFlowSpec.case_ids(num_case))

    def adapter(self, num_case: int):
        spec = self.spec(num_case)
        store = self.store()
        return build_evo_artifact_adapter(
            store,
            default_evo_ops(spec.cases),
            {
                **dataset_materializers(spec.cases, duplicate_questions=self._duplicate_case_questions),
                **eval_materializers(),
                **analysis_materializers(),
                **repair_materializers(),
                **abtest_materializers(),
            },
            concurrency_limits=self._concurrency_limits(),
        )

    def flow(self, num_case: int) -> FlowService:
        spec = self.spec(num_case)
        return FlowService(
            SQLiteFlowGate(self.store_root),
            adapter_factory=lambda: self.adapter(num_case),
            spec=spec,
            checkpoint_policy=CheckpointPolicy(('dataset', 'eval', 'analysis', 'repair')),
            tick_limit=max(50, num_case * 8 + 20),
        )

    def query(self, num_case: int) -> FlowQueryService:
        spec = self.spec(num_case)
        return FlowQueryService(SQLiteFlowGate(self.store_root), lambda: self.adapter(num_case), spec)

    def gate_state(self, run_id: str) -> FlowRunState:
        return SQLiteFlowGate(self.store_root).get(run_id) or FlowRunState(run_id)

    def seed(self, run_id: str, seed: Mapping[str, Any], request_hash: str) -> None:
        store = self.store()
        try:
            for artifact_id, value in (
                (C.RUN_CONFIG, seed['run_config']),
                (C.CORPUS_SOURCE_CONFIG, seed['source_config']),
                (C.EVAL_TARGET_CONFIG, seed['target_config']),
                (C.EVAL_POLICY, seed['eval_policy']),
                (C.REPAIR_POLICY, seed['repair_policy']),
                (C.ABTEST_CANDIDATE_CONFIG, seed['candidate_config']),
            ):
                store.commit_external(
                    run_id,
                    ArtifactKey.of(artifact_id),
                    value,
                    idempotency_key=f'seed:{run_id}:{artifact_id}:{request_hash}',
                    metadata={'kind': 'thread_seed'},
                )
        finally:
            store.close()

    def run_config(self, run_id: str) -> Mapping[str, Any] | None:
        store = self.store()
        try:
            ref = store.effective_artifacts(run_id).get(ArtifactKey.of(C.RUN_CONFIG))
            record = store.get(run_id, ref) if ref is not None else None
            return record.value if record is not None and isinstance(record.value, Mapping) else None
        finally:
            store.close()

    def _duplicate_case_questions(self, run_id: str, case_id: str, row: Mapping[str, Any]) -> list[str]:
        question = norm_text(row.get('question'))
        if not question:
            return []
        store = self.store()
        try:
            duplicates = []
            for key, ref in store.effective_artifacts(run_id).items():
                if key.artifact_id != C.EVAL_CASE or key.partition == case_id:
                    continue
                record = store.get(run_id, ref)
                value = record.value if record is not None else None
                if isinstance(value, Mapping) and norm_text(value.get('question')) == question:
                    duplicates.append(as_text(value.get('question')))
            return list(dict.fromkeys(item for item in duplicates if item))
        finally:
            store.close()

    def effective_ref(self, run_id: str, artifact_id: str) -> ArtifactRef | None:
        store = self.store()
        try:
            return store.effective_artifacts(run_id).get(ArtifactKey.of(artifact_id))
        finally:
            store.close()

    def config_artifact(self, run_id: str, target: str) -> tuple[ArtifactRef, object] | None:
        artifact_id = CONFIG_ARTIFACTS[target]
        store = self.store()
        try:
            ref = store.effective_artifacts(run_id).get(ArtifactKey.of(artifact_id))
            record = store.get(run_id, ref) if ref is not None else None
            return (ref, record.value) if ref is not None and record is not None else None
        finally:
            store.close()

    def run_ids(self) -> list[str]:
        store = self.store()
        try:
            return list(store.run_ids(ArtifactKey.of(C.RUN_CONFIG)))
        finally:
            store.close()

    def delete_run(self, run_id: str) -> None:
        store = self.store()
        try:
            store.delete_run(run_id)
        finally:
            store.close()
        SQLiteFlowGate(self.store_root).delete_run_state(run_id)
        store = self.store()
        try:
            store.gc()
        finally:
            store.close()

    def _concurrency_limits(self) -> ConcurrencyLimits:
        max_in_flight = _env_int('LAZYMIND_EVO_MAX_IN_FLIGHT', EVO_MAX_IN_FLIGHT)
        partition_limit = _env_int('LAZYMIND_EVO_PARTITION_OP_LIMIT', EVO_PARTITION_OP_LIMIT)
        eval_answer_limit = _env_int('LAZYMIND_EVO_EVAL_ANSWER_OP_LIMIT', EVO_EVAL_ANSWER_OP_LIMIT)
        return ConcurrencyLimits(
            max_in_flight=max_in_flight,
            per_materializer={
                'dataset.prepare_case': partition_limit,
                'dataset.generate_case': partition_limit,
                'eval.answer': min(eval_answer_limit, max_in_flight),
                'eval.judge': partition_limit,
                'analysis.trace_summary': partition_limit,
                'analysis.classify_case': partition_limit,
                'abtest.candidate_rag_answer': partition_limit,
                'abtest.candidate_judge': partition_limit,
            },
        )


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, '') or default)
    except ValueError:
        return default
    return value if value >= 1 else default


__all__ = ['RuntimePort']
