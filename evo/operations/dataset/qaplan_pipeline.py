from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from evo.llm import LazyLLMClient

from .chunks_build import build_chunk_candidates, build_chunks, build_chunks_manifest
from .entities import chunk_entities_extract, chunk_entities_extract_manifest
from .generate import generate, generate_manifest
from .import_cases import import_cases
from .generate_enhance import generate_enhance, generate_enhance_manifest
from .qaplan import qaplan_manifest, qaplan_plan, qaplan_spec
from .select_docs import select_docs
from .topic_discovery import (
    topic_discovery_embedding_cluster,
    topic_discovery_embedding_label,
    topic_discovery_embedding_label_cluster,
    topic_discovery_embedding_label_manifest,
    topic_discovery_entity_build_graph,
    topic_discovery_entity_cluster,
    topic_discovery_manifest,
)


def dataset_materializers(case_ids: tuple[str, ...]) -> dict[str, Any]:
    """Materializers for the dataset graph.

    LLM-backed operations read the same run_config artifact that the final
    generator uses, so live experiments do not depend on hidden process state.
    """

    if not case_ids:
        raise ValueError('case_ids must not be empty')

    def complete(inputs: Mapping[str, object]):
        config = inputs.get('run_config')
        if not isinstance(config, Mapping):
            raise ValueError('run_config must be a mapping')
        llm_config = config.get('llm_config')
        if not isinstance(llm_config, Mapping):
            raise ValueError('run_config.llm_config must be a mapping')
        return LazyLLMClient(llm_config=llm_config, model='evo_llm')

    def qaplan_context(ctx: Any) -> Any:
        return SimpleNamespace(
            case_ids=case_ids,
            input_ref_by_key=ctx.input_ref_by_key,
            output_key_by_name=ctx.output_key_by_name,
            op_id=ctx.op_id,
            run_id=ctx.run_id,
        )

    return {
        'dataset.import_cases': import_cases,
        'dataset.select_docs': select_docs,
        'dataset.build_chunk_candidates': build_chunk_candidates,
        'dataset.build_chunks': build_chunks,
        'dataset.build_chunks_manifest': build_chunks_manifest,
        'dataset.chunk_entities_extract': lambda ctx, inputs: chunk_entities_extract(
            ctx, inputs, llm_complete=complete(inputs)
        ),
        'dataset.chunk_entities_extract_manifest': chunk_entities_extract_manifest,
        'dataset.topic_discovery_entity_build_graph': topic_discovery_entity_build_graph,
        'dataset.topic_discovery_entity_cluster': topic_discovery_entity_cluster,
        'dataset.topic_discovery_embedding_cluster': topic_discovery_embedding_cluster,
        'dataset.topic_discovery_embedding_label': lambda ctx, inputs: topic_discovery_embedding_label(
            ctx, inputs, llm_complete=complete(inputs)
        ),
        'dataset.topic_discovery_embedding_label_cluster': lambda ctx, inputs: topic_discovery_embedding_label_cluster(
            ctx, inputs, llm_complete=complete(inputs)
        ),
        'dataset.topic_discovery_embedding_label_manifest': topic_discovery_embedding_label_manifest,
        'dataset.topic_discovery_manifest': topic_discovery_manifest,
        'dataset.qaplan_plan': lambda ctx, inputs: qaplan_plan(qaplan_context(ctx), inputs),
        'dataset.qaplan_spec': lambda ctx, inputs: qaplan_spec(qaplan_context(ctx), inputs),
        'dataset.qaplan_manifest': qaplan_manifest,
        'dataset.generate': generate,
        'dataset.generate_enhance': generate_enhance,
        'dataset.generate_manifest': generate_manifest,
        'dataset.generate_enhance_manifest': generate_enhance_manifest,
    }


__all__ = ['dataset_materializers']
