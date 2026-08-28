from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import SimpleNamespace
from typing import Any

from evo import artifacts as A
from evo.artifact_runtime import (
    AggregateValue,
    Operation,
    OperationContext,
    OperationResult,
    PartitionSet,
    all_items,
    each,
    keyed,
    one,
    operation,
    partitioned,
    scalar,
)

from .assemble import assemble_dataset
from .chunks_build import (
    build_chunk_candidates,
    build_chunks,
    build_chunks_manifest,
)
from .csv_loader import normalize_eval_case
from .entities import chunk_entities_extract, chunk_entities_extract_manifest
from .generate import generate, generate_manifest
from .generate_enhance import generate_enhance, generate_enhance_manifest
from .import_cases import import_cases
from .qaplan import qaplan_manifest, qaplan_plan, qaplan_spec
from .select_docs import select_docs
from .source_config import normalize_source_config
from .topic_discovery import (
    topic_discovery_embedding_cluster,
    topic_discovery_embedding_label,
    topic_discovery_embedding_label_cluster,
    topic_discovery_embedding_label_manifest,
    topic_discovery_entity_build_graph,
    topic_discovery_entity_cluster,
    topic_discovery_manifest,
)


@operation(
    op_id='dataset.import_cases',
    inputs={'source_config': one(A.CORPUS_SOURCE_CONFIG)},
    outputs={'manifest': scalar(A.DATASET_IMPORT_CASES_MANIFEST)},
)
async def import_cases_operation(ctx: OperationContext, source_config: object) -> OperationResult:
    config = normalize_source_config(source_config)
    manifest = import_cases(ctx, {'source_config': config})['import_cases_manifest']
    return await _result(ctx, 'dataset.cases_imported', {'manifest': manifest})


@operation(
    op_id='dataset.select_docs',
    inputs={
        'source_config': one(A.CORPUS_SOURCE_CONFIG),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
        'select_docs_params': one(A.DATASET_SELECT_DOCS_PARAMS),
    },
    outputs={'selected_docs': scalar(A.DATASET_SELECTED_DOCS)},
)
async def select_docs_operation(
    ctx: OperationContext,
    source_config: object,
    import_manifest: object,
    select_docs_params: object,
) -> OperationResult:
    config = normalize_source_config(source_config)
    selected = select_docs(ctx, {
        'source_config': config,
        'import_cases_manifest': _mapping(import_manifest, 'import_manifest'),
        'select_docs_params': _mapping(select_docs_params, 'select_docs_params'),
    })['selected_docs']
    return await _result(ctx, 'dataset.documents_selected', {'selected_docs': selected})


@operation(
    op_id='dataset.build_chunk_candidates',
    inputs={
        'selected_docs': one(A.DATASET_SELECTED_DOCS),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
        'build_chunks_params': one(A.DATASET_BUILD_CHUNKS_PARAMS),
    },
    outputs={
        'candidates': scalar(A.DATASET_BUILD_CHUNK_CANDIDATES),
        'partitions': scalar(A.DATASET_CHUNK_REQUESTS),
        'requests': partitioned(A.DATASET_CHUNK_REQUEST, over=A.DATASET_CHUNK_REQUESTS),
    },
)
async def build_chunk_candidates_operation(
    ctx: OperationContext,
    selected_docs: object,
    import_manifest: object,
    build_chunks_params: object,
) -> OperationResult:
    candidates = build_chunk_candidates(ctx, {
        'selected_docs': _mapping(selected_docs, 'selected_docs'),
        'import_cases_manifest': _mapping(import_manifest, 'import_manifest'),
        'build_chunks_params': _mapping(build_chunks_params, 'build_chunks_params'),
    })['build_chunk_candidates']
    partition_keys = tuple(
        str(chunk['chunk_id'])
        for chunk in candidates.get('chunks', ())
        if isinstance(chunk, Mapping) and chunk.get('selected') is True
    )
    return await _result(ctx, 'dataset.chunk_candidates_built', {
        'candidates': candidates,
        'partitions': PartitionSet(partition_keys),
        'requests': {key: {'partition_key': key} for key in partition_keys},
    }, total=len(partition_keys))


@operation(
    op_id='dataset.build_chunk',
    inputs={
        'request': each(A.DATASET_CHUNK_REQUEST, over=A.DATASET_CHUNK_REQUESTS),
        'candidates': one(A.DATASET_BUILD_CHUNK_CANDIDATES),
    },
    outputs={'chunk': partitioned(A.DATASET_CHUNK)},
    max_concurrency=10,
)
async def build_chunk_operation(
    ctx: OperationContext,
    request: object,
    candidates: object,
) -> OperationResult:
    _mapping(request, 'request')
    chunk = build_chunks(_output_context(ctx, 'chunk'), {
        'build_chunk_candidates': _mapping(candidates, 'candidates'),
    })['chunk']
    return await _result(ctx, 'dataset.chunk_built', {'chunk': chunk}, case_id=ctx.partition_key)


@operation(
    op_id='dataset.build_chunks_manifest',
    inputs={
        'selected_docs': one(A.DATASET_SELECTED_DOCS),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
        'candidates': one(A.DATASET_BUILD_CHUNK_CANDIDATES),
        'chunks': all_items(A.DATASET_CHUNK, over=A.DATASET_CHUNK_REQUESTS),
    },
    outputs={'manifest': scalar(A.DATASET_BUILD_CHUNKS_MANIFEST)},
)
async def build_chunks_manifest_operation(
    ctx: OperationContext,
    selected_docs: object,
    import_manifest: object,
    candidates: object,
    chunks: object,
) -> OperationResult:
    chunk_values, partition_keys = _chunks_with_failures(chunks)
    manifest = build_chunks_manifest(ctx, {
        'selected_docs': _mapping(selected_docs, 'selected_docs'),
        'import_cases_manifest': _mapping(import_manifest, 'import_manifest'),
        'build_chunk_candidates': _mapping(candidates, 'candidates'),
        'chunk': chunk_values,
    })['build_chunks_manifest']
    return await _result(ctx, 'dataset.chunks_manifest_built', {'manifest': manifest}, total=len(partition_keys))


@operation(
    op_id='dataset.extract_chunk_entities',
    inputs={
        'chunk': each(A.DATASET_CHUNK, over=A.DATASET_CHUNK_REQUESTS),
        'chunks_manifest': one(A.DATASET_BUILD_CHUNKS_MANIFEST),
        'run_config': one(A.RUN_CONFIG),
        'material_approval': one(A.APPROVAL_DATASET_MATERIAL_PREPARATION),
    },
    outputs={'entity': partitioned(A.DATASET_CHUNK_ENTITY)},
    max_concurrency=10,
)
async def extract_chunk_entities_operation(
    ctx: OperationContext,
    chunk: object,
    chunks_manifest: object,
    run_config: object,
    material_approval: object,
) -> OperationResult:
    del material_approval
    _mapping(chunks_manifest, 'chunks_manifest')
    entity = chunk_entities_extract(ctx, {
        'chunk': _mapping(chunk, 'chunk'),
        'chunk_entities_extract_params': {},
    }, llm_complete=_llm_complete(run_config))['chunk_entity']
    return await _result(ctx, 'dataset.chunk_entities_extracted', {'entity': entity}, case_id=ctx.partition_key)


@operation(
    op_id='dataset.chunk_entities_manifest',
    inputs={
        'chunks_manifest': one(A.DATASET_BUILD_CHUNKS_MANIFEST),
        'entities': all_items(A.DATASET_CHUNK_ENTITY, over=A.DATASET_CHUNK_REQUESTS),
    },
    outputs={'manifest': scalar(A.DATASET_CHUNK_ENTITIES_MANIFEST)},
)
async def chunk_entities_manifest_operation(
    ctx: OperationContext,
    chunks_manifest: object,
    entities: object,
) -> OperationResult:
    built = _mapping(chunks_manifest, 'chunks_manifest')
    values = _entities_with_failures(entities, built)
    if not values:
        manifest = {
            'chunks': [],
            'stats': {
                'slot_count': 0,
                'available_count': 0,
                'placeholder_count': 0,
                'entity_count': 0,
                'empty_entity_count': 0,
                'doc_count': 0,
                'group_counts': {},
            },
            'params': {'max_entities_per_chunk': 10},
        }
    else:
        manifest = chunk_entities_extract_manifest(ctx, {
            'build_chunks_manifest': built,
            'chunk_entities': values,
            'chunk_entities_extract_manifest_params': {},
        })['chunk_entities_extract_manifest']
    return await _result(ctx, 'dataset.chunk_entities_manifest_built', {'manifest': manifest}, total=len(values))


@operation(
    op_id='dataset.build_entity_graph',
    inputs={'entities_manifest': one(A.DATASET_CHUNK_ENTITIES_MANIFEST)},
    outputs={'graph': scalar(A.DATASET_ENTITY_GRAPH)},
)
async def build_entity_graph_operation(ctx: OperationContext, entities_manifest: object) -> OperationResult:
    manifest = _mapping(entities_manifest, 'entities_manifest')
    values = tuple(_mapping(item, 'entities_manifest.chunks[]') for item in manifest.get('chunks') or [])
    if values:
        graph = topic_discovery_entity_build_graph(ctx, {
            'chunk_entity': values,
            'topic_discovery_entity_build_graph_params': {},
        })['entity_graph']
    else:
        graph = {
            'nodes': [],
            'edges': [],
            'skipped_chunks': [],
            'stats': {
                'source_chunk_count': 0,
                'node_count': 0,
                'edge_count': 0,
                'skipped_chunk_count': 0,
                'noisy_entity_count': 0,
            },
            'params': {},
        }
    return await _result(ctx, 'dataset.entity_graph_built', {'graph': graph})


@operation(
    op_id='dataset.cluster_entities',
    inputs={'graph': one(A.DATASET_ENTITY_GRAPH)},
    outputs={'clusters': scalar(A.DATASET_ENTITY_CLUSTERS)},
)
async def cluster_entities_operation(ctx: OperationContext, graph: object) -> OperationResult:
    clusters = topic_discovery_entity_cluster(ctx, {
        'entity_graph': _mapping(graph, 'graph'),
        'topic_discovery_entity_cluster_params': {},
    })['entity_clusters']
    return await _result(ctx, 'dataset.entity_clusters_built', {'clusters': clusters})


@operation(
    op_id='dataset.cluster_embeddings',
    inputs={
        'chunks': all_items(A.DATASET_CHUNK, over=A.DATASET_CHUNK_REQUESTS),
        'chunks_manifest': one(A.DATASET_BUILD_CHUNKS_MANIFEST),
        'material_approval': one(A.APPROVAL_DATASET_MATERIAL_PREPARATION),
    },
    outputs={
        'candidates': scalar(A.DATASET_EMBEDDING_CLUSTER_CANDIDATES),
        'partitions': scalar(A.DATASET_EMBEDDING_LABEL_REQUESTS),
        'requests': partitioned(A.DATASET_EMBEDDING_LABEL_REQUEST, over=A.DATASET_EMBEDDING_LABEL_REQUESTS),
    },
)
async def cluster_embeddings_operation(
    ctx: OperationContext,
    chunks: object,
    chunks_manifest: object,
    material_approval: object,
) -> OperationResult:
    del material_approval
    values = _chunk_values_for_manifest(chunks, _mapping(chunks_manifest, 'chunks_manifest'))
    output = topic_discovery_embedding_cluster(ctx, {
        'chunk': values,
        'topic_discovery_embedding_cluster_params': {},
    })
    partition_ids = tuple(output['embedding_label_requests'])
    return await _result(ctx, 'dataset.embedding_candidates_built', {
        'candidates': output['embedding_cluster_candidates'],
        'partitions': PartitionSet(partition_ids),
        'requests': output['embedding_label_request'],
    }, total=len(partition_ids))


@operation(
    op_id='dataset.label_embedding_cluster',
    inputs={
        'request': each(A.DATASET_EMBEDDING_LABEL_REQUEST, over=A.DATASET_EMBEDDING_LABEL_REQUESTS),
        'run_config': one(A.RUN_CONFIG),
    },
    outputs={'cluster': partitioned(A.DATASET_EMBEDDING_CLUSTER)},
    max_concurrency=10,
)
async def label_embedding_cluster_operation(
    ctx: OperationContext,
    request: object,
    run_config: object,
) -> OperationResult:
    cluster = topic_discovery_embedding_label_cluster(ctx, {
        'request': _mapping(request, 'request'),
        'topic_discovery_embedding_label_params': {},
    }, llm_complete=_llm_complete(run_config))['embedding_cluster']
    return await _result(
        ctx,
        'dataset.embedding_cluster_labeled',
        {'cluster': cluster},
        case_id=ctx.partition_key,
    )


@operation(
    op_id='dataset.embedding_label_manifest',
    inputs={
        'candidates': one(A.DATASET_EMBEDDING_CLUSTER_CANDIDATES),
        'requests': one(A.DATASET_EMBEDDING_LABEL_REQUESTS),
        'clusters': all_items(A.DATASET_EMBEDDING_CLUSTER, over=A.DATASET_EMBEDDING_LABEL_REQUESTS),
    },
    outputs={'clusters': scalar(A.DATASET_EMBEDDING_CLUSTERS)},
)
async def embedding_label_manifest_operation(
    ctx: OperationContext,
    candidates: object,
    requests: object,
    clusters: object,
) -> OperationResult:
    values = _successful_values(clusters)
    manifest = topic_discovery_embedding_label_manifest(ctx, {
        'embedding_cluster_candidates': _mapping(candidates, 'candidates'),
        'embedding_label_requests': _partition_set_ids(requests),
        'embedding_cluster': values,
        'topic_discovery_embedding_label_params': {},
    })['embedding_clusters']
    return await _result(ctx, 'dataset.embedding_clusters_labeled', {'clusters': manifest}, total=len(values))


@operation(
    op_id='dataset.topic_manifest',
    inputs={
        'entity_clusters': one(A.DATASET_ENTITY_CLUSTERS),
        'embedding_clusters': one(A.DATASET_EMBEDDING_CLUSTERS),
    },
    outputs={'manifest': scalar(A.DATASET_TOPIC_MANIFEST)},
)
async def topic_manifest_operation(
    ctx: OperationContext,
    entity_clusters: object,
    embedding_clusters: object,
) -> OperationResult:
    manifest = topic_discovery_manifest(ctx, {
        'entity_clusters': _mapping(entity_clusters, 'entity_clusters'),
        'embedding_clusters': _mapping(embedding_clusters, 'embedding_clusters'),
    })['topic_discovery_manifest']
    return await _result(ctx, 'dataset.topic_manifest_built', {'manifest': manifest})


@operation(
    op_id='dataset.qaplan_plan',
    inputs={
        'source_config': one(A.CORPUS_SOURCE_CONFIG),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
        'topic_manifest': one(A.DATASET_TOPIC_MANIFEST),
        'chunks': all_items(A.DATASET_CHUNK, over=A.DATASET_CHUNK_REQUESTS),
        'chunks_manifest': one(A.DATASET_BUILD_CHUNKS_MANIFEST),
        'plan_params': one(A.DATASET_QAPLAN_PLAN_PARAMS),
        'topic_approval': one(A.APPROVAL_DATASET_TOPIC_DISCOVERY),
    },
    outputs={
        'plan': scalar(A.DATASET_QAPLAN_PLAN),
        'partitions': scalar(A.EVAL_CASE_REQUESTS),
        'requests': partitioned(A.EVAL_CASE_REQUEST, over=A.EVAL_CASE_REQUESTS),
    },
)
async def qaplan_plan_operation(
    ctx: OperationContext,
    source_config: object,
    import_manifest: object,
    topic_manifest: object,
    chunks: object,
    chunks_manifest: object,
    plan_params: object,
    topic_approval: object,
) -> OperationResult:
    del topic_approval
    imported = _mapping(import_manifest, 'import_manifest')
    case_ids = _case_ids(imported)
    values = _chunk_values_for_manifest(chunks, _mapping(chunks_manifest, 'chunks_manifest'))
    plan = qaplan_plan(_case_context(ctx, case_ids), {
        'source_config': normalize_source_config(source_config),
        'import_cases_manifest': imported,
        'topic_discovery_manifest': _mapping(topic_manifest, 'topic_manifest'),
        'chunk': values,
        'qaplan_plan_params': _mapping(plan_params, 'plan_params'),
    })['qaplan_plan']
    return await _result(ctx, 'dataset.qaplan_built', {
        'plan': plan,
        'partitions': PartitionSet(case_ids),
        'requests': {case_id: {'case_id': case_id} for case_id in case_ids},
    }, total=len(case_ids))


@operation(
    op_id='dataset.qaplan_spec',
    inputs={
        'request': each(A.EVAL_CASE_REQUEST, over=A.EVAL_CASE_REQUESTS),
        'plan': one(A.DATASET_QAPLAN_PLAN),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
        'topic_manifest': one(A.DATASET_TOPIC_MANIFEST),
        'chunks': all_items(A.DATASET_CHUNK, over=A.DATASET_CHUNK_REQUESTS),
        'chunks_manifest': one(A.DATASET_BUILD_CHUNKS_MANIFEST),
    },
    outputs={'specification': partitioned(A.DATASET_QAPLAN_SPEC)},
    max_concurrency=10,
)
async def qaplan_spec_operation(
    ctx: OperationContext,
    request: object,
    plan: object,
    import_manifest: object,
    topic_manifest: object,
    chunks: object,
    chunks_manifest: object,
) -> OperationResult:
    _mapping(request, 'request')
    imported = _mapping(import_manifest, 'import_manifest')
    specification = qaplan_spec(_case_context(ctx, _case_ids(imported), 'qaplan_spec'), {
        'qaplan_plan': _mapping(plan, 'plan'),
        'import_cases_manifest': imported,
        'topic_discovery_manifest': _mapping(topic_manifest, 'topic_manifest'),
        'chunk': _chunk_values_for_manifest(chunks, _mapping(chunks_manifest, 'chunks_manifest')),
    })['qaplan_spec']
    return await _result(ctx, 'dataset.qaplan_spec_built', {'specification': specification}, case_id=ctx.partition_key)


@operation(
    op_id='dataset.qaplan_manifest',
    inputs={
        'plan': one(A.DATASET_QAPLAN_PLAN),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
        'specifications': all_items(A.DATASET_QAPLAN_SPEC, over=A.EVAL_CASE_REQUESTS),
    },
    outputs={'manifest': scalar(A.DATASET_QAPLAN_MANIFEST)},
)
async def qaplan_manifest_operation(
    ctx: OperationContext,
    plan: object,
    import_manifest: object,
    specifications: object,
) -> OperationResult:
    values = _successful_values(specifications)
    manifest = qaplan_manifest(ctx, {
        'qaplan_plan': _mapping(plan, 'plan'),
        'import_cases_manifest': _mapping(import_manifest, 'import_manifest'),
        'qaplan_specs': values,
    })['qaplan_manifest']
    return await _result(ctx, 'dataset.qaplan_manifest_built', {'manifest': manifest}, total=len(values))


@operation(
    op_id='dataset.generate_case',
    inputs={
        'specification': each(A.DATASET_QAPLAN_SPEC, over=A.EVAL_CASE_REQUESTS),
        'run_config': one(A.RUN_CONFIG),
    },
    outputs={'draft': partitioned(A.DATASET_CASE_DRAFT)},
    max_concurrency=10,
)
async def generate_case_operation(
    ctx: OperationContext,
    specification: object,
    run_config: object,
) -> OperationResult:
    draft = generate(_output_context(ctx, 'case'), {
        'qaplan_spec': _mapping(specification, 'specification'),
        'run_config': _mapping(run_config, 'run_config'),
    }, llm_complete=_llm_complete(run_config))['case']
    return await _result(ctx, 'dataset.case_generated', {'draft': draft}, case_id=ctx.partition_key)


@operation(
    op_id='dataset.generate_manifest',
    inputs={
        'drafts': all_items(A.DATASET_CASE_DRAFT, over=A.EVAL_CASE_REQUESTS),
        'import_manifest': one(A.DATASET_IMPORT_CASES_MANIFEST),
    },
    outputs={'manifest': scalar(A.DATASET_GENERATE_MANIFEST)},
)
async def generate_manifest_operation(
    ctx: OperationContext,
    drafts: object,
    import_manifest: object,
) -> OperationResult:
    values = _successful_values(drafts)
    failures = _failures(drafts)
    if failures:
        manifest = _partial_generate_manifest(values, failures)
    else:
        manifest = generate_manifest(ctx, {
            'cases': values,
            'import_cases_manifest': _mapping(import_manifest, 'import_manifest'),
        })['generate_manifest']
    return await _result(ctx, 'dataset.generate_manifest_built', {'manifest': manifest}, total=len(values))


@operation(
    op_id='dataset.enhance_case',
    inputs={
        'draft': each(A.DATASET_CASE_DRAFT, over=A.EVAL_CASE_REQUESTS),
        'run_config': one(A.RUN_CONFIG),
    },
    outputs={'enhancement': partitioned(A.DATASET_CASE_ENHANCEMENT)},
    max_concurrency=10,
)
async def enhance_case_operation(
    ctx: OperationContext,
    draft: object,
    run_config: object,
) -> OperationResult:
    enhancement = generate_enhance(ctx, {
        'case': _mapping(draft, 'draft'),
        'run_config': _mapping(run_config, 'run_config'),
    }, llm_complete=_llm_complete(run_config))['case_enhance']
    return await _result(ctx, 'dataset.case_enhanced', {'enhancement': enhancement}, case_id=ctx.partition_key)


@operation(
    op_id='dataset.enhance_manifest',
    inputs={
        'enhancements': all_items(A.DATASET_CASE_ENHANCEMENT, over=A.EVAL_CASE_REQUESTS),
        'qaplan_manifest': one(A.DATASET_QAPLAN_MANIFEST),
        'generate_manifest': one(A.DATASET_GENERATE_MANIFEST),
    },
    outputs={'manifest': scalar(A.DATASET_ENHANCE_MANIFEST)},
)
async def enhance_manifest_operation(
    ctx: OperationContext,
    enhancements: object,
    qaplan_manifest: object,
    generate_manifest: object,
) -> OperationResult:
    _mapping(qaplan_manifest, 'qaplan_manifest')
    _mapping(generate_manifest, 'generate_manifest')
    values = _successful_values(enhancements)
    failures = _failures(enhancements)
    if failures:
        manifest = {'case_count': len(values), 'failed_cases': failures}
    else:
        manifest = generate_enhance_manifest(ctx, {'case_enhances': values})['generate_enhance_manifest']
    return await _result(ctx, 'dataset.enhance_manifest_built', {'manifest': manifest}, total=len(values))


@operation(
    op_id='dataset.finalize_case',
    inputs={
        'draft': each(A.DATASET_CASE_DRAFT, over=A.EVAL_CASE_REQUESTS),
        'enhancement': keyed(A.DATASET_CASE_ENHANCEMENT),
    },
    outputs={'case': partitioned(A.EVAL_CASE)},
    max_concurrency=10,
)
async def finalize_case_operation(
    ctx: OperationContext,
    draft: object,
    enhancement: object,
) -> OperationResult:
    case = _finalize_case(
        _mapping(draft, 'draft'),
        _mapping(enhancement, 'enhancement'),
        ctx.partition_key,
    )
    return await _result(ctx, 'dataset.case_finalized', {'case': case}, case_id=ctx.partition_key)


@operation(
    op_id='dataset.assemble',
    inputs={
        'cases': all_items(A.EVAL_CASE, over=A.EVAL_CASE_REQUESTS),
        'qaplan_manifest': one(A.DATASET_QAPLAN_MANIFEST),
        'generate_manifest': one(A.DATASET_GENERATE_MANIFEST),
        'enhance_manifest': one(A.DATASET_ENHANCE_MANIFEST),
    },
    outputs={'dataset': scalar(A.EVAL_DATASET)},
)
async def assemble_dataset_operation(
    ctx: OperationContext,
    cases: object,
    qaplan_manifest: object,
    generate_manifest: object,
    enhance_manifest: object,
) -> OperationResult:
    _mapping(qaplan_manifest, 'qaplan_manifest')
    _mapping(generate_manifest, 'generate_manifest')
    _mapping(enhance_manifest, 'enhance_manifest')
    case_map = dict(_successful_entries(cases))
    if not case_map:
        raise ValueError('dataset has no successful cases')
    failures = _failures(cases)
    dataset = assemble_dataset(case_map, run_id=ctx.run_id, failed_cases=failures)
    return await _result(
        ctx,
        'dataset.assembled',
        {'dataset': dataset},
        current=len(case_map),
        total=len(case_map) + len(failures),
    )


_DATASET_OPERATIONS: tuple[Operation, ...] = (
    import_cases_operation,
    select_docs_operation,
    build_chunk_candidates_operation,
    build_chunk_operation,
    build_chunks_manifest_operation,
    extract_chunk_entities_operation,
    chunk_entities_manifest_operation,
    build_entity_graph_operation,
    cluster_entities_operation,
    cluster_embeddings_operation,
    label_embedding_cluster_operation,
    embedding_label_manifest_operation,
    topic_manifest_operation,
    qaplan_plan_operation,
    qaplan_spec_operation,
    qaplan_manifest_operation,
    generate_case_operation,
    generate_manifest_operation,
    enhance_case_operation,
    enhance_manifest_operation,
    finalize_case_operation,
    assemble_dataset_operation,
)


def dataset_operations() -> tuple[Operation, ...]:
    return _DATASET_OPERATIONS


async def _result(
    ctx: OperationContext,
    event_type: str,
    values: Mapping[str, object],
    *,
    current: int | None = None,
    total: int | None = None,
    **fields: object,
) -> OperationResult:
    await ctx.record(event_type, status='completed', current=current, total=total, **fields)
    return OperationResult(values)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _output_context(ctx: OperationContext, output_name: str) -> SimpleNamespace:
    """Adapt the artifact runtime context to the pure dataset operation contract."""
    return SimpleNamespace(
        output_key_by_name={output_name: SimpleNamespace(partition=ctx.partition_key)},
    )


def _case_context(
    ctx: OperationContext,
    case_ids: tuple[str, ...],
    output_name: str | None = None,
) -> SimpleNamespace:
    values: dict[str, object] = {'case_ids': case_ids}
    if output_name is not None:
        values['output_key_by_name'] = {output_name: SimpleNamespace(partition=ctx.partition_key)}
    return SimpleNamespace(**values)


def _unavailable_chunk_payload(partition: str, group: str) -> dict[str, object]:
    return {
        'available': False,
        'chunk_id': f'unavailable:{partition}',
        'doc_id': '__unavailable__',
        'filename': '',
        'group': group,
        'type': 'placeholder',
        'text': '',
        'embedding': {},
        'metadata': {'partition': partition, 'available': False},
    }


def _successful_entries(value: object) -> tuple[tuple[str, object], ...]:
    if isinstance(value, AggregateValue):
        return tuple(sorted(value.entries))
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), item) for key, item in value.items()))
    raise ValueError('partition aggregate must be a mapping')


def _successful_values(value: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(_mapping(item, 'partition value') for _, item in _successful_entries(value))


def _partition_set_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, PartitionSet):
        raise ValueError('partition set input must be a PartitionSet')
    return value.keys


def _failures(value: object) -> list[dict[str, object]]:
    if not isinstance(value, AggregateValue):
        return []
    return [
        {
            'case_id': failure.case_id,
            'operation_id': failure.operation_id,
            'attempt_id': failure.attempt_id,
            'error_kind': failure.error.kind,
            'error_message': failure.error.message,
        }
        for _, failure in sorted(value.failures.items())
    ]


def _chunks_with_failures(value: object) -> tuple[tuple[Mapping[str, Any], ...], tuple[str, ...]]:
    entries = {key: _mapping(item, 'chunk') for key, item in _successful_entries(value)}
    failure_map = {item['case_id']: item for item in _failures(value)}
    partition_keys = tuple(sorted((*entries, *failure_map)))
    chunks = []
    for partition_key in partition_keys:
        if partition_key in entries:
            chunks.append(entries[partition_key])
            continue
        placeholder = _unavailable_chunk_payload(partition_key, 'block')
        placeholder['metadata']['failure'] = failure_map[partition_key]
        chunks.append(placeholder)
    return tuple(chunks), partition_keys


def _manifest_chunks(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw_chunks = value.get('chunks')
    if not isinstance(raw_chunks, list):
        raise ValueError('chunks_manifest.chunks must be a list')
    output = {}
    for raw in raw_chunks:
        chunk = _mapping(raw, 'chunks_manifest.chunks[]')
        partition = str(chunk.get('partition') or '')
        if not partition:
            raise ValueError('chunks_manifest chunk partition must be non-empty')
        output[partition] = chunk
    return output


def _chunk_values_for_manifest(value: object, manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = {key: _mapping(item, 'chunk') for key, item in _successful_entries(value)}
    failures = {item['case_id']: item for item in _failures(value)}
    chunks = []
    for partition, source in sorted(_manifest_chunks(manifest).items()):
        if partition in entries:
            chunks.append(entries[partition])
        else:
            placeholder = _unavailable_chunk_payload(partition, str(source.get('group') or 'block'))
            if partition in failures:
                placeholder['metadata']['failure'] = failures[partition]
            chunks.append(placeholder)
    return tuple(chunks)


def _entities_with_failures(value: object, manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    entries = {key: _mapping(item, 'entity') for key, item in _successful_entries(value)}
    failures = {item['case_id']: item for item in _failures(value)}
    output = []
    for partition, chunk in sorted(_manifest_chunks(manifest).items()):
        if partition in entries:
            output.append(entries[partition])
            continue
        output.append({
            'available': False,
            'kb_id': str(chunk.get('kb_id') or ''),
            'chunk_id': str(chunk.get('chunk_id') or f'unavailable:{partition}'),
            'doc_id': str(chunk.get('doc_id') or '__unavailable__'),
            'group': str(chunk.get('group') or 'block'),
            'entities': [],
            'failure': failures.get(partition),
        })
    return tuple(output)


def _natural_case_id_key(case_id: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r'(\d+)', case_id) if part)


def _case_ids(import_manifest: Mapping[str, Any]) -> tuple[str, ...]:
    stats = _mapping(import_manifest.get('stats'), 'import_manifest.stats')
    allocation = _mapping(stats.get('case_allocation'), 'import_manifest.stats.case_allocation')
    assignments = _mapping(allocation.get('assignments'), 'case_allocation.assignments')
    imported: list[str] = []
    generated: list[str] = []
    for case_id, raw in assignments.items():
        assignment = _mapping(raw, f'case_allocation.assignments.{case_id}')
        mode = assignment.get('mode')
        if mode == 'imported':
            imported.append(str(case_id))
        elif mode == 'generated':
            generated.append(str(case_id))
        else:
            raise ValueError(f'case assignment mode is invalid: {case_id}')
    case_ids = tuple(sorted(imported, key=_natural_case_id_key) + sorted(generated, key=_natural_case_id_key))
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError('case assignments must contain unique case ids')
    return case_ids


def _llm_complete(run_config: object) -> Callable[[str], Any]:
    config = _mapping(run_config, 'run_config')
    llm_config = _mapping(config.get('llm_config'), 'run_config.llm_config')
    from evo.llm import LazyLLMClient
    return LazyLLMClient(llm_config=llm_config, model='evo_llm')


def _partial_generate_manifest(
    values: tuple[Mapping[str, Any], ...],
    failures: list[dict[str, object]],
) -> dict[str, object]:
    cases = [
        {
            'id': str(value.get('id') or ''),
            'question_type': str(value.get('question_type') or ''),
            'difficulty': str(value.get('difficulty') or ''),
            'reference_count': len(value.get('reference_chunk_ids') or []),
        }
        for value in values
    ]
    return {
        'cases': cases,
        'stats': {
            'case_count': len(cases),
            'failed_case_count': len(failures),
        },
        'failed_cases': failures,
    }


def _finalize_case(
    draft: Mapping[str, Any],
    enhancement: Mapping[str, Any],
    case_id: str,
) -> dict[str, Any]:
    if str(draft.get('id') or '') != case_id:
        raise ValueError('draft id must match case partition')
    source_preparation = dict(_mapping(draft.get('source_preparation'), 'draft.source_preparation'))
    mode = str(source_preparation.get('dataset_mode') or 'generated')
    raw_context = draft.get('reference_context')
    if mode == 'imported' and raw_context in (None, ''):
        raw_context = []
    if not isinstance(raw_context, list) or (not raw_context and mode != 'imported'):
        raise ValueError('draft.reference_context must be a non-empty list')
    context = [dict(item) if isinstance(item, Mapping) else str(item).strip() for item in raw_context]
    context_texts = [str(item.get('text') or '').strip() if isinstance(item, Mapping) else item for item in context]
    if mode != 'imported' and any(not text for text in context_texts):
        raise ValueError('draft.reference_context text must be non-empty')

    key_points = enhancement.get('key_points')
    if not isinstance(key_points, list) or (not key_points and mode != 'imported'):
        raise ValueError('enhancement.key_points must be a non-empty list')
    reasoning_steps = [
        str(item.get('statement') or '').strip() if isinstance(item, Mapping) else str(item).strip()
        for item in key_points
    ]
    if mode != 'imported' and any(not statement for statement in reasoning_steps):
        raise ValueError('enhancement key point statements must be non-empty')

    source_preparation['context_reference'] = [dict(item) if isinstance(item, Mapping) else item for item in context]
    source_preparation['dataset_enhancement'] = dict(enhancement)
    case_source = dict(_mapping(source_preparation.get('case_source', {}), 'case_source'))
    case_source.update({
        'final_id': case_id,
        'original_id': str(case_source.get('original_id') or case_id),
        'source': str(case_source.get('source') or ('imported_csv' if mode == 'imported' else 'generated')),
    })
    source_preparation['case_source'] = case_source

    reference_doc_ids = [str(value).strip() for value in draft.get('reference_doc_ids') or []]
    if (not reference_doc_ids and mode != 'imported') or any(not value for value in reference_doc_ids):
        raise ValueError('draft.reference_doc_ids must contain non-empty values')
    question_type = str(draft.get('question_type') or '').strip()
    if question_type not in {'precision', 'reasoning'}:
        raise ValueError(f'unsupported dataset question type: {question_type}')
    difficulty = str(draft.get('difficulty') or '').strip()
    return normalize_eval_case({
        'id': case_id,
        'question': draft.get('question'),
        'answer': draft.get('answer'),
        'ground_truth': draft.get('answer'),
        'question_type': question_type,
        'difficulty': difficulty,
        'grading_guidance': draft.get('grading_guidance'),
        'key_points': [dict(item) if isinstance(item, Mapping) else item for item in key_points],
        'forbidden_claims': list(enhancement.get('forbidden_claims') or []),
        'generate_reason': draft.get('generate_reason'),
        'is_deleted': bool(draft.get('is_deleted', False)),
        'reasoning_steps': reasoning_steps,
        'difficulty_rationale': f'{difficulty} case using {len(context)} reference chunks',
        'type_rationale': f'adapted from dataset question type {draft.get("question_type")}',
        'reference_chunk_ids': list(draft.get('reference_chunk_ids') or []),
        'reference_context': context if mode == 'imported' else context_texts,
        'reference_doc': list(draft.get('reference_doc') or []) if mode == 'imported' else context_texts,
        'reference_doc_ids': reference_doc_ids,
        'source_message_id': '',
        'source_preparation': source_preparation,
    }, default_id=case_id)
__all__ = ['dataset_operations']
