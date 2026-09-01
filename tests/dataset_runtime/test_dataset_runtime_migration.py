from __future__ import annotations

import asyncio
import csv
from dataclasses import replace
import importlib
import json
from types import SimpleNamespace

from evo import artifacts as A
from evo.artifact_flow import ArtifactFlow, FlowDefinition, FlowStage
from evo.artifact_runtime import ArtifactCommit, ArtifactDraft, ArtifactKey
from evo.operations.dataset.source_config import normalize_source_config
from evo.operations.dataset.import_cases import import_cases
from evo.operations.dataset.topic_discovery import topic_discovery_embedding_cluster
from evo.operations.flow import evo_flow_definition


dataset_module = importlib.import_module('evo.operations.dataset.operations')
chunks_module = importlib.import_module('evo.operations.dataset.chunks_build')
select_docs_module = importlib.import_module('evo.operations.dataset.select_docs')


def test_current_flow_keeps_one_dataset_stage() -> None:
    definition = evo_flow_definition()
    dataset_ids = {
        operation.spec.op_id
        for operation in definition.stage_operations(0)
    }

    assert tuple(stage.name for stage in definition.stages) == A.STEPS
    assert tuple(stage.name for stage in definition.stages) == (
        'dataset', 'eval', 'analysis', 'repair', 'abtest',
    )
    assert len(definition.stage_operations(0)) == 21
    assert dataset_ids == {
        operation.spec.op_id
        for operation in dataset_module.dataset_operations()
    }
    assert dataset_ids.isdisjoint({
        'dataset.load_corpus',
        'dataset.build_corpus_snapshot',
        'dataset.case_requests',
        'dataset.prepare_case',
    })
    assert definition.stages[0].result_key == ArtifactKey.scalar(A.EVAL_DATASET)


def test_source_config_adapts_current_service_shape() -> None:
    assert normalize_source_config({
        'kb_id': ['kb-a'],
        'csv_data': [{'kb-b': '/tmp/b.csv'}],
        'target_case_count': 3,
    }) == {
        'kb_ids': ['kb-a', 'kb-b'],
        'csv_sources': [{'kb_id': 'kb-b', 'path': '/tmp/b.csv'}],
        'target_case_count': 3,
    }
    assert normalize_source_config({
        'kb_ids': ['kb-a'],
        'csv_sources': [{'kb_id': 'kb-a', 'path': '/tmp/a.csv'}],
        'target_case_count': 1,
    }) == {
        'kb_ids': ['kb-a'],
        'csv_sources': [{'kb_id': 'kb-a', 'path': '/tmp/a.csv'}],
        'target_case_count': 1,
    }


def test_default_isolated_operation_executes_in_subprocess(tmp_path) -> None:
    async def run() -> None:
        definition = FlowDefinition(
            (dataset_module.import_cases_operation,),
            (FlowStage(
                'dataset-import',
                ArtifactKey.scalar(A.DATASET_IMPORT_CASES_MANIFEST),
            ),),
        )
        flow = await ArtifactFlow.open(tmp_path / 'runtime-isolated', definition)
        try:
            source_key = ArtifactKey.scalar(A.CORPUS_SOURCE_CONFIG)
            await flow.create('run-isolated', ArtifactCommit(
                'seed:run-isolated',
                'user:create',
                (ArtifactDraft(source_key, {
                    'kb_id': ['kb-a'],
                    'csv_data': [],
                    'target_case_count': 1,
                }),),
                {source_key: None},
            ))
            await flow.start('run-isolated')
            snapshot = await flow.wait_until_boundary('run-isolated', timeout=10)

            assert snapshot.status == 'completed', snapshot.runtime.error
            record = await flow.head(
                'run-isolated',
                ArtifactKey.scalar(A.DATASET_IMPORT_CASES_MANIFEST),
            )
            assert record is not None
            manifest = await flow.read('run-isolated', record.ref)
            assert manifest['source']['csv_sources'] == []
            assert manifest['stats']['case_allocation']['target_case_count'] == 1
            assert manifest['stats']['case_allocation']['auto_case_count'] == 1
        finally:
            await flow.close()

    asyncio.run(run())


def test_multi_knowledge_base_csv_import_uses_current_source_shape(tmp_path) -> None:
    paths = []
    for index, kb_id in enumerate(('kb-a', 'kb-b'), 1):
        path = tmp_path / f'{kb_id}.csv'
        with path.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                'id', 'question', 'answer', 'question_type', 'difficulty',
                'grading_guidance', 'reference_context',
            ))
            writer.writeheader()
            writer.writerow({
                'id': f'source-{index}',
                'question': f'Question {index}?',
                'answer': f'Fact {index}',
                'question_type': 'precision',
                'difficulty': 'easy',
                'grading_guidance': f'Require Fact {index}',
                'reference_context': json.dumps([
                    {'chunk_id': f'chunk-{index}', 'text': f'Fact {index}'},
                ]),
            })
        paths.append(path)

    config = normalize_source_config({
        'kb_id': [],
        'csv_data': [
            {'kb-a': str(paths[0])},
            {'kb-b': str(paths[1])},
        ],
        'target_case_count': 2,
    })
    manifest = import_cases(
        None,
        {'source_config': config},
        kb_client=_FakeImportKnowledgeBaseClient(),
    )['import_cases_manifest']

    allocation = manifest['stats']['case_allocation']
    assert allocation['import_case_count'] == 2
    assert allocation['auto_case_count'] == 0
    assert tuple(allocation['assignments']) == ('case_0001', 'case_0002')
    assert [item['source_row_number'] for item in manifest['details']] == [1, 2]
    assert manifest['details'][1]['case']['source_preparation']['case_source']['kb_id'] == 'kb-b'


def test_real_umap_hdbscan_path_is_available() -> None:
    chunks = tuple({
        'available': True,
        'chunk_id': f'chunk-{index}',
        'embedding': {
            'model': 'embed',
            'vector': [1.0, index / 10, (index % 2) / 10],
        },
    } for index in range(1, 9))

    result = topic_discovery_embedding_cluster(None, {
        'chunk': chunks,
        'topic_discovery_embedding_cluster_params': {
            'umap_n_neighbors': 2,
            'umap_n_components': 2,
            'min_cluster_size': 2,
            'min_samples': 1,
        },
    })['embedding_cluster_candidates']

    assert result['stats']['embedding_chunk_count'] == 8
    assert result['stats']['candidate_count'] >= 1


def test_import_only_pipeline_runs_on_new_artifact_runtime(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        for operation in dataset_module.dataset_operations():
            monkeypatch.setattr(operation, 'spec', replace(operation.spec, execution='cooperative'))

        monkeypatch.setattr(dataset_module, 'import_cases', _fake_import_cases)
        monkeypatch.setattr(dataset_module, 'generate_enhance', _fake_generate_enhance)
        monkeypatch.setattr(dataset_module, '_llm_complete', lambda _config: _fake_llm_complete)

        definition = FlowDefinition(
            dataset_module.dataset_operations(),
            (FlowStage('dataset', ArtifactKey.scalar(A.EVAL_DATASET)),),
        )
        flow = await ArtifactFlow.open(tmp_path / 'runtime', definition)
        try:
            keys = (
                ArtifactKey.scalar(A.RUN_CONFIG),
                ArtifactKey.scalar(A.CORPUS_SOURCE_CONFIG),
            )
            await flow.create('run-imported', ArtifactCommit(
                'seed:run-imported',
                'user:create',
                (
                    ArtifactDraft(keys[0], {'llm_config': {'evo_llm': {}}}),
                    ArtifactDraft(keys[1], {
                        'kb_id': ['kb-a'],
                        'csv_data': [],
                        'target_case_count': 1,
                    }),
                ),
                {key: None for key in keys},
            ))
            await flow.start('run-imported')
            snapshot = await flow.wait_until_boundary('run-imported', timeout=10)

            assert snapshot.status == 'completed', (
                snapshot.runtime.error,
                [(failure.operation_id, failure.error) for failure in snapshot.runtime.case_failures],
            )
            dataset_record = await flow.head('run-imported', ArtifactKey.scalar(A.EVAL_DATASET))
            assert dataset_record is not None
            dataset = await flow.read('run-imported', dataset_record.ref)
            assert dataset['case_num'] == 1
            assert dataset['failed_case_num'] == 0
            assert dataset['cases'][0]['question_type'] == 'single_hop'
            assert dataset['cases'][0]['reasoning_steps'] == ['Imported fact']
            assert dataset['cases'][0]['key_points'][0]['statement'] == 'Imported fact'
            assert dataset['cases'][0]['forbidden_claims'] == []

            case_record = await flow.head(
                'run-imported', ArtifactKey.partition(A.EVAL_CASE, 'case_0001'),
            )
            assert case_record is not None
            case = await flow.read('run-imported', case_record.ref)
            assert case['key_points'][0]['statement'] == 'Imported fact'
            assert case['key_points'][0]['evidence_chunk_ids'] == ['chunk-imported']
            assert case['forbidden_claims'] == []
            assert case['source_preparation']['dataset_enhancement']['forbidden_claims'] == []
            assert case['source_preparation']['context_reference'][0]['chunk_id'] == 'chunk-imported'

            chunk_set_record = await flow.head(
                'run-imported', ArtifactKey.scalar(A.DATASET_CHUNK_REQUESTS),
            )
            assert chunk_set_record is not None
            chunk_set = await flow.read('run-imported', chunk_set_record.ref)
            assert chunk_set.keys == ()
        finally:
            await flow.close()

    asyncio.run(run())


def test_generated_pipeline_runs_with_dynamic_chunk_and_case_partitions(monkeypatch, tmp_path) -> None:
    async def run() -> None:
        for operation in dataset_module.dataset_operations():
            monkeypatch.setattr(operation, 'spec', replace(operation.spec, execution='cooperative'))

        monkeypatch.setattr(chunks_module, 'KnowledgeBaseClient', _FakeKnowledgeBaseClient)
        monkeypatch.setattr(select_docs_module, 'KnowledgeBaseClient', _FakeKnowledgeBaseClient)
        monkeypatch.setattr(dataset_module, '_llm_complete', lambda _config: _fake_llm_complete)

        definition = FlowDefinition(
            dataset_module.dataset_operations(),
            (FlowStage('dataset', ArtifactKey.scalar(A.EVAL_DATASET)),),
        )
        flow = await ArtifactFlow.open(tmp_path / 'runtime-generated', definition)
        try:
            keys = (
                ArtifactKey.scalar(A.RUN_CONFIG),
                ArtifactKey.scalar(A.CORPUS_SOURCE_CONFIG),
            )
            await flow.create('run-generated', ArtifactCommit(
                'seed:run-generated',
                'user:create',
                (
                    ArtifactDraft(keys[0], {'llm_config': {'evo_llm': {}}}),
                    ArtifactDraft(keys[1], {
                        'kb_id': ['kb-a'],
                        'csv_data': [],
                        'target_case_count': 1,
                    }),
                ),
                {key: None for key in keys},
            ))
            await flow.start('run-generated')
            snapshot = await flow.wait_until_boundary('run-generated', timeout=10)

            assert snapshot.status == 'completed'
            chunk_set_record = await flow.head(
                'run-generated', ArtifactKey.scalar(A.DATASET_CHUNK_REQUESTS),
            )
            assert chunk_set_record is not None
            chunk_set = await flow.read('run-generated', chunk_set_record.ref)
            assert chunk_set.keys == ('chunk_0001', 'chunk_0002')

            case_set_record = await flow.head(
                'run-generated', ArtifactKey.scalar(A.EVAL_CASE_REQUESTS),
            )
            assert case_set_record is not None
            case_set = await flow.read('run-generated', case_set_record.ref)
            assert case_set.keys == ('case_0001',)

            dataset_record = await flow.head('run-generated', ArtifactKey.scalar(A.EVAL_DATASET))
            assert dataset_record is not None
            dataset = await flow.read('run-generated', dataset_record.ref)
            assert dataset['case_num'] == 1
            assert dataset['cases'][0]['question'] == 'What is Topic?'
            assert dataset['cases'][0]['source'] == 'generated'
        finally:
            await flow.close()

    asyncio.run(run())


def _fake_import_cases(_ctx, _inputs):
    case = {
        'id': 'case_0001',
        'question': 'What is the imported fact?',
        'answer': 'Imported fact',
        'question_type': 'precision',
        'difficulty': 'easy',
        'grading_guidance': 'The answer must contain the imported fact.',
        'reference_context': [{'chunk_id': 'chunk-imported', 'text': 'Imported fact'}],
        'reference_chunk_ids': ['chunk-imported'],
        'reference_doc_ids': ['doc-imported'],
        'source_preparation': {
            'kb_ids': ['kb-a'],
            'case_source': {
                'final_id': 'case_0001',
                'original_id': 'source-1',
                'source': 'imported_csv',
                'kb_id': 'kb-a',
                'csv_path': '/tmp/import.csv',
            },
        },
    }
    return {'import_cases_manifest': {
        'source': {'csv_sources': []},
        'stats': {
            'csv_reading': {'total_row_count': 1, 'valid_row_count': 1, 'loaded_row_count': 1},
            'case_allocation': {
                'target_case_count': 1,
                'import_case_count': 1,
                'auto_case_count': 0,
                'assignments': {
                    'case_0001': {'mode': 'imported', 'source_row_number': 1},
                },
            },
        },
        'details': [{
            'source_row_number': 1,
            'load_status': 'loaded',
            'case_id': 'case_0001',
            'case': case,
        }],
    }}


def _fake_generate_enhance(_ctx, _inputs, llm_complete=None):
    assert callable(llm_complete)
    return {'case_enhance': {
        'key_points': [{
            'id': 'key_point_1',
            'statement': 'Imported fact',
            'evidence_chunk_ids': ['chunk-imported'],
        }],
        'forbidden_claims': [],
    }}


class _FakeKnowledgeBaseClient:
    def list_documents(self, kb_id):
        assert kb_id == 'kb-a'
        return [{'doc_id': 'doc-a', 'filename': 'doc-a.txt', 'file_type': 'text', 'status': 'ready'}]

    def count_valid_chunks(self, kb_id, doc_ids, groups, allowed_types, max_scan_chunks):
        assert kb_id == 'kb-a'
        assert doc_ids == ['doc-a']
        assert groups == ['block']
        assert 'text' in allowed_types
        assert max_scan_chunks >= 2
        return {
            'scanned_count': 2,
            'effective_count': 2,
            'capacities': {'block': {'doc-a': 2}},
            'filtered_count_by_type': {},
            'invalid_count_by_reason': {},
        }

    def fetch_valid_chunks(self, kb_id, doc_id, group, allowed_types, limit, *, order_by):
        assert (kb_id, doc_id, group) == ('kb-a', 'doc-a', 'block')
        assert order_by == 'stable_chunk_id_hash'
        return [
            SimpleNamespace(
                uid=f'chunk-source-{index}',
                text=f'Topic fact {index}',
                group='block',
                number=index,
                embedding={'embed': [1.0, float(index)]},
                metadata={'type': 'text'},
                global_metadata={},
            )
            for index in range(1, limit + 1)
        ]


class _FakeImportKnowledgeBaseClient:
    def list_documents(self, kb_id):
        return [{'doc_id': f'doc-{kb_id[-1]}'}]

    def iter_chunks(self, kb_id, doc_ids, groups, page_size, *, require_embeddings):
        assert groups == ['block', 'line']
        assert page_size == 200
        assert require_embeddings is False
        index = 1 if kb_id == 'kb-a' else 2
        yield [SimpleNamespace(uid=f'chunk-{index}', text=f'Fact {index}')]


def _fake_llm_complete(prompt: str) -> str:
    if '"entities"' in prompt:
        return '{"entities":["Topic"]}'
    if '"topics"' in prompt:
        return '{"topics":["Topic"]}'
    if '"key_points"' in prompt:
        return '{"key_points":[{"statement":"Topic fact","evidence_reference_ids":["ref_1"]}]}'
    if '"forbidden_claims"' in prompt:
        return '{"forbidden_claims":[]}'
    return '{"question":"What is Topic?","answer":"Topic fact","grading_guidance":"Require Topic fact."}'
