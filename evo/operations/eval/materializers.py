from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable

from evo.operations.public_contracts import build_eval_summary_root

from .answer import answer_case
from .answer_process import build_answer_process_panel
from .judge import judge_case

EVAL_FRONTEND_VIEW_VERSION = 'eval_frontend_view.v1'
UNSCORED = {'infra_failure', 'judge_contract_error', 'dataset_contract_error'}
SCORES = (
    'answer_correctness',
    'answer_relevance',
    'completeness',
    'groundedness',
    'format_compliance',
    'key_point_recall',
    'key_point_precision',
    'semantic_similarity',
    'claim_support_rate',
    'unsupported_claim_rate',
    'retrieval_hit_at_k',
    'retrieval_recall_at_k',
    'retrieval_precision_at_k',
    'retrieval_mrr',
    'retrieval_ndcg',
    'context_relevance_avg',
    'context_noise_rate',
    'answer_quality_score',
    'retrieval_quality_score',
    'overall_score',
)
OPTIONAL_SCORES = (
    'numeric_accuracy',
    'list_set_f1',
    'contradiction_rate',
)
EXPLANATIONS = (
    'matched_key_points',
    'missing_points',
    'wrong_points',
    'extra_points',
    'unsupported_claims',
    'evidence_mapping',
    'claims',
)
OPTIONAL_EXPLANATIONS = (
    'contradicted_claims',
)
FRONTEND_STAGE_NODES = (
    ('retrieval_evidence', '检索证据'),
    ('answer_generation', '生成回答'),
    ('multi_dimension_judge', '多维评测'),
    ('result_archive', '结果归档'),
)
FRONTEND_RETRIEVAL_METRICS = (
    'retrieval_hit_at_k',
    'retrieval_recall_at_k',
    'retrieval_precision_at_k',
    'retrieval_mrr',
    'retrieval_ndcg',
)


def eval_materializers() -> dict[str, Callable[[Any, Mapping[str, object]], Mapping[str, object]]]:
    def answer(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
        return {'answer': answer_case(_mapping(inputs['case'], 'case'),
                                      _mapping(inputs.get('target_config') or {}, 'target_config'))}

    def judge(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
        return {'judge': judge_case(_case_with_enhance(inputs),
                                    _mapping(inputs['answer'], 'answer'),
                                    _mapping(inputs.get('policy') or {}, 'policy'))}

    def summary(ctx: Any, inputs: Mapping[str, object]) -> Mapping[str, object]:
        judges = inputs.get('judges')
        if not isinstance(judges, tuple):
            raise ValueError('eval.summary judges input must be a partitioned tuple')
        return {'summary': build_eval_summary_root(ctx.run_id, judges)}

    return {'eval.answer': answer, 'eval.judge': judge, 'eval.summary': summary}


def build_eval_detail_summary(judges: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for index, judge in enumerate(judges, 1):
        if not isinstance(judge, Mapping):
            rows.append({
                'case_id': f'invalid_{index:04d}',
                'kb_id': '',
                'question': '',
                'question_type': '',
                'difficulty': '',
                'ground_truth': '',
                'rag_answer': '',
                **{key: 0.0 for key in ('answer_score', 'retrieval_score', *SCORES)},
                'quality_label': 'infra_failure',
                'failure_type': 'judge_contract_error',
                'retrieval_failure_type': 'not_applicable',
                'reason': 'judge result is not a mapping',
                'defect': 'judge_contract_error',
                'reference_chunk_ids': [],
                'reference_doc_ids': [],
                'retrieve_chunk_ids': [],
                'retrieve_doc_ids': [],
                'retrieve_contexts': [],
            'retrieved_contexts': [],
            **{key: [] for key in EXPLANATIONS},
            **{key: [] for key in OPTIONAL_EXPLANATIONS},
            'metric_layers': {},
                'score_breakdown': {},
                'trace_id': '',
                'target': {},
            })
            continue
        case = judge.get('case') if isinstance(judge.get('case'), Mapping) else {}
        answer = judge.get('rag_answer') if isinstance(judge.get('rag_answer'), Mapping) else {}
        target = judge.get('target') if isinstance(judge.get('target'), Mapping) else {}
        chat_error = answer.get('chat_error') if isinstance(answer.get('chat_error'), Mapping) else {}
        rows.append({
            'case_id': str(judge.get('case_id') or case.get('id') or ''),
            'kb_id': str(target.get('kb_id') or ''),
            'question': str(case.get('question') or ''),
            'question_type': str(case.get('question_type') or ''),
            'difficulty': str(case.get('difficulty') or ''),
            'ground_truth': case.get('answer'),
            'key_points': case.get('key_points') or [],
            'rag_answer': answer.get('answer'),
            **{key: judge.get(key, 0.0) for key in SCORES},
            **{key: judge[key] for key in OPTIONAL_SCORES if key in judge},
            'answer_score': judge.get('answer_quality_score', 0.0),
            'retrieval_score': judge.get('retrieval_quality_score', 0.0),
            'quality_label': str(judge.get('quality_label') or ''),
            'failure_type': str(judge.get('failure_type') or ''),
            'retrieval_failure_type': str(judge.get('retrieval_failure_type') or ''),
            'reason': str(judge.get('reason') or ''),
            'defect': str(judge.get('defect') or ''),
            'chat_error_type': str(chat_error.get('type') or ''),
            'chat_error_message': str(chat_error.get('message') or ''),
            'reference_chunk_ids': case.get('reference_chunk_ids') or [],
            'reference_doc_ids': case.get('reference_doc_ids') or [],
            'retrieve_chunk_ids': answer.get('chunk_ids') or [],
            'retrieve_doc_ids': answer.get('doc_ids') or [],
            'retrieve_contexts': answer.get('contexts') or [],
            'retrieved_contexts': answer.get('contexts') or [],
            **{key: judge.get(key) or [] for key in EXPLANATIONS},
            **{key: judge[key] for key in OPTIONAL_EXPLANATIONS if key in judge},
            'metric_layers': judge.get('metric_layers') if isinstance(judge.get('metric_layers'), Mapping) else {},
            'score_breakdown': judge.get('score_breakdown') if isinstance(judge.get('score_breakdown'), Mapping) else {},
            'trace_id': str(judge.get('trace_id') or ''),
            'answer_process': _row_answer_process(answer, judge),
            'target': dict(target),
        })
    scored = [row for row in rows if row['failure_type'] not in UNSCORED and row['quality_label'] != 'infra_failure']
    failures = [
        {
            'case_id': str(row.get('case_id') or ''),
            'kb_id': str(row.get('kb_id') or ''),
            'failure_type': str(row.get('failure_type') or ''),
            'reason': str(row.get('reason') or ''),
            'chat_error_type': str(row.get('chat_error_type') or ''),
            'chat_error_message': str(row.get('chat_error_message') or ''),
        }
        for row in rows
        if row['failure_type'] in {'infra_failure', 'judge_contract_error', 'dataset_contract_error'}
    ]
    routing_failures = [row for row in failures if row['failure_type'] == 'dataset_contract_error']
    execution_failures = [row for row in failures if row['failure_type'] != 'dataset_contract_error']
    summary = {
        'id': 'eval.summary',
        'total': len(rows),
        'case_ids': [row['case_id'] for row in rows],
        'metrics': {
            'scored_count': len(scored),
            'overall_score_avg': _avg(scored, 'overall_score'),
            'answer_quality_score_avg': _avg(scored, 'answer_quality_score'),
            'retrieval_quality_score_avg': _avg(
                [row for row in scored if row['retrieval_failure_type'] != 'not_applicable'],
                'retrieval_quality_score',
            ),
            'answer_correctness_avg': _avg(scored, 'answer_correctness'),
            'groundedness_avg': _avg(scored, 'groundedness'),
            'answer_relevance_avg': _avg(scored, 'answer_relevance'),
            'correct_rate': round(sum(1 for row in scored if row['quality_label'] == 'good') / len(scored), 4)
            if scored else 0.0,
            'key_point_recall_avg': _avg(scored, 'key_point_recall'),
            'claim_support_rate_avg': _avg(scored, 'claim_support_rate'),
            'retrieval_mrr_avg': _avg(
                [row for row in scored if row['retrieval_failure_type'] != 'not_applicable'],
                'retrieval_mrr',
            ),
            'retrieval_ndcg_avg': _avg(
                [row for row in scored if row['retrieval_failure_type'] != 'not_applicable'],
                'retrieval_ndcg',
            ),
            **_optional_metric_avgs(scored),
        },
        'by_question_type': _group_metrics(scored, 'question_type'),
        'by_difficulty': _group_metrics(scored, 'difficulty'),
        'top_missing_points': _top_items(rows, 'missing_points'),
        'top_unsupported_claims': _top_items(rows, 'unsupported_claims'),
        'quality_counts': dict(Counter(row['quality_label'] for row in rows)),
        'failure_type_counts': dict(Counter(row['failure_type'] for row in rows)),
        'retrieval_failure_type_counts': dict(Counter(row['retrieval_failure_type'] for row in rows)),
        'bad_cases': [row for row in rows if row['quality_label'] != 'good'],
        'routing_failures': routing_failures,
        'execution_failures': execution_failures,
        'checks': {'ready': not routing_failures and not execution_failures,
                   'errors': routing_failures + execution_failures},
        'rows': rows,
    }
    summary.update(build_eval_frontend_view(summary))
    return summary


def build_eval_frontend_view(
    summary: Mapping[str, Any],
    *,
    live_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable backend view model for the eval overview/list/detail UI.

    The raw eval report is intentionally rich and judge-oriented.  This wrapper
    keeps the old contract intact while giving frontend a direct shape for the
    first-week P0 UX: metric cards, a case result table, and four-tab details.

    live_progress may override totals/running/current_case using gate provenance
    (answered vs judged partitions). Cases still in answer materialization before
    an artifact exists remain event-stream only for P0.
    """
    rows = _list_of_mappings(summary.get('rows'))
    overview = _frontend_overview(summary, rows, live_progress=live_progress)
    case_overviews = [_frontend_case_overview(row) for row in rows]
    case_details = [_frontend_case_detail(row) for row in rows]
    return {
        'frontend_view_version': EVAL_FRONTEND_VIEW_VERSION,
        'overview': overview,
        'case_overviews': case_overviews,
        'case_details': case_details,
    }


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f'{name} must be a mapping')
    return value


def _case_with_enhance(inputs: Mapping[str, object]) -> Mapping[str, Any]:
    case = dict(_mapping(inputs['case'], 'case'))
    enhance = inputs.get('case_enhance')
    if enhance is None:
        return case
    enhance = _mapping(enhance, 'case_enhance')
    for key in ('key_points', 'forbidden_claims'):
        if key in enhance:
            case[key] = enhance[key]
    return case


def _avg(rows: list[Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key) or 0.0) for row in rows]
    return round(sum(values) / len(values), 4) if values else 0.0


def _optional_metric_avgs(rows: list[Mapping[str, Any]]) -> dict[str, float]:
    return {
        f'{key}_avg': _avg([row for row in rows if key in row], key)
        for key in OPTIONAL_SCORES
        if any(key in row for row in rows)
    }


def _group_metrics(rows: list[Mapping[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        label = str(row.get(key) or 'unknown')
        groups.setdefault(label, []).append(row)
    return {
        label: {
            'count': len(group),
            'overall_score_avg': _avg(group, 'overall_score'),
            'answer_quality_score_avg': _avg(group, 'answer_quality_score'),
            'retrieval_quality_score_avg': _avg(group, 'retrieval_quality_score'),
            'key_point_recall_avg': _avg(group, 'key_point_recall'),
            'claim_support_rate_avg': _avg(group, 'claim_support_rate'),
            'correct_rate': round(sum(1 for row in group if row['quality_label'] == 'good') / len(group), 4),
        }
        for label, group in sorted(groups.items())
    }


def _top_items(rows: list[Mapping[str, Any]], key: str, *, limit: int = 10) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    examples: dict[str, str] = {}
    for row in rows:
        values = row.get(key)
        if not isinstance(values, list):
            continue
        for value in values:
            text = _item_text(value)
            if not text:
                continue
            counts[text] += 1
            examples.setdefault(text, str(row.get('case_id') or ''))
    return [
        {'text': text, 'count': count, 'example_case_id': examples.get(text, '')}
        for text, count in counts.most_common(limit)
    ]


def _item_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get('statement') or value.get('text') or value.get('claim') or value)[:300]
    return str(value or '')[:300]


def _list_of_mappings(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _frontend_overview(
    summary: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    *,
    live_progress: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = summary.get('metrics') if isinstance(summary.get('metrics'), Mapping) else {}
    live = live_progress if isinstance(live_progress, Mapping) else {}
    failure_counts = Counter(_frontend_failure_label(row) for row in rows if _frontend_is_badcase(row))
    completed_cases = int(live.get('completed_cases') if live.get('completed_cases') is not None else len(rows))
    total_cases = int(live.get('total_cases') or summary.get('total') or completed_cases or 0)
    running_cases = int(live.get('running_cases') or 0)
    pending_cases = int(live.get('pending_cases') or max(0, total_cases - completed_cases - running_cases))
    current_case = live.get('current_case') if isinstance(live.get('current_case'), Mapping) else {}
    if not current_case:
        current_case = {'case_id': '', 'stage': '', 'stage_label': ''}
    percent = round(min(100.0, completed_cases * 100.0 / total_cases), 2) if total_cases else (100 if rows else 0)
    return {
        'total_cases': total_cases,
        'completed_cases': completed_cases,
        'running_cases': running_cases,
        'pending_cases': pending_cases,
        'failed_cases': len(summary.get('execution_failures') or []) + len(summary.get('routing_failures') or []),
        'average_scores': {
            'overall': _round_score(metrics.get('overall_score_avg')),
            'answer_quality': _round_score(metrics.get('answer_quality_score_avg')),
            'retrieval_quality': _round_score(metrics.get('retrieval_quality_score_avg')),
        },
        'badcase_count': sum(1 for row in rows if _frontend_is_badcase(row)),
        'main_failure_types': [
            {'type': label, 'count': count}
            for label, count in failure_counts.most_common(4)
        ],
        'current_case': {
            'case_id': str(current_case.get('case_id') or ''),
            'stage': str(current_case.get('stage') or ''),
            'stage_label': str(
                current_case.get('stage_label')
                or _frontend_stage_label(str(current_case.get('stage') or ''))
            ),
        },
        'progress': {
            'current': completed_cases,
            'total': total_cases,
            'percent': percent,
        },
        'progress_note': (
            'running_cases counts answered-but-not-judged partitions from gate provenance; '
            'cases still inside answer materialization before an artifact exists should use the event stream'
        ),
    }


def _frontend_case_overview(row: Mapping[str, Any]) -> dict[str, Any]:
    failure_type = str(row.get('failure_type') or '')
    retrieval_failure_type = str(row.get('retrieval_failure_type') or '')
    return {
        'case_id': str(row.get('case_id') or ''),
        'question': str(row.get('question') or row.get('query') or ''),
        'question_summary': _truncate(str(row.get('question') or row.get('query') or ''), 80),
        'status': _frontend_case_status(row),
        'stage_nodes': _frontend_stage_nodes(row),
        'scores': _frontend_scores(row),
        'quality_label': str(row.get('quality_label') or ''),
        'failure_type': failure_type,
        'failure_label': _frontend_failure_label(row),
        'retrieval_failure_type': retrieval_failure_type,
        'retrieval_conclusion': _frontend_retrieval_conclusion(row),
        'judge_summary': str(row.get('reason') or row.get('judge_reason') or row.get('defect') or ''),
        'trace_id': str(row.get('trace_id') or ''),
        'has_trace': bool(row.get('trace_id')),
        'has_retrieval_evidence': bool(_frontend_evidence_items(row)),
        'is_badcase': _frontend_is_badcase(row),
    }


def _frontend_case_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    question = str(row.get('question') or row.get('query') or '')
    answer = str(row.get('rag_answer') or row.get('answer') or '')
    reason = str(row.get('reason') or row.get('judge_reason') or row.get('defect') or '')
    return {
        'case_id': str(row.get('case_id') or ''),
        'tabs': {
            'overview': {
                'question': question,
                'reference_answer': row.get('ground_truth') or row.get('reference') or '',
                'key_points': _frontend_text_items(row.get('key_points') or row.get('keypoints')),
                'model_answer': answer,
                'scores': _frontend_scores(row),
                'failure_type': str(row.get('failure_type') or ''),
                'failure_label': _frontend_failure_label(row),
                'retrieval_failure_type': str(row.get('retrieval_failure_type') or ''),
                'judge_summary': reason,
                'copy_payload': {
                    'question': question,
                    'answer': answer,
                    'judge_conclusion': reason,
                },
            },
            'retrieval_evidence': {
                'conclusion': _frontend_retrieval_conclusion(row),
                'items': _frontend_evidence_items(row),
                'reference_chunk_ids': _frontend_text_items(row.get('reference_chunk_ids')),
                'reference_doc_ids': _frontend_text_items(row.get('reference_doc_ids')),
            },
            'answer_process': {
                'question': question,
                'retrieval_context_summary': _retrieval_context_summary(row),
                'final_answer': answer,
                **build_answer_process_panel(row, load_trace=not bool(row.get('answer_process'))),
            },
            'judge_evaluation': {
                'scores': _frontend_scores(row),
                'key_points': {
                    'matched': _frontend_text_items(row.get('matched_key_points')),
                    'missing': _frontend_text_items(row.get('missing_points')),
                },
                'facts': {
                    'claim_support_rate': _round_score(row.get('claim_support_rate')),
                    'unsupported_claims': _frontend_text_items(row.get('unsupported_claims')),
                    'unsupported_claim_rate': _round_score(row.get('unsupported_claim_rate')),
                },
                'retrieval_metrics': {
                    key: _round_score(row.get(key))
                    for key in FRONTEND_RETRIEVAL_METRICS
                },
                'reason': reason,
                'suggestion': _frontend_suggestion(row),
                'score_breakdown': row.get('score_breakdown') if isinstance(row.get('score_breakdown'), Mapping) else {},
                'metric_layers': row.get('metric_layers') if isinstance(row.get('metric_layers'), Mapping) else {},
            },
        },
    }


def _frontend_case_status(row: Mapping[str, Any]) -> str:
    if str(row.get('failure_type') or '') in UNSCORED or str(row.get('quality_label') or '') == 'infra_failure':
        return 'failed'
    return 'badcase' if _frontend_is_badcase(row) else 'good'


def _frontend_is_badcase(row: Mapping[str, Any]) -> bool:
    label = str(row.get('quality_label') or '')
    failure = str(row.get('failure_type') or '')
    if failure in UNSCORED or label == 'infra_failure':
        return False
    return label in {'bad', 'partial'}


def _frontend_stage_nodes(row: Mapping[str, Any]) -> list[dict[str, str]]:
    answer_failed = bool(row.get('chat_error_type') or row.get('chat_error_message')) \
        or str(row.get('failure_type') or '') in {'dataset_contract_error'}
    answer_status = 'failed' if answer_failed else 'done'
    judge_status = 'failed' if str(row.get('failure_type') or '') == 'judge_contract_error' else 'done'
    return [
        {'key': key, 'label': label, 'status': answer_status if index < 2 else judge_status}
        for index, (key, label) in enumerate(FRONTEND_STAGE_NODES)
    ]


def _frontend_stage_label(stage: str) -> str:
    return next((label for key, label in FRONTEND_STAGE_NODES if key == stage), '')


def _frontend_scores(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        'overall': _round_score(row.get('overall_score')),
        'answer_quality': _round_score(row.get('answer_quality_score') if row.get('answer_quality_score') is not None
                                      else row.get('answer_score')),
        'retrieval_quality': _round_score(row.get('retrieval_quality_score') if row.get('retrieval_quality_score') is not None
                                         else row.get('retrieval_score')),
    }


def _frontend_failure_label(row: Mapping[str, Any]) -> str:
    raw = f"{row.get('failure_type') or ''} {row.get('retrieval_failure_type') or ''}".lower()
    if 'retrieval_miss' in raw or 'miss' in raw:
        return '检索缺失'
    if 'retrieval_noise' in raw or 'noise' in raw:
        return '检索噪声'
    if 'partial' in raw or 'incomplete' in raw or 'completeness' in raw:
        return '回答不完整'
    if 'unsupported' in raw or 'hallucination' in raw or 'grounded' in raw:
        return '事实不支持'
    if 'infra' in raw or 'contract' in raw or 'chat_' in raw:
        return '执行异常'
    if 'wrong' in raw:
        return '回答错误'
    if str(row.get('quality_label') or '') == 'good':
        return '无'
    return str(row.get('failure_type') or row.get('retrieval_failure_type') or '待判断')


def _frontend_retrieval_conclusion(row: Mapping[str, Any]) -> str:
    failure = str(row.get('retrieval_failure_type') or '')
    if failure == 'none':
        return '命中'
    if failure == 'not_applicable':
        return '不适用'
    if failure == 'retrieval_miss':
        return '未命中'
    if failure == 'retrieval_noise':
        return '噪声过多'
    if failure:
        return '部分命中'
    return '待评估'


def _frontend_evidence_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    contexts = row.get('retrieve_contexts') or row.get('retrieved_contexts') or []
    if not isinstance(contexts, list):
        return []
    chunk_ids = _frontend_text_items(row.get('retrieve_chunk_ids'))
    doc_ids = _frontend_text_items(row.get('retrieve_doc_ids') or row.get('retrieve_doc'))
    reference_chunk_ids = {
        _chunk_tail(item)
        for item in _frontend_text_items(row.get('reference_chunk_ids'))
    }
    evidence_mapping = _list_of_mappings(row.get('evidence_mapping'))
    items = []
    for index, context in enumerate(contexts):
        record = context if isinstance(context, Mapping) else {}
        chunk_id = str(record.get('chunk_id') or record.get('chunkId') or record.get('id') or _safe_index(chunk_ids, index))
        doc_id = str(record.get('doc_id') or record.get('docId') or record.get('document_id') or _safe_index(doc_ids, index))
        doc_name = _doc_name(record, doc_id)
        content = str(record.get('content') or record.get('text') or context or '')
        score = record.get('score') if isinstance(record, Mapping) else None
        tail = _chunk_tail(chunk_id)
        hit_reference = bool(tail and tail in reference_chunk_ids)
        items.append({
            'rank': int(record.get('rank') or index + 1) if isinstance(record, Mapping) else index + 1,
            'doc_id': doc_id,
            'doc_name': doc_name,
            'chunk_id': chunk_id,
            'score': _round_score(score) if score is not None else None,
            'content': content,
            'hit_reference': hit_reference,
            'matched_key_points': _matched_points_for_chunk(evidence_mapping, tail),
        })
    return items


def _doc_name(record: Mapping[str, Any], doc_id: str) -> str:
    metadata = record.get('global_metadata') if isinstance(record.get('global_metadata'), Mapping) else {}
    for key in ('doc_name', 'document_name', 'title', 'file_name', 'filename', 'name'):
        text = str(record.get(key) or metadata.get(key) or '').strip()
        if text:
            return text
    return doc_id


def _matched_points_for_chunk(evidence_mapping: list[Mapping[str, Any]], chunk_tail: str) -> list[str]:
    result = []
    for item in evidence_mapping:
        support = item.get('retrieved_support') if isinstance(item.get('retrieved_support'), Mapping) else {}
        evidence_chunk_id = _chunk_tail(str(support.get('evidence_chunk_id') or support.get('chunk_id') or ''))
        if chunk_tail and evidence_chunk_id == chunk_tail:
            text = _item_text(item)
            if text:
                result.append(text)
    return result


def _retrieval_context_summary(row: Mapping[str, Any]) -> str:
    contexts = _frontend_evidence_items(row)
    return '\n\n'.join(
        f"Rank {item['rank']} · {item['doc_name'] or item['doc_id']} · {item['content']}"
        for item in contexts[:3]
        if item.get('content')
    )


def _row_answer_process(answer: Mapping[str, Any], judge: Mapping[str, Any]) -> dict[str, Any]:
    stored = answer.get('answer_process') if isinstance(answer.get('answer_process'), Mapping) else None
    if stored is None and isinstance(judge.get('answer_process'), Mapping):
        stored = judge.get('answer_process')
    row = {
        'case_id': str(judge.get('case_id') or answer.get('case_id') or ''),
        'trace_id': str(judge.get('trace_id') or answer.get('trace_id') or ''),
        'rag_answer': answer.get('answer'),
        'retrieve_contexts': answer.get('contexts') or [],
        'retrieve_chunk_ids': answer.get('chunk_ids') or [],
        'retrieve_doc_ids': answer.get('doc_ids') or [],
        'answer_process': stored or {},
    }
    return build_answer_process_panel(row, load_trace=not bool(stored), attempts=1, retry_seconds=0.0)


def _frontend_text_items(value: object) -> list[str]:
    values = value if isinstance(value, list | tuple) else [value] if value else []
    return [_item_text(item) for item in values if _item_text(item)]


def _frontend_suggestion(row: Mapping[str, Any]) -> str:
    label = _frontend_failure_label(row)
    if label == '检索缺失':
        return '优先检查召回范围、query rewrite 和参考 chunk 是否可被检索。'
    if label == '检索噪声':
        return '优先检查召回排序、rerank 阈值和噪声文档过滤。'
    if label == '回答不完整':
        return '优先检查答案是否覆盖标准答案关键点。'
    if label == '事实不支持':
        return '优先检查回答 claim 是否能在检索证据中找到支撑。'
    if label == '执行异常':
        return '优先检查 Chat 调用、路由配置和重试记录。'
    return ''


def _round_score(value: object) -> float:
    try:
        return round(max(0.0, min(1.0, float(value or 0.0))), 4)
    except (TypeError, ValueError):
        return 0.0


def _safe_index(values: list[str], index: int) -> str:
    return values[index] if 0 <= index < len(values) else ''


def _chunk_tail(value: str) -> str:
    return str(value or '').split(':')[-1]


def _truncate(value: str, limit: int) -> str:
    text = value.strip()
    return text if len(text) <= limit else f'{text[:limit]}...'
