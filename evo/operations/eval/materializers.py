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
FRONTEND_SCORE_METRIC_GUIDE = (
    {
        'key': 'overall',
        'field': 'overall_score',
        'label': '总分',
        'description': '综合答案质量与检索质量的最终分，用于总览排序和 Badcase 判断。',
    },
    {
        'key': 'answer_quality',
        'field': 'answer_quality_score',
        'label': '答案质量',
        'description': '衡量回答正确性、完整性、相关性和事实支撑程度。',
    },
    {
        'key': 'retrieval_quality',
        'field': 'retrieval_quality_score',
        'label': '检索质量',
        'description': '衡量检索是否命中参考证据，以及噪声与排序质量。不适用检索时不参与总分。',
    },
)
FRONTEND_RETRIEVAL_METRIC_GUIDE = (
    {
        'key': 'retrieval_hit_at_k',
        'label': 'Hit@K',
        'description': 'Top-K 检索结果中是否至少命中一个参考证据。',
    },
    {
        'key': 'retrieval_recall_at_k',
        'label': 'Recall@K',
        'description': '参考证据在 Top-K 中被召回的比例。',
    },
    {
        'key': 'retrieval_precision_at_k',
        'label': 'Precision@K',
        'description': 'Top-K 结果中真正相关证据的占比。',
    },
    {
        'key': 'retrieval_mrr',
        'label': 'MRR',
        'description': '第一个命中参考证据的排名倒数，越高说明相关证据排得越靠前。',
    },
    {
        'key': 'retrieval_ndcg',
        'label': 'NDCG',
        'description': '考虑排序位置的检索质量指标，相关证据越靠前分数越高。',
    },
)
FRONTEND_FACT_METRIC_GUIDE = (
    {
        'key': 'claim_support_rate',
        'label': '事实支持率',
        'description': '回答中的 claim 能被检索证据支撑的比例。',
    },
    {
        'key': 'unsupported_claim_rate',
        'label': '未支持 Claim 比例',
        'description': '回答中缺少证据支撑的 claim 占比，越高越可能幻觉。',
    },
)
FRONTEND_CASE_STATUS_GUIDE = (
    {
        'key': 'good',
        'label': '通过',
        'description': '评测通过，不计入 Badcase。',
    },
    {
        'key': 'badcase',
        'label': 'Badcase',
        'description': '质量标签为 bad 或 partial 的可评分失败样本。',
    },
    {
        'key': 'failed',
        'label': '执行异常',
        'description': '基础设施、契约或路由错误，不计入 Badcase，计入失败数。',
    },
)
FRONTEND_QUALITY_LABEL_GUIDE = (
    {
        'key': 'good',
        'label': '良好',
        'description': '回答质量达标。',
    },
    {
        'key': 'partial',
        'label': '部分正确',
        'description': '部分关键点命中或回答不完整，计入 Badcase。',
    },
    {
        'key': 'bad',
        'label': '错误',
        'description': '回答错误、事实不支持或严重缺陷，计入 Badcase。',
    },
    {
        'key': 'infra_failure',
        'label': '执行失败',
        'description': 'Chat/Judge/数据集契约异常，分数不计入均分。',
    },
)
FRONTEND_FAILURE_STATUS_GUIDE = (
    {
        'key': 'none',
        'label': '无',
        'category': 'none',
        'description': '未判定为失败，或失败类型为空。',
    },
    {
        'key': 'retrieval_miss',
        'label': '检索缺失',
        'category': 'retrieval',
        'description': '参考证据基本未被召回，优先检查召回范围与 query rewrite。',
    },
    {
        'key': 'retrieval_partial',
        'label': '检索部分命中',
        'category': 'retrieval',
        'description': '只召回了部分参考证据，可能导致回答不完整。',
    },
    {
        'key': 'retrieval_noise',
        'label': '检索噪声',
        'category': 'retrieval',
        'description': '召回了较多无关内容，优先检查 rerank 与过滤。',
    },
    {
        'key': 'wrong_answer',
        'label': '回答错误',
        'category': 'answer',
        'description': '最终回答与标准答案关键事实冲突或不正确。',
    },
    {
        'key': 'partial_answer',
        'label': '回答不完整',
        'category': 'answer',
        'description': '只覆盖了部分关键点，答案不完整。',
    },
    {
        'key': 'question_not_answered',
        'label': '未回答问题',
        'category': 'answer',
        'description': '模型没有有效回答用户问题。',
    },
    {
        'key': 'hallucination',
        'label': '事实不支持',
        'category': 'answer',
        'description': '回答包含缺少证据支撑的 claim，存在幻觉风险。',
    },
    {
        'key': 'format_error',
        'label': '格式错误',
        'category': 'answer',
        'description': '回答格式不符合题目或策略要求。',
    },
    {
        'key': 'infra_failure',
        'label': '执行异常',
        'category': 'execution',
        'description': 'Chat 调用、路由或运行时失败。',
    },
    {
        'key': 'judge_contract_error',
        'label': '评测契约异常',
        'category': 'execution',
        'description': 'Judge 输出不符合契约，无法可靠计分。',
    },
    {
        'key': 'dataset_contract_error',
        'label': '数据契约异常',
        'category': 'execution',
        'description': 'Case 缺少 kb_id 等路由/数据契约字段。',
    },
)
FRONTEND_BADCASE_RULE = (
    'Badcase 仅统计 quality_label 为 bad 或 partial 的 case；'
    'infra_failure / judge_contract_error / dataset_contract_error 计入失败数，不计入 Badcase。'
)
FRONTEND_BOUNDARY_NOTE = (
    '边界分组：good=可评分通过；badcase=可评分失败样本；'
    'exception=执行/契约异常（含 routing 与 execution），不计入均分与 Badcase。'
)
FRONTEND_BOUNDARY_GUIDE = (
    {
        'key': 'good',
        'label': '通过',
        'description': '可评分且质量达标，计入均分，不计入 Badcase/失败数。',
    },
    {
        'key': 'badcase',
        'label': 'Badcase',
        'description': '可评分但质量为 bad/partial，计入均分与 Badcase。',
    },
    {
        'key': 'exception',
        'label': '执行异常',
        'description': 'infra/judge/dataset 契约或执行失败，计入失败数，不计入均分与 Badcase。',
    },
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
        'guides': _frontend_guides(),
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
    boundaries = [_frontend_boundary(row) for row in rows]
    good_count = sum(1 for item in boundaries if item['display_group'] == 'good')
    badcase_count = sum(1 for item in boundaries if item['display_group'] == 'badcase')
    exception_count = sum(1 for item in boundaries if item['display_group'] == 'exception')
    execution_failed = sum(1 for item in boundaries if item['kind'] == 'execution_exception')
    routing_failed = sum(1 for item in boundaries if item['kind'] == 'routing_exception')
    if not execution_failed and not routing_failed:
        execution_failed = len(summary.get('execution_failures') or [])
        routing_failed = len(summary.get('routing_failures') or [])
        exception_count = max(exception_count, execution_failed + routing_failed)
    return {
        'total_cases': total_cases,
        'completed_cases': completed_cases,
        'running_cases': running_cases,
        'pending_cases': pending_cases,
        'failed_cases': exception_count,
        'exception_cases': exception_count,
        'execution_failed_cases': execution_failed,
        'routing_failed_cases': routing_failed,
        'average_scores': {
            'overall': _round_score(metrics.get('overall_score_avg')),
            'answer_quality': _round_score(metrics.get('answer_quality_score_avg')),
            'retrieval_quality': _round_score(metrics.get('retrieval_quality_score_avg')),
        },
        'badcase_count': badcase_count,
        'status_breakdown': {
            'good': good_count,
            'badcase': badcase_count,
            'exception': exception_count,
        },
        'main_failure_types': [
            {
                'type': label,
                'count': count,
                'description': next(
                    (item['description'] for item in FRONTEND_FAILURE_STATUS_GUIDE if item['label'] == label),
                    '',
                ),
            }
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
        'boundary_note': FRONTEND_BOUNDARY_NOTE,
    }


def _frontend_case_overview(row: Mapping[str, Any]) -> dict[str, Any]:
    failure_type = str(row.get('failure_type') or '')
    retrieval_failure_type = str(row.get('retrieval_failure_type') or '')
    failure_status = _frontend_failure_status(row)
    boundary = _frontend_boundary(row)
    answer_process = row.get('answer_process') if isinstance(row.get('answer_process'), Mapping) else {}
    trace = answer_process.get('trace') if isinstance(answer_process.get('trace'), Mapping) else {}
    return {
        'case_id': str(row.get('case_id') or ''),
        'question': str(row.get('question') or row.get('query') or ''),
        'question_summary': _truncate(str(row.get('question') or row.get('query') or ''), 80),
        'status': _frontend_case_status(row),
        'status_label': _frontend_case_status_label(row),
        'boundary': boundary,
        'scores_applicable': boundary['scores_applicable'],
        'stage_nodes': _frontend_stage_nodes(row),
        'scores': _frontend_scores(row) if boundary['scores_applicable'] else {
            'overall': 0.0,
            'answer_quality': 0.0,
            'retrieval_quality': 0.0,
        },
        'score_metrics': _frontend_score_metrics(row) if boundary['scores_applicable'] else [],
        'quality_label': str(row.get('quality_label') or ''),
        'failure_type': failure_type,
        'failure_label': failure_status['label'],
        'failure_status': failure_status,
        'retrieval_failure_type': retrieval_failure_type,
        'retrieval_conclusion': _frontend_retrieval_conclusion(row),
        'judge_summary': str(row.get('reason') or row.get('judge_reason') or row.get('defect') or ''),
        'trace_id': str(row.get('trace_id') or ''),
        'has_trace': bool(row.get('trace_id')),
        'trace_readable': bool(trace.get('readable')),
        'latency_available': bool(trace.get('latency_available')),
        'has_retrieval_evidence': bool(_frontend_evidence_items(row)),
        'is_badcase': _frontend_is_badcase(row),
        'is_exception': boundary['display_group'] == 'exception',
    }


def _frontend_case_detail(row: Mapping[str, Any]) -> dict[str, Any]:
    question = str(row.get('question') or row.get('query') or '')
    answer = str(row.get('rag_answer') or row.get('answer') or '')
    reason = str(row.get('reason') or row.get('judge_reason') or row.get('defect') or '')
    failure_status = _frontend_failure_status(row)
    boundary = _frontend_boundary(row)
    scores = _frontend_scores(row) if boundary['scores_applicable'] else {
        'overall': 0.0,
        'answer_quality': 0.0,
        'retrieval_quality': 0.0,
    }
    return {
        'case_id': str(row.get('case_id') or ''),
        'boundary': boundary,
        'tabs': {
            'overview': {
                'question': question,
                'reference_answer': row.get('ground_truth') or row.get('reference') or '',
                'key_points': _frontend_text_items(row.get('key_points') or row.get('keypoints')),
                'model_answer': answer,
                'scores': scores,
                'score_metrics': _frontend_score_metrics(row) if boundary['scores_applicable'] else [],
                'scores_applicable': boundary['scores_applicable'],
                'failure_type': str(row.get('failure_type') or ''),
                'failure_label': failure_status['label'],
                'failure_status': failure_status,
                'boundary': boundary,
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
                **build_answer_process_panel(row, load_trace=True, attempts=1, retry_seconds=0.0),
            },
            'judge_evaluation': {
                'scores': scores,
                'score_metrics': _frontend_score_metrics(row) if boundary['scores_applicable'] else [],
                'scores_applicable': boundary['scores_applicable'],
                'key_points': {
                    'matched': _frontend_text_items(row.get('matched_key_points')),
                    'missing': _frontend_text_items(row.get('missing_points')),
                },
                'facts': {
                    'claim_support_rate': _round_score(row.get('claim_support_rate')),
                    'unsupported_claims': _frontend_text_items(row.get('unsupported_claims')),
                    'unsupported_claim_rate': _round_score(row.get('unsupported_claim_rate')),
                    'metric_guide': [dict(item) for item in FRONTEND_FACT_METRIC_GUIDE],
                },
                'retrieval_metrics': {
                    key: _round_score(row.get(key))
                    for key in FRONTEND_RETRIEVAL_METRICS
                },
                'retrieval_metric_guide': [
                    {
                        **item,
                        'value': _round_score(row.get(item['key'])),
                    }
                    for item in FRONTEND_RETRIEVAL_METRIC_GUIDE
                ],
                'reason': reason,
                'suggestion': failure_status['suggestion'],
                'failure_status': failure_status,
                'boundary': boundary,
                'score_breakdown': row.get('score_breakdown') if isinstance(row.get('score_breakdown'), Mapping) else {},
                'metric_layers': row.get('metric_layers') if isinstance(row.get('metric_layers'), Mapping) else {},
            },
        },
    }


def _frontend_case_status(row: Mapping[str, Any]) -> str:
    if str(row.get('failure_type') or '') in UNSCORED or str(row.get('quality_label') or '') == 'infra_failure':
        return 'failed'
    return 'badcase' if _frontend_is_badcase(row) else 'good'


def _frontend_case_status_label(row: Mapping[str, Any]) -> str:
    status = _frontend_case_status(row)
    return next((item['label'] for item in FRONTEND_CASE_STATUS_GUIDE if item['key'] == status), status)


def _frontend_boundary(row: Mapping[str, Any]) -> dict[str, Any]:
    failure_type = str(row.get('failure_type') or '')
    quality_label = str(row.get('quality_label') or '')
    if failure_type == 'dataset_contract_error':
        kind = 'routing_exception'
        display_group = 'exception'
    elif failure_type in UNSCORED or quality_label == 'infra_failure':
        kind = 'execution_exception'
        display_group = 'exception'
    elif _frontend_is_badcase(row):
        kind = 'scored_badcase'
        display_group = 'badcase'
    else:
        kind = 'scored_good'
        display_group = 'good'
    scores_applicable = display_group != 'exception'
    return {
        'kind': kind,
        'display_group': display_group,
        'label': next(
            (item['label'] for item in FRONTEND_BOUNDARY_GUIDE if item['key'] == display_group),
            display_group,
        ),
        'description': next(
            (item['description'] for item in FRONTEND_BOUNDARY_GUIDE if item['key'] == display_group),
            '',
        ),
        'scores_applicable': scores_applicable,
        'counts_in_badcase': display_group == 'badcase',
        'counts_in_score_avg': scores_applicable,
        'counts_in_failed_cases': display_group == 'exception',
    }


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


def _frontend_score_metrics(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    scores = _frontend_scores(row)
    return [
        {
            **item,
            'value': scores[item['key']],
        }
        for item in FRONTEND_SCORE_METRIC_GUIDE
    ]


def _frontend_guides() -> dict[str, Any]:
    return {
        'score_metrics': [dict(item) for item in FRONTEND_SCORE_METRIC_GUIDE],
        'retrieval_metrics': [dict(item) for item in FRONTEND_RETRIEVAL_METRIC_GUIDE],
        'fact_metrics': [dict(item) for item in FRONTEND_FACT_METRIC_GUIDE],
        'case_statuses': [dict(item) for item in FRONTEND_CASE_STATUS_GUIDE],
        'quality_labels': [dict(item) for item in FRONTEND_QUALITY_LABEL_GUIDE],
        'failure_statuses': [dict(item) for item in FRONTEND_FAILURE_STATUS_GUIDE],
        'boundary_statuses': [dict(item) for item in FRONTEND_BOUNDARY_GUIDE],
        'badcase_rule': FRONTEND_BADCASE_RULE,
        'boundary_note': FRONTEND_BOUNDARY_NOTE,
    }


def _frontend_failure_status(row: Mapping[str, Any]) -> dict[str, Any]:
    failure_type = str(row.get('failure_type') or '')
    retrieval_failure = str(row.get('retrieval_failure_type') or '')
    code = _frontend_failure_code(failure_type, retrieval_failure, str(row.get('quality_label') or ''))
    guide = next((item for item in FRONTEND_FAILURE_STATUS_GUIDE if item['key'] == code), None)
    label = guide['label'] if guide else _frontend_failure_label(row)
    description = guide['description'] if guide else ''
    category = guide['category'] if guide else 'none'
    suggestion = _frontend_suggestion_for_label(label)
    boundary = _frontend_boundary(row)
    return {
        'code': code,
        'label': label,
        'description': description,
        'category': category,
        'failure_type': failure_type,
        'retrieval_failure_type': retrieval_failure,
        'quality_label': str(row.get('quality_label') or ''),
        'is_badcase': boundary['counts_in_badcase'],
        'is_exception': boundary['display_group'] == 'exception',
        'display_group': boundary['display_group'],
        'counts_in_score_avg': boundary['counts_in_score_avg'],
        'counts_in_failed_cases': boundary['counts_in_failed_cases'],
        'suggestion': suggestion,
    }


def _frontend_failure_code(failure_type: str, retrieval_failure: str, quality_label: str) -> str:
    if failure_type in {
        'infra_failure',
        'judge_contract_error',
        'dataset_contract_error',
        'wrong_answer',
        'partial_answer',
        'question_not_answered',
        'hallucination',
        'format_error',
    }:
        return failure_type
    if retrieval_failure in {'retrieval_miss', 'retrieval_partial', 'retrieval_noise'}:
        return retrieval_failure
    if quality_label == 'good' or failure_type in {'', 'none'}:
        return 'none'
    if 'noise' in f'{failure_type} {retrieval_failure}'.lower():
        return 'retrieval_noise'
    if 'miss' in f'{failure_type} {retrieval_failure}'.lower():
        return 'retrieval_miss'
    if 'partial' in f'{failure_type} {quality_label}'.lower() or 'incomplete' in failure_type.lower():
        return 'partial_answer'
    if 'hallucination' in failure_type.lower() or 'unsupported' in failure_type.lower():
        return 'hallucination'
    if failure_type:
        return failure_type
    return 'none'


def _frontend_failure_label(row: Mapping[str, Any]) -> str:
    code = _frontend_failure_code(
        str(row.get('failure_type') or ''),
        str(row.get('retrieval_failure_type') or ''),
        str(row.get('quality_label') or ''),
    )
    guide = next((item for item in FRONTEND_FAILURE_STATUS_GUIDE if item['key'] == code), None)
    if guide:
        return guide['label']
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
    return build_answer_process_panel(row, load_trace=True, attempts=3, retry_seconds=0.5)


def _frontend_text_items(value: object) -> list[str]:
    values = value if isinstance(value, list | tuple) else [value] if value else []
    return [_item_text(item) for item in values if _item_text(item)]


def _frontend_suggestion(row: Mapping[str, Any]) -> str:
    return _frontend_suggestion_for_label(_frontend_failure_label(row))


def _frontend_suggestion_for_label(label: str) -> str:
    if label == '检索缺失':
        return '优先检查召回范围、query rewrite 和参考 chunk 是否可被检索。'
    if label == '检索噪声':
        return '优先检查召回排序、rerank 阈值和噪声文档过滤。'
    if label in {'回答不完整', '检索部分命中'}:
        return '优先检查答案是否覆盖标准答案关键点。'
    if label == '事实不支持':
        return '优先检查回答 claim 是否能在检索证据中找到支撑。'
    if label in {'执行异常', '评测契约异常', '数据契约异常'}:
        return '优先检查 Chat 调用、路由配置和重试记录。'
    if label == '回答错误':
        return '优先核对标准答案关键事实与模型回答差异。'
    if label == '未回答问题':
        return '优先检查模型是否产出有效最终回答。'
    if label == '格式错误':
        return '优先检查输出格式约束与后处理。'
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
