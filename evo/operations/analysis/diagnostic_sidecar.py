from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evo.operations.public_contracts import clean_text as _text, mapping_or_empty as _mapping

from . import _as_list as _list, _clip_text as _clip, _ids, _stable_id
from .confirmation import build_confirmation_plan
from .diagnosis import (
    build_diagnosis_targets,
    build_judge_adapter_report,
    build_target_gate,
    build_target_paths,
    build_target_results,
    STAGE_ORDER,
)
from .mechanism_registry import MECHANISM_REGISTRY, registered_probes_for


def build_diagnostic_plan(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    max_review_calls: int = 2,
) -> dict[str, Any]:
    obligations = build_diagnosis_targets(case, answer, judge)
    target_gate = build_target_gate(obligations, judge)
    judge_adapter = build_judge_adapter_report(case, answer, judge, target_gate)
    timeline = build_evidence_timeline(obligations, case, answer, judge, trace)
    target_paths = build_target_paths(obligations, timeline)
    agenda = build_review_agenda(obligations, timeline, case, answer, judge, trace)
    review_plan = build_review_plan(agenda, target_paths=target_paths, max_review_calls=max_review_calls)
    confirmation_plan = build_confirmation_plan(target_paths, agenda, trace)
    return {
        'id': 'analysis.diagnostic_plan',
        'case_id': _text(case.get('id') or answer.get('case_id') or judge.get('case_id')),
        'trace_id': _text(trace.get('trace_id') or answer.get('trace_id') or judge.get('trace_id')),
        'judge_interface': {
            'required_change': False,
            'consumed_fields_only': True,
            'contract_note': 'diagnostic sidecar is derived from existing case, answer, judge, and trace inputs',
            'adapter_status': judge_adapter['status'],
            'target_gate': target_gate,
        },
        'judge_adapter': judge_adapter,
        'diagnosis_targets': obligations,
        'evidence_timeline': timeline,
        'target_paths': target_paths,
        'agenda': agenda,
        'review_plan': review_plan,
        'confirmation_plan': confirmation_plan,
        'checks': {
            'ready': True,
            'errors': [],
            'judge_interface_unchanged': True,
            'agenda_non_exclusive': len(agenda) >= 1,
            'diagnosis_target_count': len(obligations),
            'target_path_count': len(target_paths),
            'confirmation_step_count': len(confirmation_plan.get('steps') or ()),
            'judge_adapter_status': judge_adapter['status'],
            'target_gate_status': target_gate['status'],
            'review_call_budget': max_review_calls,
        },
    }


def finalize_diagnostic_sidecar(
    diagnostic_plan: Mapping[str, Any],
    *,
    semantic_reviews: Sequence[Mapping[str, Any]] | None = None,
    probe_observations: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    obligations = _mapping_sequence(diagnostic_plan.get('diagnosis_targets'))
    target_paths = _mapping_sequence(diagnostic_plan.get('target_paths'))
    agenda = _mapping_sequence(diagnostic_plan.get('agenda'))
    target_results = build_target_results(
        obligations,
        target_paths,
        agenda,
        semantic_reviews=semantic_reviews or (),
        probe_observations=probe_observations or (),
    )
    target_gate = diagnostic_plan.get('judge_interface')
    target_gate = target_gate.get('target_gate') if isinstance(target_gate, Mapping) else {}
    result = resolve_diagnostic_result(
        target_gate=target_gate if isinstance(target_gate, Mapping) else {},
        target_results=target_results,
    )
    result.setdefault('status', _text(result.get('actionability')))
    checks = dict(diagnostic_plan.get('checks') or {})
    checks['target_result_count'] = len(target_results)
    return {
        'id': 'analysis.diagnostic_sidecar',
        'case_id': _text(diagnostic_plan.get('case_id')),
        'trace_id': _text(diagnostic_plan.get('trace_id')),
        'diagnostic_plan_ref': {
            'artifact_id': 'analysis.diagnostic_plans',
            'partition_key': _text(diagnostic_plan.get('case_id')),
            'content_hash': _stable_id(diagnostic_plan),
        },
        'target_results': target_results,
        'semantic_reviews': [_compact_review(item) for item in semantic_reviews or ()],
        'probe_observations': [_compact_probe(item) for item in probe_observations or ()],
        'diagnostic_result': result,
        'checks': checks,
    }


def build_evidence_timeline(
    obligations: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    retrieved_docs = _ids(trace.get('retrieved_doc_ids'))
    retrieved_chunks = _ids(trace.get('retrieved_chunk_ids'))
    final_docs = _ids(trace.get('final_context_doc_ids')) or _ids(answer.get('doc_ids'))
    final_chunks = _ids(trace.get('final_context_chunk_ids')) or _ids(answer.get('chunk_ids'))
    stages = _stage_map(trace)
    missing_ids = {_text(item.get('id')) for item in _list(judge.get('missing_points')) if isinstance(item, Mapping)}
    wrong_ids = {_text(item.get('id')) for item in _list(judge.get('wrong_points')) if isinstance(item, Mapping)}
    matched_ids = {_text(item.get('id')) for item in _list(judge.get('matched_key_points')) if isinstance(item, Mapping)}
    items = []
    for obligation in obligations:
        ref_docs = set(_ids(obligation.get('reference_doc_ids')) or _ids(case.get('reference_doc_ids')))
        ref_chunks = set(_ids(obligation.get('reference_chunk_ids')) or _ids(case.get('reference_chunk_ids')))
        retrieved_overlap = sorted(ref_docs & set(retrieved_docs) or ref_chunks & set(retrieved_chunks))
        final_overlap = sorted(ref_docs & set(final_docs) or ref_chunks & set(final_chunks))
        oid = _text(obligation.get('id'))
        observations = [
            _observation('query_rewrite', _query_status(stages), stages.get('query_rewrite', []), {}),
            _observation(
                'retrieve',
                _retrieval_status(ref_docs, ref_chunks, retrieved_docs, retrieved_chunks, trace),
                stages.get('retrieve', []),
                {'reference_overlap': retrieved_overlap, 'retrieved_doc_ids': retrieved_docs[:12],
                 'retrieved_chunk_ids': retrieved_chunks[:12]},
            ),
            _observation(
                'rerank',
                _rerank_status(ref_docs, ref_chunks, retrieved_docs, retrieved_chunks, final_docs, final_chunks, trace),
                stages.get('rerank', []),
                {'reference_overlap': retrieved_overlap},
            ),
            _observation(
                'context_assembly',
                _context_status(retrieved_overlap, final_overlap, final_docs, final_chunks),
                stages.get('context_assembly', []) + stages.get('prompt_build', []),
                {'reference_overlap': final_overlap, 'final_doc_ids': final_docs[:12],
                 'final_chunk_ids': final_chunks[:12]},
            ),
            _observation(
                'llm_generate',
                _answer_status(oid, obligation, answer, missing_ids, wrong_ids, matched_ids, judge),
                stages.get('llm_generate', []),
                {'answer_preview': _clip(answer.get('answer'), 420)},
            ),
            _observation(
                'judge',
                _judge_status(oid, missing_ids, wrong_ids, matched_ids, judge),
                [],
                {'failure_type': _text(judge.get('failure_type')),
                 'retrieval_failure_type': _text(judge.get('retrieval_failure_type'))},
            ),
        ]
        items.append({
            'obligation_id': oid,
            'statement': _clip(obligation.get('statement'), 420),
            'required': bool(obligation.get('required')),
            'reference_doc_ids': sorted(ref_docs),
            'reference_chunk_ids': sorted(ref_chunks),
            'observations': observations,
            'earliest_observable_failure': _earliest_failure(observations),
        })
    return items


def build_review_agenda(
    obligations: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ctx = _context(obligations, timeline, case, judge, trace)
    agenda = []
    for spec in sorted(MECHANISM_REGISTRY, key=lambda item: int(item.get('order') or 0)):
        status, evidence, counterevidence = _mechanism_state(spec['id'], ctx)
        review_package = _text(spec.get('review_package'))
        item = {
            'mechanism_id': spec['id'],
            'surface': spec['surface'],
            'stage': spec['stage'],
            'affected_block': spec['affected_block'],
            'failure_mode': spec['failure_mode'],
            'status': status,
            'review_package': review_package if status == 'needs_semantic_review' else '',
            'confirmation_mode': spec['confirmation_mode'],
            'requires_probe': bool(spec.get('requires_probe')) and status in {'needs_probe', 'needs_semantic_review'},
            'repair_owner': spec['repair_owner'],
            'validation_focus': list(spec.get('validation_focus') or ()),
            'evidence': evidence[:8],
            'counterevidence': counterevidence[:8],
            'probe_plan': registered_probes_for(spec['id']) if status == 'needs_probe' else [],
        }
        agenda.append(item)
    return agenda


def build_review_plan(
    agenda: Sequence[Mapping[str, Any]],
    *,
    target_paths: Sequence[Mapping[str, Any]] | None = None,
    max_review_calls: int = 2,
) -> dict[str, Any]:
    packages = []
    for path in target_paths or ():
        package = _text(path.get('next_review_package'))
        if not package:
            continue
        packages.append(package)
    for item in agenda:
        if item.get('status') != 'needs_semantic_review' or not _text(item.get('review_package')):
            continue
        package = _text(item.get('review_package'))
        packages.append(package)
    unique_packages = list(dict.fromkeys(packages))
    selected_packages = unique_packages[:max(0, max_review_calls)]
    return {
        'max_review_calls': max_review_calls,
        'review_packages': selected_packages,
        'delayed_packages': unique_packages[max(0, max_review_calls):],
    }


def resolve_diagnostic_result(
    *,
    target_gate: Mapping[str, Any] | None = None,
    target_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gate_status = _text((target_gate or {}).get('status'))
    if gate_status == 'judge_diagnosis_incomplete':
        return {
            'primary_mechanism': {},
            'alternatives': [],
            'missing_evidence': ['judge diagnosis target is incomplete'],
            'repair_ready': False,
            'actionability': 'judge_diagnosis_incomplete',
        }
    if gate_status == 'not_required':
        return {
            'primary_mechanism': {},
            'alternatives': [],
            'missing_evidence': [],
            'repair_ready': False,
            'actionability': 'not_required',
            'resolved_target_count': 0,
            'target_count': 0,
            'fully_resolved': True,
        }
    return _resolve_from_targets(target_results)


def _resolve_from_targets(
    target_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    confirmed = [
        item for item in target_results
        if isinstance(item.get('primary_mechanism'), Mapping)
        and item.get('primary_mechanism')
    ]
    unresolved = [item for item in target_results if item not in confirmed]
    if confirmed:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for item in confirmed:
            mechanism = item.get('primary_mechanism')
            mechanism_id = _text(mechanism.get('mechanism_id')) if isinstance(mechanism, Mapping) else ''
            if mechanism_id:
                grouped.setdefault(mechanism_id, []).append(item)
        ranked = sorted(
            grouped.values(),
            key=lambda items: (
                -sum(max(0.1, float(item.get('weight') or 0.0)) for item in items),
                -max(float(_mapping(item.get('primary_mechanism')).get('confidence') or 0.0) for item in items),
                STAGE_ORDER.get(
                    _text(_mapping(items[0].get('primary_mechanism')).get('stage')),
                    999,
                ),
            ),
        )
        selected = ranked[0]
        first = selected[0]
        primary = {
            **dict(first['primary_mechanism']),
            'target_ids': [_text(item.get('target_id')) for item in selected],
        }
        alternatives = [
            {
                **dict(items[0]['primary_mechanism']),
                'target_ids': [_text(item.get('target_id')) for item in items],
            }
            for items in ranked[1:]
        ]
        missing = [
            f'{_text(item.get("target_id"))}: {reason}'
            for item in unresolved
            for reason in _list(item.get('missing_evidence'))
        ]
        return {
            'primary_mechanism': primary,
            'alternatives': alternatives[:4],
            'missing_evidence': missing[:8],
            'repair_ready': any(bool(item.get('repair_ready')) for item in selected),
            'actionability': (
                'repair_ready'
                if any(bool(item.get('repair_ready')) for item in selected)
                else 'insufficient_evidence'
            ),
            'resolved_target_count': len(confirmed),
            'target_count': len(target_results),
            'fully_resolved': not unresolved,
        }
    actionabilities = {_text(item.get('actionability')) for item in target_results}
    if 'needs_probe' in actionabilities:
        actionability = 'needs_probe'
    elif 'needs_semantic_review' in actionabilities:
        actionability = 'needs_semantic_review'
    else:
        actionability = 'insufficient_evidence'
    return {
        'primary_mechanism': {},
        'alternatives': [
            dict(mechanism)
            for item in target_results
            for mechanism in _list(item.get('alternatives'))
            if isinstance(mechanism, Mapping)
        ][:4],
        'missing_evidence': [
            f'{_text(item.get("target_id"))}: {reason}'
            for item in target_results
            for reason in _list(item.get('missing_evidence'))
        ][:8],
        'repair_ready': False,
        'actionability': actionability,
        'resolved_target_count': 0,
        'target_count': len(target_results),
        'fully_resolved': False,
    }


def _context(
    obligations: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
    case: Mapping[str, Any],
    judge: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    retrieved_overlap = _any_status(timeline, 'retrieve', {'present'})
    retrieve_missing = _any_status(timeline, 'retrieve', {'missing'})
    retrieve_unknown = _any_status(timeline, 'retrieve', {'unknown', 'insufficient_evidence'})
    final_present = _any_status(timeline, 'context_assembly', {'present'})
    final_dropped = _any_status(timeline, 'context_assembly', {'dropped'})
    final_missing = _any_status(timeline, 'context_assembly', {'missing'})
    answer_bad = _text(judge.get('failure_type')) not in {'', 'none', 'infra_failure'}
    refs_exist = bool(_ids(case.get('reference_doc_ids')) or _ids(case.get('reference_chunk_ids')))
    return {
        'judge': judge,
        'trace': trace,
        'retrieved_overlap': retrieved_overlap,
        'retrieve_missing': retrieve_missing,
        'retrieve_unknown': retrieve_unknown,
        'final_present': final_present,
        'final_dropped': final_dropped,
        'final_missing': final_missing,
        'answer_bad': answer_bad,
        'has_diagnosis_targets': bool(obligations),
        'refs_exist': refs_exist,
        'retrieved_count': len(_ids(trace.get('retrieved_doc_ids')) + _ids(trace.get('retrieved_chunk_ids'))),
        'final_count': len(_ids(trace.get('final_context_doc_ids')) + _ids(trace.get('final_context_chunk_ids'))),
        'error_stages': _list(trace.get('error_stages')),
        'unknown_stage_count': int(trace.get('unknown_stage_count') or 0),
    }


def _mechanism_state(mechanism_id: str, ctx: Mapping[str, Any]) -> tuple[str, list[str], list[str]]:
    judge = ctx['judge']
    trace = ctx['trace']
    evidence: list[str] = []
    counter: list[str] = []
    if mechanism_id == 'execution.stage_error':
        if ctx['error_stages']:
            return 'confirmed', ['trace.error_stages present'], []
        return 'ruled_out', [], ['no trace error stage observed']
    if mechanism_id == 'trace.metrics_missing':
        if ctx['unknown_stage_count']:
            return 'insufficient_evidence', ['trace unknown stages present'], []
        if ctx['retrieve_unknown']:
            return 'insufficient_evidence', ['trace retrieval ids unavailable'], []
        return 'ruled_out', [], ['trace ids sufficient for local agenda']
    if mechanism_id == 'judge.judge_conflict':
        issues = _judge_conflict_signals(ctx)
        if issues:
            return 'needs_semantic_review', issues, []
        return 'ruled_out', [], ['no local judge conflict signal']
    if ctx['answer_bad'] and not ctx['has_diagnosis_targets']:
        return 'insufficient_evidence', ['judge_diagnosis_incomplete'], []
    if mechanism_id == 'query.intent_lost':
        if not ctx['answer_bad']:
            return 'ruled_out', [], ['answer judged good']
        if 'query_rewrite' in _list(trace.get('diagnostic_stage_sequence')):
            return 'needs_semantic_review', ['query rewrite stage observed on bad answer'], []
        return 'insufficient_evidence', ['no query rewrite stage evidence'], []
    if mechanism_id == 'retrieve.reference_absent':
        if not ctx['refs_exist']:
            return 'ruled_out', [], ['case has no reference ids']
        if ctx['retrieved_overlap']:
            return 'ruled_out', [], ['reference evidence observed during retrieval']
        if ctx['retrieve_unknown']:
            return 'insufficient_evidence', ['retrieval id trace evidence missing'], []
        if ctx['retrieved_count']:
            return 'needs_semantic_review', ['reference ids absent but retrieved candidates exist'], []
        return 'confirmed', ['reference ids absent from empty retrieval'], []
    if mechanism_id == 'rerank.relevant_candidate_demoted':
        if ctx['retrieved_overlap'] and not ctx['final_present']:
            return 'needs_probe', ['reference evidence retrieved but absent from final context'], []
        if _text(judge.get('retrieval_failure_type')) == 'retrieval_noise':
            return 'needs_semantic_review', ['judge retrieval_noise signal'], []
        return 'ruled_out', [], ['no retrieved reference evidence to demote']
    if mechanism_id == 'context.required_evidence_dropped':
        if ctx['final_dropped']:
            return 'confirmed', ['reference evidence retrieved then dropped before final context'], []
        if ctx['final_missing'] and ctx['final_count']:
            return 'needs_semantic_review', ['final context exists without reference ids'], []
        return 'ruled_out', [], ['no context drop signal']
    if mechanism_id == 'context.context_insufficient':
        if ctx['final_present'] and ctx['answer_bad']:
            return 'needs_semantic_review', ['final context contains reference ids but answer is bad'], []
        if ctx['final_missing'] and ctx['final_count']:
            return 'needs_semantic_review', ['final context may be semantically sufficient despite id miss'], []
        return 'ruled_out', [], ['context sufficiency not applicable before final context']
    if mechanism_id == 'answer.available_context_ignored':
        if ctx['final_present'] and ctx['answer_bad']:
            return 'needs_semantic_review', ['answer failed despite visible reference context'], []
        return 'ruled_out', [], ['required context not visible to answer generation']
    if mechanism_id == 'answer.unsupported_or_contradicted':
        if _list(judge.get('unsupported_claims')) or _list(judge.get('contradicted_claims')):
            return 'needs_semantic_review', ['judge unsupported or contradicted claims present'], []
        return 'ruled_out', [], ['no unsupported or contradicted claim evidence']
    return 'insufficient_evidence', evidence, counter


def _judge_conflict_signals(ctx: Mapping[str, Any]) -> list[str]:
    judge = ctx['judge']
    signals = []
    if ctx['answer_bad'] and not ctx['has_diagnosis_targets']:
        signals.append('judge bad label lacks diagnosis targets')
    if judge.get('quality_label') == 'good' and judge.get('failure_type') != 'none':
        signals.append('quality_label=good conflicts with failure_type')
    if judge.get('is_correct') is True and judge.get('failure_type') != 'none':
        signals.append('is_correct=true conflicts with failure_type')
    if judge.get('retrieval_failure_type') == 'not_applicable' and ctx['refs_exist']:
        signals.append('retrieval not_applicable conflicts with reference ids')
    if ctx['final_present'] and judge.get('retrieval_failure_type') in {'retrieval_miss', 'retrieval_partial'}:
        signals.append('judge retrieval failure conflicts with final reference context')
    return signals


def _any_status(
    timeline: Sequence[Mapping[str, Any]],
    stage: str,
    statuses: set[str],
) -> bool:
    for item in timeline:
        for observation in _list(item.get('observations')):
            if not isinstance(observation, Mapping):
                continue
            if observation.get('stage') == stage and observation.get('status') in statuses:
                return True
    return False


def _stage_map(trace: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    stages: dict[str, list[Mapping[str, Any]]] = {}
    for item in _list(trace.get('stages')):
        if isinstance(item, Mapping):
            stages.setdefault(_text(item.get('stage')), []).append(item)
    return stages


def _observation(
    stage: str,
    status: str,
    stage_items: Sequence[Mapping[str, Any]],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        'stage': stage,
        'status': status,
        'stage_refs': [_text(item.get('id')) for item in stage_items if _text(item.get('id'))][:6],
        'observed': dict(observed),
    }


def _query_status(stage_items: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return 'needs_semantic_review' if stage_items.get('query_rewrite') else 'not_observed'


def _retrieval_status(
    ref_docs: set[str],
    ref_chunks: set[str],
    retrieved_docs: list[str],
    retrieved_chunks: list[str],
    trace: Mapping[str, Any],
) -> str:
    if not (ref_docs or ref_chunks):
        return 'not_applicable'
    if ref_docs & set(retrieved_docs) or ref_chunks & set(retrieved_chunks):
        return 'present'
    if trace.get('semantic_metric_keys') and not (retrieved_docs or retrieved_chunks):
        return 'missing'
    if (
        not trace.get('semantic_metric_keys')
        and trace.get('retrieval_steps')
        and not (retrieved_docs or retrieved_chunks)
    ):
        return 'unknown'
    return 'missing'


def _rerank_status(
    ref_docs: set[str],
    ref_chunks: set[str],
    retrieved_docs: list[str],
    retrieved_chunks: list[str],
    final_docs: list[str],
    final_chunks: list[str],
    trace: Mapping[str, Any],
) -> str:
    retrieved = bool(ref_docs & set(retrieved_docs) or ref_chunks & set(retrieved_chunks))
    final = bool(ref_docs & set(final_docs) or ref_chunks & set(final_chunks))
    if retrieved and not final:
        return 'needs_probe'
    if 'rerank' in _list(trace.get('diagnostic_stage_sequence')):
        return 'needs_semantic_review'
    return 'not_observed'


def _context_status(
    retrieved_overlap: Sequence[str],
    final_overlap: Sequence[str],
    final_docs: list[str],
    final_chunks: list[str],
) -> str:
    if final_overlap:
        return 'present'
    if retrieved_overlap:
        return 'dropped'
    if final_docs or final_chunks:
        return 'missing'
    return 'unknown'


def _answer_status(
    obligation_id: str,
    obligation: Mapping[str, Any],
    answer: Mapping[str, Any],
    missing_ids: set[str],
    wrong_ids: set[str],
    matched_ids: set[str],
    judge: Mapping[str, Any],
) -> str:
    if obligation_id in wrong_ids:
        return 'wrong'
    if obligation_id in missing_ids:
        return 'missing'
    if obligation_id in matched_ids:
        return 'addressed'
    statement = _text(obligation.get('statement')).lower()
    answer_text = _text(answer.get('answer')).lower()
    if statement and statement in answer_text:
        return 'addressed'
    if _text(judge.get('failure_type')) in {'wrong_answer', 'partial_answer', 'question_not_answered'}:
        return 'needs_semantic_review'
    return 'unknown'


def _judge_status(
    obligation_id: str,
    missing_ids: set[str],
    wrong_ids: set[str],
    matched_ids: set[str],
    judge: Mapping[str, Any],
) -> str:
    if obligation_id in wrong_ids:
        return 'wrong'
    if obligation_id in missing_ids:
        return 'missing'
    if obligation_id in matched_ids:
        return 'matched'
    if _text(judge.get('failure_type')) == 'none':
        return 'passed'
    return 'not_mapped'


def _earliest_failure(observations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failure_statuses = {'missing', 'dropped', 'wrong', 'needs_probe', 'needs_semantic_review'}
    candidates = [
        observation for observation in observations
        if observation.get('status') in failure_statuses
    ]
    if not candidates:
        return {}
    item = min(candidates, key=lambda observation: STAGE_ORDER.get(_text(observation.get('stage')), 999))
    return {
        'stage': _text(item.get('stage')),
        'status': _text(item.get('status')),
        'observed': item.get('observed') or {},
    }


def _compact_review(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'id': _text(value.get('id')),
        'review_package': _text(value.get('review_package')),
        'findings': _list(value.get('findings') or value.get('semantic_findings'))[:8],
        'rule_alignment': _text(value.get('rule_alignment')),
    }


def _compact_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        'probe_id': _text(value.get('probe_id')),
        'mechanism_ids': _list(value.get('mechanism_ids') or value.get('mechanism_id')),
        'target_ids': _list(value.get('target_ids') or value.get('target_id')),
        'decision': _text(value.get('decision')),
        'confidence': value.get('confidence'),
        'evidence_refs': _list(value.get('evidence_refs'))[:8],
        'controlled_variables': _list(value.get('controlled_variables'))[:8],
        'baseline': value.get('baseline') if isinstance(value.get('baseline'), Mapping) else {},
        'treatment': value.get('treatment') if isinstance(value.get('treatment'), Mapping) else {},
        'cost': value.get('cost') if isinstance(value.get('cost'), Mapping) else {},
        'observation': value.get('observation'),
    }


def _mapping_sequence(value: Any) -> list[Mapping[str, Any]]:
    return [item for item in _list(value) if isinstance(item, Mapping)]
