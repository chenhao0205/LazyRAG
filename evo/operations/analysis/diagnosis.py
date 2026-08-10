from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from evo.operations.public_contracts import clean_text as _text

from . import (
    _as_list as _list,
    _clip_text as _clip,
    _ids,
    _stable_id,
    _unique_text_values as _text_values,
)

STAGE_ORDER = {
    'runtime': 0,
    'query_rewrite': 10,
    'retrieve': 20,
    'rerank': 30,
    'context_assembly': 40,
    'prompt_build': 50,
    'llm_generate': 60,
    'judge': 70,
    'trace': 80,
}
SEMANTIC_CONFIRM_THRESHOLD = 0.85
PROBE_CONFIRM_THRESHOLD = 0.8
REPAIRABLE_EVIDENCE_LEVELS = {
    'trace_fact',
    'controlled_probe',
    'corroborated_semantic',
}
EVIDENCE_LEVEL_ORDER = {
    'heuristic': 0,
    'semantic_review': 1,
    'corroborated_semantic': 2,
    'trace_fact': 3,
    'controlled_probe': 4,
}


def build_diagnosis_targets(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_claim_texts: set[str] = set()
    missing_ids = {
        _text(item.get('id'))
        for item in _list(judge.get('missing_points'))
        if isinstance(item, Mapping) and _text(item.get('id'))
    }
    wrong_ids = {
        _text(item.get('id'))
        for item in _list(judge.get('wrong_points'))
        if isinstance(item, Mapping) and _text(item.get('id'))
    }
    for index, item in enumerate(_list(case.get('key_points'))):
        if isinstance(item, Mapping):
            oid = _target_id(item, f'kp_{index + 1}')
            if oid not in missing_ids and oid not in wrong_ids:
                continue
            targets.append({
                'id': oid,
                'source': 'case.key_points',
                'target_type': 'wrong_point' if oid in wrong_ids else 'missing_point',
                'statement': _clip(item.get('statement') or item.get('text') or item.get('answer'), 420),
                'required': item.get('required') is not False,
                'weight': _number(item.get('weight'), 1.0),
                'acceptable_variants': _list(item.get('acceptable_variants'))[:8],
                'reference_doc_ids': _merge_ids(case.get('reference_doc_ids'), item.get('evidence_doc_ids')),
                'reference_chunk_ids': _merge_ids(case.get('reference_chunk_ids'), item.get('evidence_chunk_ids')),
                'source_refs': ['case.key_points'],
            })
            seen.add(oid)
    for source in ('missing_points', 'wrong_points'):
        for index, item in enumerate(_list(judge.get(source))):
            if not isinstance(item, Mapping):
                continue
            oid = _target_id(item, f'{source}_{index + 1}')
            if oid in seen:
                continue
            targets.append({
                'id': oid,
                'source': f'judge.{source}',
                'target_type': 'missing_point' if source == 'missing_points' else 'wrong_point',
                'statement': _clip(item.get('statement') or item.get('text') or item.get('claim'), 420),
                'required': item.get('required') is not False,
                'weight': _number(item.get('weight'), 1.0),
                'acceptable_variants': _list(item.get('acceptable_variants'))[:8],
                'reference_doc_ids': _merge_ids(case.get('reference_doc_ids'), item.get('evidence_doc_ids')),
                'reference_chunk_ids': _merge_ids(case.get('reference_chunk_ids'), item.get('evidence_chunk_ids')),
                'source_refs': [f'judge.{source}'],
            })
            seen.add(oid)
    for source, target_type in (
        ('unsupported_claims', 'unsupported_claim'),
        ('contradicted_claims', 'contradicted_claim'),
    ):
        for index, item in enumerate(_list(judge.get(source))[:6]):
            text = (
                _text(item.get('text') or item.get('claim'))
                if isinstance(item, Mapping)
                else _text(item)
            )
            if not text:
                continue
            oid = f'answer_claim_{_stable_id({"source": source, "text": text})[:8]}'
            if oid in seen:
                continue
            targets.append({
                'id': oid,
                'source': f'judge.{source}',
                'target_type': target_type,
                'statement': _clip(text, 420),
                'required': False,
                'weight': 0.0,
                'acceptable_variants': [],
                'reference_doc_ids': _ids(case.get('reference_doc_ids')),
                'reference_chunk_ids': _ids(case.get('reference_chunk_ids')),
                'source_refs': [f'judge.{source}[{index}]'],
            })
            seen.add(oid)
            seen_claim_texts.add(text)
    for index, item in enumerate(_list(judge.get('claims'))[:6]):
        if not isinstance(item, Mapping):
            continue
        if item.get('supported') is not False and item.get('contradicted') is not True:
            continue
        text = _text(item.get('text') or item.get('claim'))
        if not text or text in seen_claim_texts:
            continue
        oid = f'answer_claim_{_stable_id(text)[:8]}'
        if oid in seen:
            continue
        targets.append({
            'id': oid,
            'source': 'judge.claims',
            'target_type': 'contradicted_claim' if item.get('contradicted') is True else 'unsupported_claim',
            'statement': _clip(text, 420),
            'required': False,
            'weight': 0.0,
            'acceptable_variants': [],
            'reference_doc_ids': _ids(case.get('reference_doc_ids')),
            'reference_chunk_ids': _ids(case.get('reference_chunk_ids')),
            'source_refs': [f'judge.claims[{index}]'],
        })
        seen.add(oid)
    return targets


def build_target_gate(
    targets: Sequence[Mapping[str, Any]],
    judge: Mapping[str, Any],
) -> dict[str, Any]:
    failure = _text(judge.get('failure_type'))
    if failure in {'', 'none'}:
        return {
            'status': 'not_required',
            'reason': 'judge marked answer correct; diagnosis targets are only built for observed symptoms',
            'usable_target_count': len(targets),
            'limitations': [],
        }
    if failure in {'infra_failure', 'judge_contract_error', 'dataset_contract_error'}:
        return {
            'status': 'terminal',
            'reason': f'failure_type={failure} is handled by guard logic before business diagnosis',
            'usable_target_count': len(targets),
            'limitations': [],
        }
    if targets:
        sources = sorted({_text(item.get('source')) for item in targets if _text(item.get('source'))})
        return {
            'status': 'valid',
            'reason': 'judge supplied at least one concrete missing/wrong point or abnormal claim',
            'usable_target_count': len(targets),
            'limitations': [],
            'target_sources': sources,
        }
    return {
        'status': 'judge_diagnosis_incomplete',
        'reason': 'judge marked answer bad but supplied no usable missing/wrong point or abnormal claim',
        'usable_target_count': 0,
        'limitations': ['business mechanism agenda is fail-closed until a concrete diagnosis target exists'],
    }


def build_judge_adapter_report(
    case: Mapping[str, Any],
    answer: Mapping[str, Any],
    judge: Mapping[str, Any],
    target_gate: Mapping[str, Any],
) -> dict[str, Any]:
    gate_status = _text(target_gate.get('status'))
    if gate_status in {'terminal'}:
        status = 'invalid'
    elif gate_status in {'judge_diagnosis_incomplete'}:
        status = 'degraded'
    else:
        status = 'valid'
    limitations = list(target_gate.get('limitations') or [])
    if status == 'degraded':
        limitations.append('summary fields may guide audit priority but cannot drive business mechanism findings')
    return {
        'id': 'analysis.judge_adapter',
        'status': status,
        'target_gate_status': gate_status,
        'usable_target_count': int(target_gate.get('usable_target_count') or 0),
        'identity': {
            'case_id': _text(case.get('id') or judge.get('case_id')),
            'answer_case_id': _text(answer.get('case_id')),
            'judge_case_id': _text(judge.get('case_id')),
            'answer_trace_id': _text(answer.get('trace_id')),
            'judge_trace_id': _text(judge.get('trace_id')),
        },
        'runtime_status': {
            'rag_answer_status': _text(answer.get('status')),
            'failure_type': _text(judge.get('failure_type')),
            'quality_label': _text(judge.get('quality_label')),
            'retrieval_failure_type': _text(judge.get('retrieval_failure_type')),
        },
        'consumed_symptom_fields': [
            'missing_points',
            'wrong_points',
            'unsupported_claims',
            'contradicted_claims',
            'claims',
        ],
        'disabled_summary_fields_when_degraded': (
            ['failure_type', 'quality_label', 'scores'] if status == 'degraded' else []
        ),
        'limitations': list(dict.fromkeys(limitations)),
    }


def build_target_paths(
    targets: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    timeline_by_id = {
        _text(item.get('obligation_id')): item
        for item in timeline
        if isinstance(item, Mapping)
    }
    paths = []
    for target in targets:
        target_id = _text(target.get('id'))
        item = timeline_by_id.get(target_id, {})
        context_obs = _timeline_observation(item, 'context_assembly')
        retrieve_obs = _timeline_observation(item, 'retrieve')
        answer_obs = _timeline_observation(item, 'llm_generate')
        path = _target_path_from_observations(target, context_obs, retrieve_obs, answer_obs)
        paths.append(path)
    return paths


def build_target_results(
    targets: Sequence[Mapping[str, Any]],
    target_paths: Sequence[Mapping[str, Any]],
    agenda: Sequence[Mapping[str, Any]],
    *,
    semantic_reviews: Sequence[Mapping[str, Any]] = (),
    probe_observations: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    paths = {_text(item.get('target_id')): item for item in target_paths}
    multiple_targets = len(targets) > 1
    results = []
    for target in targets:
        target_id = _text(target.get('id'))
        path = paths.get(target_id, {})
        candidate_ids = _target_mechanism_ids(target, path)
        mechanisms = {
            _text(item.get('mechanism_id')): dict(item)
            for item in agenda
            if _text(item.get('mechanism_id')) in candidate_ids
        }
        votes = _semantic_votes(
            target_id,
            mechanisms,
            semantic_reviews,
            multiple_targets=multiple_targets,
        )
        votes.extend(_probe_votes(
            target_id,
            mechanisms,
            probe_observations,
            multiple_targets=multiple_targets,
        ))
        resolved = [
            _resolve_target_mechanism(item, votes)
            for item in mechanisms.values()
        ]
        active = [item for item in resolved if item['status'] != 'ruled_out']
        confirmed = sorted(
            (item for item in active if item['status'] == 'confirmed'),
            key=lambda item: (
                -EVIDENCE_LEVEL_ORDER.get(_text(item.get('evidence_level')), 0),
                -_number(item.get('confidence'), 0.0),
                STAGE_ORDER.get(_text(item.get('stage')), 999),
            ),
        )
        primary = confirmed[0] if confirmed else None
        if primary:
            repair_ready = (
                _text(primary.get('evidence_level')) in REPAIRABLE_EVIDENCE_LEVELS
                and _text(primary.get('affected_block')) not in {
                    'eval_contract',
                    'tracing_observability',
                }
            )
            actionability = 'repair_ready' if repair_ready else 'insufficient_evidence'
            missing_evidence: list[str] = (
                [] if repair_ready else ['confirmed mechanism is an analysis guard, not a repair target']
            )
        elif any(item['status'] == 'needs_probe' for item in active):
            repair_ready = False
            actionability = 'needs_probe'
            missing_evidence = ['registered probe required before confirming target mechanism']
        elif any(item['status'] == 'needs_semantic_review' for item in active):
            repair_ready = False
            actionability = 'needs_semantic_review'
            missing_evidence = ['semantic review required before confirming target mechanism']
        else:
            repair_ready = False
            actionability = 'insufficient_evidence'
            missing_evidence = ['no mechanism confirmed for target by available evidence']
        problem = {
            'target_id': target_id,
            'target_type': _text(target.get('target_type')),
            'statement': _clip(target.get('statement'), 420),
            'source': _text(target.get('source')),
            'weight': _number(target.get('weight'), 1.0),
        }
        root_cause = _compact_mechanism(primary or {})
        evidence_values = list(primary.get('evidence') or ())[:8] if primary else []
        results.append({
            'target_id': target_id,
            'target_type': problem['target_type'],
            'statement': problem['statement'],
            'weight': problem['weight'],
            'investigation_direction': _text(path.get('investigation_direction')),
            'status': 'confirmed' if primary else actionability,
            'problem': problem,
            'root_cause': root_cause,
            'primary_mechanism': root_cause,
            'alternatives': [
                _compact_mechanism(item)
                for item in active
                if item is not primary
            ][:4],
            'missing_evidence': missing_evidence,
            'repair_ready': repair_ready,
            'actionability': actionability,
            'evidence': evidence_values,
            'evidence_records': _evidence_records(
                target_id,
                evidence_values,
                root_cause,
            ),
            'counterevidence': list(primary.get('counterevidence') or ())[:8] if primary else [],
            'observation_updates': [
                vote for vote in votes
                if vote.get('mechanism_id') in mechanisms
            ][:12],
            'next_action': {
                'status': actionability,
                'missing_evidence': missing_evidence,
                'requires_probe': actionability == 'needs_probe',
                'requires_semantic_review': actionability == 'needs_semantic_review',
            },
        })
    return results


def _target_mechanism_ids(
    target: Mapping[str, Any],
    path: Mapping[str, Any],
) -> set[str]:
    direction = _text(path.get('investigation_direction'))
    target_type = _text(target.get('target_type'))
    ids = {'execution.stage_error', 'judge.judge_conflict'}
    if direction == 'evidence_backtrack':
        ids.update({
            'query.intent_lost',
            'retrieve.reference_absent',
            'rerank.relevant_candidate_demoted',
            'context.required_evidence_dropped',
        })
    elif direction == 'needs_review':
        ids.update({
            'query.intent_lost',
            'retrieve.reference_absent',
            'context.context_insufficient',
        })
    elif direction == 'blocked_by_trace':
        ids.update({
            'query.intent_lost',
            'retrieve.reference_absent',
            'trace.metrics_missing',
        })
    elif target_type in {'unsupported_claim', 'contradicted_claim'}:
        ids.add('answer.unsupported_or_contradicted')
    else:
        ids.update({
            'context.context_insufficient',
            'answer.available_context_ignored',
        })
    return ids


def _semantic_votes(
    target_id: str,
    mechanisms: Mapping[str, Mapping[str, Any]],
    reviews: Sequence[Mapping[str, Any]],
    *,
    multiple_targets: bool,
) -> list[dict[str, Any]]:
    votes = []
    for review in reviews:
        package = _text(review.get('review_package'))
        alignment = _text(review.get('rule_alignment'))
        findings = [
            item for item in _list(review.get('findings') or review.get('semantic_findings'))
            if isinstance(item, Mapping)
        ]
        explicit_targets = list(dict.fromkeys(
            _text(item.get('obligation_id'))
            for item in findings
            if _text(item.get('obligation_id'))
        ))
        explicit_targets.extend(
            item for item in _text_values(review.get('target_ids'), review.get('target_id'))
            if item not in explicit_targets
        )
        if explicit_targets and target_id not in explicit_targets:
            continue
        if multiple_targets and not explicit_targets:
            continue
        status = {
            'supports_candidate': 'confirmed',
            'contradicts_candidate': 'ruled_out',
        }.get(alignment)
        if not status:
            continue
        matching_findings = [
            item for item in findings
            if not _text(item.get('obligation_id')) or _text(item.get('obligation_id')) == target_id
        ]
        package_mechanisms = [
            mechanism_id
            for mechanism_id, mechanism in mechanisms.items()
            if _text(mechanism.get('review_package')) == package
        ]
        for finding in matching_findings:
            mechanism_id = _text(finding.get('mechanism_id'))
            if mechanism_id:
                if mechanism_id not in package_mechanisms:
                    continue
            elif len(package_mechanisms) == 1:
                mechanism_id = package_mechanisms[0]
            else:
                continue
            confidence = _number(finding.get('confidence'), 0.0)
            if confidence < SEMANTIC_CONFIRM_THRESHOLD:
                continue
            votes.append({
                'source': 'semantic_review',
                'source_id': package,
                'mechanism_id': mechanism_id,
                'status': status,
                'confidence': confidence,
                'evidence_refs': _text_values(finding.get('evidence_refs')),
            })
    return votes


def _probe_votes(
    target_id: str,
    mechanisms: Mapping[str, Mapping[str, Any]],
    observations: Sequence[Mapping[str, Any]],
    *,
    multiple_targets: bool,
) -> list[dict[str, Any]]:
    votes = []
    for item in observations:
        target_ids = _text_values(
            item.get('target_ids'),
            item.get('target_id'),
            _mapping_value(item.get('observation'), 'target_ids'),
            _mapping_value(item.get('observation'), 'target_id'),
        )
        if target_ids and target_id not in target_ids:
            continue
        if multiple_targets and not target_ids:
            continue
        decision = _text(
            item.get('decision')
            or _mapping_value(item.get('observation'), 'decision')
            or _mapping_value(item.get('observation'), 'mechanism_status')
        )
        if decision not in {'confirmed', 'ruled_out'}:
            continue
        confidence = _number(item.get('confidence'), 0.0)
        if confidence < PROBE_CONFIRM_THRESHOLD:
            continue
        mechanism_ids = _text_values(
            item.get('mechanism_ids'),
            item.get('mechanism_id'),
        )
        for mechanism_id in mechanism_ids:
            if mechanism_id not in mechanisms:
                continue
            votes.append({
                'source': 'probe',
                'source_id': _text(item.get('probe_id')),
                'mechanism_id': mechanism_id,
                'status': decision,
                'confidence': confidence,
                'evidence_refs': _text_values(item.get('evidence_refs')),
            })
    return votes


def _resolve_target_mechanism(
    mechanism: Mapping[str, Any],
    votes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    resolved = dict(mechanism)
    baseline_status = _text(resolved.get('status'))
    mechanism_votes = [
        item for item in votes
        if item.get('mechanism_id') == mechanism.get('mechanism_id')
    ]
    probe_votes = [item for item in mechanism_votes if item.get('source') == 'probe']
    decisive = probe_votes or [
        item for item in mechanism_votes
        if item.get('source') == 'semantic_review'
    ]
    statuses = {_text(item.get('status')) for item in decisive}
    if len(statuses) == 1:
        resolved['status'] = next(iter(statuses))
    elif len(statuses) > 1:
        resolved['status'] = 'insufficient_evidence'
        resolved['counterevidence'] = [
            *list(resolved.get('counterevidence') or ()),
            'conflicting experiment observations',
        ]
    evidence_refs = list(dict.fromkeys(
        ref
        for item in decisive
        for ref in _text_values(item.get('evidence_refs'))
    ))
    if evidence_refs:
        resolved['evidence'] = [
            *list(resolved.get('evidence') or ()),
            *evidence_refs,
        ]
    if decisive:
        resolved['confidence'] = max(_number(item.get('confidence'), 0.0) for item in decisive)
        resolved['decision_source'] = _text(decisive[0].get('source'))
        resolved['evidence_level'] = (
            'controlled_probe'
            if decisive[0].get('source') == 'probe'
            else 'corroborated_semantic'
        )
    elif baseline_status == 'confirmed':
        resolved['confidence'] = 1.0
        resolved['decision_source'] = 'trace'
        resolved['evidence_level'] = 'trace_fact'
    else:
        resolved['evidence_level'] = 'heuristic'
    return resolved


def _compact_mechanism(item: Mapping[str, Any]) -> dict[str, Any]:
    if not item:
        return {}
    return {
        'mechanism_id': _text(item.get('mechanism_id')),
        'surface': _text(item.get('surface')),
        'stage': _text(item.get('stage')),
        'affected_block': _text(item.get('affected_block')),
        'failure_mode': _text(item.get('failure_mode')),
        'status': _text(item.get('status')),
        'repair_owner': _text(item.get('repair_owner')),
        'validation_focus': _list(item.get('validation_focus')),
        'confidence': item.get('confidence'),
        'decision_source': _text(item.get('decision_source')),
        'evidence_level': _text(item.get('evidence_level')),
    }


def _evidence_records(
    target_id: str,
    values: Sequence[Any],
    mechanism: Mapping[str, Any],
) -> list[dict[str, Any]]:
    level = _text(mechanism.get('evidence_level'))
    source = _text(mechanism.get('decision_source')) or 'rule'
    return [
        {
            'id': f'ev_{_stable_id({"target_id": target_id, "value": value})[:12]}',
            'target_id': target_id,
            'kind': 'artifact_ref' if '.' in _text(value) else 'observation',
            'source': source,
            'evidence_level': level,
            'label': _clip(value, 240),
            'ref': _clip(value, 240) if '.' in _text(value) else '',
        }
        for value in values
        if _text(value)
    ]


def _target_path_from_observations(
    target: Mapping[str, Any],
    context_obs: Mapping[str, Any],
    retrieve_obs: Mapping[str, Any],
    answer_obs: Mapping[str, Any],
) -> dict[str, Any]:
    target_id = _text(target.get('id'))
    target_type = _text(target.get('target_type'))
    retrieval_status = _text(retrieve_obs.get('status'))
    if target_type in {'unsupported_claim', 'contradicted_claim'}:
        return {
            'target_id': target_id,
            'target_type': target_type,
            'initial_question': 'answer claim support and evidence use',
            'final_context_status': _claim_context_status(context_obs),
            'retrieval_status': retrieval_status,
            'investigation_direction': 'answer_side',
            'next_review_package': 'answer_faithfulness_review',
            'checkpoint': 'llm_generate',
            'answer_status': _text(answer_obs.get('status')),
            'breakpoint_window': {},
            'pending_reason': 'claim target should be checked against visible evidence before backtracking retrieval',
        }
    status = _text(context_obs.get('status'))
    if status == 'present':
        return {
            'target_id': target_id,
            'target_type': target_type,
            'initial_question': 'final context already contains exact required evidence',
            'final_context_status': 'sufficient_by_fact',
            'retrieval_status': retrieval_status,
            'investigation_direction': 'answer_side',
            'next_review_package': 'answer_faithfulness_review',
            'checkpoint': 'llm_generate',
            'answer_status': _text(answer_obs.get('status')),
            'breakpoint_window': {},
            'pending_reason': 'answer failed despite visible required context',
        }
    if status == 'dropped':
        return {
            'target_id': target_id,
            'target_type': target_type,
            'initial_question': 'required evidence was present upstream but absent from final context',
            'final_context_status': 'insufficient_by_fact',
            'retrieval_status': retrieval_status,
            'investigation_direction': 'evidence_backtrack',
            'next_review_package': '',
            'checkpoint': 'context_assembly',
            'answer_status': _text(answer_obs.get('status')),
            'breakpoint_window': {'from': 'retrieve', 'to': 'context_assembly'},
            'pending_reason': 'local before/after evidence can explain the first loss window',
        }
    if status == 'missing':
        return {
            'target_id': target_id,
            'target_type': target_type,
            'initial_question': 'final context has candidates but exact required evidence is absent',
            'final_context_status': 'semantic_ambiguous',
            'retrieval_status': retrieval_status,
            'investigation_direction': 'needs_review',
            'next_review_package': 'context_completeness_review',
            'checkpoint': 'context_assembly',
            'answer_status': _text(answer_obs.get('status')),
            'breakpoint_window': {},
            'pending_reason': 'semantic sufficiency must be checked before assigning retrieval or generation blame',
        }
    return {
        'target_id': target_id,
        'target_type': target_type,
        'initial_question': 'final context is not observable for this target',
        'final_context_status': 'not_observed',
        'retrieval_status': retrieval_status,
        'investigation_direction': 'blocked_by_trace',
        'next_review_package': '',
        'checkpoint': 'context_assembly',
        'answer_status': _text(answer_obs.get('status')),
        'breakpoint_window': {},
        'pending_reason': 'actual final request/context is unavailable',
    }


def _target_id(item: Mapping[str, Any], fallback: str) -> str:
    text = _text(item.get('id'))
    if text:
        return text
    statement = _text(item.get('statement') or item.get('text') or item.get('claim'))
    return f'{fallback}_{_stable_id(statement)[:8]}' if statement else fallback


def _claim_context_status(context_obs: Mapping[str, Any]) -> str:
    status = _text(context_obs.get('status'))
    if status in {'present', 'missing', 'dropped'}:
        return 'requires_faithfulness_review'
    return 'not_observed'


def _timeline_observation(item: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    for observation in _list(item.get('observations')):
        if isinstance(observation, Mapping) and observation.get('stage') == stage:
            return observation
    return {}


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _merge_ids(*values: Any) -> list[str]:
    ids: list[str] = []
    for value in values:
        ids.extend(_ids(value))
    return list(dict.fromkeys(ids))


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
