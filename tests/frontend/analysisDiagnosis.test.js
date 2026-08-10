import { describe, expect, it } from 'vitest';
import {
  evidenceRecordLabel,
  normalizeAnalysisDiagnosis,
} from '../../frontend/src/modules/selfEvolution/components/analysis/analysisDiagnosis.ts';


function diagnosticCase(index) {
  const caseId = `case-${String(index).padStart(3, '0')}`;
  return {
    case_id: caseId,
    trace_id: `trace-${index}`,
    cluster_id: 'cluster-context',
    analysis_status: 'repair_ready',
    problem: {
      issue_type: 'context_assembly_failure',
      affected_block: 'context_assembly',
      failure_mode: 'required_evidence_dropped',
      target_count: 1,
      target_types: ['missing_point'],
      statements: ['Paris is the capital'],
    },
    investigation: {
      stage_sequence: ['retrieve', 'rerank', 'context_assembly', 'llm_generate'],
      checkpoint_stage: 'context_assembly',
      route_signature: 'retrieve>rerank>context_assembly>llm_generate',
      trace_complete: true,
      review_status: 'not_required',
      probe_status: 'not_required',
      unavailable_probes: [],
    },
    root_cause: {
      mechanism_id: 'context.required_evidence_dropped',
      affected_block: 'context_assembly',
      stage: 'context_assembly',
      confidence: 0.98,
      evidence_level: 'trace_fact',
      decision_source: 'trace_diff',
    },
    evidence: {
      count: 1,
      records: [{ source: 'trace', summary: 'gold chunk absent from final context' }],
      missing: [],
    },
    repair: { ready: true, group_ids: ['grp-context'] },
  };
}


describe('Analysis diagnosis projection', () => {
  it('normalizes 100 cases without duplicating source artifacts', () => {
    const cases = Array.from({ length: 100 }, (_, index) => diagnosticCase(index));
    const model = normalizeAnalysisDiagnosis({
      total: 100,
      diagnostic_overview: {
        total_cases: 100,
        trace_complete: 100,
        progress_counts: {
          problem_observed: 100,
          root_cause_confirmed: 100,
          evidence_backed: 100,
          repair_ready: 100,
        },
      },
      root_cause_groups: [{
        group_id: 'root:context_assembly:context.required_evidence_dropped',
        mechanism_id: 'context.required_evidence_dropped',
        affected_block: 'context_assembly',
        failure_mode: 'required_evidence_dropped',
        evidence_level: 'trace_fact',
        case_count: 100,
        case_ids: cases.map((item) => item.case_id),
        representative_case_id: 'case-000',
        average_confidence: 0.98,
        repair_ready_count: 100,
      }],
      case_diagnostics: cases,
    });

    expect(model.overview).toEqual({
      totalCases: 100,
      problemObserved: 100,
      rootConfirmed: 100,
      evidenceBacked: 100,
      repairReady: 100,
      traceComplete: 100,
    });
    expect(model.groups).toHaveLength(1);
    expect(model.groups[0].caseIds).toHaveLength(100);
    expect(model.cases).toHaveLength(100);
    expect(model.cases[0].mechanismId).toBe('context.required_evidence_dropped');
    expect(model.cases[0].stageSequence).toEqual([
      'retrieve', 'rerank', 'context_assembly', 'llm_generate',
    ]);
    expect(evidenceRecordLabel(model.cases[0].evidenceRecords[0])).toBe(
      'gold chunk absent from final context',
    );
  });

  it('fails soft on an empty or partial summary', () => {
    expect(normalizeAnalysisDiagnosis(undefined).cases).toEqual([]);
    expect(normalizeAnalysisDiagnosis({ diagnostic_overview: {} }).overview.totalCases).toBe(0);
  });
});
