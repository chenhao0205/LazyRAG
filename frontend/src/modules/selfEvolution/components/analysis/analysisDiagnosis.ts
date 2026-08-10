export type AnalysisOverview = {
  totalCases: number;
  problemObserved: number;
  rootConfirmed: number;
  evidenceBacked: number;
  repairReady: number;
  traceComplete: number;
};

export type AnalysisRootCauseGroup = {
  groupId: string;
  mechanismId: string;
  affectedBlock: string;
  failureMode: string;
  evidenceLevel: string;
  caseCount: number;
  caseIds: string[];
  representativeCaseId: string;
  averageConfidence: number;
  repairReadyCount: number;
};

export type AnalysisCaseDiagnostic = {
  caseId: string;
  traceId: string;
  clusterId: string;
  analysisStatus: string;
  issueType: string;
  affectedBlock: string;
  failureMode: string;
  statements: string[];
  targetCount: number;
  targetTypes: string[];
  stageSequence: string[];
  checkpointStage: string;
  routeSignature: string;
  traceComplete: boolean;
  reviewStatus: string;
  probeStatus: string;
  unavailableProbes: unknown[];
  mechanismId: string;
  rootStage: string;
  confidence: number;
  evidenceLevel: string;
  decisionSource: string;
  evidenceCount: number;
  evidenceRecords: Record<string, unknown>[];
  missingEvidence: string[];
  repairReady: boolean;
  repairGroupIds: string[];
};

export type AnalysisDiagnosisModel = {
  overview: AnalysisOverview;
  groups: AnalysisRootCauseGroup[];
  cases: AnalysisCaseDiagnostic[];
};

function record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record).filter((item) => Object.keys(item).length > 0) : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => String(item ?? "").trim()).filter(Boolean)
    : [];
}

function text(value: unknown): string {
  return String(value ?? "").trim();
}

function number(value: unknown): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

export function normalizeAnalysisDiagnosis(
  content: Record<string, unknown> | undefined,
): AnalysisDiagnosisModel {
  const summary = record(content);
  const overview = record(summary.diagnostic_overview);
  const progress = record(overview.progress_counts);
  const cases = records(summary.case_diagnostics).map((item): AnalysisCaseDiagnostic => {
    const problem = record(item.problem);
    const investigation = record(item.investigation);
    const rootCause = record(item.root_cause);
    const evidence = record(item.evidence);
    const repair = record(item.repair);
    return {
      caseId: text(item.case_id),
      traceId: text(item.trace_id),
      clusterId: text(item.cluster_id),
      analysisStatus: text(item.analysis_status),
      issueType: text(problem.issue_type),
      affectedBlock: text(problem.affected_block || rootCause.affected_block),
      failureMode: text(problem.failure_mode || rootCause.failure_mode),
      statements: strings(problem.statements),
      targetCount: number(problem.target_count),
      targetTypes: strings(problem.target_types),
      stageSequence: strings(investigation.stage_sequence),
      checkpointStage: text(investigation.checkpoint_stage),
      routeSignature: text(investigation.route_signature),
      traceComplete: Boolean(investigation.trace_complete),
      reviewStatus: text(investigation.review_status),
      probeStatus: text(investigation.probe_status),
      unavailableProbes: Array.isArray(investigation.unavailable_probes)
        ? investigation.unavailable_probes
        : [],
      mechanismId: text(rootCause.mechanism_id),
      rootStage: text(rootCause.stage),
      confidence: number(rootCause.confidence),
      evidenceLevel: text(rootCause.evidence_level),
      decisionSource: text(rootCause.decision_source),
      evidenceCount: number(evidence.count),
      evidenceRecords: records(evidence.records),
      missingEvidence: strings(evidence.missing),
      repairReady: Boolean(repair.ready),
      repairGroupIds: strings(repair.group_ids),
    };
  }).filter((item) => item.caseId);

  return {
    overview: {
      totalCases: number(overview.total_cases || summary.total),
      problemObserved: number(progress.problem_observed),
      rootConfirmed: number(progress.root_cause_confirmed),
      evidenceBacked: number(progress.evidence_backed),
      repairReady: number(progress.repair_ready),
      traceComplete: number(overview.trace_complete),
    },
    groups: records(summary.root_cause_groups).map((item): AnalysisRootCauseGroup => ({
      groupId: text(item.group_id),
      mechanismId: text(item.mechanism_id),
      affectedBlock: text(item.affected_block),
      failureMode: text(item.failure_mode),
      evidenceLevel: text(item.evidence_level),
      caseCount: number(item.case_count),
      caseIds: strings(item.case_ids),
      representativeCaseId: text(item.representative_case_id),
      averageConfidence: number(item.average_confidence),
      repairReadyCount: number(item.repair_ready_count),
    })).filter((item) => item.groupId),
    cases,
  };
}

export function evidenceRecordLabel(value: Record<string, unknown>): string {
  return text(
    value.summary || value.observation || value.statement || value.evidence_ref ||
    value.reference || value.path || value.source,
  ) || JSON.stringify(value);
}
