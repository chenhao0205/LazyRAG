import type { CaseStageKey, OperationStatus, VisualStatus } from './types';
import { acceptsAttemptUpdate, mergeStableExecutionStatus } from './executionStore';

export const CASE_GENERATION_STAGE = 'dataset.case_generation';

const CASE_STAGE_BY_OPERATION: Record<string, CaseStageKey> = {
  'dataset.qaplan_spec': 'plan',
  'dataset.generate_case': 'generate',
  'dataset.enhance_case': 'grading',
};

const CASE_STAGE_ORDER: CaseStageKey[] = ['plan', 'generate', 'grading'];

export type CaseGenerationEvent = {
  stage: string;
  tab?: string;
  operationId?: string;
  attemptId?: string;
  event: string;
  stepId: string;
  partition?: { id?: string; total?: number };
  status?: string;
};

export type CaseGenerationStep = {
  key: CaseStageKey;
  completed: number;
  total: number;
  running: number;
  failed: number;
  canceled: number;
  status: VisualStatus;
};

export type CaseGenerationProgress = {
  stepId: string;
  totals: Partial<Record<CaseStageKey, number>>;
  partitions: Partial<Record<CaseStageKey, Record<string, OperationStatus>>>;
  attempts: Partial<Record<CaseStageKey, Record<string, string>>>;
};

type StableCaseGenerationStep = {
  status?: string;
  completed?: number | null;
  total?: number | null;
  status_counts?: Partial<Record<OperationStatus, number | null>> | null;
};

export type CaseGenerationDisplayStep = {
  completed: number;
  total: number;
  running: number;
  failed: number;
  canceled: number;
  pending: number;
  status: VisualStatus;
};

type CaseExecutionReconciliation = {
  reconciliationToken: number;
  lastReconciledToken: number;
  expectedOverviewToken: number;
  loadedOverviewToken?: number;
  expectedListToken: number;
  loadedListToken?: number;
  overviewExecutionRevision?: string;
  listExecutionRevision?: string;
};

export type CaseGenerationEventResult = {
  progress: CaseGenerationProgress | undefined;
  /** Overview baseline should reload when live progress starts or switches step. */
  shouldRefreshBaseline: boolean;
};

/**
 * Apply one partition event. A new step_id replaces the live store — that is the
 * anti-flash boundary; callers should refresh the overview baseline afterward.
 */
export function applyCaseGenerationPartitionEvent(
  current: CaseGenerationProgress | undefined,
  event: CaseGenerationEvent,
): CaseGenerationEventResult {
  if (event.stage !== CASE_GENERATION_STAGE && event.tab !== 'cases') {
    return { progress: current, shouldRefreshBaseline: false };
  }
  const key = CASE_STAGE_BY_OPERATION[event.operationId || event.event];
  const caseId = event.partition?.id;
  const status = toOperationStatus(event.status);
  if (!key || !event.stepId || !caseId || !status) {
    return { progress: current, shouldRefreshBaseline: false };
  }

  const stepChanged = Boolean(current && current.stepId !== event.stepId);
  const startedLive = !current;
  const next = !current || stepChanged ? emptyProgress(event.stepId) : cloneProgress(current);
  const currentStatus = next.partitions[key]?.[caseId];
  const currentAttemptId = next.attempts[key]?.[caseId];
  if (!acceptsAttemptUpdate(
    currentStatus ? { attemptId: currentAttemptId, status: currentStatus } : undefined,
    { attemptId: event.attemptId, status },
  )) {
    return { progress: current, shouldRefreshBaseline: false };
  }
  next.partitions[key] = { ...(next.partitions[key] || {}), [caseId]: status };
  if (event.attemptId) {
    next.attempts[key] = { ...(next.attempts[key] || {}), [caseId]: event.attemptId };
  }
  if (event.partition?.total != null) {
    next.totals[key] = Math.max(next.totals[key] || 0, event.partition.total);
  }
  return {
    progress: next,
    shouldRefreshBaseline: stepChanged || startedLive,
  };
}

export function caseGenerationSteps(progress: CaseGenerationProgress | undefined): CaseGenerationStep[] {
  return CASE_STAGE_ORDER.map((key) => summarizeStep(
    key,
    progress?.partitions[key] || {},
    progress?.totals[key] || 0,
  ));
}

/**
 * Overview rings: overview counts are the baseline (imports already completed);
 * SSE only covers cases that emitted events. Unobserved cases keep the baseline.
 * `importedCompleted` is a floor so imported placeholders survive a new run.
 */
export function caseGenerationDisplayStep(
  live: CaseGenerationStep | undefined,
  baseline: StableCaseGenerationStep | undefined,
  importedCompleted = 0,
): CaseGenerationDisplayStep {
  const total = Math.max(baseline?.total ?? 0, live?.total ?? 0);
  const importedFloor = Math.min(Math.max(0, importedCompleted), total);
  const base = {
    completed: Math.max(baseline?.completed ?? 0, importedFloor),
    running: baseline?.status_counts?.running ?? 0,
    failed: baseline?.status_counts?.failed ?? 0,
    canceled: baseline?.status_counts?.canceled ?? 0,
  };

  const liveSeen = live
    ? live.completed + live.running + live.failed + live.canceled
    : 0;
  if (!live || liveSeen === 0) {
    return {
      completed: base.completed,
      total,
      running: base.running,
      failed: base.failed,
      canceled: base.canceled,
      pending: Math.max(0, total - base.completed - base.running - base.failed - base.canceled),
      status: toVisualStatus(baseline?.status),
    };
  }

  // Previous-run "all completed" snapshot while a new step is already live.
  // Drop generated leftovers, but keep the imported completed floor.
  if (base.completed >= total && total > 0 && liveSeen < total) {
    return finalizeCounts({
      completed: Math.min(total, live.completed + importedFloor),
      total,
      running: live.running,
      failed: live.failed,
      canceled: live.canceled,
      pending: Math.max(0, total - liveSeen - importedFloor),
    });
  }

  const unobserved = Math.max(0, total - liveSeen);
  let remaining = unobserved;
  const unobservedCompleted = Math.min(remaining, base.completed);
  remaining -= unobservedCompleted;
  const unobservedFailed = Math.min(remaining, base.failed);
  remaining -= unobservedFailed;
  const unobservedCanceled = Math.min(remaining, base.canceled);
  remaining -= unobservedCanceled;

  return finalizeCounts({
    completed: Math.min(total, live.completed + Math.max(unobservedCompleted, importedFloor)),
    total,
    running: live.running,
    failed: live.failed + unobservedFailed,
    canceled: live.canceled + unobservedCanceled,
    pending: remaining,
  });
}

export function shouldReconcileCaseExecution(input: CaseExecutionReconciliation): boolean {
  return input.reconciliationToken > input.lastReconciledToken
    && input.loadedOverviewToken === input.expectedOverviewToken
    && input.loadedListToken === input.expectedListToken
    && Boolean(input.overviewExecutionRevision)
    && input.overviewExecutionRevision === input.listExecutionRevision;
}

export function overlayCaseProgress<T extends { case_id: string; stages: Record<CaseStageKey, OperationStatus> }>(
  rows: T[],
  progress: CaseGenerationProgress | undefined,
): T[] {
  if (!progress) return rows;
  return rows.map((row) => {
    let changed = false;
    const stages = { ...row.stages };
    for (const key of CASE_STAGE_ORDER) {
      const next = progress.partitions[key]?.[row.case_id];
      if (next && next !== stages[key]) {
        const merged = mergeStableExecutionStatus(stages[key], next);
        if (merged === stages[key]) continue;
        stages[key] = merged;
        changed = true;
      }
    }
    return changed ? { ...row, stages } : row;
  });
}

function finalizeCounts(input: {
  completed: number;
  total: number;
  running: number;
  failed: number;
  canceled: number;
  pending: number;
}): CaseGenerationDisplayStep {
  const pending = Math.max(0, input.total - input.completed - input.running - input.failed - input.canceled);
  return {
    ...input,
    pending,
    status: input.failed || input.canceled
      ? 'failed'
      : input.running
        ? 'running'
        : input.total > 0 && input.completed === input.total
          ? 'done'
          : 'pending',
  };
}

function toOperationStatus(status?: string): OperationStatus | undefined {
  if (status === 'running' || status === 'completed' || status === 'failed' || status === 'canceled') {
    return status;
  }
  return undefined;
}

function toVisualStatus(status?: string): VisualStatus {
  if (status === 'completed' || status === 'succeeded') return 'done';
  if (status === 'running') return 'running';
  if (status === 'paused') return 'paused';
  if (status === 'failed' || status === 'canceled') return 'failed';
  return 'pending';
}

function emptyProgress(stepId: string): CaseGenerationProgress {
  return { stepId, totals: {}, partitions: {}, attempts: {} };
}

function cloneProgress(progress: CaseGenerationProgress): CaseGenerationProgress {
  return {
    stepId: progress.stepId,
    totals: { ...progress.totals },
    partitions: {
      plan: { ...progress.partitions.plan },
      generate: { ...progress.partitions.generate },
      grading: { ...progress.partitions.grading },
    },
    attempts: {
      plan: { ...progress.attempts.plan },
      generate: { ...progress.attempts.generate },
      grading: { ...progress.attempts.grading },
    },
  };
}

function summarizeStep(
  key: CaseStageKey,
  statuses: Record<string, OperationStatus>,
  total: number,
): CaseGenerationStep {
  const values = Object.values(statuses);
  const completed = values.filter((status) => status === 'completed').length;
  const running = values.filter((status) => status === 'running').length;
  const failed = values.filter((status) => status === 'failed').length;
  const canceled = values.filter((status) => status === 'canceled').length;
  const resolvedTotal = Math.max(total, values.length);
  const status: VisualStatus = failed
    ? 'failed'
    : running
      ? 'running'
      : resolvedTotal > 0 && completed === resolvedTotal
        ? 'done'
        : canceled === resolvedTotal && resolvedTotal > 0
          ? 'failed'
          : 'pending';
  return { key, completed, total: resolvedTotal, running, failed, canceled, status };
}
