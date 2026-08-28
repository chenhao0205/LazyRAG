import type { DatasetTab, ThreadStepsResponse, VisualStatus } from "./types";
import { toVisualStatus } from "./primitives";

const STAGE_BY_TAB: Record<DatasetTab, string> = {
  materials: "dataset.material_preparation",
  topics: "dataset.topic_discovery",
  cases: "dataset.case_generation",
};

const TAB_BY_STAGE: Record<string, DatasetTab> = Object.fromEntries(
  Object.entries(STAGE_BY_TAB).map(([tab, stage]) => [stage, tab as DatasetTab]),
) as Record<string, DatasetTab>;

export function datasetTabForStage(stage: string): DatasetTab | undefined {
  return TAB_BY_STAGE[stage];
}

/** Returns an active tab only when it was read for the displayed Thread. */
export function activeDatasetTabForThread(
  threadId: string | undefined,
  sourceThreadId: string | undefined,
  activeTab: DatasetTab | undefined,
): DatasetTab | undefined {
  return threadId && threadId === sourceThreadId ? activeTab : undefined;
}

export const INITIAL_STAGE_STATUSES: Record<DatasetTab, VisualStatus> = {
  materials: "pending",
  topics: "pending",
  cases: "pending",
};

export type DatasetStageState = {
  statuses: Record<DatasetTab, VisualStatus>;
  activeTab?: DatasetTab;
  activeStepId?: string;
};

export function deriveDatasetStageState(response: ThreadStepsResponse): DatasetStageState {
  const next = { ...INITIAL_STAGE_STATUSES };
  let activeTab: DatasetTab | undefined;
  let activeStepId = response.active_step_id || undefined;
  for (const item of response.items || []) {
    const tab = datasetTabForStage(item.stage);
    if (!tab) continue;
    next[tab] = toVisualStatus(item.status);
    if (item.step_id === response.active_step_id || item.status === "paused") {
      activeTab = tab;
      activeStepId = item.step_id;
    }
  }
  return { statuses: next, activeTab, activeStepId };
}

/** Historical rounds are present in the initial SSE response but cannot write the current store. */
export function isCurrentDatasetExecutionEvent(
  activeStepId: string | undefined,
  eventStepId: string,
): boolean {
  return Boolean(activeStepId && eventStepId && activeStepId === eventStepId);
}
