import type { DatasetDraft, DatasetTab } from './types';

const DATASET_TAB_ORDER: DatasetTab[] = ['materials', 'topics', 'cases'];

export const TERMINAL_STAGE_EVENTS = new Set([
  'step.finish',
  'checkpoint.continue',
  'done',
]);

export function draftAffectsTab(draft: DatasetDraft | undefined, tab: DatasetTab): boolean {
  if (!draft) return false;
  if (tab === 'materials') {
    return draft.kind === 'materials-config' || draft.kind === 'chunk-selection';
  }
  if (tab === 'topics') {
    return draft.kind === 'topic-names';
  }
  return draft.kind === 'generation-plan';
}

/** Stages whose SSE-derived execution progress is invalidated by a write. */
export function executionImpactTabs(kind: DatasetDraft['kind']): DatasetTab[] {
  const first = kind === 'materials-config' || kind === 'chunk-selection'
    ? 'materials'
    : kind === 'topic-names'
      ? 'topics'
      : 'cases';
  return DATASET_TAB_ORDER.slice(DATASET_TAB_ORDER.indexOf(first));
}

export function shouldShowGenerationPlanPause(
  status: string | undefined,
  generateCompleted: number | null | undefined,
  importedCompleted: number,
): boolean {
  if (status !== 'paused') return false;
  return (generateCompleted ?? 0) <= importedCompleted;
}

export function shouldResumeDatasetStream(lastHandledToken: number, nextToken: number): boolean {
  return nextToken !== lastHandledToken;
}

export function shouldPublishRefresh(
  previous: string | null | undefined,
  next: string | null,
): boolean {
  if (previous !== undefined && previous !== next) return true;
  return previous === undefined && next !== null;
}

export type RevisionRefreshAction = 'auto' | 'stale' | 'pending' | 'none';

/** Decide how a stage should react when its published revision changes. */
export function resolveRevisionRefreshAction(
  stageTab: DatasetTab,
  currentTab: DatasetTab,
  previous: string | null | undefined,
  next: string | null,
  draft: DatasetDraft | undefined,
): RevisionRefreshAction {
  if (!shouldPublishRefresh(previous, next)) {
    return 'none';
  }
  if (stageTab === currentTab) {
    return draftAffectsTab(draft, stageTab) ? 'stale' : 'auto';
  }
  return 'pending';
}
