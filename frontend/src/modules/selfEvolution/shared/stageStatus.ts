import type { StepStatus } from './types';

/** The latest /steps snapshot overrides historical SSE terminal events. */
export function resolveCurrentStageStatus(
  stepStatus: StepStatus | undefined,
  terminalStatus: StepStatus | undefined,
  checkpointCompleted: boolean,
  fallbackStatus: StepStatus,
): StepStatus {
  return stepStatus ?? terminalStatus ?? (checkpointCompleted ? 'done' : fallbackStatus);
}
