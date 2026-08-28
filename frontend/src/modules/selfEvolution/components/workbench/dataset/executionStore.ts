export type AttemptExecutionStatus = 'running' | 'completed' | 'failed' | 'canceled';

/**
 * A partition has one current attempt. Terminal events from an earlier attempt
 * must never overwrite a later retry that is already running.
 */
export function acceptsAttemptUpdate(
  current: { attemptId?: string; status?: AttemptExecutionStatus } | undefined,
  incoming: { attemptId?: string; status: AttemptExecutionStatus },
): boolean {
  if (!current?.attemptId || !incoming.attemptId || current.attemptId === incoming.attemptId) {
    return !(isTerminal(current?.status) && incoming.status === 'running');
  }
  return incoming.status === 'running' && isTerminal(current.status);
}

export function mergeStableExecutionStatus<T extends string>(
  stable: T,
  transient: T,
): T {
  return stable === 'completed' && transient !== 'completed' ? stable : transient;
}

function isTerminal(status?: AttemptExecutionStatus): boolean {
  return status === 'completed' || status === 'failed' || status === 'canceled';
}
