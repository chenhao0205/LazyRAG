import { useCallback, useEffect, useRef } from "react";
import { AgentAppsAuth } from "@/components/auth";
import { getJson, threadRoot } from "./api";
import type { DatasetTab, ThreadStepsResponse } from "./types";
import {
  datasetTabForStage,
  deriveDatasetStageState,
  isCurrentDatasetExecutionEvent,
} from "./stageState";

export const DATASET_TABS: Array<{ id: DatasetTab; label: string }> = [
  { id: "materials", label: "材料准备" },
  { id: "topics", label: "主题发现" },
  { id: "cases", label: "用例生成" },
];

export type DatasetStreamEvent = {
  event: string;
  tab: DatasetTab;
  stage: string;
  stepId: string;
  operationId?: string;
  attemptId?: string;
  status?: string;
  partition?: { id?: string; index?: number; total?: number };
  progress?: { current?: number | null; total?: number | null };
};

type UseDatasetStagesOptions = {
  /** Push every /steps snapshot to the Workbench owner (sole nav state writer). */
  onStepsSnapshot: (response: ThreadStepsResponse) => void;
  onStageEvent: (event: DatasetStreamEvent) => void;
};

/**
 * Dataset SSE + /steps refresh trigger.
 *
 * Navigation status is NOT stored here — the parent writes threadStepList from
 * onStepsSnapshot and derives top / sub-nav / continue via deriveDatasetView.
 * This hook only keeps the stream alive and filters stale step_id events.
 */
export function useDatasetStages(
  threadId: string | undefined,
  { onStepsSnapshot, onStageEvent }: UseDatasetStagesOptions,
) {
  const eventHandler = useRef(onStageEvent);
  eventHandler.current = onStageEvent;
  const snapshotHandler = useRef(onStepsSnapshot);
  snapshotHandler.current = onStepsSnapshot;
  const resumeStream = useRef<(force?: boolean) => void>(() => undefined);
  const activeStepId = useRef<string>();
  const inactiveStepIds = useRef(new Set<string>());

  const refreshSteps = useCallback(async () => {
    if (!threadId) return undefined;
    try {
      const response = await getJson<ThreadStepsResponse>(`${threadRoot(threadId)}/steps`);
      snapshotHandler.current(response);
      const next = deriveDatasetStageState(response);
      activeStepId.current = next.activeStepId;
      inactiveStepIds.current = new Set(
        (response.items || [])
          .map((item) => item.step_id)
          .filter((stepId) => stepId && stepId !== next.activeStepId),
      );
      return next;
    } catch {
      return undefined;
    }
  }, [threadId]);

  const resumeAfterWrite = useCallback(() => {
    resumeStream.current(true);
  }, []);

  useEffect(() => {
    activeStepId.current = undefined;
    inactiveStepIds.current.clear();
  }, [threadId]);

  useEffect(() => {
    if (!threadId) return undefined;
    let stopped = false;
    let round = 0;
    let activeController: AbortController | undefined;
    let lastEventId: string | undefined;
    let streamEnded = false;

    const consume = async () => {
      const myRound = ++round;
      streamEnded = false;
      activeController?.abort();
      const controller = new AbortController();
      activeController = controller;
      try {
        await refreshSteps();
        if (stopped || myRound !== round) return;
        const headers: Record<string, string> = {
          Accept: "text/event-stream",
          ...AgentAppsAuth.getAuthHeaders(),
        };
        if (lastEventId) headers["Last-Event-ID"] = lastEventId;
        const response = await fetch(`${threadRoot(threadId)}/events:stream`, {
          headers,
          signal: controller.signal,
        });
        if (!response.ok || !response.body) return;
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        while (!stopped && myRound === round) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const frames = buffer.split(/\r?\n\r?\n/);
          buffer = frames.pop() || "";
          let reachedDone = false;
          for (const frame of frames) {
            const parsed = parseDatasetSseFrame(frame);
            if (!parsed) continue;
            if (parsed.cursor) lastEventId = parsed.cursor;
            const event = toDatasetStreamEvent(parsed);
            if (parsed.event === "done") {
              streamEnded = true;
              await refreshSteps();
              eventHandler.current(
                event ?? {
                  event: "done",
                  tab: "materials",
                  stage: parsed.payload.stage || parsed.payload.current_step || "dataset.material_preparation",
                  stepId: parsed.payload.step_id || "",
                  status: parsed.payload.status,
                },
              );
              reachedDone = true;
              break;
            }
            if (!event) continue;
            const isFlowEvent = event.event === "step.finish" || event.event === "checkpoint.continue";
            if (!isCurrentDatasetExecutionEvent(activeStepId.current, event.stepId)) {
              if (inactiveStepIds.current.has(event.stepId)) continue;
              const next = await refreshSteps();
              if (!isCurrentDatasetExecutionEvent(next?.activeStepId, event.stepId)) continue;
            }
            eventHandler.current(event);
            if (isFlowEvent) {
              await refreshSteps();
            }
          }
          if (reachedDone) {
            await reader.cancel().catch(() => undefined);
            break;
          }
        }
      } catch {
        // Aborted on unmount / reconnect, or Evo closed the stream when the run ended.
      } finally {
        if (myRound === round) streamEnded = true;
      }
    };

    void consume();
    resumeStream.current = (force?: boolean) => {
      if (stopped) return;
      if (!force && !streamEnded) return;
      void consume();
    };

    return () => {
      stopped = true;
      resumeStream.current = () => undefined;
      activeController?.abort();
    };
  }, [refreshSteps, threadId]);

  return {
    refreshSteps,
    resumeAfterWrite,
  };
}

type ParsedDatasetFrame = {
  event: string;
  cursor?: string;
  payload: {
    stage?: string;
    current_step?: string;
    step_id?: string;
    operation_id?: string;
    attempt_id?: string;
    status?: string;
    last_event_id?: string;
    partition?: { id?: string; index?: number; total?: number };
    progress?: { current?: number | null; total?: number | null };
  };
};

function parseDatasetSseFrame(frame: string): ParsedDatasetFrame | undefined {
  const lines = frame.split(/\r?\n/);
  let event = "";
  let id: string | undefined;
  const data = lines
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("");
  for (const line of lines) {
    if (line.startsWith("id:")) id = line.slice(3).trim() || undefined;
    if (line.startsWith("event:")) event = line.slice(6).trim();
  }
  if (!data) return undefined;
  try {
    const payload = JSON.parse(data) as ParsedDatasetFrame["payload"];
    const cursor = id || (typeof payload.last_event_id === "string" ? payload.last_event_id : undefined);
    return { event: event || "message", cursor, payload };
  } catch {
    return undefined;
  }
}

function toDatasetStreamEvent(parsed: ParsedDatasetFrame): DatasetStreamEvent | undefined {
  const { event, payload } = parsed;
  const stage = payload.stage || payload.current_step;
  const tab = stage ? datasetTabForStage(stage) : undefined;
  return stage && tab && event
    ? {
        event,
        tab,
        stage,
        stepId: payload.step_id || "",
        operationId: payload.operation_id,
        attemptId: payload.attempt_id,
        status: payload.status,
        partition: payload.partition,
        progress: payload.progress,
      }
    : undefined;
}
