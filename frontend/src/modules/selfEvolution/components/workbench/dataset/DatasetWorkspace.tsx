import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Empty, Modal, message } from "antd";
import { CasesStage } from "./CasesStage";
import { DatasetResultModal } from "./DatasetResultModal";
import { MaterialsStage } from "./MaterialsStage";
import { TopicsStage } from "./TopicsStage";
import { datasetRoot, describeRequestError, newRequestId, postJson } from "./api";
import { STATUS_TEXT } from "./primitives";
import {
  draftAffectsTab,
  executionImpactTabs,
  resolveRevisionRefreshAction,
  shouldResumeDatasetStream,
  TERMINAL_STAGE_EVENTS,
} from "./datasetRefresh";
import { applyTopicLabelPartitionEvent, type TopicDiscoveryProgress } from "./topicLabelProgress";
import { applyCaseGenerationPartitionEvent, type CaseGenerationProgress } from "./caseGenerationProgress";
import { DATASET_TABS, useDatasetStages, type DatasetStreamEvent } from "./useDatasetStages";
import { INITIAL_STAGE_STATUSES } from "./stageState";
import { usePublishDatasetResultAction } from "./stageAction";
import "./dataset.scss";
import {
  DRAFT_IMPACT_DETAIL,
  DRAFT_IMPACT_START,
  DRAFT_LABELS,
  type DatasetDraft,
  type DatasetTab,
  type ThreadStepsResponse,
  type VisualStatus,
} from "./types";

const STEP_SYMBOL: Record<string, string> = {
  done: "✓",
  running: "●",
  paused: "⏸",
  stale: "↻",
  failed: "!",
};

export function DatasetWorkspace({
  threadId,
  stageStatuses = INITIAL_STAGE_STATUSES,
  suggestedTab,
  onStepsSnapshot,
  onWriteApplied,
  executionResumeToken = 0,
}: {
  threadId?: string;
  /** Derived from the Workbench-owned /steps list (deriveDatasetView). */
  stageStatuses?: Record<DatasetTab, VisualStatus>;
  suggestedTab?: DatasetTab;
  onStepsSnapshot?: (response: ThreadStepsResponse) => void;
  onWriteApplied?: () => void;
  executionResumeToken?: number;
}) {
  const [tab, setTab] = useState<DatasetTab>("materials");
  const [refreshToken, setRefreshToken] = useState(0);
  const [overviewToken, setOverviewToken] = useState(0);
  const [topicLabelProgress, setTopicLabelProgress] = useState<TopicDiscoveryProgress>();
  const [caseGenerationProgress, setCaseGenerationProgress] = useState<CaseGenerationProgress>();
  const [caseReconciliationToken, setCaseReconciliationToken] = useState(0);
  const [staleTab, setStaleTab] = useState<DatasetTab>();
  const [draft, setDraft] = useState<DatasetDraft>();
  const [applying, setApplying] = useState(false);
  const [resultOpen, setResultOpen] = useState(false);
  const followActiveStage = useRef(true);
  const revisions = useRef<Partial<Record<DatasetTab, string | null>>>({});
  // Set while an event-triggered overview reload is in flight, so that manual
  // refreshes and stage switches never raise the "data changed" banner.
  const probing = useRef<DatasetTab>();
  const pendingRefresh = useRef(new Set<DatasetTab>());
  const tabRef = useRef(tab);
  tabRef.current = tab;
  // Read by callbacks that must stay referentially stable for the stage panels.
  const draftRef = useRef<DatasetDraft>();
  draftRef.current = draft;
  const topicProgressRef = useRef<TopicDiscoveryProgress>();
  const caseProgressRef = useRef<CaseGenerationProgress>();
  const progressFlushRef = useRef<number>();
  const handledExecutionResumeToken = useRef(executionResumeToken);

  const scheduleProgressFlush = useCallback(() => {
    if (progressFlushRef.current) return;
    progressFlushRef.current = requestAnimationFrame(() => {
      progressFlushRef.current = 0;
      setTopicLabelProgress(topicProgressRef.current);
      setCaseGenerationProgress(caseProgressRef.current);
    });
  }, []);

  const flushPendingRefresh = useCallback((targetTab: DatasetTab) => {
    if (!pendingRefresh.current.has(targetTab)) return;
    pendingRefresh.current.delete(targetTab);
    if (draftAffectsTab(draftRef.current, targetTab)) {
      setStaleTab(targetTab);
      setOverviewToken((token) => token + 1);
      return;
    }
    setStaleTab(undefined);
    setRefreshToken((token) => token + 1);
    setOverviewToken((token) => token + 1);
  }, []);

  // Terminal SSE events reload overview for the finished stage; when the
  // published revision changes, lists auto-refresh unless a draft would be lost.
  // Navigation status comes from the parent /steps snapshot (onStepsSnapshot).
  const handleStageEvent = useCallback((event: DatasetStreamEvent) => {
    if (event.tab === "topics" || event.stage === "dataset.topic_discovery") {
      const next = applyTopicLabelPartitionEvent(topicProgressRef.current, event);
      if (next !== topicProgressRef.current) {
        topicProgressRef.current = next;
        scheduleProgressFlush();
      }
    }
    if (event.tab === "cases" || event.stage === "dataset.case_generation") {
      const { progress: next, shouldRefreshBaseline } = applyCaseGenerationPartitionEvent(
        caseProgressRef.current,
        event,
      );
      if (next !== caseProgressRef.current) {
        caseProgressRef.current = next;
        scheduleProgressFlush();
      }
      if (shouldRefreshBaseline) {
        setOverviewToken((token) => token + 1);
      }
    }
    if (!TERMINAL_STAGE_EVENTS.has(event.event)) return;
    if (event.tab === "cases") {
      setCaseReconciliationToken((token) => token + 1);
    }
    if (event.tab === tabRef.current) {
      probing.current = event.tab;
      setOverviewToken((token) => token + 1);
      return;
    }
    pendingRefresh.current.add(event.tab);
  }, [scheduleProgressFlush]);

  const handleStepsSnapshot = useCallback(
    (response: ThreadStepsResponse) => {
      onStepsSnapshot?.(response);
    },
    [onStepsSnapshot],
  );

  const clearExecutionProgress = useCallback((tabs: DatasetTab[]) => {
    if (tabs.includes("topics")) topicProgressRef.current = undefined;
    if (tabs.includes("cases")) caseProgressRef.current = undefined;
    scheduleProgressFlush();
  }, [scheduleProgressFlush]);

  const { refreshSteps, resumeAfterWrite } = useDatasetStages(threadId, {
    onStepsSnapshot: handleStepsSnapshot,
    onStageEvent: handleStageEvent,
  });

  const openResult = useCallback(() => setResultOpen(true), []);
  const resultAvailable = stageStatuses.cases === "done" || stageStatuses.cases === "partial";
  usePublishDatasetResultAction(
    resultAvailable ? { label: "查看生成结果", onClick: openResult } : undefined,
  );

  const handleOverviewRevision = useCallback((stageTab: DatasetTab, revision: string | null) => {
    const previous = revisions.current[stageTab];
    const wasProbing = probing.current === stageTab;
    if (wasProbing) {
      probing.current = undefined;
    }
    const action = resolveRevisionRefreshAction(
      stageTab,
      tabRef.current,
      previous,
      revision,
      draftRef.current,
    );
    revisions.current[stageTab] = revision;
    const draftBlocks = draftAffectsTab(draftRef.current, stageTab);
    if (action === "auto" || (wasProbing && action === "none" && !draftBlocks)) {
      setStaleTab(undefined);
      setRefreshToken((token) => token + 1);
    } else if (action === "stale" || (wasProbing && draftBlocks)) {
      setStaleTab(stageTab);
    } else if (action === "pending") {
      pendingRefresh.current.add(stageTab);
    }
  }, []);

  const handleCaseExecutionReconciled = useCallback(() => {
    if (!caseProgressRef.current) return;
    caseProgressRef.current = undefined;
    scheduleProgressFlush();
  }, [scheduleProgressFlush]);

  useEffect(() => {
    if (!shouldResumeDatasetStream(handledExecutionResumeToken.current, executionResumeToken)) return;
    handledExecutionResumeToken.current = executionResumeToken;
    resumeAfterWrite();
  }, [executionResumeToken, resumeAfterWrite]);

  useEffect(() => {
    followActiveStage.current = true;
    setTab("materials");
    topicProgressRef.current = undefined;
    caseProgressRef.current = undefined;
    pendingRefresh.current.clear();
    revisions.current = {};
    probing.current = undefined;
    if (progressFlushRef.current) {
      cancelAnimationFrame(progressFlushRef.current);
      progressFlushRef.current = 0;
    }
    setTopicLabelProgress(undefined);
    setCaseGenerationProgress(undefined);
    setCaseReconciliationToken(0);
    setStaleTab(undefined);
  }, [threadId]);

  useEffect(
    () => () => {
      if (progressFlushRef.current) {
        cancelAnimationFrame(progressFlushRef.current);
      }
    },
    [],
  );

  // The executing stage only decides the default tab on first entry; later
  // progress must not pull the view away from what the user is reading.
  useEffect(() => {
    if (suggestedTab && followActiveStage.current) {
      followActiveStage.current = false;
      setTab(suggestedTab);
      flushPendingRefresh(suggestedTab);
    }
  }, [suggestedTab, flushPendingRefresh]);

  const selectTab = (next: DatasetTab) => {
    followActiveStage.current = false;
    setStaleTab(undefined);
    setTab(next);
    void refreshSteps();
    flushPendingRefresh(next);
  };

  const refreshNow = () => {
    setStaleTab(undefined);
    // A draft targets the revision the page was showing, so it cannot survive.
    setDraft(undefined);
    setRefreshToken((token) => token + 1);
  };

  const saveDraft = useCallback((next: DatasetDraft) => {
    const current = draftRef.current;
    if (current && current.kind !== next.kind) {
      message.warning("已有待应用的修改，请先应用或放弃后再编辑其他内容。");
      return false;
    }
    if (current?.kind === "topic-names" && next.kind === "topic-names") {
      setDraft({ ...next, names: { ...current.names, ...next.names } });
    } else {
      setDraft(next);
    }
    message.success("修改已暂存，应用前不会影响当前结果。");
    return true;
  }, []);

  const applyDraft = async () => {
    if (!threadId || !draft) return;
    setApplying(true);
    try {
      const affectedTabs = executionImpactTabs(draft.kind);
      const root = datasetRoot(threadId);
      const requestId = newRequestId();
      if (draft.kind === "materials-config") {
        await postJson(`${root}/materials:apply`, {
          request_id: requestId,
          expected_revision: draft.revision,
          changes: draft.changes,
        });
      } else if (draft.kind === "chunk-selection") {
        await postJson(`${root}/materials:apply`, {
          request_id: requestId,
          expected_revision: draft.revision,
          changes: { chunk_selection_changes: draft.changes },
        });
      } else if (draft.kind === "topic-names") {
        await postJson(`${root}/topics:apply`, {
          request_id: requestId,
          expected_revision: draft.revision,
          changes: Object.entries(draft.names).map(([topic_id, name]) => ({ topic_id, name })),
        });
      } else {
        await postJson(`${root}/generation-plan:apply`, {
          request_id: requestId,
          expected_revision: draft.revision,
          distribution: draft.distribution,
        });
      }
      setDraft(undefined);
      setStaleTab(undefined);
      clearExecutionProgress(affectedTabs);
      setRefreshToken((token) => token + 1);
      setOverviewToken((token) => token + 1);
      resumeAfterWrite();
      onWriteApplied?.();
      message.success("修改已应用，受影响的步骤将重新执行。");
    } catch (error) {
      message.error(describeRequestError(error, "应用修改失败"));
    } finally {
      setApplying(false);
    }
  };

  const confirmApply = () => {
    if (!draft) return;
    const start = DRAFT_IMPACT_START[draft.kind];
    Modal.confirm({
      className: "dataset-impact-modal",
      title: "确认修改影响",
      width: 520,
      content: (
        <div className="dataset-impact-copy">
          <p>系统会保留当前结果，完成更新后再替换为新版本。</p>
          <div className="dataset-impact-flow">
            {DATASET_TABS.map((item, index) => (
              <div
                className={`dataset-impact-node${
                  index === start ? " is-changed" : index > start ? " is-affected" : ""
                }`}
                key={item.id}
              >
                {item.label}
                <span>{index === start ? "本次修改" : index > start ? "需更新" : "不受影响"}</span>
              </div>
            ))}
          </div>
          <div className="dataset-impact-detail">{DRAFT_IMPACT_DETAIL[draft.kind]}</div>
        </div>
      ),
      okText: "确认并更新受影响步骤",
      cancelText: "暂不应用",
      onOk: applyDraft,
    });
  };

  if (!threadId) {
    return (
      <Empty
        className="dataset-workspace-empty"
        description="创建或打开一个自进化任务后，可在这里查看 Dataset 过程。"
      />
    );
  }

  return (
    <section className="dataset-workspace" aria-label="数据集自动构建">
      <nav className="dataset-stepper" aria-label="数据集内部步骤">
        {DATASET_TABS.map((item, index) => {
          const status = stageStatuses[item.id];
          return (
            <button
              type="button"
              key={item.id}
              className={`dataset-step is-${status}${tab === item.id ? " is-selected" : ""}`}
              onClick={() => selectTab(item.id)}
            >
              <span className="dataset-step-dot">{STEP_SYMBOL[status] || index + 1}</span>
              <span className="dataset-step-copy">
                <span className="dataset-step-name">{item.label}</span>
                <span className="dataset-step-status">{STATUS_TEXT[status]}</span>
              </span>
            </button>
          );
        })}
      </nav>

      {staleTab === tab ? (
        <div className="dataset-stale-banner">
          <span>
            该阶段结果已更新，当前列表与详情仍是你打开时的数据。
            {draft ? "刷新会丢弃尚未应用的修改。" : ""}
          </span>
          <Button size="small" onClick={refreshNow}>
            刷新
          </Button>
        </div>
      ) : null}

      <div className="dataset-content">
        {tab === "materials" ? (
          <MaterialsStage
            threadId={threadId}
            refreshToken={refreshToken}
            overviewToken={overviewToken}
            onOverviewRevision={handleOverviewRevision}
            draft={draft}
            onSaveDraft={saveDraft}
          />
        ) : tab === "topics" ? (
          <TopicsStage
            threadId={threadId}
            refreshToken={refreshToken}
            overviewToken={overviewToken}
            labelProgress={topicLabelProgress}
            onOverviewRevision={handleOverviewRevision}
            draft={draft}
            onSaveDraft={saveDraft}
          />
        ) : (
          <CasesStage
            threadId={threadId}
            refreshToken={refreshToken}
            overviewToken={overviewToken}
            progress={caseGenerationProgress}
            reconciliationToken={caseReconciliationToken}
            onOverviewRevision={handleOverviewRevision}
            onExecutionReconciled={handleCaseExecutionReconciled}
            onSaveDraft={saveDraft}
            onCaseSaved={() => {
              clearExecutionProgress(["cases"]);
              setRefreshToken((token) => token + 1);
              setOverviewToken((token) => token + 1);
              resumeAfterWrite();
              onWriteApplied?.();
            }}
          />
        )}
      </div>

      <DatasetResultModal
        threadId={threadId}
        open={resultOpen}
        onClose={() => setResultOpen(false)}
      />

      {draft ? (
        <footer className="dataset-change-bar">
          <div className="dataset-change-copy">
            <strong>{DRAFT_LABELS[draft.kind]}修改尚未应用</strong>
            <span>暂未影响正在运行的流程；确认影响范围后才会更新结果。</span>
          </div>
          <Button size="small" type="text" onClick={() => setDraft(undefined)}>
            放弃修改
          </Button>
          <Button size="small" type="primary" loading={applying} onClick={confirmApply}>
            查看影响并应用
          </Button>
        </footer>
      ) : null}
    </section>
  );
}
