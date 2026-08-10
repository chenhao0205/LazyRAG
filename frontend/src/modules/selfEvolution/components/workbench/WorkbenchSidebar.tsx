import { type ReactNode } from "react";
import { Typography } from "antd";
import { CloseOutlined, DownOutlined, HistoryOutlined, PlusOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { type SelfEvolutionWorkbenchTab } from "../types";
import type { SelfEvolutionChatMessage } from "../types";
import type { SelfEvolutionSessionSummary } from "./types";

const { Paragraph, Text, Title } = Typography;

export function WorkbenchSidebar({
  activeStepText,
  routeThreadId,
  isRestoringThread,
  threadRestoreError,
  activeWorkbenchTab,
  activeStageLabel,
  activeSession,
  displayedMessages,
  chatSessionsCount,
  artifactNavigationPanel,
  isArtifactPanelOpen,
  onCloseArtifactPanel,
  onWorkbenchTabChange,
  onRetryRestoreThread,
  onOpenHistorySessionModal,
  onCloseSession,
  onCreateSession,
  onMessageAnchorClick,
}: {
  activeStepText: string;
  routeThreadId?: string;
  isRestoringThread: boolean;
  threadRestoreError: string;
  activeWorkbenchTab?: SelfEvolutionWorkbenchTab;
  activeStageLabel: string;
  activeSession: SelfEvolutionSessionSummary;
  displayedMessages: SelfEvolutionChatMessage[];
  chatSessionsCount: number;
  artifactNavigationPanel: ReactNode;
  isArtifactPanelOpen: boolean;
  onCloseArtifactPanel: () => void;
  onWorkbenchTabChange: (tab?: SelfEvolutionWorkbenchTab) => void;
  onRetryRestoreThread: () => void;
  onOpenHistorySessionModal: () => void;
  onCloseSession: (sessionId: string) => void;
  onCreateSession: () => void;
  onMessageAnchorClick: (messageId: string) => void;
}) {
  const { t } = useTranslation();
  const userMessageAnchors = displayedMessages
    .map((item, index) => ({ ...item, index }))
    .filter((item) => item.role === "user");
  const activeNavigationTab = activeWorkbenchTab === "messages" || activeWorkbenchTab === "processes"
    ? activeWorkbenchTab
    : undefined;
  const activeNavigationTitle = activeNavigationTab === "messages"
    ? t("selfEvolutionRun.navInteractionTitle")
    : t("selfEvolutionRun.navStageOverviewTitle");
  const activeNavigationDesc = activeNavigationTab === "messages"
    ? t("selfEvolutionRun.navInteractionDesc")
    : activeStageLabel;
  const getMessageNavTitle = (content: string) => content.replace(/\s+/g, " ").trim() || t("selfEvolutionRun.emptyMessage");

  const renderStageNavigationPanel = () => (
    <div className="self-evolution-artifact-sidebar is-navigation">
      {artifactNavigationPanel}
    </div>
  );
  const renderSidebarToggle = (key: SelfEvolutionWorkbenchTab, title: string, desc: string) => {
    const isExpanded = activeNavigationTab === key;
    return (
      <section className={`self-evolution-workbench-accordion-section${isExpanded ? " is-active" : ""}`}>
        <button
          type="button"
          className="self-evolution-workbench-accordion-toggle"
          onClick={() => onWorkbenchTabChange(isExpanded ? undefined : key)}
          aria-expanded={isExpanded}
          aria-controls={`self-evolution-workbench-sidebar-${key}`}
        >
          <DownOutlined className="self-evolution-workbench-accordion-arrow" />
          <span>
            <strong>{title}</strong>
            <small>{desc}</small>
          </span>
        </button>
      </section>
    );
  };
  const renderMessagesNavigationPanel = () => (
    <div className="self-evolution-message-nav-card">
      <div className="self-evolution-message-nav-summary">
        <strong>{activeSession.title}</strong>
        <span>{routeThreadId ? t("selfEvolutionRun.threadLabelShort", { id: routeThreadId }) : t("selfEvolutionRun.localSession")}</span>
        <span>{displayedMessages.length ? t("selfEvolutionRun.messageCountLabel", { count: displayedMessages.length }) : t("selfEvolutionRun.waitingMessages")}</span>
      </div>
      <div className="self-evolution-message-nav-list">
        {userMessageAnchors.length ? (
          userMessageAnchors.map((item, index) => (
            <button key={item.id} type="button" onClick={() => onMessageAnchorClick(item.id)}>
              <strong>{t("selfEvolutionRun.userMessageLabel", { index: index + 1 })}</strong>
              <span>{getMessageNavTitle(item.content)}</span>
              <em>{item.time}</em>
            </button>
          ))
        ) : (
          <span className="self-evolution-message-nav-empty">{t("selfEvolutionRun.noUserMessages")}</span>
        )}
      </div>
    </div>
  );

  return (
    <aside
      className={`self-evolution-workbench-nav${activeNavigationTab ? " has-open-panel" : ""}`}
      aria-label={t("selfEvolutionRun.workbenchNavAria")}
      onClick={isArtifactPanelOpen ? onCloseArtifactPanel : undefined}
    >
      <div className="self-evolution-workbench-nav-head">
        <Title level={3}>{t("selfEvolutionRun.executionOrchestration")}</Title>
        <Paragraph>{t("selfEvolutionRun.currentFocus", { step: activeStepText })}</Paragraph>
        {routeThreadId && (
          <Text className="self-evolution-detail-thread">
            {t("selfEvolutionRun.threadIdWithRestore", { id: routeThreadId, restoring: isRestoringThread ? t("selfEvolutionRun.restoringDetailSuffix") : "" })}
          </Text>
        )}
        {threadRestoreError && routeThreadId && (
          <div className="self-evolution-restore-error" role="alert">
            <span>{threadRestoreError}</span>
            <button type="button" onClick={onRetryRestoreThread}>
              {t("selfEvolutionRun.retry")}
            </button>
          </div>
        )}
      </div>
      <div className="self-evolution-workbench-accordion">
        <div className="self-evolution-workbench-accordion-toggles">
          {renderSidebarToggle("messages", t("selfEvolutionRun.navInteractionTitle"), t("selfEvolutionRun.navInteractionDesc"))}
          {renderSidebarToggle("processes", t("selfEvolutionRun.navStageOverviewTitle"), activeStageLabel)}
        </div>
        {activeNavigationTab && (
          <section
            id={`self-evolution-workbench-sidebar-${activeNavigationTab}`}
            className={`self-evolution-workbench-navigation-panel is-${activeNavigationTab}`}
          >
            <header className="self-evolution-workbench-navigation-panel-head">
              <div>
                <strong>{activeNavigationTitle}</strong>
                <span>{activeNavigationDesc}</span>
              </div>
              <button
                type="button"
                onClick={() => onWorkbenchTabChange(undefined)}
                title={t("selfEvolutionRun.collapse")}
                aria-label={t("selfEvolutionRun.collapse")}
              >
                <CloseOutlined />
              </button>
            </header>
            <div className="self-evolution-workbench-navigation-panel-content">
              {activeNavigationTab === "messages" ? renderMessagesNavigationPanel() : renderStageNavigationPanel()}
            </div>
          </section>
        )}
      </div>
      <div className="self-evolution-workbench-sidebar-actions">
        {chatSessionsCount > 1 && (
          <button type="button" onClick={() => onCloseSession(activeSession.id)} title={t("selfEvolutionRun.closeCurrentSession")}>
            <CloseOutlined />
          </button>
        )}
        <button type="button" onClick={onCreateSession} title={t("selfEvolutionRun.newSession")}>
          <PlusOutlined />
          <span>{t("selfEvolutionRun.new")}</span>
        </button>
        <button type="button" onClick={onOpenHistorySessionModal} title={t("selfEvolutionRun.openHistoryAria")}>
          <HistoryOutlined />
          <span>{t("selfEvolutionRun.history")}</span>
        </button>
      </div>
    </aside>
  );
}
