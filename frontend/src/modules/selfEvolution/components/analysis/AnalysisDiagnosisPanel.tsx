import { useEffect, useMemo, useState, type ReactNode } from "react";
import { Input, Select, Table, Tag, Tooltip, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  AimOutlined,
  CheckCircleFilled,
  ExperimentOutlined,
  FileSearchOutlined,
  SearchOutlined,
  WarningFilled,
} from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { getAffectedBlockLabel } from "../../shared";
import {
  evidenceRecordLabel,
  normalizeAnalysisDiagnosis,
  type AnalysisCaseDiagnostic,
  type AnalysisRootCauseGroup,
} from "./analysisDiagnosis";

const { Text } = Typography;

type Props = {
  content: Record<string, unknown> | undefined;
};

function percent(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function statusTone(value: string): "success" | "processing" | "warning" | "default" {
  if (["repair_ready", "completed", "not_required"].includes(value)) return "success";
  if (["needs_probe", "needs_review", "pending_analysis"].includes(value)) return "warning";
  if (value) return "processing";
  return "default";
}

export function AnalysisDiagnosisPanel({ content }: Props) {
  const { t } = useTranslation();
  const model = useMemo(() => normalizeAnalysisDiagnosis(content), [content]);
  const [selectedGroupId, setSelectedGroupId] = useState("");
  const [selectedCaseId, setSelectedCaseId] = useState("");
  const [keyword, setKeyword] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  useEffect(() => {
    if (!model.groups.some((item) => item.groupId === selectedGroupId)) {
      setSelectedGroupId(model.groups[0]?.groupId || "");
    }
  }, [model.groups, selectedGroupId]);

  const selectedGroup = model.groups.find((item) => item.groupId === selectedGroupId);
  const visibleCases = useMemo(() => {
    const groupCaseIds = new Set(selectedGroup?.caseIds || []);
    const query = keyword.trim().toLowerCase();
    return model.cases.filter((item) => {
      if (selectedGroup && !groupCaseIds.has(item.caseId)) return false;
      if (statusFilter !== "all" && item.analysisStatus !== statusFilter) return false;
      if (!query) return true;
      return [item.caseId, item.issueType, item.mechanismId, item.affectedBlock, ...item.statements]
        .some((value) => value.toLowerCase().includes(query));
    });
  }, [keyword, model.cases, selectedGroup, statusFilter]);

  useEffect(() => {
    if (!visibleCases.some((item) => item.caseId === selectedCaseId)) {
      const representative = selectedGroup?.representativeCaseId;
      setSelectedCaseId(
        visibleCases.find((item) => item.caseId === representative)?.caseId ||
        visibleCases[0]?.caseId || "",
      );
    }
  }, [selectedCaseId, selectedGroup, visibleCases]);

  const selectedCase = visibleCases.find((item) => item.caseId === selectedCaseId);
  const statusOptions = Array.from(new Set(model.cases.map((item) => item.analysisStatus).filter(Boolean)));
  const columns = useMemo<ColumnsType<AnalysisCaseDiagnostic>>(() => [
    { title: "Case", dataIndex: "caseId", key: "caseId", width: 128 },
    {
      title: t("selfEvolutionRun.analysisDiagnosis.problem"),
      dataIndex: "issueType",
      key: "issueType",
      width: 220,
      render: (value: string) => <span className="self-evolution-analysis-ellipsis" title={value}>{value || "-"}</span>,
    },
    {
      title: t("selfEvolutionRun.analysisDiagnosis.rootCause"),
      dataIndex: "mechanismId",
      key: "mechanismId",
      width: 240,
      render: (value: string) => <span className="self-evolution-analysis-ellipsis" title={value}>{value || "-"}</span>,
    },
    {
      title: t("selfEvolutionRun.analysisDiagnosis.evidence"),
      dataIndex: "evidenceCount",
      key: "evidenceCount",
      width: 108,
      render: (value: number, row) => <span>{value} · {row.evidenceLevel || "-"}</span>,
    },
    {
      title: t("selfEvolutionRun.analysisDiagnosis.status"),
      dataIndex: "analysisStatus",
      key: "analysisStatus",
      width: 130,
      render: (value: string) => <Tag color={statusTone(value)}>{value || "-"}</Tag>,
    },
  ], [t]);

  return (
    <div className="self-evolution-analysis-diagnosis">
      <section className="self-evolution-analysis-progress" aria-label={t("selfEvolutionRun.analysisDiagnosis.progressAria")}>
        <ProgressItem label={t("selfEvolutionRun.analysisDiagnosis.problemObserved")} value={model.overview.problemObserved} total={model.overview.totalCases} icon={<WarningFilled />} tone="problem" />
        <ProgressItem label={t("selfEvolutionRun.analysisDiagnosis.rootConfirmed")} value={model.overview.rootConfirmed} total={model.overview.totalCases} icon={<AimOutlined />} tone="root" />
        <ProgressItem label={t("selfEvolutionRun.analysisDiagnosis.evidenceBacked")} value={model.overview.evidenceBacked} total={model.overview.totalCases} icon={<FileSearchOutlined />} tone="evidence" />
        <ProgressItem label={t("selfEvolutionRun.analysisDiagnosis.repairReady")} value={model.overview.repairReady} total={model.overview.totalCases} icon={<CheckCircleFilled />} tone="repair" />
      </section>

      <div className="self-evolution-analysis-workspace">
        <aside className="self-evolution-analysis-groups">
          <div className="self-evolution-analysis-pane-head">
            <div>
              <Text strong>{t("selfEvolutionRun.analysisDiagnosis.rootGroups")}</Text>
              <span>{t("selfEvolutionRun.analysisDiagnosis.rootGroupsCount", { count: model.groups.length })}</span>
            </div>
            <Tooltip title={t("selfEvolutionRun.analysisDiagnosis.traceCoverage")}>
              <span className="self-evolution-analysis-trace-rate">
                {model.overview.traceComplete}/{model.overview.totalCases}
              </span>
            </Tooltip>
          </div>
          <div className="self-evolution-analysis-group-list">
            {model.groups.map((group) => (
              <RootCauseGroupItem
                key={group.groupId}
                group={group}
                active={group.groupId === selectedGroupId}
                onSelect={() => setSelectedGroupId(group.groupId)}
              />
            ))}
          </div>
        </aside>

        <section className="self-evolution-analysis-cases">
          <div className="self-evolution-analysis-filter-row">
            <Input
              allowClear
              prefix={<SearchOutlined />}
              value={keyword}
              placeholder={t("selfEvolutionRun.analysisDiagnosis.searchPlaceholder")}
              onChange={(event) => setKeyword(event.target.value)}
            />
            <Select
              value={statusFilter}
              onChange={setStatusFilter}
              options={[
                { value: "all", label: t("selfEvolutionRun.analysisDiagnosis.allStatuses") },
                ...statusOptions.map((value) => ({ value, label: value })),
              ]}
            />
          </div>
          <Table<AnalysisCaseDiagnostic>
            className="self-evolution-analysis-case-table"
            size="small"
            rowKey="caseId"
            columns={columns}
            dataSource={visibleCases}
            pagination={{ pageSize: 10, size: "small", showSizeChanger: false }}
            scroll={{ x: 826 }}
            rowClassName={(row) => row.caseId === selectedCaseId ? "is-selected" : ""}
            onRow={(row) => ({ onClick: () => setSelectedCaseId(row.caseId) })}
          />
        </section>
      </div>

      {selectedCase && <CaseDiagnosisDetail item={selectedCase} />}
    </div>
  );
}

function ProgressItem({ label, value, total, icon, tone }: {
  label: string;
  value: number;
  total: number;
  icon: ReactNode;
  tone: string;
}) {
  return (
    <div className={`self-evolution-analysis-progress-item is-${tone}`}>
      <span className="self-evolution-analysis-progress-icon">{icon}</span>
      <div><strong>{value}</strong><span>/ {total}</span><small>{label}</small></div>
    </div>
  );
}

function RootCauseGroupItem({ group, active, onSelect }: {
  group: AnalysisRootCauseGroup;
  active: boolean;
  onSelect: () => void;
}) {
  const { t } = useTranslation();
  return (
    <button type="button" className={`self-evolution-analysis-group${active ? " is-active" : ""}`} onClick={onSelect}>
      <span className="self-evolution-analysis-group-block">{getAffectedBlockLabel(group.affectedBlock) || group.affectedBlock}</span>
      <strong title={group.mechanismId}>{group.mechanismId || group.failureMode}</strong>
      <span className="self-evolution-analysis-group-meta">
        {t("selfEvolutionRun.analysisDiagnosis.affectedCases", { count: group.caseCount })}
        <b>{percent(group.averageConfidence)}</b>
      </span>
      <span className="self-evolution-analysis-group-bar"><i style={{ width: percent(group.repairReadyCount / Math.max(1, group.caseCount)) }} /></span>
      <small>{group.evidenceLevel || t("selfEvolutionRun.analysisDiagnosis.unconfirmed")}</small>
    </button>
  );
}

function CaseDiagnosisDetail({ item }: { item: AnalysisCaseDiagnostic }) {
  const { t } = useTranslation();
  const stages = item.stageSequence.length ? item.stageSequence : ["retrieve", "rerank", "context_assembly", "llm_generate"];
  return (
    <section className="self-evolution-analysis-detail" aria-label={t("selfEvolutionRun.analysisDiagnosis.caseDetailAria", { caseId: item.caseId })}>
      <div className="self-evolution-analysis-detail-head">
        <div><Text strong>{item.caseId}</Text><span>{item.traceId || "-"}</span></div>
        <div><Tag color={statusTone(item.analysisStatus)}>{item.analysisStatus}</Tag><span>{percent(item.confidence)}</span></div>
      </div>
      <div className="self-evolution-analysis-causal-chain">
        <DiagnosisNode icon={<WarningFilled />} label={t("selfEvolutionRun.analysisDiagnosis.problem")} title={item.issueType} detail={item.statements[0] || item.failureMode} tone="problem" />
        <DiagnosisNode icon={<AimOutlined />} label={t("selfEvolutionRun.analysisDiagnosis.rootCause")} title={item.mechanismId} detail={`${getAffectedBlockLabel(item.affectedBlock) || item.affectedBlock} · ${item.rootStage || item.checkpointStage}`} tone="root" />
        <DiagnosisNode icon={<FileSearchOutlined />} label={t("selfEvolutionRun.analysisDiagnosis.evidence")} title={`${item.evidenceCount} · ${item.evidenceLevel || "-"}`} detail={item.decisionSource || item.routeSignature} tone="evidence" />
        <DiagnosisNode icon={<ExperimentOutlined />} label="Repair" title={item.repairReady ? t("selfEvolutionRun.analysisDiagnosis.ready") : t("selfEvolutionRun.analysisDiagnosis.blocked")} detail={item.repairGroupIds.join(", ") || "-"} tone="repair" />
      </div>
      <div className="self-evolution-analysis-detail-grid">
        <div className="self-evolution-analysis-lifecycle">
          <Text strong>{t("selfEvolutionRun.analysisDiagnosis.evidenceLifecycle")}</Text>
          <div className="self-evolution-analysis-stage-list">
            {stages.map((stage, index) => {
              const checkpoint = stage === item.checkpointStage || stage === item.rootStage;
              return (
                <div key={`${stage}-${index}`} className={`self-evolution-analysis-stage${checkpoint ? " is-checkpoint" : ""}`}>
                  {checkpoint ? <WarningFilled /> : <CheckCircleFilled />}
                  <span>{stage}</span>
                </div>
              );
            })}
          </div>
          <span className="self-evolution-analysis-route">{item.routeSignature || "-"}</span>
        </div>
        <div className="self-evolution-analysis-evidence-list">
          <Text strong>{t("selfEvolutionRun.analysisDiagnosis.keyEvidence")}</Text>
          {item.evidenceRecords.length > 0 ? item.evidenceRecords.map((record, index) => (
            <div key={`${item.caseId}-evidence-${index}`}>
              <FileSearchOutlined />
              <span>{evidenceRecordLabel(record)}</span>
              <small>{String(record.source || record.evidence_level || "trace")}</small>
            </div>
          )) : <span className="self-evolution-analysis-empty">{t("selfEvolutionRun.analysisDiagnosis.noEvidence")}</span>}
          {item.missingEvidence.map((value) => <div key={value} className="is-missing"><WarningFilled /><span>{value}</span></div>)}
        </div>
      </div>
    </section>
  );
}

function DiagnosisNode({ icon, label, title, detail, tone }: {
  icon: ReactNode;
  label: string;
  title: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className={`self-evolution-analysis-node is-${tone}`}>
      <span>{icon}</span>
      <div><small>{label}</small><strong title={title}>{title || "-"}</strong><p title={detail}>{detail || "-"}</p></div>
    </div>
  );
}
