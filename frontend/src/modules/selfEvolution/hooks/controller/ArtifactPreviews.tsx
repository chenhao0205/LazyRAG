import { LoadingOutlined } from "@ant-design/icons";
import { Table, Typography } from "antd";
import type { ColumnsType } from "antd/es/table";
import {
  getNestedRecordField,
  getResultItems,
  getStringField,
  getStructuredArrayField,
  getStructuredRecordField,
  isEmptyResultPayload,
  stringifyResultPayload,
  type WorkflowResultKind,
  type WorkflowResultState,
} from "../../shared";
import {
  buildDatasetQuestionTypeCounts,
  getDatasetTotalCaseCount,
} from "./helpers";
import type { DatasetCasePreviewRow, TFunction } from "./types";

const { Paragraph, Text } = Typography;

export function WorkflowResultPayload({
  kind,
  state,
  label,
  t,
  onRetry,
}: {
  kind: WorkflowResultKind;
  state: WorkflowResultState;
  label: string;
  t: TFunction;
  onRetry: (kind: WorkflowResultKind) => void;
}) {
  if (state.loading) {
    return (
      <div className="self-evolution-result-state is-loading">
        <LoadingOutlined spin />
        <span>{t("selfEvolutionRun.resultLoading", { label })}</span>
      </div>
    );
  }

  if (state.error) {
    return (
      <div className="self-evolution-result-state is-error" role="alert">
        <span>{state.error}</span>
        <button type="button" onClick={() => onRetry(kind)}>
          {t("selfEvolutionRun.resultRetry")}
        </button>
      </div>
    );
  }

  if (!state.loaded || isEmptyResultPayload(state.data)) {
    return (
      <Paragraph className="self-evolution-px-empty">
        {t(
          state.loaded
            ? "selfEvolutionRun.resultEmptyHint"
            : "selfEvolutionRun.resultNotLoadedHint",
          { label },
        )}
      </Paragraph>
    );
  }

  return (
    <div className="self-evolution-result-json">
      <div className="self-evolution-result-json-head">
        <Text>{t("selfEvolutionRun.resultJsonHead", { label })}</Text>
        <Text>
          {t("selfEvolutionRun.resultItemCount", {
            count: getResultItems(state.data).length || 1,
          })}
        </Text>
      </div>
      <pre>{stringifyResultPayload(state.data)}</pre>
    </div>
  );
}

export function DatasetArtifactPreview({
  state,
  artifactData,
  rows,
  columns,
  t,
  fallback,
}: {
  state: WorkflowResultState;
  artifactData: Record<string, unknown> | undefined;
  rows: DatasetCasePreviewRow[];
  columns: ColumnsType<DatasetCasePreviewRow>;
  t: TFunction;
  fallback: React.ReactNode;
}) {
  if (state.loading || state.error || !state.loaded || isEmptyResultPayload(state.data)) {
    return fallback;
  }

  const checks =
    getStructuredRecordField(artifactData, ["checks"]) ||
    getNestedRecordField(artifactData, ["checks"]);
  const typeCounts = buildDatasetQuestionTypeCounts(artifactData);
  const errors = getStructuredArrayField(checks, ["errors"]) || [];
  const warnings = getStructuredArrayField(checks, ["warnings"]) || [];
  const totalCases = getDatasetTotalCaseCount(artifactData, rows.length);
  const runId = getStringField(artifactData, ["run_id"]);

  return (
    <section
      className="self-evolution-dataset-preview"
      aria-label={t("selfEvolutionRun.datasetResultAria")}
    >
      <div className="self-evolution-dataset-cases-head">
        <Text>{t("selfEvolutionRun.finalEvalDataset")}</Text>
        <Text>
          {t("selfEvolutionRun.datasetSampleStats", {
            total: totalCases,
            shown: rows.length,
          })}
        </Text>
      </div>
      <div className="self-evolution-dataset-metrics">
        {runId ? <span>run_id：{runId}</span> : null}
        {checks ? (
          <>
            <span>
              ready：
              {checks.ready === false
                ? t("selfEvolutionRun.datasetReadyNo")
                : t("selfEvolutionRun.datasetReadyYes")}
            </span>
            <span>
              {t("selfEvolutionRun.datasetWarningError", {
                warnings: warnings.length,
                errors: errors.length,
              })}
            </span>
          </>
        ) : null}
        <span>
          {t("selfEvolutionRun.datasetTypeCount", {
            count: Object.keys(typeCounts).length,
          })}
        </span>
      </div>
      <Table<DatasetCasePreviewRow>
        className="self-evolution-dataset-table"
        size="small"
        rowKey="key"
        columns={columns}
        dataSource={rows}
        locale={{ emptyText: t("selfEvolutionRun.datasetCaseTableEmpty") }}
        pagination={
          rows.length > 10
            ? { pageSize: 10, size: "small", showSizeChanger: false }
            : false
        }
        scroll={{ x: 1250, y: 360 }}
      />
    </section>
  );
}
