import type { MouseEvent } from "react";
import { DownloadOutlined, LoadingOutlined } from "@ant-design/icons";
import { Typography } from "antd";
import {
  normalizeTraceObservation,
  TraceObservationView,
} from "../../components/TraceObservationView";
import {
  getResultItems,
  getWorkflowResultLabels,
  stringifyResultPayload,
} from "../../shared";
import type {
  ArtifactPanelItem,
  CaseArtifactState,
  TFunction,
} from "./types";

const { Text } = Typography;

function CaseArtifactPreview({
  artifact,
  t,
  onRetry,
}: {
  artifact: CaseArtifactState;
  t: TFunction;
  onRetry: () => void;
}) {
  if (artifact.loading) {
    return (
      <div className="self-evolution-result-state is-loading">
        <LoadingOutlined spin />
        <span>
          {t("selfEvolutionRun.caseArtifactLoading", {
            id: artifact.artifactId,
          })}
        </span>
      </div>
    );
  }
  if (artifact.error) {
    return (
      <div className="self-evolution-result-state is-error" role="alert">
        <span>{artifact.error}</span>
        <button type="button" onClick={onRetry}>
          {t("selfEvolutionRun.resultRetry")}
        </button>
      </div>
    );
  }

  const observation = normalizeTraceObservation(artifact.data);
  if (observation) {
    return (
      <TraceObservationView
        observation={observation}
        title={
          observation.kind === "compare"
            ? `${artifact.title}${t("selfEvolutionRun.caseTraceABObservationSuffix")}`
            : `${artifact.title}${t("selfEvolutionRun.caseObservationDetailSuffix")}`
        }
      />
    );
  }

  return (
    <div className="self-evolution-result-json">
      <div className="self-evolution-result-json-head">
        <Text>{artifact.artifactId}</Text>
        <Text>
          {t("selfEvolutionRun.resultItemCount", {
            count: getResultItems(artifact.data).length || 1,
          })}
        </Text>
      </div>
      <pre>{stringifyResultPayload(artifact.data)}</pre>
    </div>
  );
}

export function ArtifactDetailPanel({
  caseArtifact,
  activeArtifact,
  t,
  onRetryCaseArtifact,
  onDownloadArtifact,
}: {
  caseArtifact?: CaseArtifactState;
  activeArtifact?: ArtifactPanelItem;
  t: TFunction;
  onRetryCaseArtifact: () => void;
  onDownloadArtifact: (event: MouseEvent<HTMLButtonElement>) => void;
}) {
  if (caseArtifact) {
    return (
      <section
        className="self-evolution-artifact-detail"
        aria-label={t("selfEvolutionRun.artifactDetailAria")}
      >
        <div className="self-evolution-artifact-detail-head">
          <div>
            <Text strong>{caseArtifact.title}</Text>
            <span>{`${getWorkflowResultLabels()[caseArtifact.kind]} · ${t("selfEvolutionRun.singleCaseArtifact")}`}</span>
          </div>
        </div>
        <div className="self-evolution-artifact-detail-body">
          <CaseArtifactPreview
            artifact={caseArtifact}
            t={t}
            onRetry={onRetryCaseArtifact}
          />
        </div>
      </section>
    );
  }

  if (!activeArtifact) {
    return null;
  }

  return (
    <section
      className="self-evolution-artifact-detail"
      aria-label={t("selfEvolutionRun.artifactProductDetail")}
    >
      <div className="self-evolution-artifact-detail-head">
        <div>
          <Text strong>{activeArtifact.title}</Text>
          <span>{activeArtifact.desc}</span>
        </div>
        <button type="button" onClick={onDownloadArtifact}>
          <DownloadOutlined />
          <span>{t("selfEvolutionRun.downloadArtifact")}</span>
        </button>
      </div>
      <div
        className={`self-evolution-artifact-detail-body${activeArtifact.kind === "analysis-reports" ? " is-analysis-report" : ""}`}
      >
        {activeArtifact.preview}
      </div>
    </section>
  );
}
