import type { CaseSource, CaseStageKey, OperationStatus } from "./types";

export type CaseRetryAction =
  | {
      mode: "rerun";
      artifactId: "dataset.case_draft" | "dataset.case_enhancement";
      label: "重试生成";
      description: string;
    }
  | {
      mode: "retry";
      label: "重试生成";
      description: string;
    };

const STAGE_COPY: Partial<Record<CaseStageKey, {
  artifactId: "dataset.case_draft" | "dataset.case_enhancement";
  description: string;
}>> = {
  generate: {
    artifactId: "dataset.case_draft",
    description: "将重新生成问答，并更新该用例的判分规则。",
  },
  grading: {
    artifactId: "dataset.case_enhancement",
    description: "将仅重新生成该用例的判分规则，问答内容保持不变。",
  },
};

/**
 * A retry is only actionable after the selected LLM operation has settled.
 * Import cases have no LLM-generated artifact to rerun.
 * Failed/canceled steps have no published artifact, so they retry the recorded failure.
 */
export function getCaseRetryAction(
  stage: CaseStageKey,
  status: OperationStatus,
  source: CaseSource,
): CaseRetryAction | undefined {
  if (source === "imported" || status === "pending" || status === "running") {
    return undefined;
  }
  const copy = STAGE_COPY[stage];
  if (!copy) return undefined;
  if (status === "completed") {
    return {
      mode: "rerun",
      artifactId: copy.artifactId,
      label: "重试生成",
      description: copy.description,
    };
  }
  return {
    mode: "retry",
    label: "重试生成",
    description: copy.description,
  };
}

export function caseRetryRequest(
  threadRoot: string,
  caseId: string,
  action: CaseRetryAction,
): { path: string; body: Record<string, string> } {
  const encoded = encodeURIComponent(caseId);
  if (action.mode === "retry") {
    return { path: `${threadRoot}/cases/${encoded}/retry`, body: {} };
  }
  return {
    path: `${threadRoot}/cases/${encoded}/rerun`,
    body: { artifact_id: action.artifactId },
  };
}
