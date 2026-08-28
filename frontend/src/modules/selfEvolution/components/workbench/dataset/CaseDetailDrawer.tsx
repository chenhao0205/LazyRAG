import { useCallback, useEffect, useMemo, useState } from "react";
import { Alert, Button, Checkbox, Drawer, Input, Modal, message } from "antd";
import { DeleteOutlined, ReloadOutlined } from "@ant-design/icons";
import { datasetRoot, describeRequestError, getJson, newRequestId, patchJson, postJson, threadRoot } from "./api";
import { caseRetryRequest, getCaseRetryAction } from "./caseRetry";
import type { CaseGenerationProgress } from "./caseGenerationProgress";
import { PAGE_SIZE, useDatasetResource } from "./hooks";
import {
  Chip,
  ChunkText,
  DIFFICULTY_TEXT,
  DrawerAttributes,
  QUESTION_TYPE_TEXT,
  toVisualStatus,
} from "./primitives";
import type {
  CaseDetail,
  CaseKeyPoint,
  CaseRow,
  CaseStageKey,
  CaseTopicOption,
  PagedResponse,
} from "./types";

const STAGE_LABEL: Record<CaseStageKey, string> = {
  plan: "生成规划",
  generate: "问答生成",
  grading: "判分规则",
};

type Draft = {
  topicId?: string;
  question: string;
  answer: string;
  guidance: string;
  keyPoints: CaseKeyPoint[];
  forbiddenClaims: string[];
};

const toDraft = (detail: CaseDetail): Draft => ({
  topicId: detail.topic?.topic_id,
  question: detail.stages.generate.question || "",
  answer: detail.stages.generate.answer || "",
  guidance: detail.stages.generate.grading_guidance || "",
  keyPoints: (detail.stages.grading.key_points || []).map((point) => ({
    statement: point.statement,
    evidence_chunk_ids: [...point.evidence_chunk_ids],
  })),
  forbiddenClaims: [...(detail.stages.grading.forbidden_claims || [])],
});

/** Opens on the sub-stage that most needs attention: the first unfinished one. */
const focusStage = (detail: CaseDetail): CaseStageKey => {
  const order: CaseStageKey[] = ["plan", "generate", "grading"];
  return order.find((stage) => detail.stages[stage].status !== "completed") || "generate";
};

export function CaseDetailDrawer({
  threadId,
  row,
  progress,
  onClose,
  onSaved,
}: {
  threadId: string;
  row?: CaseRow;
  progress?: CaseGenerationProgress;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [stage, setStage] = useState<CaseStageKey>("generate");
  const [draft, setDraft] = useState<Draft>();
  const [saving, setSaving] = useState(false);
  const [retrying, setRetrying] = useState(false);
  const [evidenceEditorIndex, setEvidenceEditorIndex] = useState<number>();

  const caseId = row?.case_id;
  const root = datasetRoot(threadId);

  const fetchDetail = useCallback(
    () => getJson<CaseDetail>(`${root}/cases/${encodeURIComponent(caseId || "")}`),
    [caseId, root],
  );
  const detailResource = useDatasetResource(
    caseId ? fetchDetail : undefined,
    0,
    "读取用例详情失败",
    true,
  );
  const detail = detailResource.data?.case_id === caseId ? detailResource.data : undefined;

  useEffect(() => {
    if (!caseId) {
      setDraft(undefined);
      return;
    }
    if (!detail) return;
    setDraft(toDraft(detail));
    setStage(focusStage(detail));
    setEvidenceEditorIndex(undefined);
  }, [caseId, detail]);

  const loadTopicOptions = useCallback(
    () => getJson<PagedResponse<CaseTopicOption>>(
      `${root}/cases/${encodeURIComponent(caseId || "")}/topic-options`,
      { page_size: PAGE_SIZE },
    ),
    [caseId, root],
  );
  const canLoadTopicOptions = (() => {
    if (stage !== "plan" || !detail || detail.source === "imported") return;
    return true;
  })();
  const topicOptionsResource = useDatasetResource(
    canLoadTopicOptions ? loadTopicOptions : undefined,
    0,
    "读取可替换主题失败",
    true,
  );
  const topicOptions = topicOptionsResource.data?.items || [];

  const imported = detail?.source === "imported";
  const initial = useMemo(() => (detail ? toDraft(detail) : undefined), [detail]);
  const planChanged = Boolean(draft && initial && draft.topicId !== initial.topicId);
  const generateChanged = Boolean(
    draft &&
      initial &&
      (draft.question !== initial.question ||
        draft.answer !== initial.answer ||
        draft.guidance !== initial.guidance),
  );
  const gradingChanged = Boolean(
    draft &&
      initial &&
      (JSON.stringify(draft.keyPoints) !== JSON.stringify(initial.keyPoints) ||
        JSON.stringify(draft.forbiddenClaims) !== JSON.stringify(initial.forbiddenClaims)),
  );
  const dirty = planChanged || generateChanged || gradingChanged;

  const selectedTopic = topicOptions.find((option) => option.topic_id === draft?.topicId);
  const currentTopicName =
    selectedTopic?.name ||
    (draft?.topicId === detail?.topic?.topic_id ? detail?.topic?.name : undefined) ||
    "—";

  const patch = (partial: Partial<Draft>) =>
    setDraft((prev) => (prev ? { ...prev, ...partial } : prev));

  const effectiveStageStatus = (key: CaseStageKey) =>
    detail && progress?.partitions[key]?.[detail.case_id]
      ? progress.partitions[key][detail.case_id]
      : detail?.stages[key].status;

  const retryAction = detail
    ? getCaseRetryAction(stage, effectiveStageStatus(stage) || detail.stages[stage].status, detail.source)
    : undefined;

  const submit = async () => {
    if (!detail || !draft || !dirty) return;
    if (planChanged && gradingChanged) {
      message.warning("更换主题后判分规则会基于新的引用重新生成，请先撤销判分规则的修改。");
      return;
    }
    const forbidden = draft.forbiddenClaims.map((line) => line.trim()).filter(Boolean);
    if (gradingChanged) {
      if (draft.keyPoints.some((point) => !point.statement.trim())) {
        message.warning("关键得分点内容不能为空。");
        return;
      }
      if (draft.keyPoints.some((point) => !point.evidence_chunk_ids.length)) {
        message.warning("每个关键得分点至少绑定一个依据片段。");
        return;
      }
      if (forbidden.length > 3) {
        message.warning("错误结论最多 3 条。");
        return;
      }
    }

    const affected = [
      planChanged ? "生成规划" : undefined,
      generateChanged || planChanged ? "问答生成" : undefined,
      "判分规则",
    ].filter(Boolean);

    Modal.confirm({
      title: "保存当前用例修改",
      content: (
        <div className="dataset-impact-copy">
          <p>
            本次修改：
            {[
              planChanged ? "更换主题" : undefined,
              generateChanged ? "问答内容" : undefined,
              gradingChanged ? "判分规则" : undefined,
            ]
              .filter(Boolean)
              .join("、")}
          </p>
          <p>可能受影响的阶段：{affected.join(" → ")}</p>
          <p>保存后 Evo 只会重新执行当前用例受影响的下游阶段，不影响其他用例。</p>
        </div>
      ),
      okText: "确认保存",
      cancelText: "返回继续编辑",
      onOk: async () => {
        setSaving(true);
        try {
          const changes: Record<string, unknown> = {};
          if (planChanged && draft.topicId) changes.plan = { topic_id: draft.topicId };
          if (generateChanged) {
            changes.generate = {
              question: draft.question,
              answer: draft.answer,
              grading_guidance: draft.guidance,
            };
          }
          if (gradingChanged) {
            changes.grading = {
              key_points: draft.keyPoints,
              forbidden_claims: forbidden,
            };
          }
          await patchJson(`${root}/cases/${encodeURIComponent(detail.case_id)}`, {
            request_id: newRequestId(),
            expected_revision: detail.revision,
            changes,
          });
          message.success("用例修改已保存");
          onSaved();
          detailResource.reload();
        } catch (caught) {
          message.error(describeRequestError(caught, "保存用例失败"));
          throw caught;
        } finally {
          setSaving(false);
        }
      },
    });
  };

  const retryGeneration = async () => {
    if (!detail || !retryAction || retrying) return;
    setRetrying(true);
    try {
      const request = caseRetryRequest(threadRoot(threadId), detail.case_id, retryAction);
      await postJson(request.path, {
        command_id: newRequestId(),
        ...request.body,
      });
      message.success("已提交当前用例的重试生成");
      onSaved();
      detailResource.reload();
    } catch (caught) {
      message.error(describeRequestError(caught, "重试生成失败"));
    } finally {
      setRetrying(false);
    }
  };

  const retryControl = retryAction ? (
    <div className="dataset-case-retry-control">
      <span>{retryAction.description}</span>
      <Button
        size="small"
        icon={<ReloadOutlined />}
        loading={retrying}
        onClick={() => void retryGeneration()}
      >
        {retryAction.label}
      </Button>
    </div>
  ) : null;

  return (
    <Drawer
      className="dataset-drawer"
      rootClassName="dataset-drawer-root"
      title={row ? `${row.case_id} · 用例详情` : "用例详情"}
      open={Boolean(row)}
      width={560}
      onClose={onClose}
      destroyOnClose
      footer={
        <div className="dataset-drawer-foot">
          <Button onClick={onClose}>取消</Button>
          <Button type="primary" disabled={!dirty} loading={saving} onClick={() => void submit()}>
            保存
          </Button>
        </div>
      }
    >
      {detailResource.error ? (
        <Alert
          type="error"
          showIcon
          message="用例详情不可用"
          description={detailResource.error}
          action={
            <Button size="small" onClick={detailResource.reload}>
              重试
            </Button>
          }
        />
      ) : !detail || !draft ? (
        <div className="dataset-skeleton-lines">
          <span />
          <span />
          <span />
          <span />
        </div>
      ) : (
        <>
          <DrawerAttributes
            items={[
              { label: "Case ID", value: detail.case_id },
              {
                label: "来源",
                value: (
                  <Chip tone={imported ? "imported" : "neutral"}>
                    {imported ? "外部导入" : "自动生成"}
                  </Chip>
                ),
              },
              {
                label: "题型与难度",
                value: `${QUESTION_TYPE_TEXT[detail.question_type]} · ${
                  detail.difficulty ? DIFFICULTY_TEXT[detail.difficulty] : "—"
                }`,
              },
              { label: "主题", value: currentTopicName },
            ]}
          />

          {dirty ? (
            <div className="dataset-case-unsaved">
              当前 Case 有修改尚未保存；可继续切换子阶段编辑。
            </div>
          ) : null}

          <nav className="dataset-case-roadmap" aria-label="用例阶段">
            {(Object.keys(STAGE_LABEL) as CaseStageKey[]).map((key, index) => {
              const stageStatus = effectiveStageStatus(key) || detail.stages[key].status;
              return (
                <button
                  type="button"
                  key={key}
                  className={`dataset-roadmap-step${stage === key ? " is-selected" : ""}`}
                  onClick={() => setStage(key)}
                >
                  <span
                    className={`dataset-roadmap-mark is-${toVisualStatus(stageStatus)}`}
                    aria-hidden="true"
                  >
                    {toVisualStatus(stageStatus) === "done" ? "✓" : index + 1}
                  </span>
                  <strong>{STAGE_LABEL[key]}</strong>
                </button>
              );
            })}
          </nav>

          {stage === "plan" && (
            <section>
              <div className="dataset-case-panel-summary">
                {imported
                  ? "该用例随外部导入，不绑定主题。"
                  : "只展示符合当前题型、难度下限及占用规则的可选主题。"}
              </div>
              {imported ? (
                <div className="dataset-detail-block">
                  <h4>主题</h4>
                  <p>—</p>
                </div>
              ) : (
                <>
                  <div className="dataset-detail-block">
                    <h4>当前绑定主题</h4>
                    <div className="dataset-current-topic">
                      <div>
                        <strong>{currentTopicName}</strong>
                        <small>
                          {draft.topicId || "—"} · {selectedTopic?.chunk_count ?? detail.topic?.chunk_count ?? 0} 个片段
                        </small>
                      </div>
                      <span>从下方更换</span>
                    </div>
                  </div>
                  <div className="dataset-detail-block">
                    <h4>可选主题</h4>
                    <div className="dataset-topic-option-list">
                      {topicOptions
                        .filter((option) => option.topic_id !== detail.topic?.topic_id)
                        .map((option) => (
                        <button
                          type="button"
                          key={option.topic_id}
                          className={`dataset-topic-option${
                            draft.topicId === option.topic_id ? " is-selected" : ""
                          }`}
                          onClick={() => patch({ topicId: option.topic_id })}
                        >
                          <strong>{option.name}</strong>
                          <div className="dataset-topic-option-meta">
                            <span>{option.topic_id}</span>
                            <span>{option.chunk_count} 个片段</span>
                          </div>
                        </button>
                      ))}
                      {topicOptionsResource.loading ? (
                        <p className="dataset-note">正在读取可替换主题…</p>
                      ) : topicOptionsResource.error ? (
                        <p className="dataset-note">{topicOptionsResource.error}</p>
                      ) : !topicOptions.length ? (
                        <p className="dataset-note">当前没有可替换的主题。</p>
                      ) : null}
                    </div>
                  </div>
                  <div className="dataset-warning-note">
                    参考片段由算法随主题自动调整，不在此处单独展示或修改。
                  </div>
                </>
              )}
            </section>
          )}

          {stage === "generate" && (
            <section>
              <div className="dataset-case-panel-summary">
                {imported
                  ? "问答内容随外部导入，在本阶段只读。"
                  : "问题、标准答案与评分说明共同属于当前 Case 修改。"}
              </div>
              {retryControl}
              <div className="dataset-inline-editable-stack">
                <InlineEditableField
                  label="问题"
                  value={draft.question}
                  readOnly={imported}
                  onChange={(value) => patch({ question: value })}
                />
                <InlineEditableField
                  label="标准答案"
                  value={draft.answer}
                  readOnly={imported}
                  onChange={(value) => patch({ answer: value })}
                />
                <InlineEditableField
                  label="评分说明"
                  value={draft.guidance}
                  readOnly={imported}
                  onChange={(value) => patch({ guidance: value })}
                />
              </div>
            </section>
          )}

          {stage === "grading" && (
            <section>
              {retryControl}
              <div className="dataset-section-heading">
                <h4>关键得分点</h4>
                <button
                  type="button"
                  className="dataset-text-action"
                  onClick={() =>
                    patch({ keyPoints: [...draft.keyPoints, { statement: "", evidence_chunk_ids: [] }] })
                  }
                >
                  + 添加得分点
                </button>
              </div>
              {draft.keyPoints.map((point, index) => (
                <div className="dataset-score-point" key={`key-point-${index}`}>
                  <div className="dataset-score-point-head">
                    <span className="dataset-score-number">{String(index + 1).padStart(2, "0")}</span>
                    <button
                      type="button"
                      className="dataset-icon-action"
                      aria-label={`删除关键得分点 ${index + 1}`}
                      onClick={() =>
                        patch({ keyPoints: draft.keyPoints.filter((_, at) => at !== index) })
                      }
                    >
                      <DeleteOutlined />
                    </button>
                  </div>
                  <InlineEditableField
                    label=""
                    value={point.statement}
                    onChange={(value) =>
                      patch({
                        keyPoints: draft.keyPoints.map((item, at) =>
                          at === index ? { ...item, statement: value } : item,
                        ),
                      })
                    }
                  />
                  <button
                    type="button"
                    className="dataset-evidence-selector"
                    onClick={() => setEvidenceEditorIndex(index)}
                  >
                    <span>依据片段</span>
                    <strong>已绑定 {point.evidence_chunk_ids.length} 个 · 点击选择</strong>
                  </button>
                </div>
              ))}
              <div className="dataset-forbidden-list">
                <div className="dataset-section-heading">
                  <h4>错误结论（最多 3 条）</h4>
                  <button
                    type="button"
                    className="dataset-text-action"
                    disabled={draft.forbiddenClaims.length >= 3}
                    onClick={() => patch({ forbiddenClaims: [...draft.forbiddenClaims, ""] })}
                  >
                    + 添加错误结论
                  </button>
                </div>
                {draft.forbiddenClaims.map((claim, index) => (
                  <div className="dataset-forbidden-item" key={`forbidden-${index}`}>
                    <span className="dataset-forbidden-mark">×</span>
                    <InlineEditableField
                      label=""
                      value={claim}
                      placeholder="点击填写错误结论"
                      onChange={(value) =>
                        patch({
                          forbiddenClaims: draft.forbiddenClaims.map((item, at) =>
                            at === index ? value : item,
                          ),
                        })
                      }
                    />
                    <button type="button" className="dataset-icon-action" aria-label={`删除错误结论 ${index + 1}`}
                      onClick={() => patch({ forbiddenClaims: draft.forbiddenClaims.filter((_, at) => at !== index) })}>
                      <DeleteOutlined />
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}
          <EvidencePicker
            open={evidenceEditorIndex != null}
            point={evidenceEditorIndex == null ? undefined : draft.keyPoints[evidenceEditorIndex]}
            references={detail.references}
            onClose={() => setEvidenceEditorIndex(undefined)}
            onChange={(ids) => evidenceEditorIndex != null && patch({
              keyPoints: draft.keyPoints.map((item, at) => at === evidenceEditorIndex
                ? { ...item, evidence_chunk_ids: ids } : item),
            })}
          />
        </>
      )}
    </Drawer>
  );
}

function InlineEditableField({
  label,
  value,
  placeholder = "点击填写",
  readOnly = false,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  readOnly?: boolean;
  onChange: (value: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const displayValue = value || placeholder;

  return (
    <div className={`dataset-inline-editable${editing ? " is-editing" : ""}`}>
      {label ? <label>{label}</label> : null}
      {readOnly ? (
        <p>{value || "—"}</p>
      ) : editing ? (
        <Input.TextArea
          autoFocus
          rows={label ? 3 : 2}
          value={value}
          placeholder={placeholder}
          onChange={(event) => onChange(event.target.value)}
          onBlur={() => setEditing(false)}
        />
      ) : (
        <button type="button" onClick={() => setEditing(true)}>
          {displayValue}
        </button>
      )}
    </div>
  );
}

function EvidencePicker({
  open,
  point,
  references,
  onClose,
  onChange,
}: {
  open: boolean;
  point?: CaseKeyPoint;
  references: CaseDetail["references"];
  onClose: () => void;
  onChange: (ids: string[]) => void;
}) {
  return (
    <Modal open={open} title="选择依据片段" footer={null} onCancel={onClose} width={680}>
      <p className="dataset-evidence-picker-note">至少选择 1 个片段；同一片段可绑定多个关键得分点。</p>
      <div className="dataset-chunk-card-list">
        {references.map((reference) => {
          const checked = Boolean(point?.evidence_chunk_ids.includes(reference.chunk_id));
          return (
            <article className="dataset-chunk-card" key={reference.chunk_id}>
              <div className="dataset-chunk-card-head">
                <strong title={reference.document.name}>{reference.document.name}</strong>
                <Checkbox checked={checked} onChange={(event) => onChange(
                  event.target.checked
                    ? [...(point?.evidence_chunk_ids || []), reference.chunk_id]
                    : (point?.evidence_chunk_ids || []).filter((id) => id !== reference.chunk_id),
                )}>绑定</Checkbox>
              </div>
              <div className="dataset-chunk-card-meta">
                <Chip>{reference.knowledge_base.name}</Chip>
                <span className="dataset-chunk-id">{reference.chunk_id}</span>
              </div>
              <ChunkText text={reference.text} />
            </article>
          );
        })}
        {!references.length ? <p className="dataset-note">当前用例还没有参考片段。</p> : null}
      </div>
    </Modal>
  );
}
