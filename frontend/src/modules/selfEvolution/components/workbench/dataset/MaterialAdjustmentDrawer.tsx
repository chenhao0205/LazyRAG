import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, Button, Checkbox, Drawer, InputNumber, Tooltip } from "antd";
import { datasetRoot, getJson } from "./api";
import { useDatasetList } from "./hooks";
import type {
  AdjustmentOptions,
  DatasetDraft,
  DocumentRow,
  MaterialsConfigChanges,
  PagedResponse,
} from "./types";

type Props = {
  threadId: string;
  open: boolean;
  /** Capability catalog owned by the stage, shared with the document detail. */
  options?: AdjustmentOptions;
  optionsError?: string;
  onReloadOptions: () => void;
  onClose: () => void;
  onSaveDraft: (draft: DatasetDraft) => boolean;
};

const documentKey = (row: Pick<DocumentRow, "document_id" | "knowledge_base">) =>
  `${row.knowledge_base.id}/${row.document_id}`;

/**
 * Adjust materials: target case count, per-knowledge-base document scope and
 * the chunk candidate configuration. Every control is initialised from
 * `materials/adjustment-options`, including whether the current material
 * sources actually support a split rule or layout type.
 */
export function MaterialAdjustmentDrawer({
  threadId,
  open,
  options,
  optionsError,
  onReloadOptions,
  onClose,
  onSaveDraft,
}: Props) {
  const [target, setTarget] = useState<number | null>(null);
  const [knowledgeBases, setKnowledgeBases] = useState<Record<string, boolean>>({});
  const [documentChanges, setDocumentChanges] = useState<Record<string, boolean>>({});
  const [ruleOrder, setRuleOrder] = useState<string[]>([]);
  const [enabledRules, setEnabledRules] = useState<Record<string, boolean>>({});
  const [enabledLayouts, setEnabledLayouts] = useState<Record<string, boolean>>({});

  const root = datasetRoot(threadId);
  const fetchDocuments = useCallback(
    (pageToken?: string) => getJson<PagedResponse<DocumentRow>>(`${root}/materials/documents`, {
      page_size: 200,
      page_token: pageToken,
    }),
    [root],
  );
  const documentPage = useDatasetList(
    open ? fetchDocuments : undefined,
    0,
    "读取文档列表失败",
  );
  const error = optionsError || documentPage.error;
  const reload = () => {
    onReloadOptions();
    documentPage.reload();
  };

  // Controls always reset to the catalog state when the drawer opens, so an
  // abandoned edit never leaks into the next session.
  useEffect(() => {
    if (!open || !options) return;
    setTarget(options.target_case_count ?? null);
    setDocumentChanges({});
    setKnowledgeBases(
      Object.fromEntries(options.knowledge_bases.map((item) => [item.id, item.included])),
    );
    setRuleOrder([...options.split_rules].sort(compareByPriority).map((item) => item.id));
    setEnabledRules(Object.fromEntries(options.split_rules.map((item) => [item.id, item.enabled])));
    setEnabledLayouts(Object.fromEntries(options.layout_types.map((item) => [item.id, item.enabled])));
  }, [open, options]);
  const rules = useMemo(() => {
    if (!options) return [];
    const byId = new Map(options.split_rules.map((item) => [item.id, item]));
    return ruleOrder.flatMap((id) => {
      const item = byId.get(id);
      return item ? [item] : [];
    });
  }, [options, ruleOrder]);

  const documentsByKnowledgeBase = useMemo(() => {
    const groups = new Map<string, { name: string; items: DocumentRow[] }>();
    for (const item of documentPage.items) {
      const group = groups.get(item.knowledge_base.id) || {
        name: item.knowledge_base.name,
        items: [],
      };
      group.items.push(item);
      groups.set(item.knowledge_base.id, group);
    }
    // Keep the knowledge base order declared by the source config.
    return (options?.knowledge_bases || []).map((kb) => ({
      id: kb.id,
      name: kb.name,
      documents: groups.get(kb.id)?.items || [],
    }));
  }, [documentPage.items, options?.knowledge_bases]);

  const selectedRuleIds = rules.filter((item) => enabledRules[item.id]).map((item) => item.id);
  const selectedLayoutIds = (options?.layout_types || [])
    .filter((item) => enabledLayouts[item.id])
    .map((item) => item.id);

  // Only changed entries are submitted; the service merges them into the full config.
  const changes = useMemo(() => {
    if (!options) return {};
    const payload: MaterialsConfigChanges = {};
    if (target != null && target !== options.target_case_count) {
      payload.target_case_count = target;
    }
    const changedKnowledgeBases = options.knowledge_bases
      .filter((item) => Boolean(knowledgeBases[item.id]) !== item.included)
      .map((item) => ({ id: item.id, included: Boolean(knowledgeBases[item.id]) }));
    if (changedKnowledgeBases.length) payload.knowledge_bases = changedKnowledgeBases;

    const changedDocuments = documentPage.items
      .filter((item) => documentChanges[documentKey(item)] !== undefined)
      .map((item) => ({
        knowledge_base_id: item.knowledge_base.id,
        document_id: item.document_id,
        included: documentChanges[documentKey(item)],
      }));
    if (changedDocuments.length) payload.documents = changedDocuments;

    const originalRuleIds = [...options.split_rules]
      .filter((item) => item.enabled)
      .sort(compareByPriority)
      .map((item) => item.id);
    if (originalRuleIds.join("|") !== selectedRuleIds.join("|")) {
      payload.split_rule_ids = selectedRuleIds;
    }

    const originalLayoutIds = options.layout_types
      .filter((item) => item.enabled)
      .map((item) => item.id);
    if ([...originalLayoutIds].sort().join("|") !== [...selectedLayoutIds].sort().join("|")) {
      payload.layout_type_ids = selectedLayoutIds;
    }
    return payload;
  }, [documentChanges, documentPage.items, knowledgeBases, options, selectedLayoutIds, selectedRuleIds, target]);

  const canSave =
    Boolean(options) &&
    target != null &&
    target >= options.min_target_case_count &&
    selectedRuleIds.length > 0 &&
    selectedLayoutIds.length > 0 &&
    Object.keys(changes).length > 0;

  const moveRule = (from: number, to: number) =>
    setRuleOrder((prev) => {
      if (to < 0 || to >= prev.length) return prev;
      const next = [...prev];
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      return next;
    });

  const save = () => {
    if (!options || !canSave) return;
    const accepted = onSaveDraft({
      kind: "materials-config",
      revision: options.revision,
      changes,
    });
    if (accepted) onClose();
  };

  return (
    <Drawer
      className="dataset-drawer"
      rootClassName="dataset-drawer-root"
      title="调整材料"
      open={open}
      width={520}
      onClose={onClose}
      destroyOnClose
      footer={
        <div className="dataset-drawer-foot">
          <Button onClick={onClose}>取消</Button>
          <Button type="primary" disabled={!canSave} onClick={save}>
            保存为待应用修改
          </Button>
        </div>
      }
    >
      {error ? (
        <Alert
          type="error"
          showIcon
          message="材料调整选项不可用"
          description={error}
          action={
            <Button size="small" onClick={reload}>
              重试
            </Button>
          }
        />
      ) : (documentPage.loading && !documentPage.items.length) || !options ? (
        <div className="dataset-skeleton-lines">
          <span />
          <span />
          <span />
          <span />
          <span />
        </div>
      ) : (
        <>
          <div className="dataset-form-note">
            调整材料只有三个业务可控点；切分类型和版面类型共同属于“片段候选配置”。
          </div>

          <section className="dataset-control-point">
            <div className="dataset-control-heading">
              <span>1</span>整体用例生成计划
            </div>
            <div className="dataset-form-group">
              <label htmlFor="dataset-target-cases">目标用例</label>
              <InputNumber
                id="dataset-target-cases"
                min={options.min_target_case_count}
                max={100000}
                value={target}
                onChange={setTarget}
              />
            </div>
            <div className="dataset-warning-note">
              {options.min_target_case_count > 1
                ? `已导入 ${options.min_target_case_count} 个用例；目标不能低于该数量，只生成其余用例。`
                : "修改目标用例将重新执行数据集全部步骤。"}
            </div>
          </section>

          <section className="dataset-control-point">
            <div className="dataset-control-heading">
              <span>2</span>片段来源分布
            </div>
            {documentsByKnowledgeBase.length ? (
              documentsByKnowledgeBase.map((group) => (
                <div className="dataset-source-group" key={group.id}>
                  <div className="dataset-source-group-title">
                    <Checkbox
                      checked={Boolean(knowledgeBases[group.id])}
                      onChange={(event) =>
                        setKnowledgeBases((prev) => ({ ...prev, [group.id]: event.target.checked }))
                      }
                    >
                      {group.name}
                    </Checkbox>
                  </div>
                  <div className="dataset-check-list">
                    {group.documents.length ? (
                      group.documents.map((item) => {
                        const key = documentKey(item);
                        return (
                          <label className="dataset-check is-nested" key={key}>
                            <Checkbox
                              checked={documentChanges[key] ?? item.included}
                              disabled={!knowledgeBases[group.id]}
                              onChange={(event) =>
                                setDocumentChanges((prev) => {
                                  const next = { ...prev };
                                  if (event.target.checked === item.included) delete next[key];
                                  else next[key] = event.target.checked;
                                  return next;
                                })
                              }
                            />
                            <span>{item.name}</span>
                          </label>
                        );
                      })
                    ) : (
                      <span className="dataset-note">当前已加载范围内暂无文档。</span>
                    )}
                  </div>
                </div>
              ))
            ) : (
              <span className="dataset-note">当前 Thread 未配置材料来源。</span>
            )}
            {documentPage.nextPageToken ? (
              <Button loading={documentPage.loading} onClick={documentPage.loadMore}>
                加载更多文档
              </Button>
            ) : null}
          </section>

          <section className="dataset-control-point">
            <div className="dataset-control-heading">
              <span>3</span>片段候选配置
            </div>
            <div className="dataset-note">
              下列类型来自当前已启用知识库的解析能力；勾选状态即当前参与片段候选的类型，与文档详情中的切分规则配额、片段版面类型一致。改动知识库范围后，支持情况会在应用后重新计算。
            </div>
            <div className="dataset-candidate-config">
              <div className="dataset-candidate-part">
                <strong>切分类型</strong>
                {rules.length ? (
                  <SortableRuleList
                    rules={rules}
                    enabled={enabledRules}
                    onToggle={(id, checked) =>
                      setEnabledRules((prev) => ({ ...prev, [id]: checked }))
                    }
                    onMove={moveRule}
                  />
                ) : (
                  <span className="dataset-note">当前材料来源未返回可用的切分类型。</span>
                )}
                <div className="dataset-warning-note">拖动或使用箭头改变候选优先级。</div>
              </div>
              <div className="dataset-candidate-part">
                <strong>版面类型</strong>
                {options.layout_types.length ? (
                  <div className="dataset-check-list">
                    {options.layout_types.map((item) => (
                      <label className="dataset-check" key={item.id}>
                        <Checkbox
                          checked={Boolean(enabledLayouts[item.id])}
                          disabled={!item.supported}
                          onChange={(event) =>
                            setEnabledLayouts((prev) => ({ ...prev, [item.id]: event.target.checked }))
                          }
                        />
                        <span>
                          {item.name}
                          {item.supported ? null : <small>当前材料来源不支持</small>}
                        </span>
                      </label>
                    ))}
                  </div>
                ) : (
                  <span className="dataset-note">当前材料来源未返回可用的版面类型。</span>
                )}
              </div>
            </div>
          </section>
        </>
      )}
    </Drawer>
  );
}

function SortableRuleList({
  rules,
  enabled,
  onToggle,
  onMove,
}: {
  rules: AdjustmentOptions["split_rules"];
  enabled: Record<string, boolean>;
  onToggle: (id: string, checked: boolean) => void;
  onMove: (from: number, to: number) => void;
}) {
  const draggedIndex = useRef<number>();
  let priority = 0;

  return (
    <div className="dataset-sortable-list">
      {rules.map((item, index) => {
        const isEnabled = Boolean(enabled[item.id]);
        if (isEnabled) priority += 1;
        return (
          <div
            className="dataset-sortable-item"
            key={item.id}
            draggable
            onDragStart={() => {
              draggedIndex.current = index;
            }}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              if (draggedIndex.current != null && draggedIndex.current !== index) {
                onMove(draggedIndex.current, index);
              }
              draggedIndex.current = undefined;
            }}
          >
            <span className="dataset-drag-handle" aria-hidden>
              ⠿
            </span>
            <Tooltip title={item.supported ? undefined : "当前材料来源不支持该切分类型"}>
              <Checkbox
                checked={isEnabled}
                disabled={!item.supported}
                aria-label={`启用 ${item.name}`}
                onChange={(event) => onToggle(item.id, event.target.checked)}
              />
            </Tooltip>
            <span className="dataset-sortable-label">
              {item.name}
              {item.supported ? null : <small>不支持</small>}
            </span>
            <span className="dataset-sortable-priority">{isEnabled ? priority : "—"}</span>
            <span className="dataset-sortable-move">
              <button
                type="button"
                className="dataset-icon-action"
                aria-label={`上移 ${item.name}`}
                disabled={index === 0}
                onClick={() => onMove(index, index - 1)}
              >
                ↑
              </button>
              <button
                type="button"
                className="dataset-icon-action"
                aria-label={`下移 ${item.name}`}
                disabled={index === rules.length - 1}
                onClick={() => onMove(index, index + 1)}
              >
                ↓
              </button>
            </span>
          </div>
        );
      })}
    </div>
  );
}

function compareByPriority(
  left: AdjustmentOptions["split_rules"][number],
  right: AdjustmentOptions["split_rules"][number],
) {
  if (left.enabled !== right.enabled) return left.enabled ? -1 : 1;
  if (left.enabled && right.enabled) {
    return (left.priority ?? Number.MAX_SAFE_INTEGER) - (right.priority ?? Number.MAX_SAFE_INTEGER);
  }
  return left.name.localeCompare(right.name);
}
