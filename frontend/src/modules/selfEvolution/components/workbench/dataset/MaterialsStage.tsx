import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, Button, Checkbox, Drawer, Select } from "antd";
import { datasetRoot, getJson } from "./api";
import {
  capabilityFilterOptions,
  capabilityNote,
  chunkTags,
  describeCapability,
} from "./capabilities";
import {
  CHUNK_PAGE_SIZE,
  PAGE_SIZE,
  useDatasetList,
  useDatasetPagedDetail,
  useDatasetResource,
} from "./hooks";
import { MaterialAdjustmentDrawer } from "./MaterialAdjustmentDrawer";
import {
  Chip,
  ChunkCard,
  ColumnFilter,
  DrawerAttributes,
  ListPlaceholder,
  OverviewMetrics,
  OverviewPane,
  ScrollSentinel,
  percentText,
  ratio,
} from "./primitives";
import { usePublishDatasetStageAction } from "./stageAction";
import type {
  AdjustmentOptions,
  ChunkSelectionChange,
  DatasetDraft,
  DocumentChunk,
  DocumentDetail,
  DocumentRow,
  MaterialsOverview,
  PagedResponse,
} from "./types";

type Props = {
  threadId: string;
  refreshToken: number;
  overviewToken: number;
  onOverviewRevision: (tab: "materials", revision: string | null) => void;
  draft?: DatasetDraft;
  onSaveDraft: (draft: DatasetDraft) => boolean;
};

type IncludedFilter = "included" | "excluded";

const documentKey = (row: Pick<DocumentRow, "document_id" | "knowledge_base">) =>
  `${row.knowledge_base.id}/${row.document_id}`;

export function MaterialsStage({
  threadId,
  refreshToken,
  overviewToken,
  onOverviewRevision,
  draft,
  onSaveDraft,
}: Props) {
  const [includedFilter, setIncludedFilter] = useState<IncludedFilter>();
  const [knowledgeBaseId, setKnowledgeBaseId] = useState<string>();
  const [openDocument, setOpenDocument] = useState<DocumentRow>();
  const [adjustmentOpen, setAdjustmentOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const root = datasetRoot(threadId);

  const fetchOverview = useCallback(
    () => getJson<MaterialsOverview>(`${root}/materials/overview`),
    [root],
  );
  const overview = useDatasetResource(
    fetchOverview,
    overviewToken,
    "材料概览加载失败",
  );

  // The capability catalog is the single source of truth for split rule and
  // layout type names plus their support and participation state. It is loaded
  // once here and shared with the document detail and the adjustment drawer, so
  // the two never disagree about a type.
  const fetchCapabilities = useCallback(
    () => getJson<AdjustmentOptions>(`${root}/materials/adjustment-options`),
    [root],
  );
  const capabilities = useDatasetResource(
    fetchCapabilities,
    refreshToken,
    "材料调整选项加载失败",
  );

  useEffect(() => {
    if (overview.data) onOverviewRevision("materials", overview.data.revision);
  }, [onOverviewRevision, overview.data]);

  const fetchDocuments = useCallback(
    (pageToken?: string) =>
      getJson<PagedResponse<DocumentRow>>(`${root}/materials/documents`, {
        page_size: PAGE_SIZE,
        page_token: pageToken,
        included: includedFilter ? includedFilter === "included" : undefined,
        knowledge_base_id: knowledgeBaseId,
      }),
    [includedFilter, knowledgeBaseId, root],
  );
  const documents = useDatasetList(fetchDocuments, refreshToken, "文档列表加载失败");

  usePublishDatasetStageAction(
    useMemo(
      () => ({ label: "调整材料", onClick: () => setAdjustmentOpen(true) }),
      [],
    ),
  );

  const chunks = overview.data?.chunks;
  const casePlan = overview.data?.case_plan;
  const pendingDocumentId = draft?.kind === "chunk-selection" ? draft.documentId : undefined;
  const hasFilters = Boolean(includedFilter || knowledgeBaseId);
  const clearFilters = () => {
    setIncludedFilter(undefined);
    setKnowledgeBaseId(undefined);
  };

  return (
    <>
      <div className="dataset-overview-row">
        <OverviewPane title="用例计划">
          {overview.error ? (
            <p className="dataset-pane-error">{overview.error}</p>
          ) : (
            <OverviewMetrics
              items={[
                { label: "目标", value: casePlan?.target, unit: "个" },
                { label: "外部导入", value: casePlan?.imported, unit: "个" },
                { label: "自动生成", value: casePlan?.automatic, unit: "个" },
              ]}
            />
          )}
        </OverviewPane>
        <OverviewPane
          title="片段准备情况"
          extra={documents.items.length ? `${documents.items.length} 份文档` : undefined}
        >
          {chunks ? (
            <>
              <div className="dataset-funnel-list">
                <FunnelRow label="扫描" tone="scanned" count={chunks.scanned} percent={100} />
                <FunnelRow
                  label="有效"
                  tone="effective"
                  count={chunks.effective}
                  percent={ratio(chunks.effective, chunks.scanned)}
                />
                <FunnelRow
                  label="入选"
                  tone="selected"
                  count={chunks.selected}
                  percent={ratio(chunks.selected, chunks.scanned)}
                />
              </div>
              <div className="dataset-conversion-note">
                有效率 {chunks.effective} / {chunks.scanned} · 有效片段入选率 {chunks.selected} /{" "}
                {chunks.effective}
              </div>
            </>
          ) : (
            <p className="dataset-pane-error">{overview.error || "尚无片段统计"}</p>
          )}
        </OverviewPane>
      </div>

      {overview.data?.warnings?.map((warning) => (
        <Alert key={warning} className="dataset-inline-alert" type="warning" showIcon message={warning} />
      ))}

      <section className="dataset-list-card" aria-label="文档列表">
        <div className="dataset-table-wrap" ref={listRef}>
          <table className="dataset-object-table dataset-document-table">
            <thead>
              <tr>
                <th>文档</th>
                <th>
                  <ColumnFilter<IncludedFilter>
                    label="导入状态"
                    value={includedFilter}
                    onChange={setIncludedFilter}
                    options={[
                      { value: "included", label: "已导入" },
                      { value: "excluded", label: "未导入" },
                    ]}
                  />
                </th>
                <th>
                  <ColumnFilter
                    label="知识库"
                    value={knowledgeBaseId}
                    onChange={setKnowledgeBaseId}
                    options={(capabilities.data?.knowledge_bases || []).map(({ id, name }) => ({
                      value: id,
                      label: name,
                    }))}
                  />
                </th>
                <th>片段概况</th>
              </tr>
            </thead>
            <tbody>
              {documents.items.map((row) => (
                <tr
                  key={documentKey(row)}
                  tabIndex={0}
                  role="button"
                  onClick={() => setOpenDocument(row)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      setOpenDocument(row);
                    }
                  }}
                >
                  <td>
                    <strong className="dataset-identity" title={row.name}>
                      {row.name}
                    </strong>
                    {pendingDocumentId === row.document_id ? (
                      <Chip tone="pending">待应用</Chip>
                    ) : null}
                  </td>
                  <td>
                    <span className={`dataset-status-text${row.included ? " is-ok" : " is-wait"}`}>
                      {row.included ? "已导入" : "未导入"}
                    </span>
                  </td>
                  <td>
                    <Chip>{row.knowledge_base.name}</Chip>
                  </td>
                  <td>
                    {row.chunks ? (
                      <div className="dataset-relation">
                        <div className="dataset-relation-track">
                          <span className="is-effective" style={{ width: "100%" }} />
                          <span
                            className="is-selected"
                            style={{ width: `${ratio(row.chunks.selected, row.chunks.effective)}%` }}
                          />
                        </div>
                        <div className="dataset-relation-values">
                          有效 {row.chunks.effective} · 入选 {row.chunks.selected}（
                          {percentText(row.chunks.selection_rate)}）
                        </div>
                      </div>
                    ) : (
                      "—"
                    )}
                  </td>
                </tr>
              ))}
              {!documents.items.length && (
                <ListPlaceholder
                  colSpan={4}
                  loading={documents.loading}
                  error={documents.error}
                  filtered={hasFilters}
                  emptyText="尚未发现可用文档"
                  onRetry={documents.reload}
                  onClearFilters={clearFilters}
                />
              )}
            </tbody>
          </table>
          {documents.error && documents.items.length ? (
            <p className="dataset-pane-error">{documents.error}</p>
          ) : null}
          <ScrollSentinel
            rootRef={listRef}
            hasMore={!!documents.nextPageToken}
            loading={documents.loading}
            onLoadMore={() => void documents.loadMore()}
          />
        </div>
      </section>

      <DocumentDrawer
        threadId={threadId}
        row={openDocument}
        capabilities={capabilities.data}
        draft={draft?.kind === "chunk-selection" ? draft : undefined}
        onClose={() => setOpenDocument(undefined)}
        onSaveDraft={onSaveDraft}
      />

      <MaterialAdjustmentDrawer
        threadId={threadId}
        open={adjustmentOpen}
        options={capabilities.data}
        optionsError={capabilities.error}
        onReloadOptions={capabilities.reload}
        onClose={() => setAdjustmentOpen(false)}
        onSaveDraft={onSaveDraft}
      />
    </>
  );
}

function FunnelRow({
  label,
  count,
  percent,
  tone,
}: {
  label: string;
  count: number;
  percent: number;
  tone: string;
}) {
  return (
    <div className="dataset-funnel-row">
      <span>{label}</span>
      <div className="dataset-funnel-track">
        <div className={`dataset-funnel-fill is-${tone}`} style={{ width: `${percent}%` }} />
      </div>
      <span className="dataset-funnel-value">
        {count} · {percent.toFixed(1)}%
      </span>
    </div>
  );
}

type ChunkDraft = Record<string, boolean>;

function DocumentDrawer({
  threadId,
  row,
  capabilities,
  draft,
  onClose,
  onSaveDraft,
}: {
  threadId: string;
  row?: DocumentRow;
  capabilities?: AdjustmentOptions;
  draft?: Extract<DatasetDraft, { kind: "chunk-selection" }>;
  onClose: () => void;
  onSaveDraft: (draft: DatasetDraft) => boolean;
}) {
  const [changes, setChanges] = useState<ChunkDraft>({});
  const [selectionFilter, setSelectionFilter] = useState<string>();
  const [ruleFilter, setRuleFilter] = useState<string>();

  const detailUrl = row
    ? `${datasetRoot(threadId)}/materials/knowledge-bases/${encodeURIComponent(
        row.knowledge_base.id,
      )}/documents/${encodeURIComponent(row.document_id)}`
    : undefined;

  const fetchDetailPage = useCallback(
    (pageToken?: string) => getJson<DocumentDetail>(detailUrl || "", {
      page_size: CHUNK_PAGE_SIZE,
      page_token: pageToken,
    }),
    [detailUrl],
  );
  const detailPage = useDatasetPagedDetail(
    detailUrl ? fetchDetailPage : undefined,
    (value) => value.chunks?.items || [],
    (value) => value.chunks?.next_page_token || "",
    "读取文档详情失败",
  );
  const { data: detail, items: chunks } = detailPage;

  useEffect(() => {
    setSelectionFilter(undefined);
    setRuleFilter(undefined);
  }, [detailUrl]);

  // Reopening a document restores its unapplied selection so the draft stays visible.
  useEffect(() => {
    if (!row) return;
    const restored: ChunkDraft = {};
    if (draft?.documentId === row.document_id) {
      for (const change of draft.changes) {
        restored[change.chunk_id] = change.selected;
      }
    }
    setChanges(restored);
  }, [draft, row]);

  const isSelected = (chunk: DocumentChunk) => changes[chunk.chunk_id] ?? chunk.selected;
  const deltaFor = (splitRule: string) =>
    chunks
      .filter((chunk) => chunk.split_rule === splitRule && changes[chunk.chunk_id] !== undefined)
      .reduce((sum, chunk) => sum + (isSelected(chunk) ? 1 : 0) - (chunk.selected ? 1 : 0), 0);
  const totalDelta = chunks
    .filter((chunk) => changes[chunk.chunk_id] !== undefined)
    .reduce((sum, chunk) => sum + (isSelected(chunk) ? 1 : 0) - (chunk.selected ? 1 : 0), 0);

  const quotas = (detail?.quotas || []).map((quota) => ({
    ...quota,
    current: quota.selected + deltaFor(quota.split_rule),
  }));
  const quotaValid = quotas.every((quota) => quota.current === quota.required);
  const dirtyChunkIds = Object.keys(changes).filter((chunkId) => {
    const chunk = chunks.find((item) => item.chunk_id === chunkId);
    return chunk ? changes[chunkId] !== chunk.selected : false;
  });

  const visibleChunks = chunks.filter(
    (chunk) =>
      (selectionFilter === undefined || String(isSelected(chunk)) === selectionFilter) &&
      (!ruleFilter || chunk.split_rule === ruleFilter),
  );

  const save = () => {
    if (!detail?.revision || !row || !dirtyChunkIds.length || !quotaValid) return;
    const payload: ChunkSelectionChange[] = dirtyChunkIds.map((chunkId) => ({
      knowledge_base_id: row.knowledge_base.id,
      document_id: row.document_id,
      chunk_id: chunkId,
      selected: changes[chunkId],
    }));
    const accepted = onSaveDraft({
      kind: "chunk-selection",
      revision: detail.revision,
      documentId: row.document_id,
      documentName: row.name,
      changes: payload,
    });
    if (accepted) onClose();
  };

  return (
    <Drawer
      className="dataset-drawer"
      rootClassName="dataset-drawer-root"
      title={row?.name || "文档详情"}
      open={Boolean(row)}
      width={520}
      onClose={onClose}
      destroyOnClose
      footer={
        <div className="dataset-drawer-foot">
          <Button onClick={onClose}>取消</Button>
          <Button
            type="primary"
            disabled={!dirtyChunkIds.length || !quotaValid}
            title={quotaValid ? undefined : "每种切分规则的当前入选数必须等于配额"}
            onClick={save}
          >
            保存为待应用修改
          </Button>
        </div>
      }
    >
      {detailPage.error ? (
        <Alert
          type="error"
          showIcon
          message="文档详情不可用"
          description={detailPage.error}
          action={
            <Button size="small" onClick={detailPage.reload}>
              重试
            </Button>
          }
        />
      ) : !detail ? (
        <div className="dataset-skeleton-lines">
          <span />
          <span />
          <span />
          <span />
        </div>
      ) : (
        <>
          <div className="dataset-detail-meta">
            <span className={`dataset-status-text${detail.document.included ? " is-ok" : " is-wait"}`}>
              {detail.document.included ? "已导入" : "未导入"}
            </span>
          </div>
          <DrawerAttributes
            items={[
              { label: "所属知识库", value: detail.document.knowledge_base.name },
              { label: "文档状态", value: detail.document.included ? "已导入" : "未导入" },
              { label: "有效片段", value: detail.chunk_summary?.effective ?? "—" },
              {
                label: "入选片段",
                value:
                  detail.chunk_summary == null
                    ? "—"
                    : detail.chunk_summary.selected + totalDelta,
              },
            ]}
          />

          <div className="dataset-quota-list">
            <strong>切分规则配额</strong>
            {quotas.length ? (
              quotas.map((quota) => {
                const rule = describeCapability(capabilities, "split_rules", quota.split_rule);
                const note = capabilityNote(rule);
                return (
                  <div
                    className={`dataset-quota-row${quota.current === quota.required ? "" : " is-invalid"}`}
                    key={quota.split_rule}
                  >
                    <span title={quota.split_rule}>
                      {rule.label}
                      {note ? <small className="dataset-note">{note}</small> : null}
                    </span>
                    <b>
                      {quota.current} / {quota.required}
                    </b>
                  </div>
                );
              })
            ) : (
              <span className="dataset-note">该文档当前未参与材料构建。</span>
            )}
          </div>

          {detail.chunk_summary ? (
            <div className="dataset-detail-block">
              <h4>有效片段</h4>
              <div className="dataset-chunk-filter-row">
                <label>
                  入选状态
                  <Select
                    size="small"
                    value={selectionFilter}
                    allowClear
                    placeholder="全部"
                    onChange={setSelectionFilter}
                    options={[
                      { value: "true", label: "已入选" },
                      { value: "false", label: "未入选" },
                    ]}
                  />
                </label>
                <label>
                  切分规则
                  <Select
                    size="small"
                    value={ruleFilter}
                    allowClear
                    placeholder="全部"
                    onChange={setRuleFilter}
                    options={capabilityFilterOptions(
                      capabilities,
                      "split_rules",
                      chunks.map((chunk) => chunk.split_rule),
                    )}
                  />
                </label>
              </div>
              {!quotaValid && (
                <Alert
                  className="dataset-inline-alert"
                  type="warning"
                  showIcon
                  message="每种切分规则的入选数量必须等于冻结配额后才能保存。"
                />
              )}
              <div className="dataset-chunk-card-list">
                {visibleChunks.map((chunk) => (
                  <ChunkCard
                    key={chunk.chunk_id}
                    tags={chunkTags(capabilities, chunk)}
                    chunkId={chunk.chunk_id}
                    text={chunk.text}
                    pending={changes[chunk.chunk_id] !== undefined && changes[chunk.chunk_id] !== chunk.selected}
                    action={
                      <Checkbox
                        checked={isSelected(chunk)}
                        onChange={(event) =>
                          setChanges((prev) => ({ ...prev, [chunk.chunk_id]: event.target.checked }))
                        }
                      >
                        {isSelected(chunk) ? "已入选" : "未入选"}
                      </Checkbox>
                    }
                  />
                ))}
                {!visibleChunks.length && !detailPage.loading && (
                  <p className="dataset-note">没有符合当前筛选条件的片段。</p>
                )}
              </div>
              <ScrollSentinel
                hasMore={!!detailPage.nextPageToken}
                loading={detailPage.loading}
                onLoadMore={detailPage.loadMore}
              />
              {!detailPage.nextPageToken && !detailPage.loading ? (
                <div className="dataset-note is-centered">已按原文顺序加载全部有效片段</div>
              ) : null}
            </div>
          ) : (
            <p className="dataset-note">该文档未参与本次数据集构建，没有可调整的片段。</p>
          )}
        </>
      )}
    </Drawer>
  );
}
