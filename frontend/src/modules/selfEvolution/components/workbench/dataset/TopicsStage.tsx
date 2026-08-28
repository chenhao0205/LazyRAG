import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, Button, Drawer, Input } from "antd";
import { datasetRoot, getJson } from "./api";
import { chunkTags } from "./capabilities";
import {
  CHUNK_PAGE_SIZE,
  PAGE_SIZE,
  useDatasetList,
  useDatasetPagedDetail,
  useDatasetResource,
} from "./hooks";
import {
  Chip,
  ChunkCard,
  ColumnFilter,
  DrawerAttributes,
  ListPlaceholder,
  OverviewPane,
  QUESTION_TYPE_TEXT,
  ScrollSentinel,
  StageProgressTrack,
  StatusIcon,
  percentText,
  ratio,
} from "./primitives";
import { usePublishDatasetStageAction } from "./stageAction";
import { topicDiscoverySteps, type TopicDiscoveryProgress } from "./topicLabelProgress";
import type {
  AdjustmentOptions,
  DatasetDraft,
  PagedResponse,
  QuestionType,
  TopicDetail,
  TopicRow,
  TopicsOverview,
} from "./types";

type Props = {
  threadId: string;
  refreshToken: number;
  overviewToken: number;
  labelProgress?: TopicDiscoveryProgress;
  onOverviewRevision: (tab: "topics", revision: string | null) => void;
  draft?: DatasetDraft;
  onSaveDraft: (draft: DatasetDraft) => boolean;
};

type ChunkBucket = "one" | "two" | "threePlus";

const CHUNK_BUCKETS: Record<ChunkBucket, { label: string; min?: number; max?: number }> = {
  one: { label: "1", min: 1, max: 1 },
  two: { label: "2", min: 2, max: 2 },
  threePlus: { label: "3+", min: 3 },
};

export function TopicsStage({
  threadId,
  refreshToken,
  overviewToken,
  labelProgress,
  onOverviewRevision,
  draft,
  onSaveDraft,
}: Props) {
  const [questionType, setQuestionType] = useState<QuestionType>();
  const [chunkBucket, setChunkBucket] = useState<ChunkBucket>();
  const [openTopic, setOpenTopic] = useState<TopicRow>();
  const listRef = useRef<HTMLDivElement>(null);

  const root = datasetRoot(threadId);
  const nameDrafts = draft?.kind === "topic-names" ? draft.names : undefined;

  // The topic stage has no page-level action; names are edited from a row.
  usePublishDatasetStageAction(undefined);

  const fetchOverview = useCallback(
    () => getJson<TopicsOverview>(`${root}/topics/overview`),
    [root],
  );
  const overview = useDatasetResource(
    fetchOverview,
    overviewToken,
    "主题概览加载失败",
  );

  useEffect(() => {
    if (overview.data) onOverviewRevision("topics", overview.data.revision);
  }, [onOverviewRevision, overview.data]);

  const fetchTopics = useCallback(
    (pageToken?: string) =>
      getJson<PagedResponse<TopicRow>>(`${root}/topics`, {
        page_size: PAGE_SIZE,
        page_token: pageToken,
        question_type: questionType,
        min_chunk_count: chunkBucket ? CHUNK_BUCKETS[chunkBucket].min : undefined,
        max_chunk_count: chunkBucket ? CHUNK_BUCKETS[chunkBucket].max : undefined,
      }),
    [chunkBucket, questionType, root],
  );
  const topics = useDatasetList(fetchTopics, refreshToken, "主题列表加载失败");

  const distribution = overview.data?.question_types;
  const hasFilters = Boolean(questionType || chunkBucket);
  const discoverySteps = useMemo(
    () => topicDiscoverySteps(labelProgress, overview.data),
    [labelProgress, overview.data],
  );
  const clearFilters = () => {
    setQuestionType(undefined);
    setChunkBucket(undefined);
  };

  return (
    <>
      <div className="dataset-overview-row">
        <OverviewPane
          title="发现进度"
          extra={
            overview.data?.total_topics != null ? `共 ${overview.data.total_topics} 个主题` : undefined
          }
        >
          {overview.error ? (
            <p className="dataset-pane-error">{overview.error}</p>
          ) : (
            <StageProgressTrack steps={discoverySteps} />
          )}
        </OverviewPane>
        <OverviewPane
          title="适用题型分布"
          extra={overview.data?.total_topics != null ? `${overview.data.total_topics} 个` : undefined}
        >
          {distribution ? (
            <div className="dataset-overview-ratio">
              <div
                className="dataset-stacked-bar"
                role="img"
                aria-label={`准确型 ${distribution.precision.count ?? 0} 个，推理型 ${
                  distribution.reasoning.count ?? 0
                } 个`}
              >
                <span
                  className="is-precision"
                  style={{
                    width: `${ratio(distribution.precision.count, overview.data?.total_topics)}%`,
                  }}
                />
                <span
                  className="is-reasoning"
                  style={{
                    width: `${ratio(distribution.reasoning.count, overview.data?.total_topics)}%`,
                  }}
                />
              </div>
              <div className="dataset-ratio-labels">
                <span>
                  <strong>准确型 {distribution.precision.count ?? 0}</strong> ·{" "}
                  {percentText(distribution.precision.rate)}
                </span>
                <span>
                  <strong>推理型 {distribution.reasoning.count ?? 0}</strong> ·{" "}
                  {percentText(distribution.reasoning.rate)}
                </span>
              </div>
            </div>
          ) : (
            <p className="dataset-pane-error">{overview.error || "尚无主题结果"}</p>
          )}
        </OverviewPane>
      </div>

      <section className="dataset-list-card" aria-label="主题列表">
        <div className="dataset-table-wrap" ref={listRef}>
          <table className="dataset-object-table dataset-topic-table">
            <thead>
              <tr>
                <th>主题编号</th>
                <th>状态</th>
                <th>主题</th>
                <th>
                  <ColumnFilter<QuestionType>
                    label="适用题型"
                    value={questionType}
                    onChange={setQuestionType}
                    options={[
                      { value: "precision", label: "准确型" },
                      { value: "reasoning", label: "推理型" },
                    ]}
                  />
                </th>
                <th>
                  <ColumnFilter<ChunkBucket>
                    label="支撑片段数"
                    value={chunkBucket}
                    onChange={setChunkBucket}
                    options={(Object.keys(CHUNK_BUCKETS) as ChunkBucket[]).map((value) => ({
                      value,
                      label: CHUNK_BUCKETS[value].label,
                    }))}
                  />
                </th>
              </tr>
            </thead>
            <tbody>
              {topics.items.map((row) => {
                const pendingName = nameDrafts?.[row.topic_id];
                const displayName = pendingName || row.name;
                return (
                  <tr
                    key={row.topic_id}
                    tabIndex={0}
                    role="button"
                    onClick={() => setOpenTopic(row)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        setOpenTopic(row);
                      }
                    }}
                  >
                    <td>
                      <strong className="dataset-identity">{row.topic_id}</strong>
                    </td>
                    <td>
                      <StatusIcon status={pendingName ? "stale" : "done"} />
                    </td>
                    <td>
                      <div className="dataset-ellipsis" title={displayName}>
                        {displayName}
                        {pendingName ? <Chip tone="pending">待应用</Chip> : null}
                      </div>
                    </td>
                    <td>
                      <Chip tone={row.question_type === "reasoning" ? "reasoning" : "neutral"}>
                        {QUESTION_TYPE_TEXT[row.question_type]}
                      </Chip>
                    </td>
                    <td>{row.chunk_count}</td>
                  </tr>
                );
              })}
              {!topics.items.length && (
                <ListPlaceholder
                  colSpan={5}
                  loading={topics.loading}
                  error={topics.error}
                  filtered={hasFilters}
                  emptyText="尚未生成主题"
                  onRetry={topics.reload}
                  onClearFilters={clearFilters}
                />
              )}
            </tbody>
          </table>
          {topics.error && topics.items.length ? (
            <p className="dataset-pane-error">{topics.error}</p>
          ) : null}
          <ScrollSentinel
            rootRef={listRef}
            hasMore={!!topics.nextPageToken}
            loading={topics.loading}
            onLoadMore={() => void topics.loadMore()}
          />
        </div>
      </section>

      <TopicDrawer
        threadId={threadId}
        row={openTopic}
        listRevision={topics.revision}
        nameDrafts={nameDrafts}
        draftRevision={draft?.kind === "topic-names" ? draft.revision : undefined}
        onClose={() => setOpenTopic(undefined)}
        onSaveDraft={onSaveDraft}
      />
    </>
  );
}

function TopicDrawer({
  threadId,
  row,
  listRevision,
  nameDrafts,
  draftRevision,
  onClose,
  onSaveDraft,
}: {
  threadId: string;
  row?: TopicRow;
  listRevision: string | null;
  nameDrafts?: Record<string, string>;
  draftRevision?: string;
  onClose: () => void;
  onSaveDraft: (draft: DatasetDraft) => boolean;
}) {
  const [name, setName] = useState("");
  const [editingName, setEditingName] = useState(false);

  // Chunk tags name their split rule and layout type from the same capability
  // catalog the material stage uses, so a type never reads differently here.
  const root = datasetRoot(threadId);
  const fetchCapabilities = useCallback(
    () => getJson<AdjustmentOptions>(`${root}/materials/adjustment-options`),
    [root],
  );
  const capabilities = useDatasetResource(row ? fetchCapabilities : undefined);

  const topicId = row?.topic_id;
  const detailUrl = topicId ? `${root}/topics/${encodeURIComponent(topicId)}` : undefined;

  const fetchDetailPage = useCallback(
    (pageToken?: string) => getJson<TopicDetail>(detailUrl || "", {
      page_size: CHUNK_PAGE_SIZE,
      page_token: pageToken,
    }),
    [detailUrl],
  );
  const detailPage = useDatasetPagedDetail(
    detailUrl ? fetchDetailPage : undefined,
    (value) => value.chunks?.items || [],
    (value) => value.chunks?.next_page_token || "",
    "读取主题详情失败",
  );
  const { data: detail, items: chunks } = detailPage;

  useEffect(() => {
    if (!row) return;
    setName(nameDrafts?.[row.topic_id] || row.name);
    setEditingName(false);
  }, [nameDrafts, row]);

  // Renames apply against the topic collection version, which the list owns.
  const revision = draftRevision || listRevision;
  const trimmed = name.trim();
  const changed = Boolean(row && trimmed && trimmed !== row.name);

  const save = () => {
    if (!row || !revision || !changed) return;
    const accepted = onSaveDraft({
      kind: "topic-names",
      revision,
      names: { [row.topic_id]: trimmed },
    });
    if (accepted) onClose();
  };

  return (
    <Drawer
      className="dataset-drawer"
      rootClassName="dataset-drawer-root"
      title={row ? nameDrafts?.[row.topic_id] || row.name : "主题详情"}
      open={Boolean(row)}
      width={520}
      onClose={onClose}
      destroyOnClose
      footer={
        <div className="dataset-drawer-foot">
          <Button onClick={onClose}>取消</Button>
          <Button type="primary" disabled={!changed || !revision} onClick={save}>
            保存为待应用修改
          </Button>
        </div>
      }
    >
      {detailPage.error ? (
        <Alert
          type="error"
          showIcon
          message="主题详情不可用"
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
        </div>
      ) : (
        <>
          <div className="dataset-topic-name-editor">
            <label>主题名称</label>
            {editingName ? (
              <Input
                autoFocus
                id="dataset-topic-name"
                value={name}
                maxLength={120}
                onBlur={() => setEditingName(false)}
                onChange={(event) => setName(event.target.value)}
              />
            ) : (
              <button type="button" className="dataset-inline-title" onClick={() => setEditingName(true)}>
                {name || "未命名主题"}
              </button>
            )}
          </div>
          <DrawerAttributes
            items={[
              { label: "主题编号", value: detail.topic.topic_id },
              { label: "适用题型", value: QUESTION_TYPE_TEXT[detail.topic.question_type] },
              { label: "支撑片段", value: detail.topic.chunk_count },
            ]}
          />
          <div className="dataset-detail-block">
            <h4>片段信息</h4>
            <div className="dataset-chunk-card-list">
              {chunks.map((chunk) => (
                <ChunkCard
                  key={chunk.chunk_id}
                  documentName={chunk.document.name}
                  tags={chunkTags(capabilities.data, chunk)}
                  chunkId={chunk.chunk_id}
                  text={chunk.text}
                />
              ))}
              {!chunks.length && !detailPage.loading && <p className="dataset-note">该主题没有支撑片段。</p>}
            </div>
            <ScrollSentinel
              hasMore={!!detailPage.nextPageToken}
              loading={detailPage.loading}
              onLoadMore={detailPage.loadMore}
            />
          </div>
        </>
      )}
    </Drawer>
  );
}
