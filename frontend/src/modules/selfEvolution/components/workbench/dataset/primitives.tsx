import { Fragment, useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { Button, Spin } from "antd";
import type {
  Difficulty,
  FlowStatus,
  OperationStatus,
  QuestionType,
  VisualStatus,
} from "./types";

export const STATUS_TEXT: Record<VisualStatus, string> = {
  done: "已完成",
  running: "执行中",
  paused: "等待调整",
  stale: "待更新",
  pending: "未开始",
  failed: "失败",
  partial: "部分失败",
};

const STATUS_SYMBOL: Record<VisualStatus, string> = {
  done: "✓",
  running: "●",
  paused: "⏸",
  stale: "↻",
  pending: "○",
  failed: "!",
  partial: "!",
};

export const FLOW_STATUS_TEXT: Record<FlowStatus, string> = {
  pending: "等待中",
  running: "执行中",
  paused: "等待调整",
  completed: "已完成",
  awaiting_approval: "等待确认",
  failed: "执行失败",
};

export const QUESTION_TYPE_TEXT: Record<QuestionType, string> = {
  precision: "准确型",
  reasoning: "推理型",
};

export const DIFFICULTY_TEXT: Record<Difficulty, string> = {
  easy: "简单",
  medium: "中等",
  hard: "困难",
};

export function toVisualStatus(status: OperationStatus | FlowStatus | string): VisualStatus {
  switch (status) {
    case "completed":
      return "done";
    case "running":
    case "pausing":
      return "running";
    case "paused":
    case "awaiting_approval":
      return "paused";
    case "failed":
    case "cancelled":
    case "canceled":
    case "cancelling":
      return "failed";
    case "partial_failed":
    case "partial":
      return "partial";
    default:
      return "pending";
  }
}

export const percentText = (value: number | null | undefined) =>
  value == null ? "—" : `${(value * 100).toFixed(1)}%`;

export const ratio = (value: number | null | undefined, total: number | null | undefined) =>
  !total ? 0 : Math.max(0, Math.min(100, ((value || 0) / total) * 100));

export function StatusIcon({ status, note }: { status: VisualStatus; note?: string }) {
  const label = `${STATUS_TEXT[status]}${note ? `，${note}` : ""}`;
  return (
    <span className="dataset-status-wrap" title={label}>
      <span className={`dataset-status-icon is-${status}`} role="img" aria-label={label}>
        {STATUS_SYMBOL[status]}
      </span>
    </span>
  );
}

export function Chip({
  tone = "neutral",
  children,
}: {
  tone?: "neutral" | "imported" | "reasoning" | "medium" | "hard" | "pending";
  children: ReactNode;
}) {
  return <span className={`dataset-chip is-${tone}`}>{children}</span>;
}

export function OverviewPane({
  title,
  extra,
  children,
}: {
  title: string;
  extra?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="dataset-overview-pane" aria-label={title}>
      <div className="dataset-overview-title">
        <span>{title}</span>
        {extra ? <span>{extra}</span> : null}
      </div>
      {children}
    </section>
  );
}

export function OverviewMetrics({
  items,
}: {
  items: Array<{ label: string; value: number | null | undefined; unit?: string }>;
}) {
  return (
    <div className={`dataset-overview-metrics${items.length === 1 ? " is-single" : ""}`}>
      {items.map((item) => (
        <div className="dataset-overview-metric" key={item.label}>
          <small>{item.label}</small>
          <strong>
            {item.value ?? "—"}
            {item.unit ? <span>{item.unit}</span> : null}
          </strong>
        </div>
      ))}
    </div>
  );
}

/**
 * Column header with a lightweight filter entry, per the list filter spec:
 * no persistent filter bar, the entry lives next to the column name.
 */
export function ColumnFilter<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value?: T;
  options: Array<{ value: T; label: string }>;
  onChange: (value?: T) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return undefined;
    const close = (event: MouseEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("click", close);
    return () => document.removeEventListener("click", close);
  }, [open]);

  const select = (next?: T) => {
    setOpen(false);
    onChange(next);
  };

  return (
    <span className="dataset-column-heading" ref={rootRef}>
      <span>{label}</span>
      <button
        type="button"
        className={`dataset-column-filter${value ? " is-active" : ""}`}
        aria-label={`筛选${label}`}
        aria-expanded={open}
        onClick={(event) => {
          event.stopPropagation();
          setOpen((prev) => !prev);
        }}
      >
        ▾
      </button>
      {open && (
        <span className="dataset-column-filter-menu">
          <button
            type="button"
            className={value ? "" : "is-selected"}
            onClick={() => select(undefined)}
          >
            全部
          </button>
          {options.map((option) => (
            <button
              type="button"
              key={option.value}
              className={value === option.value ? "is-selected" : ""}
              onClick={() => select(option.value)}
            >
              {option.label}
            </button>
          ))}
        </span>
      )}
    </span>
  );
}

/** Placeholder row that keeps the table header and sizing while a list is empty. */
export function ListPlaceholder({
  colSpan,
  loading,
  error,
  filtered,
  emptyText,
  onRetry,
  onClearFilters,
}: {
  colSpan: number;
  loading: boolean;
  error?: string;
  filtered: boolean;
  emptyText: string;
  onRetry: () => void;
  onClearFilters?: () => void;
}) {
  return (
    <tr className="dataset-list-placeholder">
      <td colSpan={colSpan}>
        {loading ? (
          <span className="dataset-skeleton-lines">
            <span />
            <span />
            <span />
          </span>
        ) : error ? (
          <span className="dataset-placeholder-copy">
            {error}
            <Button size="small" onClick={onRetry}>
              重新加载
            </Button>
          </span>
        ) : filtered ? (
          <span className="dataset-placeholder-copy">
            没有符合当前筛选条件的内容
            {onClearFilters ? (
              <Button size="small" onClick={onClearFilters}>
                清空筛选
              </Button>
            ) : null}
          </span>
        ) : (
          <span className="dataset-placeholder-copy">{emptyText}</span>
        )}
      </td>
    </tr>
  );
}

/**
 * Chunk text is clamped by default; the whole preview is a button so a click
 * expands the full content in place.
 */
export function ChunkText({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <button
      type="button"
      className={`dataset-chunk-preview${expanded ? " is-expanded" : ""}`}
      aria-expanded={expanded}
      onClick={(event) => {
        event.stopPropagation();
        setExpanded((prev) => !prev);
      }}
    >
      {text}
    </button>
  );
}

export function ChunkCard({
  documentName,
  tags,
  chunkId,
  text,
  action,
  pending,
}: {
  documentName?: string;
  tags: string[];
  chunkId: string;
  text: string;
  action?: ReactNode;
  pending?: boolean;
}) {
  return (
    <article className={`dataset-chunk-card${pending ? " is-pending" : ""}`}>
      {documentName || action ? (
        <div className={`dataset-chunk-card-head${documentName ? "" : " is-action-only"}`}>
          {documentName ? <strong title={documentName}>{documentName}</strong> : null}
          {action}
        </div>
      ) : null}
      <div className="dataset-chunk-card-meta">
        {tags.filter(Boolean).map((tag) => (
          <Chip key={tag}>{tag}</Chip>
        ))}
        {pending ? <Chip tone="pending">待应用</Chip> : null}
        <span className="dataset-chunk-id">{chunkId}</span>
      </div>
      <ChunkText text={text} />
    </article>
  );
}

export function DrawerAttributes({
  items,
}: {
  items: Array<{ label: string; value: ReactNode }>;
}) {
  return (
    <div className="dataset-drawer-attributes">
      {items.map((item) => (
        <div className="dataset-drawer-attribute" key={item.label}>
          <small>{item.label}</small>
          <strong>{item.value}</strong>
        </div>
      ))}
    </div>
  );
}

export type StageProgressStep = {
  key: string;
  label: string;
  completed: number;
  total: number | null;
  status: VisualStatus;
  summary: string;
};

/** Keeps a dense live-state breakdown readable in the compact three-step card. */
export function progressSummaryLines(summary: string): string[] {
  const notes = summary.split(" · ").filter(Boolean);
  const running = notes.find((note) => note.includes("执行中"));
  const failed = notes.find((note) => note.includes("失败"));
  const liveNotes = [running, failed].filter((note): note is string => Boolean(note));
  return liveNotes.length ? liveNotes : notes.slice(0, 1);
}

export function StageProgressTrack({ steps }: { steps: StageProgressStep[] }) {
  return (
    <div className="dataset-stage-progress">
      {steps.map((step, index) => (
        <Fragment key={step.key}>
          {index ? <span className="dataset-stage-progress-link" aria-hidden /> : null}
          <div className="dataset-stage-progress-step">
            <div
              className={`dataset-progress-ring is-${step.status}`}
              style={{ "--dataset-progress": `${step.total ? Math.round((step.completed / step.total) * 360) : 0}deg` } as CSSProperties}
            >
              <strong>
                {step.total == null ? "—" : `${step.completed}/${step.total}`}
              </strong>
            </div>
            <b>{step.label}</b>
            <small className={`is-${step.status}`} title={step.summary}>
              {progressSummaryLines(step.summary).map((note) => (
                <span key={note}>{note}</span>
              ))}
            </small>
          </div>
        </Fragment>
      ))}
    </div>
  );
}

export function ScrollSentinel({
  hasMore,
  loading,
  onLoadMore,
  rootRef,
}: {
  hasMore: boolean;
  loading: boolean;
  onLoadMore: () => void;
  rootRef?: { current: HTMLElement | null };
}) {
  const ref = useRef<HTMLDivElement>(null);
  const loadMoreRef = useRef(onLoadMore);
  loadMoreRef.current = onLoadMore;

  const handleIntersect = useCallback((entries: IntersectionObserverEntry[]) => {
    if (entries[0]?.isIntersecting) {
      loadMoreRef.current();
    }
  }, []);

  useEffect(() => {
    const el = ref.current;
    if (!el || !hasMore || loading) return undefined;
    const observer = new IntersectionObserver(handleIntersect, {
      root: rootRef?.current ?? null,
      rootMargin: '200px',
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loading, handleIntersect, rootRef]);

  if (!hasMore) return null;
  return (
    <div ref={ref} className="dataset-load-more">
      {loading ? <Spin size="small" /> : null}
    </div>
  );
}
