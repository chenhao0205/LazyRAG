export type DatasetTab = "materials" | "topics" | "cases";

/** Flow-level status shared by the three dataset stages. */
export type FlowStatus =
  | "pending"
  | "running"
  | "paused"
  | "completed"
  | "awaiting_approval"
  | "failed";

/** Per-operation status used by the three case sub-stages. */
export type OperationStatus = "pending" | "running" | "completed" | "failed" | "canceled";

/** Visual status vocabulary shared by the stepper, status icons and rings. */
export type VisualStatus = "done" | "running" | "paused" | "stale" | "pending" | "failed" | "partial";

export type QuestionType = "precision" | "reasoning";
export type Difficulty = "easy" | "medium" | "hard";
export type CaseSource = "imported" | "generated";
export type CaseStageKey = "plan" | "generate" | "grading";

export type Ref = { id: string; name: string };

export type PagedResponse<T> = {
  thread_id: string;
  revision: string | null;
  execution_revision?: string;
  items: T[];
  next_page_token: string;
};

export type ThreadStep = {
  step_id: string;
  stage: string;
  title: string;
  order_index: number;
  status: string;
  active: boolean;
};

export type ThreadStepsResponse = {
  thread_id: string;
  active_step_id: string;
  items: ThreadStep[];
};

export type MaterialsOverview = {
  thread_id: string;
  revision: string | null;
  status: FlowStatus;
  case_plan: { target: number; imported: number; automatic: number } | null;
  chunks: {
    scanned: number;
    effective: number;
    selected: number;
    effective_rate: number | null;
    selection_rate: number | null;
  } | null;
  warnings: string[];
};

export type DocumentRow = {
  document_id: string;
  name: string;
  included: boolean;
  knowledge_base: Ref;
  chunks: { effective: number; selected: number; selection_rate: number | null } | null;
};

export type DocumentChunk = {
  chunk_id: string;
  split_rule: string;
  layout_type: string;
  text: string;
  selected: boolean;
};

export type DocumentDetail = {
  thread_id: string;
  revision: string | null;
  document: { id: string; name: string; included: boolean; knowledge_base: Ref };
  chunk_summary: { effective: number; selected: number } | null;
  quotas: Array<{ split_rule: string; required: number; selected: number }>;
  chunks: { items: DocumentChunk[]; next_page_token: string };
};

/**
 * One entry of the split rule / layout type catalog. `supported` reflects the
 * currently included knowledge bases; `enabled` reflects the saved candidate
 * configuration. Ids share the namespace of chunk `split_rule` / `layout_type`.
 */
export type CapabilityEntry = {
  id: string;
  name: string;
  supported: boolean;
  enabled: boolean;
  priority?: number | null;
};

export type AdjustmentOptions = {
  thread_id: string;
  revision: string;
  target_case_count: number;
  min_target_case_count: number;
  knowledge_bases: Array<{ id: string; name: string; included: boolean }>;
  split_rules: CapabilityEntry[];
  layout_types: CapabilityEntry[];
};

export type TopicsOverview = {
  thread_id: string;
  revision: string | null;
  status: FlowStatus;
  total_topics: number | null;
  question_types: Record<QuestionType, { count: number | null; rate: number | null }> | null;
  stages: Record<"entities" | "semantic" | "topics", {
    status: FlowStatus;
    completed: number;
    total: number | null;
    failed?: number;
  }>;
};

export type TopicRow = {
  topic_id: string;
  name: string;
  question_type: QuestionType;
  chunk_count: number;
};

export type TopicChunk = {
  chunk_id: string;
  knowledge_base: Ref;
  document: Ref;
  split_rule: string;
  layout_type: string;
  text: string;
};

export type TopicDetail = {
  thread_id: string;
  revision: string;
  topic: TopicRow;
  chunks: { items: TopicChunk[]; next_page_token: string };
};

export type CaseStageProgress = {
  status: OperationStatus;
  completed: number | null;
  total: number | null;
  status_counts: Record<OperationStatus, number | null> | null;
};

export type CaseDifficultyCounts = Record<Difficulty, number | null>;

export type CasesOverview = {
  thread_id: string;
  revision: string | null;
  execution_revision: string;
  status: FlowStatus;
  stages: Record<CaseStageKey, CaseStageProgress>;
  automatic_plan: {
    total: number | null;
    question_types: Record<QuestionType, {
      total: number | null;
      difficulties: CaseDifficultyCounts;
      capacities?: CaseDifficultyCounts;
    }>;
  } | null;
};

export type CaseRow = {
  case_id: string;
  stages: Record<CaseStageKey, OperationStatus>;
  source: CaseSource;
  question_type: QuestionType;
  difficulty: Difficulty | null;
  topic: { topic_id: string; name: string } | null;
};

export type CaseReference = {
  chunk_id: string;
  knowledge_base: Ref;
  document: Ref;
  text: string;
};

export type CaseKeyPoint = { statement: string; evidence_chunk_ids: string[] };

export type CaseDetail = {
  thread_id: string;
  revision: string;
  case_id: string;
  source: CaseSource;
  question_type: QuestionType;
  difficulty: Difficulty | null;
  topic: { topic_id: string; name: string; chunk_count: number } | null;
  references: CaseReference[];
  stages: {
    plan: { status: OperationStatus };
    generate: {
      status: OperationStatus;
      question: string | null;
      answer: string | null;
      grading_guidance: string | null;
    };
    grading: {
      status: OperationStatus;
      key_points: CaseKeyPoint[] | null;
      forbidden_claims: string[] | null;
    };
  };
};

export type CaseTopicOption = { topic_id: string; name: string; chunk_count: number };

export type DatasetResultCase = {
  case_id: string;
  question: string;
  question_type: QuestionType;
  difficulty: Difficulty | "";
  ground_truth: string;
  grading_guidance: string;
  key_points: CaseKeyPoint[];
  forbidden_claims: string[];
  reference_context: unknown;
  reference_doc: string[];
  reference_doc_ids: string[];
  reference_chunk_ids: string[];
  generate_reason: string;
  is_deleted: boolean;
};

export type DatasetResultResponse = PagedResponse<DatasetResultCase> & {
  revision: string;
  completed_with_problems: boolean;
  total_size: number;
  failed_case_count: number;
};

export type PlanDistribution = Record<QuestionType, Record<Difficulty, number>>;

export type ChunkSelectionChange = {
  knowledge_base_id: string;
  document_id: string;
  chunk_id: string;
  selected: boolean;
};

/** Only the scan configuration entries the user actually changed. */
export type MaterialsConfigChanges = {
  target_case_count?: number;
  knowledge_bases?: Array<{ id: string; included: boolean }>;
  documents?: Array<{ knowledge_base_id: string; document_id: string; included: boolean }>;
  split_rule_ids?: string[];
  layout_type_ids?: string[];
};

/**
 * A single unapplied edit. Only one draft can be pending at a time because
 * every kind invalidates a different point of the downstream flow.
 */
export type DatasetDraft =
  | {
      kind: "materials-config";
      revision: string;
      changes: MaterialsConfigChanges;
    }
  | {
      kind: "chunk-selection";
      revision: string;
      documentId: string;
      documentName: string;
      changes: ChunkSelectionChange[];
    }
  | {
      kind: "topic-names";
      revision: string;
      names: Record<string, string>;
    }
  | {
      kind: "generation-plan";
      revision: string;
      distribution: PlanDistribution;
    };

export const DRAFT_LABELS: Record<DatasetDraft["kind"], string> = {
  "materials-config": "材料范围与片段候选配置",
  "chunk-selection": "文档片段入选",
  "topic-names": "主题名称",
  "generation-plan": "自动生成用例计划分布",
};

/** Index of the dataset stage a draft first invalidates. */
export const DRAFT_IMPACT_START: Record<DatasetDraft["kind"], number> = {
  "materials-config": 0,
  "chunk-selection": 0,
  "topic-names": 1,
  "generation-plan": 2,
};

export const DRAFT_IMPACT_DETAIL: Record<DatasetDraft["kind"], string> = {
  "materials-config":
    "材料范围或片段候选配置变化后，将重新准备材料，并使主题发现与用例生成进入待更新状态。",
  "chunk-selection":
    "具体入选片段变化后，将更新材料结果；后续实际重新执行范围和顺序由 Evo 框架控制。",
  "topic-names":
    "多个主题名称将一次写入新的整体主题产物；后续实际重新执行范围和顺序由 Evo 框架控制。",
  "generation-plan":
    "组合数量修改后，将重新生成全部自动用例计划并覆盖逐个用例的主题调整；全部自动生成用例的问答生成与判分规则需要更新。",
};
