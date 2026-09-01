from __future__ import annotations

# The three query-anchored logics checked by eval_qc. Code-side source of truth
# for the logic_id enum (schema, judging, aggregation). The human-readable scoring
# table lives in evo/eval_qc/prompts/eval_qc_judge.md and is kept in sync by hand.
LOGIC_IDS: tuple[str, ...] = (
    'query_to_gt_answer',
    'query_to_gt_text',
    'query_to_key_points',
)
