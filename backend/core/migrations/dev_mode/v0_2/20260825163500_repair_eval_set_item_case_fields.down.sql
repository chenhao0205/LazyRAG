-- +migrate Dialect postgres
ALTER TABLE public.eval_set_items
    DROP COLUMN IF EXISTS forbidden_claims,
    DROP COLUMN IF EXISTS grading_guidance,
    DROP COLUMN IF EXISTS difficulty;

-- +migrate Dialect sqlite
ALTER TABLE eval_set_items DROP COLUMN forbidden_claims;
ALTER TABLE eval_set_items DROP COLUMN grading_guidance;
ALTER TABLE eval_set_items DROP COLUMN difficulty;
