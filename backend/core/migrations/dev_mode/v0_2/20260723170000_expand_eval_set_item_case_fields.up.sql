-- +migrate Dialect postgres
ALTER TABLE public.eval_set_items
    ADD COLUMN IF NOT EXISTS difficulty character varying(32) NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS grading_guidance text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS forbidden_claims text NOT NULL DEFAULT '';

-- +migrate Dialect sqlite
ALTER TABLE eval_set_items ADD COLUMN difficulty varchar(32) NOT NULL DEFAULT '';
ALTER TABLE eval_set_items ADD COLUMN grading_guidance text NOT NULL DEFAULT '';
ALTER TABLE eval_set_items ADD COLUMN forbidden_claims text NOT NULL DEFAULT '';
