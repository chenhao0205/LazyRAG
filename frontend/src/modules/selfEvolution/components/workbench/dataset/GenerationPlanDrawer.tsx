import { useEffect, useState } from "react";
import { Alert, Button, Drawer, InputNumber } from "antd";
import type {
  CasesOverview,
  DatasetDraft,
  Difficulty,
  PlanDistribution,
  QuestionType,
} from "./types";
import { DIFFICULTY_TEXT, QUESTION_TYPE_TEXT } from "./primitives";

const QUESTION_TYPES: QuestionType[] = ["precision", "reasoning"];
const DIFFICULTIES: Difficulty[] = ["easy", "medium", "hard"];

const emptyDistribution = (): PlanDistribution => ({
  precision: { easy: 0, medium: 0, hard: 0 },
  reasoning: { easy: 0, medium: 0, hard: 0 },
});

function toDistribution(overview?: CasesOverview): PlanDistribution {
  const plan = overview?.automatic_plan;
  if (!plan) return emptyDistribution();
  const result = emptyDistribution();
  for (const type of QUESTION_TYPES) {
    for (const difficulty of DIFFICULTIES) {
      result[type][difficulty] = plan.question_types?.[type]?.difficulties?.[difficulty] ?? 0;
    }
  }
  return result;
}

/**
 * Edits the exact integer split of automatically generated cases. The total is
 * fixed by the current automatic case count, so the form validates against it.
 */
export function GenerationPlanDrawer({
  open,
  overview,
  onClose,
  onSaveDraft,
}: {
  open: boolean;
  overview?: CasesOverview;
  onClose: () => void;
  onSaveDraft: (draft: DatasetDraft) => boolean;
}) {
  const [distribution, setDistribution] = useState<PlanDistribution>(emptyDistribution);

  useEffect(() => {
    if (open) setDistribution(toDistribution(overview));
  }, [open, overview]);

  const expected = overview?.automatic_plan?.total ?? 0;
  const total = QUESTION_TYPES.reduce(
    (sum, type) => sum + DIFFICULTIES.reduce((inner, difficulty) => inner + distribution[type][difficulty], 0),
    0,
  );
  const revision = overview?.revision;
  const withinCapacities = QUESTION_TYPES.every((type) =>
    DIFFICULTIES.every((difficulty) => {
      const capacity = overview?.automatic_plan?.question_types?.[type]?.capacities?.[difficulty];
      return capacity == null || distribution[type][difficulty] <= capacity;
    }),
  );
  const unchanged =
    JSON.stringify(distribution) === JSON.stringify(toDistribution(overview));
  const canSave = Boolean(revision) && total === expected && withinCapacities && !unchanged;

  const save = () => {
    if (!revision || !canSave) return;
    if (onSaveDraft({ kind: "generation-plan", revision, distribution })) {
      onClose();
    }
  };

  return (
    <Drawer
      className="dataset-drawer"
      rootClassName="dataset-drawer-root"
      title="调整生成计划"
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
      {!overview?.automatic_plan ? (
        <Alert type="info" showIcon message="当前还没有自动生成用例计划，暂不能调整分布。" />
      ) : (
        <>
          <div className="dataset-form-note">
            六种题型与难度组合均填写非负整数，合计必须等于 {expected} 个自动生成用例。
          </div>
          <div className="dataset-lane-inputs">
            {QUESTION_TYPES.map((type) =>
              DIFFICULTIES.map((difficulty) => (
                <div className="dataset-lane-input" key={`${type}-${difficulty}`}>
                  <label htmlFor={`plan-${type}-${difficulty}`}>
                    {QUESTION_TYPE_TEXT[type]} · {DIFFICULTY_TEXT[difficulty]} · 上限 {overview.automatic_plan.question_types?.[type]?.capacities?.[difficulty] ?? "—"}
                  </label>
                  <InputNumber
                    id={`plan-${type}-${difficulty}`}
                    min={0}
                    max={overview.automatic_plan.question_types?.[type]?.capacities?.[difficulty] ?? expected}
                    value={distribution[type][difficulty]}
                    onChange={(value) =>
                      setDistribution((prev) => ({
                        ...prev,
                        [type]: { ...prev[type], [difficulty]: value ?? 0 },
                      }))
                    }
                  />
                </div>
              )),
            )}
          </div>
          <div className={`dataset-plan-total${total === expected ? " is-ok" : " is-invalid"}`}>
            当前合计 {total} / {expected}
          </div>
          <div className="dataset-warning-note">
            应用后会重新分配全部自动生成用例，并覆盖此前逐个用例的主题更换；外部导入用例不受影响。
          </div>
        </>
      )}
    </Drawer>
  );
}
