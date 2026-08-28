import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { Button, Space } from "antd";

type StageAction = { label: string; onClick: () => void };

type StageActionStore = {
  action?: StageAction;
  resultAction?: StageAction;
  setAction: (action?: StageAction) => void;
  setResultAction: (action?: StageAction) => void;
};

const DatasetStageActionContext = createContext<StageActionStore>({
  setAction: () => undefined,
  setResultAction: () => undefined,
});

/**
 * The page-level dataset action lives in the workbench stage header, while the
 * drawer it opens belongs to the dataset workspace below it. The provider wraps
 * both so the active sub-stage can publish its own action.
 */
export function DatasetStageActionProvider({ children }: { children: ReactNode }) {
  const [action, setAction] = useState<StageAction>();
  const [resultAction, setResultAction] = useState<StageAction>();
  const store = useMemo(
    () => ({ action, resultAction, setAction, setResultAction }),
    [action, resultAction],
  );
  return (
    <DatasetStageActionContext.Provider value={store}>
      {children}
    </DatasetStageActionContext.Provider>
  );
}

export function DatasetStageActionButton() {
  const { action, resultAction } = useContext(DatasetStageActionContext);
  if (!action && !resultAction) return null;
  return (
    <Space>
      {resultAction ? (
        <Button
          onClick={(event) => {
            event.stopPropagation();
            resultAction.onClick();
          }}
        >
          {resultAction.label}
        </Button>
      ) : null}
      {action ? (
        <Button
          className="self-evolution-dataset-adjust-button"
          onClick={(event) => {
            event.stopPropagation();
            action.onClick();
          }}
        >
          {action.label}
        </Button>
      ) : null}
    </Space>
  );
}

export function usePublishDatasetResultAction(action?: StageAction) {
  const { setResultAction } = useContext(DatasetStageActionContext);
  const label = action?.label;
  const onClick = action?.onClick;
  useEffect(() => {
    setResultAction(label && onClick ? { label, onClick } : undefined);
    return () => setResultAction(undefined);
  }, [label, onClick, setResultAction]);
}

export function usePublishDatasetStageAction(action?: StageAction) {
  const { setAction } = useContext(DatasetStageActionContext);
  const label = action?.label;
  const onClick = action?.onClick;
  useEffect(() => {
    setAction(label && onClick ? { label, onClick } : undefined);
    return () => setAction(undefined);
  }, [label, onClick, setAction]);
}
