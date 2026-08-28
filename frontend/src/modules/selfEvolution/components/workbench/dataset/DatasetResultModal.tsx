import { useCallback, useState } from "react";
import { Alert, Button, Modal, Space, Table, Tag, Typography, message } from "antd";
import type { ColumnsType } from "antd/es/table";
import { DownloadOutlined } from "@ant-design/icons";
import { datasetRoot, describeRequestError, downloadDatasetResult, getJson } from "./api";
import { useDatasetPagedDetail } from "./hooks";
import type { DatasetResultCase, DatasetResultResponse } from "./types";

const { Text } = Typography;

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

const columns: ColumnsType<DatasetResultCase> = [
  { title: "用例编号", dataIndex: "case_id", width: 140, fixed: "left" },
  { title: "问题", dataIndex: "question", width: 300, ellipsis: true },
  {
    title: "题型",
    dataIndex: "question_type",
    width: 100,
    render: (value: string) => <Tag color={value === "reasoning" ? "purple" : "blue"}>{value}</Tag>,
  },
  { title: "难度", dataIndex: "difficulty", width: 90, render: (value: string) => value || "—" },
  { title: "标准答案", dataIndex: "ground_truth", width: 320, ellipsis: true },
  { title: "评分说明", dataIndex: "grading_guidance", width: 260, ellipsis: true },
];

const resultItems = (result: DatasetResultResponse) => result.items || [];
const resultNextToken = (result: DatasetResultResponse) => result.next_page_token || "";

export function DatasetResultModal({
  threadId,
  open,
  onClose,
}: {
  threadId: string;
  open: boolean;
  onClose: () => void;
}) {
  const [downloading, setDownloading] = useState(false);
  const fetchResult = useCallback(
    (pageToken?: string) =>
      getJson<DatasetResultResponse>(`${datasetRoot(threadId)}/result`, {
        page_size: 50,
        page_token: pageToken,
      }),
    [threadId],
  );
  const resultPage = useDatasetPagedDetail(
    open ? fetchResult : undefined,
    resultItems,
    resultNextToken,
    "生成结果加载失败",
  );
  const result = resultPage.data;

  const download = async () => {
    if (!result?.revision) return;
    setDownloading(true);
    try {
      const blob = await downloadDatasetResult(threadId, result.revision);
      saveBlob(blob, `dataset-${threadId}.csv`);
    } catch (requestError) {
      message.error(describeRequestError(requestError, "数据集下载失败"));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Modal
      title="生成结果"
      open={open}
      onCancel={onClose}
      width={1120}
      footer={null}
      destroyOnClose
    >
      {resultPage.error ? (
        <Alert type="error" showIcon message={resultPage.error} action={<Button onClick={resultPage.reload}>重试</Button>} />
      ) : null}
      <div className="dataset-result-summary">
        <Space size="large">
          <Text>共 {result?.total_size ?? 0} 个用例</Text>
          {result?.completed_with_problems ? (
            <Text type="warning">{result.failed_case_count} 个计划用例生成失败，当前结果仍可使用</Text>
          ) : null}
        </Space>
        <Button
          icon={<DownloadOutlined />}
          disabled={!result?.revision}
          loading={downloading}
          onClick={download}
        >
          下载 CSV
        </Button>
      </div>
      <Table<DatasetResultCase>
        rowKey="case_id"
        columns={columns}
        dataSource={resultPage.items}
        loading={resultPage.loading && !resultPage.items.length}
        pagination={false}
        scroll={{ x: 1210, y: 520 }}
      />
      {resultPage.nextPageToken ? (
        <div className="dataset-result-more">
          <Button loading={resultPage.loading} onClick={resultPage.loadMore}>加载更多</Button>
        </div>
      ) : null}
    </Modal>
  );
}
