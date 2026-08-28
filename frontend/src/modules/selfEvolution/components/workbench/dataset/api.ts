import { axiosInstance } from "@/components/request";
import { EVO_API_BASE } from "../../../shared/constants";

type QueryParams = Record<string, string | number | boolean | undefined>;

// `silentError` keeps the global axios interceptor from raising a toast; the
// dataset panels render their own inline error states instead.
const silentConfig = (params?: QueryParams) => ({ params, silentError: true }) as never;

export const threadRoot = (threadId: string) =>
  `${EVO_API_BASE}/threads/${encodeURIComponent(threadId)}`;

export const datasetRoot = (threadId: string) => `${threadRoot(threadId)}/dataset`;

export const newRequestId = () =>
  globalThis.crypto?.randomUUID?.() ||
  `dataset-${Date.now()}-${Math.random().toString(36).slice(2)}`;

/**
 * Evo returns FastAPI style `{ detail }` payloads. Axios only exposes the HTTP
 * status in `error.message`, which is useless in the UI, so unwrap the body.
 */
export function describeRequestError(error: unknown, fallback: string): string {
  const payload = (error as { response?: { data?: unknown } })?.response?.data;
  const detail = (payload as { detail?: unknown })?.detail;
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) =>
        typeof item === "string" ? item : (item as { msg?: string })?.msg,
      )
      .filter(Boolean);
    if (messages.length) {
      return messages.join("；");
    }
  }
  const status = (error as { response?: { status?: number } })?.response?.status;
  return status ? `${fallback}（HTTP ${status}）` : fallback;
}

export async function getJson<T>(url: string, params?: QueryParams): Promise<T> {
  const response = await axiosInstance.get<T>(url, silentConfig(params));
  return response.data;
}

export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await axiosInstance.post<T>(url, body, silentConfig());
  return response.data;
}

export async function patchJson<T>(url: string, body: unknown): Promise<T> {
  const response = await axiosInstance.patch<T>(url, body, silentConfig());
  return response.data;
}

export async function downloadDatasetResult(threadId: string, revision: string) {
  const response = await axiosInstance.get<Blob>(`${datasetRoot(threadId)}/result:download`, {
    params: { format: "csv", revision },
    responseType: "blob",
    silentError: true,
  } as never);
  return response.data;
}
