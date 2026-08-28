import { useCallback, useEffect, useRef, useState } from "react";
import { describeRequestError } from "./api";
import type { PagedResponse } from "./types";

export const PAGE_SIZE = 50;
export const CHUNK_PAGE_SIZE = 100;

type ResourceState<T> = { data?: T; loading: boolean; error?: string; loadedToken?: number };

/** Loads a single dataset object (an overview, a detail, an option set). */
export function useDatasetResource<T>(
  fetchOne: (() => Promise<T>) | undefined,
  refreshToken = 0,
  failureText = "数据加载失败",
  clearOnLoad = false,
) {
  const [state, setState] = useState<ResourceState<T>>({ loading: false });
  const [localToken, setLocalToken] = useState(0);

  useEffect(() => {
    if (!fetchOne) {
      setState({ loading: false });
      return undefined;
    }
    let cancelled = false;
    setState((prev) => ({ data: clearOnLoad ? undefined : prev.data, loading: true }));
    fetchOne()
      .then((data) => {
        if (!cancelled) setState({ data, loading: false, loadedToken: refreshToken });
      })
      .catch((error) => {
        if (!cancelled) setState({ loading: false, error: describeRequestError(error, failureText) });
      });
    return () => {
      cancelled = true;
    };
  }, [clearOnLoad, fetchOne, refreshToken, localToken, failureText]);

  const reload = useCallback(() => setLocalToken((token) => token + 1), []);

  return { ...state, reload };
}

type PagedState<TPage, TItem> = {
  page?: TPage;
  items: TItem[];
  nextPageToken: string;
  loading: boolean;
  error?: string;
  loadedToken?: number;
};

type PageSlice<TItem> = { items: TItem[]; nextPageToken: string };
const emptyPagedState = <TPage, TItem>(): PagedState<TPage, TItem> => ({
  items: [],
  nextPageToken: "",
  loading: false,
});

function usePagedResource<TPage, TItem>(
  fetchPage: ((pageToken?: string) => Promise<TPage>) | undefined,
  project: (page: TPage) => PageSlice<TItem>,
  refreshToken: number,
  failureText: string,
) {
  const [state, setState] = useState<PagedState<TPage, TItem>>(emptyPagedState);
  const [reloadToken, setReloadToken] = useState(0);
  const version = useRef(0);
  const loading = useRef(false);
  const projectRef = useRef(project);
  projectRef.current = project;

  const load = useCallback(async (pageToken = "", requestVersion = version.current) => {
    if (!fetchPage || loading.current) return;
    loading.current = true;
    setState((prev) => ({ ...prev, loading: true, error: undefined }));
    try {
      const page = await fetchPage(pageToken || undefined);
      if (requestVersion !== version.current) return;
      const slice = projectRef.current(page);
      setState((prev) => ({
        page,
        items: pageToken ? [...prev.items, ...slice.items] : slice.items,
        nextPageToken: slice.nextPageToken,
        loading: false,
        loadedToken: pageToken ? prev.loadedToken : refreshToken,
      }));
    } catch (error) {
      if (requestVersion !== version.current) return;
      setState((prev) => ({ ...prev, loading: false, error: describeRequestError(error, failureText) }));
    } finally {
      if (requestVersion === version.current) loading.current = false;
    }
  }, [failureText, fetchPage, refreshToken]);

  useEffect(() => {
    const requestVersion = ++version.current;
    loading.current = false;
    if (!fetchPage) {
      setState(emptyPagedState);
      return undefined;
    }
    setState({ ...emptyPagedState<TPage, TItem>(), loading: true });
    void load("", requestVersion);
    return () => {
      if (version.current === requestVersion) version.current += 1;
    };
  }, [fetchPage, load, reloadToken]);

  const loadMore = useCallback(() => {
    if (state.nextPageToken) void load(state.nextPageToken);
  }, [load, state.nextPageToken]);
  const reload = useCallback(() => setReloadToken((token) => token + 1), []);
  return { ...state, loadMore, reload };
}

/** Shared state machine for drawers whose detail response embeds a paged list. */
export function useDatasetPagedDetail<TDetail, TItem>(
  fetchPage: ((pageToken?: string) => Promise<TDetail>) | undefined,
  itemsOf: (detail: TDetail) => TItem[],
  nextTokenOf: (detail: TDetail) => string,
  failureText: string,
) {
  const result = usePagedResource(
    fetchPage,
    (page) => ({ items: itemsOf(page), nextPageToken: nextTokenOf(page) }),
    0,
    failureText,
  );
  const { page, ...state } = result;
  return { ...state, data: page };
}

/**
 * Cursor paginated list. `fetchPage` must be stable (memoised on its filters);
 * changing it restarts from the first page, as required by the paging contract.
 */
export function useDatasetList<T>(
  fetchPage: ((pageToken?: string) => Promise<PagedResponse<T>>) | undefined,
  refreshToken = 0,
  failureText = "列表加载失败",
) {
  const result = usePagedResource(
    fetchPage,
    (page) => ({ items: page.items || [], nextPageToken: page.next_page_token || "" }),
    refreshToken,
    failureText,
  );
  const { page, ...state } = result;
  return {
    ...state,
    revision: page?.revision ?? null,
    executionRevision: page?.execution_revision,
  };
}
