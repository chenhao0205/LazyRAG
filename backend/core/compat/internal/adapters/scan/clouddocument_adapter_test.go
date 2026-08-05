package scan

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"lazymind/core/compat/clouddocument"
	"lazymind/core/compat/contract"
)

func TestCloudDocumentAdapterListSourcesRequestAndMapping(t *testing.T) {
	var gotMethod, gotPath, gotUserID, gotQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotUserID = r.Header.Get("X-User-ID")
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{{
				"source_id":              "source-1",
				"name":                   "Docs",
				"dataset_id":             "dataset-1",
				"status":                 "ACTIVE",
				"config_version":         3,
				"binding_count":          2,
				"source_options":         map[string]any{"token": "secret"},
				"summary":                map[string]any{"total_document_count": 7},
				"auth_connection_status": map[string]any{"status": "ACTIVE", "connection_ids": []string{"conn-secret"}},
			}},
			"total": 41,
		})
	}))
	defer server.Close()
	adapter := mustAdapter(t, server.URL)
	result, err := adapter.ListSources(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.ListInput{
		Keyword: "doc",
		Status:  "ACTIVE",
		Page:    contract.PageRequest{PageSize: 20, PageToken: contract.EncodeOffsetPageToken(40)},
	})
	if err != nil {
		t.Fatalf("ListSources returned error: %v", err)
	}
	if gotMethod != http.MethodGet || gotPath != "/api/scan/sources" {
		t.Fatalf("request = %s %s, want GET /api/scan/sources", gotMethod, gotPath)
	}
	if gotUserID != "user-1" {
		t.Fatalf("X-User-ID = %q, want user-1", gotUserID)
	}
	for _, want := range []string{"keyword=doc", "status=ACTIVE", "page=3", "page_size=20"} {
		if !strings.Contains(gotQuery, want) {
			t.Fatalf("query = %q, want contains %s", gotQuery, want)
		}
	}
	if len(result.Sources) != 1 || result.Sources[0].ID != "source-1" || result.Sources[0].ConfigVersion != 3 || result.Sources[0].DocumentCount == nil || *result.Sources[0].DocumentCount != 7 {
		t.Fatalf("result = %#v, want mapped source", result)
	}
	if result.Sources[0].AuthConnectionStatus != "ACTIVE" {
		t.Fatalf("auth status = %q, want ACTIVE", result.Sources[0].AuthConnectionStatus)
	}
	if result.Page.NextPageToken != "" {
		t.Fatalf("next token = %q, want empty on final page", result.Page.NextPageToken)
	}
	assertNoSensitiveOutput(t, result)
}

func TestCloudDocumentAdapterGetSourceRequestAndMapping(t *testing.T) {
	var gotMethod, gotPath, gotQuery, gotUserID string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotUserID = r.Header.Get("X-User-ID")
		writeTestJSON(t, w, map[string]any{
			"source": map[string]any{
				"source_id":      "source-1",
				"name":           "Docs",
				"dataset_id":     "dataset-1",
				"status":         "ACTIVE",
				"config_version": 9,
				"source_options": map[string]any{"app_secret": "secret"},
			},
			"bindings": []map[string]any{{
				"binding_id":         "binding-1",
				"auth_connection_id": "conn-secret",
				"provider_options":   map[string]any{"refresh_token": "secret"},
			}},
			"summary": map[string]any{"document_objects": 5},
		})
	}))
	defer server.Close()
	adapter := mustAdapter(t, server.URL)
	result, err := adapter.GetSource(context.Background(), contract.CallContext{UserID: "user-1"}, "source-1")
	if err != nil {
		t.Fatalf("GetSource returned error: %v", err)
	}
	if gotMethod != http.MethodGet || gotPath != "/api/scan/sources/source-1" {
		t.Fatalf("request = %s %s, want GET source", gotMethod, gotPath)
	}
	for _, want := range []string{"include_bindings=false", "include_summary=true"} {
		if !strings.Contains(gotQuery, want) {
			t.Fatalf("query = %q, want contains %s", gotQuery, want)
		}
	}
	if gotUserID != "user-1" || result.ID != "source-1" || result.ConfigVersion != 9 || result.DocumentCount == nil || *result.DocumentCount != 5 {
		t.Fatalf("user=%q result=%#v, want mapped source", gotUserID, result)
	}
	assertNoSensitiveOutput(t, result)
}

func TestCloudDocumentAdapterListDocumentsRequestAndMapping(t *testing.T) {
	var gotPath, gotQuery, gotUserID string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotUserID = r.Header.Get("X-User-ID")
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{{
				"document_id":            "doc-1",
				"source_id":              "source-1",
				"binding_id":             "binding-1",
				"object_key":             "obj-1",
				"display_name":           "Spec",
				"name":                   "Spec.md",
				"path":                   "/local/secret/spec.md",
				"file_type":              "md",
				"size_bytes":             1024,
				"source_version":         "source-v2",
				"baseline_version":       "source-v1",
				"core_document_id":       "core-doc-1",
				"parse_status":           "SUCCEEDED",
				"parse_state":            "SUCCEEDED",
				"effective_parse_status": "PARSED",
				"source_state":           "NEW",
				"sync_state":             "READY",
				"pending_action":         "UPSERT",
				"parse_queue_state":      "PENDING",
				"has_update":             true,
				"update_type":            "new",
				"source_modified_at":     "2026-08-06T01:02:03Z",
				"last_synced_at":         "2026-08-06T02:03:04Z",
			}},
			"total":     22,
			"page":      2,
			"page_size": 20,
		})
	}))
	defer server.Close()
	adapter := mustAdapter(t, server.URL)
	result, err := adapter.ListDocuments(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.GetInput{
		SourceID:        "source-1",
		BindingID:       "binding-1",
		DocumentKeyword: "spec",
		StateFilter:     []string{"NEW", "MODIFIED"},
		ParseStatuses:   []string{"SUCCEEDED"},
		DocumentsPage:   contract.PageRequest{PageSize: 20, PageToken: contract.EncodeOffsetPageToken(20)},
	})
	if err != nil {
		t.Fatalf("ListDocuments returned error: %v", err)
	}
	if gotPath != "/api/scan/sources/source-1/documents" || gotUserID != "user-1" {
		t.Fatalf("path=%q user=%q, want documents path/user", gotPath, gotUserID)
	}
	for _, want := range []string{"binding_id=binding-1", "keyword=spec", "page=2", "page_size=20", "refresh_state=false", "state_filter=NEW", "state_filter=MODIFIED", "parse_status=SUCCEEDED"} {
		if !strings.Contains(gotQuery, want) {
			t.Fatalf("query = %q, want contains %s", gotQuery, want)
		}
	}
	if len(result.Documents) != 1 || result.Documents[0].ID != "doc-1" || result.Documents[0].EffectiveParseStatus != "PARSED" {
		t.Fatalf("documents = %#v, want mapped doc", result.Documents)
	}
	doc := result.Documents[0]
	if doc.SourceID != "source-1" || doc.BindingID != "binding-1" || doc.Name != "Spec.md" || doc.SizeBytes != 1024 || doc.SourceVersion != "source-v2" || doc.BaselineVersion != "source-v1" {
		t.Fatalf("document identity/version fields = %#v, want mapped scan fields", doc)
	}
	if doc.SyncState != "READY" || doc.PendingAction != "UPSERT" || doc.ParseQueueState != "PENDING" || !doc.HasUpdate || doc.UpdateType != "new" || doc.SourceModifiedAt == nil || doc.LastSyncedAt == nil {
		t.Fatalf("document state fields = %#v, want mapped state fields", doc)
	}
	if result.Page.NextPageToken == "" {
		t.Fatalf("next token is empty, want next page")
	}
	assertNoSensitiveOutput(t, result)
}

func TestCloudDocumentAdapterSearchRequestAndMapping(t *testing.T) {
	var gotMethod, gotPath, gotUserID string
	var gotBody scanSearchSourceTreeRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotUserID = r.Header.Get("X-User-ID")
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode search body: %v", err)
		}
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{{
				"key":               "binding-1:obj-1",
				"display_name":      "Spec",
				"search_name":       "spec",
				"source_id":         "source-1",
				"binding_id":        "binding-1",
				"tree_key":          "tree-1",
				"object_key":        "obj-1",
				"parent_key":        "parent-1",
				"object_type":       "docx",
				"is_document":       true,
				"is_container":      false,
				"has_children":      true,
				"selectable":        true,
				"source_state":      "NEW",
				"sync_state":        "READY",
				"pending_action":    "UPSERT",
				"parse_queue_state": "PENDING",
				"has_update":        true,
				"update_type":       "modified",
				"provider_meta": map[string]any{
					"token": "secret",
				},
			}},
			"next_cursor": "obj-2",
			"has_more":    true,
			"search_mode": "indexed",
		})
	}))
	defer server.Close()
	adapter := mustAdapter(t, server.URL)
	result, err := adapter.Search(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.SearchInput{
		SourceID:    "source-1",
		Query:       "spec",
		BindingID:   "binding-1",
		TreeKey:     "tree-1",
		StateFilter: []string{"NEW"},
		Page:        contract.PageRequest{PageSize: 50, PageToken: "obj-0"},
	})
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if gotMethod != http.MethodPost || gotPath != "/api/scan/sources/source-1/tree/search" || gotUserID != "user-1" {
		t.Fatalf("request = %s %s user=%q, want POST search with user", gotMethod, gotPath, gotUserID)
	}
	if gotBody.Keyword != "spec" || gotBody.BindingID != "binding-1" || gotBody.TreeKey != "tree-1" || gotBody.ListMode != "page" || gotBody.PageSize != 50 || gotBody.Cursor != "obj-0" {
		t.Fatalf("body = %#v, want scan search fields", gotBody)
	}
	if gotBody.RefreshState == nil || *gotBody.RefreshState {
		t.Fatalf("refresh_state = %#v, want false", gotBody.RefreshState)
	}
	if !gotBody.IncludeDocuments || !gotBody.IncludeContainers {
		t.Fatalf("includes = docs:%v containers:%v, want both", gotBody.IncludeDocuments, gotBody.IncludeContainers)
	}
	if len(result.Hits) != 1 || result.Hits[0].ObjectKey != "obj-1" || result.Page.NextPageToken != "obj-2" {
		t.Fatalf("result = %#v, want mapped hit/cursor", result)
	}
	hit := result.Hits[0]
	if !hit.HasChildren || !hit.Selectable || hit.SyncState != "READY" || hit.PendingAction != "UPSERT" || hit.ParseQueueState != "PENDING" || !hit.HasUpdate || hit.UpdateType != "modified" {
		t.Fatalf("hit state fields = %#v, want mapped scan tree state", hit)
	}
	assertNoSensitiveOutput(t, result)
}

func TestCloudDocumentAdapterMapsHTTPAndJSONErrors(t *testing.T) {
	tests := []struct {
		name     string
		status   int
		body     string
		run      func(*CloudDocumentAdapter) error
		wantCode contract.ErrorCode
	}{
		{name: "not found", status: http.StatusNotFound, body: `{"code":"SOURCE_NOT_FOUND","message":"source not found"}`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.GetSource(context.Background(), contract.CallContext{UserID: "user-1"}, "missing")
			return err
		}, wantCode: contract.NotFound},
		{name: "server error", status: http.StatusInternalServerError, body: `{"code":"INTERNAL_ERROR","message":"scan failed"}`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.ListSources(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.ListInput{})
			return err
		}, wantCode: contract.BackendUnavailable},
		{name: "invalid json", status: http.StatusOK, body: `{`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.Search(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.SearchInput{SourceID: "source-1", Query: "doc"})
			return err
		}, wantCode: contract.BackendUnavailable},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			adapter := mustAdapter(t, server.URL)
			code, ok := contract.CodeOf(tc.run(adapter))
			if !ok || code != tc.wantCode {
				t.Fatalf("code = %s, %v; want %s", code, ok, tc.wantCode)
			}
		})
	}
}

func TestCloudDocumentAdapterInvalidPageToken(t *testing.T) {
	adapter := mustAdapter(t, "http://scan.test")
	_, err := adapter.ListSources(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.ListInput{
		Page: contract.PageRequest{PageSize: 20, PageToken: "bad"},
	})
	if code, ok := contract.CodeOf(err); !ok || code != contract.InvalidArgument {
		t.Fatalf("code = %s, %v; want INVALID_ARGUMENT", code, ok)
	}
}

func mustAdapter(t *testing.T, baseURL string) *CloudDocumentAdapter {
	t.Helper()
	adapter, err := NewCloudDocumentAdapter(baseURL, http.DefaultClient, 0)
	if err != nil {
		t.Fatalf("NewCloudDocumentAdapter returned error: %v", err)
	}
	return adapter
}

func writeTestJSON(t *testing.T, w http.ResponseWriter, body any) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(body); err != nil {
		t.Fatalf("encode response: %v", err)
	}
}

func assertNoSensitiveOutput(t *testing.T, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal output: %v", err)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range []string{"secret", "provider_options", "authorization", "refresh_token", "auth_connection_id", "/local/secret"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("output contains sensitive marker %q: %s", forbidden, raw)
		}
	}
}
