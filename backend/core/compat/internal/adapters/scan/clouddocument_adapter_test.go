package scan

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"lazymind/core/compat/clouddocument"
	"lazymind/core/compat/contract"
)

func TestCloudDocumentAdapterListSourcesRequestAndMapping(t *testing.T) {
	var gotMethod, gotPath, gotUserID, gotTenantID, gotQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotUserID = r.Header.Get("X-User-ID")
		gotTenantID = r.Header.Get("X-Tenant-ID")
		// Shape transcribed from Scan list source response DTO. Summary is raw
		// backend data; Compat only whitelists document count from it.
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{{
				"source_id":              "source-1",
				"name":                   "Docs",
				"dataset_id":             "dataset-1",
				"status":                 "ACTIVE",
				"config_version":         3,
				"binding_count":          2,
				"source_options":         map[string]any{"token": "source-secret"},
				"summary":                map[string]any{"total_document_count": 7, "document_objects": 99, "last_error": "/home/internal/secret", "bindings": []any{map[string]any{"auth_connection_id": "conn-secret"}}},
				"auth_connection_status": map[string]any{"status": "ACTIVE", "connection_ids": []string{"conn-secret"}},
			}},
			"total": 41,
		})
	}))
	defer server.Close()
	adapter := mustAdapter(t, server.URL)
	result, err := adapter.ListSources(context.Background(), contract.CallContext{UserID: "user-1", TenantID: "tenant-1"}, clouddocument.ListInput{
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
	if gotTenantID != "tenant-1" {
		t.Fatalf("X-Tenant-ID = %q, want tenant-1", gotTenantID)
	}
	for _, want := range []string{"keyword=doc", "status=ACTIVE", "connector_type=feishu%2Cnotion", "page=3", "page_size=20"} {
		if !strings.Contains(gotQuery, want) {
			t.Fatalf("query = %q, want contains %s", gotQuery, want)
		}
	}
	source := result.Sources[0]
	if len(result.Sources) != 1 || source.ID != "source-1" || source.DatasetID != "dataset-1" || source.DocumentCount == nil || *source.DocumentCount != 7 {
		t.Fatalf("result = %#v, want mapped source with whitelisted document count", result)
	}
	if source.AuthConnectionStatus != "ACTIVE" {
		t.Fatalf("auth status = %q, want ACTIVE", source.AuthConnectionStatus)
	}
	if result.Page.NextPageToken != "" {
		t.Fatalf("next token = %q, want empty on final page", result.Page.NextPageToken)
	}
	assertNoSensitiveOutput(t, result)
	assertNoFields(t, reflect.TypeOf(source), "Summary", "ConfigVersion")
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
				"source_options": map[string]any{"app_secret": "source-secret"},
			},
			"bindings": []map[string]any{{
				"binding_id":         "binding-1",
				"auth_connection_id": "conn-secret",
				"provider_options":   map[string]any{"refresh_token": "source-secret"},
			}},
			"summary": map[string]any{"document_objects": 5, "last_error": "/home/internal/secret"},
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
	if gotUserID != "user-1" || result.ID != "source-1" || result.DocumentCount == nil || *result.DocumentCount != 5 {
		t.Fatalf("user=%q result=%#v, want mapped source", gotUserID, result)
	}
	assertNoSensitiveOutput(t, result)
	assertNoFields(t, reflect.TypeOf(result), "Summary", "ConfigVersion")
}

func TestCloudDocumentAdapterListDocumentsRequestAndMapping(t *testing.T) {
	var gotPath, gotQuery, gotUserID string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotUserID = r.Header.Get("X-User-ID")
		// Shape transcribed from Scan SourceDocumentItem. Compat deliberately
		// ignores path, directory, binding, version, and state-machine fields.
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{{
				"document_id":            "doc-1",
				"source_id":              "source-1",
				"binding_id":             "binding-1",
				"object_key":             "obj-1",
				"display_name":           "Spec",
				"name":                   "Spec.md",
				"path":                   "/local/secret/spec.md",
				"directory":              "/local/secret",
				"file_type":              "md",
				"size_bytes":             1024,
				"source_version":         "source-v2",
				"baseline_version":       "source-v1",
				"core_document_id":       "core-doc-1",
				"lazyllm_doc_id":         "lazyllm-doc-should-not-map",
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
	result, err := adapter.ListDocuments(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.SourceDetail{ID: "source-1", DatasetID: "dataset-1"}, clouddocument.GetInput{
		SourceID:      "source-1",
		DocumentsPage: contract.PageRequest{PageSize: 20, PageToken: contract.EncodeOffsetPageToken(20)},
	})
	if err != nil {
		t.Fatalf("ListDocuments returned error: %v", err)
	}
	if gotPath != "/api/scan/sources/source-1/documents" || gotUserID != "user-1" {
		t.Fatalf("path=%q user=%q, want documents path/user", gotPath, gotUserID)
	}
	for _, want := range []string{"page=2", "page_size=20", "refresh_state=false"} {
		if !strings.Contains(gotQuery, want) {
			t.Fatalf("query = %q, want contains %s", gotQuery, want)
		}
	}
	for _, absent := range []string{"binding_id=", "keyword=", "state_filter=", "parse_status="} {
		if strings.Contains(gotQuery, absent) {
			t.Fatalf("query = %q, want no unsupported Get document filter %s", gotQuery, absent)
		}
	}
	doc := result.Documents[0]
	if len(result.Documents) != 1 || doc.ID != "doc-1" || doc.SourceID != "source-1" || doc.Name != "Spec.md" || doc.SizeBytes == nil || *doc.SizeBytes != 1024 {
		t.Fatalf("documents = %#v, want mapped stable document fields", result.Documents)
	}
	if doc.SourceModifiedAt == nil || doc.LastSyncedAt == nil {
		t.Fatalf("document timestamps = %#v, want mapped optional times", doc)
	}
	if doc.KnowledgeDocument == nil || doc.KnowledgeDocument.KnowledgeID != "dataset-1" || doc.KnowledgeDocument.DocumentID != "core-doc-1" {
		t.Fatalf("KnowledgeDocument = %#v, want dataset/core doc ref", doc.KnowledgeDocument)
	}
	if result.Page.NextPageToken == "" {
		t.Fatalf("next token is empty, want next page")
	}
	assertNoSensitiveOutput(t, result)
	assertNoFields(t, reflect.TypeOf(doc), "BindingID", "SourceVersion", "BaselineVersion", "CoreDocumentID", "ParseStatus", "ParseState", "EffectiveParseStatus", "SourceState", "SyncState", "PendingAction", "ParseQueueState", "HasUpdate", "UpdateType")
}

func TestCloudDocumentAdapterListDocumentsAllowsNullAndMissingOptionalFields(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{
				{
					"document_id":      "doc-missing",
					"source_id":        "source-1",
					"object_key":       "obj-missing",
					"display_name":     "Missing",
					"core_document_id": "core-doc-missing",
					"parse_status":     "SUCCEEDED",
				},
				{
					"document_id":        "doc-null",
					"source_id":          "source-1",
					"object_key":         "obj-null",
					"display_name":       "Null",
					"size_bytes":         nil,
					"source_modified_at": nil,
					"last_synced_at":     nil,
					"core_document_id":   "core-doc-null",
					"parse_status":       "SUCCEEDED",
				},
			},
			"total": 2,
		})
	}))
	defer server.Close()
	adapter := mustAdapter(t, server.URL)
	result, err := adapter.ListDocuments(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.SourceDetail{ID: "source-1", DatasetID: "dataset-1"}, clouddocument.GetInput{SourceID: "source-1"})
	if err != nil {
		t.Fatalf("ListDocuments returned error: %v", err)
	}
	if len(result.Documents) != 2 {
		t.Fatalf("documents = %#v, want two docs", result.Documents)
	}
	for _, doc := range result.Documents {
		if doc.SizeBytes != nil || doc.SourceModifiedAt != nil || doc.LastSyncedAt != nil {
			t.Fatalf("doc = %#v, want nil optional numeric/timestamps for missing/null values", doc)
		}
	}
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
		// Shape transcribed from Scan TreeNode. There is no object_type field.
		writeTestJSON(t, w, map[string]any{
			"items": []map[string]any{{
				"key":               "binding-1:obj-1",
				"node_ref":          "node-1",
				"display_name":      "Spec",
				"search_name":       "spec",
				"connector_type":    "feishu",
				"target_type":       "file",
				"target_ref":        "target-1",
				"source_id":         "source-1",
				"binding_id":        "binding-1",
				"tree_key":          "tree-1",
				"object_key":        "obj-1",
				"parent_key":        "parent-1",
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
				"snippet":           "body text should not map",
				"text":              "body text should not map",
				"score":             0.99,
				"provider_meta": map[string]any{
					"token": "tree-secret",
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
	hit := result.Hits[0]
	if len(result.Hits) != 1 || hit.ObjectKey != "obj-1" || result.Page.NextPageToken != "obj-2" {
		t.Fatalf("result = %#v, want mapped hit/cursor", result)
	}
	if !hit.IsDocument || hit.IsContainer || !hit.HasChildren || !hit.Selectable {
		t.Fatalf("hit document/container flags = %#v, want mapped scan tree flags", hit)
	}
	assertNoFields(t, reflect.TypeOf(hit), "ObjectType", "BindingID", "SourceState", "SyncState", "PendingAction", "ParseQueueState", "HasUpdate", "UpdateType")
	assertNoSensitiveOutput(t, result)
	assertNoBodySearchOutput(t, result)
}

func TestCloudDocumentAdapterMapsHTTPJSONAndPermissionErrorsSafely(t *testing.T) {
	tests := []struct {
		name        string
		status      int
		body        string
		run         func(*CloudDocumentAdapter) error
		wantCode    contract.ErrorCode
		wantMessage string
		retryable   bool
	}{
		{name: "not found", status: http.StatusNotFound, body: `{"code":"SOURCE_NOT_FOUND","message":"source not found"}`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.GetSource(context.Background(), contract.CallContext{UserID: "user-1"}, "missing")
			return err
		}, wantCode: contract.NotFound, wantMessage: "scan resource not found"},
		{name: "server error", status: http.StatusInternalServerError, body: `{"code":"INTERNAL_ERROR","message":"/home/internal/feishu/token-secret failed"}`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.ListSources(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.ListInput{})
			return err
		}, wantCode: contract.BackendUnavailable, wantMessage: "scan backend unavailable", retryable: true},
		{name: "unauthorized", status: http.StatusUnauthorized, body: `{"code":"UNAUTHORIZED","message":"/home/internal/feishu/token-secret failed"}`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.ListSources(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.ListInput{})
			return err
		}, wantCode: contract.Internal, wantMessage: "scan access denied"},
		{name: "forbidden", status: http.StatusForbidden, body: `{"code":"FORBIDDEN","message":"/home/internal/feishu/token-secret failed"}`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.ListSources(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.ListInput{})
			return err
		}, wantCode: contract.Internal, wantMessage: "scan access denied"},
		{name: "invalid json", status: http.StatusOK, body: `{`, run: func(a *CloudDocumentAdapter) error {
			_, err := a.Search(context.Background(), contract.CallContext{UserID: "user-1"}, clouddocument.SearchInput{SourceID: "source-1", Query: "doc"})
			return err
		}, wantCode: contract.BackendUnavailable, wantMessage: "scan response is invalid", retryable: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()
			adapter := mustAdapter(t, server.URL)
			err := tc.run(adapter)
			var compatErr *contract.Error
			if !errors.As(err, &compatErr) {
				t.Fatalf("err = %v, want compat error", err)
			}
			if compatErr.Code != tc.wantCode || compatErr.Message != tc.wantMessage || compatErr.Retryable != tc.retryable {
				t.Fatalf("compat error = %#v, want code=%s message=%q retryable=%v", compatErr, tc.wantCode, tc.wantMessage, tc.retryable)
			}
			public := strings.ToLower(err.Error())
			for _, forbidden := range []string{"/home/", "token-secret"} {
				if strings.Contains(public, forbidden) {
					t.Fatalf("public error leaks backend text %q: %s", forbidden, public)
				}
			}
		})
	}
}

func TestKnowledgeDocumentRefEligibility(t *testing.T) {
	size := int64(1)
	base := scanDocumentItem{
		DocumentID:           "doc-1",
		SourceID:             "source-1",
		ObjectKey:            "obj-1",
		DisplayName:          "Spec",
		SizeBytes:            &size,
		CoreDocumentID:       "core-doc-1",
		ParseStatus:          "SUCCEEDED",
		EffectiveParseStatus: "PARSED",
		SourceState:          "NEW",
		PendingAction:        "UPSERT",
	}
	tests := []struct {
		name        string
		knowledgeID string
		item        scanDocumentItem
		wantRef     bool
	}{
		{name: "complete ids and parsed", knowledgeID: "dataset-1", item: base, wantRef: true},
		{name: "missing knowledge id", knowledgeID: "", item: base},
		{name: "missing core document id", knowledgeID: "dataset-1", item: withDoc(base, func(item *scanDocumentItem) { item.CoreDocumentID = "" })},
		{name: "pending delete", knowledgeID: "dataset-1", item: withDoc(base, func(item *scanDocumentItem) { item.PendingAction = "DELETE" })},
		{name: "deleted source state", knowledgeID: "dataset-1", item: withDoc(base, func(item *scanDocumentItem) { item.SourceState = "DELETED" })},
		{name: "out of scope source state", knowledgeID: "dataset-1", item: withDoc(base, func(item *scanDocumentItem) { item.SourceState = "OUT_OF_SCOPE" })},
		{name: "parse failed", knowledgeID: "dataset-1", item: withDoc(base, func(item *scanDocumentItem) {
			item.ParseStatus = "FAILED"
			item.EffectiveParseStatus = "FAILED"
		})},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			doc := mapDocumentSummary(tc.knowledgeID, tc.item)
			if (doc.KnowledgeDocument != nil) != tc.wantRef {
				t.Fatalf("KnowledgeDocument = %#v, want ref=%v", doc.KnowledgeDocument, tc.wantRef)
			}
			if doc.KnowledgeDocument != nil && doc.KnowledgeDocument.DocumentID != "core-doc-1" {
				t.Fatalf("DocumentID = %q, want core_document_id", doc.KnowledgeDocument.DocumentID)
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

func withDoc(item scanDocumentItem, edit func(*scanDocumentItem)) scanDocumentItem {
	edit(&item)
	return item
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

func assertNoFields(t *testing.T, typ reflect.Type, names ...string) {
	t.Helper()
	for _, name := range names {
		if _, ok := typ.FieldByName(name); ok {
			t.Fatalf("%s exposes removed/internal field %s", typ.Name(), name)
		}
	}
}

func assertNoSensitiveOutput(t *testing.T, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal output: %v", err)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range []string{"source-secret", "tree-secret", "provider_options", "provider_meta", "authorization", "refresh_token", "auth_connection_id", "lazyllm", "/local/secret", "/home/internal", "directory"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("output contains sensitive marker %q: %s", forbidden, raw)
		}
	}
}

func assertNoBodySearchOutput(t *testing.T, value any) {
	t.Helper()
	raw, err := json.Marshal(value)
	if err != nil {
		t.Fatalf("marshal output: %v", err)
	}
	lower := strings.ToLower(string(raw))
	for _, forbidden := range []string{"snippet", "body text", "semantic", "score"} {
		if strings.Contains(lower, forbidden) {
			t.Fatalf("output contains unsupported search marker %q: %s", forbidden, raw)
		}
	}
}
