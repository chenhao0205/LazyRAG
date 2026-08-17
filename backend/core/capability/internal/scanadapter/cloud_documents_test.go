package scanadapter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"lazymind/core/capability"
)

func cloudCall() capability.InvocationContext {
	return capability.InvocationContext{Principal: capability.Principal{UserID: "user", TenantID: "tenant", Permissions: capability.NewPermissionSet(capability.RequiredPermission)}}
}

func TestCloudReaderScopesListGetAndSearchToCloudConnectors(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-User-ID") != "user" || r.Header.Get("X-Tenant-ID") != "tenant" {
			t.Fatalf("identity headers missing: %#v", r.Header)
		}
		switch {
		case r.URL.Path == "/api/scan/sources":
			if r.URL.Query().Get("connector_type") != "feishu,notion" {
				t.Fatalf("connector filter=%q", r.URL.Query().Get("connector_type"))
			}
			_, _ = io.WriteString(w, `{"items":[{"source_id":"cloud","name":"Feishu","dataset_id":"kb"}],"total":1}`)
		case r.URL.Path == "/api/scan/sources/cloud":
			_, _ = io.WriteString(w, `{"source":{"source_id":"cloud","name":"Feishu","dataset_id":"kb"},"bindings":[{"connector_type":"feishu"}]}`)
		case r.URL.Path == "/api/scan/sources/cloud/tree/search":
			var body map[string]any
			_ = jsonNewDecoder(r.Body).Decode(&body)
			got := body["connector_types"].([]any)
			if len(got) != 2 || got[0] != "feishu" {
				t.Fatalf("connector_types=%#v", got)
			}
			_, _ = io.WriteString(w, `{"items":[{"key":"x","source_id":"cloud","is_document":true}],"next_cursor":""}`)
		case r.URL.Path == "/api/scan/sources/local":
			_, _ = io.WriteString(w, `{"source":{"source_id":"local","name":"Local"},"bindings":[{"connector_type":"localfs"}]}`)
		default:
			t.Fatalf("unexpected %s", r.URL.Path)
		}
	}))
	defer server.Close()
	reader, err := NewCloudDocumentReader(server.URL, server.Client(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if r, err := reader.ListCloudDocuments(context.Background(), cloudCall(), capability.CloudDocumentListQuery{Limit: 20}); err != nil || len(r.Items) != 1 {
		t.Fatalf("list=%#v err=%v", r, err)
	}
	if _, err := reader.GetCloudDocument(context.Background(), cloudCall(), capability.GetCloudDocumentInput{SourceID: "local"}); err == nil {
		t.Fatal("non-cloud get unexpectedly succeeded")
	}
	if r, err := reader.SearchCloudDocuments(context.Background(), cloudCall(), capability.SearchCloudDocumentsInput{SourceID: "cloud", Query: "fixture", Page: capability.PageRequest{PageSize: 20}}); err != nil || len(r.Hits) != 1 {
		t.Fatalf("search=%#v err=%v", r, err)
	}
}

// Kept local to make the test's JSON use explicit without exporting adapter internals.
func jsonNewDecoder(r io.Reader) *json.Decoder { return json.NewDecoder(r) }
