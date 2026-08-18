package scanadapter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
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
		case r.URL.Path == "/api/scan/sources/cloud/documents":
			if r.URL.Query().Get("connector_type") != cloudConnectors {
				t.Fatalf("document connector filter=%q", r.URL.Query().Get("connector_type"))
			}
			_, _ = io.WriteString(w, `{"items":[{"document_id":"feishu-doc","source_id":"cloud","display_name":"Feishu document"}],"total":1}`)
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
	if r, err := reader.GetCloudDocument(context.Background(), cloudCall(), capability.GetCloudDocumentInput{SourceID: "cloud", IncludeDocuments: true, DocumentsPage: capability.PageRequest{PageSize: 20}}); err != nil || len(r.Documents) != 1 || r.Documents[0].ID != "feishu-doc" {
		t.Fatalf("get=%#v err=%v", r, err)
	}
	if r, err := reader.SearchCloudDocuments(context.Background(), cloudCall(), capability.SearchCloudDocumentsInput{SourceID: "cloud", Query: "fixture", Page: capability.PageRequest{PageSize: 20}}); err != nil || len(r.Hits) != 1 {
		t.Fatalf("search=%#v err=%v", r, err)
	}
}

func TestCloudReaderGetsOnlyCloudBindingDocuments(t *testing.T) {
	tests := []struct {
		name      string
		sourceID  string
		bindings  string
		documents string
		want      []string
	}{
		{
			name:      "feishu only",
			sourceID:  "feishu",
			bindings:  `[ {"binding_id":"feishu-binding","connector_type":"feishu"} ]`,
			documents: `[{"document_id":"feishu-doc","source_id":"feishu"}]`,
			want:      []string{"feishu-doc"},
		},
		{
			name:      "feishu and localfs",
			sourceID:  "mixed",
			bindings:  `[ {"binding_id":"feishu-binding","connector_type":"feishu"}, {"binding_id":"local-binding","connector_type":"localfs"} ]`,
			documents: `[{"document_id":"feishu-doc","source_id":"mixed"}]`,
			want:      []string{"feishu-doc"},
		},
		{
			name:      "feishu notion and localfs",
			sourceID:  "multi",
			bindings:  `[ {"binding_id":"feishu-binding","connector_type":"feishu"}, {"binding_id":"notion-binding","connector_type":"notion"}, {"binding_id":"local-binding","connector_type":"localfs"} ]`,
			documents: `[{"document_id":"feishu-doc","source_id":"multi"},{"document_id":"notion-doc","source_id":"multi"}]`,
			want:      []string{"feishu-doc", "notion-doc"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/scan/sources/" + test.sourceID:
					_, _ = io.WriteString(w, `{"source":{"source_id":"`+test.sourceID+`"},"bindings":`+test.bindings+`}`)
				case "/api/scan/sources/" + test.sourceID + "/documents":
					if got := r.URL.Query().Get("connector_type"); got != cloudConnectors {
						t.Fatalf("connector_type=%q, want %q", got, cloudConnectors)
					}
					_, _ = io.WriteString(w, `{"items":`+test.documents+`,"total":`+strconv.Itoa(len(test.want))+`}`)
				default:
					t.Fatalf("unexpected path %q", r.URL.Path)
				}
			}))
			defer server.Close()
			reader, err := NewCloudDocumentReader(server.URL, server.Client(), 0)
			if err != nil {
				t.Fatal(err)
			}
			got, err := reader.GetCloudDocument(context.Background(), cloudCall(), capability.GetCloudDocumentInput{SourceID: test.sourceID, IncludeDocuments: true, DocumentsPage: capability.PageRequest{PageSize: 20}})
			if err != nil {
				t.Fatal(err)
			}
			if len(got.Documents) != len(test.want) {
				t.Fatalf("documents=%#v, want IDs %v", got.Documents, test.want)
			}
			for i, want := range test.want {
				if got.Documents[i].ID != want {
					t.Fatalf("document[%d]=%q, want %q", i, got.Documents[i].ID, want)
				}
			}
		})
	}

	t.Run("localfs only is rejected", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/scan/sources/local" {
				t.Fatalf("unexpected path %q", r.URL.Path)
			}
			_, _ = io.WriteString(w, `{"source":{"source_id":"local"},"bindings":[{"binding_id":"local-binding","connector_type":"localfs"}]}`)
		}))
		defer server.Close()
		reader, err := NewCloudDocumentReader(server.URL, server.Client(), 0)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := reader.GetCloudDocument(context.Background(), cloudCall(), capability.GetCloudDocumentInput{SourceID: "local", IncludeDocuments: true, DocumentsPage: capability.PageRequest{PageSize: 20}}); err == nil {
			t.Fatal("non-cloud get unexpectedly succeeded")
		}
	})
}

// Kept local to make the test's JSON use explicit without exporting adapter internals.
func jsonNewDecoder(r io.Reader) *json.Decoder { return json.NewDecoder(r) }
