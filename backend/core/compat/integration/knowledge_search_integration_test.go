package integration_test

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"lazymind/core/acl"
	"lazymind/core/common/orm"
	"lazymind/core/compat/contract"
	adaptercore "lazymind/core/compat/internal/adapters/core"
	compatknowledge "lazymind/core/compat/knowledge"
	compatruntime "lazymind/core/compat/runtime"
)

// TestKnowledgeRuntimeWithRealSearchBackend exercises the real PostgreSQL
// resolver/document mapper and the configured HTTP knowledge-search backend.
// The search fixture is provisioned by the explicitly selected non-production
// search environment because Core has no supported API for creating and
// deleting an indexed knowledge fixture synchronously.
func TestKnowledgeRuntimeWithRealSearchBackend(t *testing.T) {
	fixture, ok := realSearchIntegrationFixture(t)
	if !ok {
		return
	}

	db, err := orm.Connect(orm.DriverPostgres, fixture.dsn)
	if err != nil {
		t.Fatalf("connect core db: %v", err)
	}
	sqlDB, err := db.DB.DB()
	if err != nil {
		t.Fatalf("get sql db: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	acl.InitStore(db)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	var dataset orm.Dataset
	if err := db.DB.WithContext(ctx).
		Where("id = ? AND deleted_at IS NULL", fixture.datasetID).
		First(&dataset).Error; err != nil {
		t.Fatalf("load search fixture dataset: %v", err)
	}
	if strings.TrimSpace(dataset.KbID) == "" {
		t.Fatal("search fixture dataset has empty kb_id")
	}
	if dataset.CreateUserID != fixture.userID || dataset.TenantID != fixture.tenantID {
		t.Fatalf("search fixture ownership mismatch: dataset owner/tenant do not match test identity")
	}

	var document orm.Document
	if err := db.DB.WithContext(ctx).
		Where("id = ? AND dataset_id = ? AND deleted_at IS NULL", fixture.documentID, fixture.datasetID).
		First(&document).Error; err != nil {
		t.Fatalf("load search fixture document: %v", err)
	}
	if strings.TrimSpace(document.LazyllmDocID) == "" {
		t.Fatal("search fixture document has empty lazyllm_doc_id")
	}
	installKnowledgeCatalogScanStub(t, fixture.userID, fixture.tenantID)

	adapter, err := adaptercore.NewKnowledgeSearchAdapterForDB(db.DB, fixture.searchURL)
	if err != nil {
		t.Fatalf("NewKnowledgeSearchAdapterForDB: %v", err)
	}
	rt, err := compatruntime.New(compatruntime.Dependencies{KnowledgeSearch: adapter})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	callCtx := contract.CallContext{UserID: fixture.userID, TenantID: fixture.tenantID}

	result, err := rt.Knowledge.Search(ctx, callCtx, compatknowledge.SearchInput{
		Query:        fixture.query,
		KnowledgeIDs: []string{fixture.datasetID},
		TopK:         10,
	})
	if err != nil {
		t.Fatalf("Knowledge.Search: %v", err)
	}
	if !hasExpectedSearchFixtureHit(result, fixture) {
		t.Fatalf("Knowledge.Search did not return the configured indexed fixture")
	}

	noHit, err := rt.Knowledge.Search(ctx, callCtx, compatknowledge.SearchInput{
		Query:        "compat-knowledge-search-no-hit-" + uuid.NewString(),
		KnowledgeIDs: []string{fixture.datasetID},
		TopK:         10,
	})
	if err != nil {
		t.Fatalf("Knowledge.Search no-hit query: %v", err)
	}
	if noHit.Hits == nil || len(noHit.Hits) != 0 {
		t.Fatalf("Knowledge.Search no-hit result = %#v, want empty hits", noHit)
	}

	_, err = rt.Knowledge.Search(ctx, callCtx, compatknowledge.SearchInput{
		Query:        fixture.query,
		KnowledgeIDs: []string{uuid.NewString()},
		TopK:         1,
	})
	if code, ok := contract.CodeOf(err); !ok || code != contract.NotFound {
		t.Fatalf("Knowledge.Search unknown dataset error = %v, want NOT_FOUND", err)
	}

	unavailable, err := adaptercore.NewKnowledgeSearchAdapterForDB(db.DB, "http://127.0.0.1:1")
	if err != nil {
		t.Fatalf("NewKnowledgeSearchAdapterForDB unavailable backend: %v", err)
	}
	unavailableRT, err := compatruntime.New(compatruntime.Dependencies{KnowledgeSearch: unavailable})
	if err != nil {
		t.Fatalf("Runtime.New unavailable backend: %v", err)
	}
	failureCtx, failureCancel := context.WithTimeout(context.Background(), time.Second)
	defer failureCancel()
	_, err = unavailableRT.Knowledge.Search(failureCtx, callCtx, compatknowledge.SearchInput{
		Query:        fixture.query,
		KnowledgeIDs: []string{fixture.datasetID},
		TopK:         1,
	})
	if code, ok := contract.CodeOf(err); !ok || code != contract.BackendUnavailable {
		t.Fatalf("Knowledge.Search unavailable backend error = %v, want BACKEND_UNAVAILABLE", err)
	}

	t.Logf("Knowledge Search integration: dataset_id=%s document_id=%s hit_count=%d", fixture.datasetID, fixture.documentID, len(result.Hits))
}

type realSearchFixture struct {
	userID     string
	tenantID   string
	dsn        string
	searchURL  string
	datasetID  string
	documentID string
	query      string
	marker     string
}

func realSearchIntegrationFixture(t *testing.T) (realSearchFixture, bool) {
	t.Helper()
	if strings.TrimSpace(os.Getenv("COMPAT_INTEGRATION")) != "1" ||
		strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_INTEGRATION")) != "1" {
		t.Skip("set COMPAT_INTEGRATION=1 and COMPAT_KNOWLEDGE_SEARCH_INTEGRATION=1 to run real knowledge search integration")
	}
	values := map[string]string{
		"COMPAT_TEST_USER_ID":                   strings.TrimSpace(os.Getenv("COMPAT_TEST_USER_ID")),
		"COMPAT_TEST_TENANT_ID":                 strings.TrimSpace(os.Getenv("COMPAT_TEST_TENANT_ID")),
		"ACL_DB_DSN":                            strings.TrimSpace(os.Getenv("ACL_DB_DSN")),
		"LAZYMIND_CHAT_SERVICE_URL":             strings.TrimSpace(os.Getenv("LAZYMIND_CHAT_SERVICE_URL")),
		"LAZYMIND_AUTH_SERVICE_INTERNAL_TOKEN":  strings.TrimSpace(os.Getenv("LAZYMIND_AUTH_SERVICE_INTERNAL_TOKEN")),
		"COMPAT_KNOWLEDGE_SEARCH_DATASET_ID":    strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_DATASET_ID")),
		"COMPAT_KNOWLEDGE_SEARCH_DOCUMENT_ID":   strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_DOCUMENT_ID")),
		"COMPAT_KNOWLEDGE_SEARCH_QUERY":         strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_QUERY")),
		"COMPAT_KNOWLEDGE_SEARCH_EXPECTED_TEXT": strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_EXPECTED_TEXT")),
	}
	if strings.TrimSpace(os.Getenv("ACL_DB_DRIVER")) != orm.DriverPostgres {
		t.Skip("set ACL_DB_DRIVER=postgres for real knowledge search integration")
	}
	missing := make([]string, 0, len(values))
	for key, value := range values {
		if value == "" {
			missing = append(missing, key)
		}
	}
	if len(missing) > 0 {
		t.Skipf("real knowledge search integration requires non-production indexed fixture and env: %s", strings.Join(missing, ", "))
	}
	return realSearchFixture{
		userID: values["COMPAT_TEST_USER_ID"], tenantID: values["COMPAT_TEST_TENANT_ID"], dsn: values["ACL_DB_DSN"],
		searchURL: values["LAZYMIND_CHAT_SERVICE_URL"], datasetID: values["COMPAT_KNOWLEDGE_SEARCH_DATASET_ID"],
		documentID: values["COMPAT_KNOWLEDGE_SEARCH_DOCUMENT_ID"], query: values["COMPAT_KNOWLEDGE_SEARCH_QUERY"],
		marker: values["COMPAT_KNOWLEDGE_SEARCH_EXPECTED_TEXT"],
	}, true
}

func hasExpectedSearchFixtureHit(result compatknowledge.SearchResult, fixture realSearchFixture) bool {
	for _, hit := range result.Hits {
		if hit.KnowledgeID == fixture.datasetID && hit.DocumentID == fixture.documentID && strings.Contains(hit.Text, fixture.marker) {
			return true
		}
	}
	return false
}
