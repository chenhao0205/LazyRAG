package integration_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"

	"lazymind/core/acl"
	"lazymind/core/common/orm"
	"lazymind/core/compat/contract"
	adaptercore "lazymind/core/compat/internal/adapters/core"
	compatknowledge "lazymind/core/compat/knowledge"
	compatruntime "lazymind/core/compat/runtime"

	"gorm.io/gorm"
)

// TestKnowledgeRuntimeWithRealPostgreSQLCatalog verifies the production-like
// Catalog wiring with self-contained rows in an explicitly gated test database.
func TestKnowledgeRuntimeWithRealPostgreSQLCatalog(t *testing.T) {
	if strings.TrimSpace(os.Getenv("COMPAT_INTEGRATION")) != "1" {
		t.Skip("set COMPAT_INTEGRATION=1 to run compat integration tests")
	}
	userID := strings.TrimSpace(os.Getenv("COMPAT_TEST_USER_ID"))
	if userID == "" {
		t.Skip("set COMPAT_TEST_USER_ID to run knowledge catalog integration")
	}
	driver := strings.TrimSpace(os.Getenv("ACL_DB_DRIVER"))
	dsn := strings.TrimSpace(os.Getenv("ACL_DB_DSN"))
	if driver == "" || dsn == "" {
		t.Skip("set ACL_DB_DRIVER and ACL_DB_DSN to run knowledge catalog integration")
	}
	if driver != orm.DriverPostgres {
		t.Fatalf("Knowledge PostgreSQL integration requires ACL_DB_DRIVER=%q, got %q", orm.DriverPostgres, driver)
	}

	db, err := orm.Connect(driver, dsn)
	if err != nil {
		t.Fatalf("connect core db: %v", err)
	}
	sqlDB, err := db.DB.DB()
	if err != nil {
		t.Fatalf("get sql db: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	acl.InitStore(db)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	prefix := uniqueIntegrationPrefix("knowledge-catalog")
	tenantID := uuid.NewString()
	ids := []string{uuid.NewString(), uuid.NewString(), uuid.NewString()}
	createKnowledgeCatalogFixtures(t, ctx, db.DB, userID, tenantID, prefix, ids)
	t.Cleanup(func() { cleanupKnowledgeCatalogFixtures(t, context.Background(), db.DB, ids) })
	installKnowledgeCatalogScanStub(t, userID, tenantID)

	adapter, err := adaptercore.NewKnowledgeCatalogAdapterForDB(db.DB)
	if err != nil {
		t.Fatalf("NewKnowledgeCatalogAdapterForDB: %v", err)
	}
	rt, err := compatruntime.New(compatruntime.Dependencies{KnowledgeCatalog: adapter})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	if rt.Knowledge == nil {
		t.Fatal("Runtime.Knowledge is nil")
	}
	callCtx := contract.CallContext{UserID: userID, TenantID: tenantID}

	firstPage, err := rt.Knowledge.List(ctx, callCtx, compatknowledge.ListInput{
		Keyword: prefix,
		Tags:    []string{"compat-integration", prefix},
		Page:    contract.PageRequest{PageSize: 1},
	})
	if err != nil {
		t.Fatalf("Knowledge.List first page: %v", err)
	}
	if len(firstPage.Items) != 1 || firstPage.Page.Total == nil || *firstPage.Page.Total != 3 || firstPage.Page.NextPageToken == "" {
		t.Fatalf("Knowledge.List first page = %#v", firstPage)
	}
	first := firstPage.Items[0]
	if first.ID != ids[2] || first.Name != prefix+"-three" || first.Description != "compat knowledge catalog fixture" || len(first.Tags) != 2 {
		t.Fatalf("first knowledge metadata = %#v", first)
	}

	secondPage, err := rt.Knowledge.List(ctx, callCtx, compatknowledge.ListInput{
		Keyword: prefix,
		Tags:    []string{"compat-integration", prefix},
		Page:    contract.PageRequest{PageSize: 2, PageToken: firstPage.Page.NextPageToken},
	})
	if err != nil {
		t.Fatalf("Knowledge.List second page: %v", err)
	}
	if len(secondPage.Items) != 2 || secondPage.Page.Total == nil || *secondPage.Page.Total != 3 || secondPage.Page.NextPageToken != "" {
		t.Fatalf("Knowledge.List second page = %#v", secondPage)
	}
	for _, item := range secondPage.Items {
		if item.ID == first.ID {
			t.Fatalf("pagination repeated knowledge ID %q", item.ID)
		}
	}

	got, err := rt.Knowledge.Get(ctx, callCtx, compatknowledge.GetInput{KnowledgeID: first.ID})
	if err != nil {
		t.Fatalf("Knowledge.Get(%q): %v", first.ID, err)
	}
	if got.Knowledge.ID != first.ID || got.Knowledge.Name != first.Name || got.Knowledge.Description != first.Description {
		t.Fatalf("Knowledge.Get metadata = %#v, list item = %#v", got.Knowledge, first)
	}
	_, err = rt.Knowledge.Get(ctx, callCtx, compatknowledge.GetInput{KnowledgeID: uuid.NewString()})
	if code, ok := contract.CodeOf(err); !ok || code != contract.NotFound {
		t.Fatalf("Knowledge.Get missing error = %v, want NOT_FOUND", err)
	}
	t.Logf("Knowledge Catalog PostgreSQL integration: fixture=%s total=%d tenant_propagation=verified", first.ID, *firstPage.Page.Total)
}

func createKnowledgeCatalogFixtures(t *testing.T, ctx context.Context, db *gorm.DB, userID, tenantID, prefix string, ids []string) {
	t.Helper()
	if len(ids) != 3 {
		t.Fatalf("fixture IDs = %d, want 3", len(ids))
	}
	base := time.Now().UTC().Add(-3 * time.Second)
	for index, suffix := range []string{"one", "two", "three"} {
		tags, err := json.Marshal(map[string]any{"tags": []string{"compat-integration", prefix}})
		if err != nil {
			t.Fatalf("marshal dataset tags: %v", err)
		}
		updatedAt := base.Add(time.Duration(index) * time.Second)
		row := orm.Dataset{
			ID:                     ids[index],
			KbID:                   "kb-" + ids[index],
			DisplayName:            prefix + "-" + suffix,
			Desc:                   "compat knowledge catalog fixture",
			CoverImage:             "",
			ResourceUID:            "",
			BucketName:             "",
			OssPath:                "",
			DatasetInfo:            json.RawMessage(`{}`),
			DatasetState:           0,
			EmbeddingModel:         "",
			EmbeddingModelProvider: "",
			ShareType:              0,
			TenantID:               tenantID,
			Type:                   1,
			Ext:                    tags,
			BaseModel: orm.BaseModel{
				CreateUserID: userID, CreateUserName: userID, CreatedAt: updatedAt, UpdatedAt: updatedAt,
			},
		}
		if err := db.WithContext(ctx).Create(&row).Error; err != nil {
			t.Fatalf("create knowledge fixture %q: %v", row.ID, err)
		}
	}
}

func cleanupKnowledgeCatalogFixtures(t *testing.T, ctx context.Context, db *gorm.DB, ids []string) {
	t.Helper()
	if len(ids) == 0 {
		return
	}
	if err := db.WithContext(ctx).Where("id IN ?", ids).Delete(&orm.Dataset{}).Error; err != nil {
		t.Errorf("cleanup knowledge catalog fixtures: %v", err)
		return
	}
	t.Logf("cleanup completed for knowledge catalog fixtures: %s", strings.Join(ids, ","))
}

func installKnowledgeCatalogScanStub(t *testing.T, expectedUserID, expectedTenantID string) {
	t.Helper()
	var mu sync.Mutex
	var callerChecks int
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		switch request.URL.Path {
		case "/api/scan/internal/source-access/by-dataset:batch":
			if request.Header.Get("X-User-ID") != expectedUserID || request.Header.Get("X-Tenant-ID") != expectedTenantID {
				http.Error(writer, "unexpected caller identity", http.StatusForbidden)
				return
			}
			mu.Lock()
			callerChecks++
			mu.Unlock()
			var payload struct {
				DatasetIDs []string `json:"dataset_ids"`
			}
			if err := json.NewDecoder(request.Body).Decode(&payload); err != nil {
				http.Error(writer, "invalid request", http.StatusBadRequest)
				return
			}
			items := make([]map[string]any, 0, len(payload.DatasetIDs))
			for _, id := range payload.DatasetIDs {
				items = append(items, map[string]any{"dataset_id": id, "exists": true, "allowed": true})
			}
			writeKnowledgeCatalogScanJSON(t, writer, map[string]any{"items": items})
		case "/api/scan/internal/sources/by-datasets":
			writeKnowledgeCatalogScanJSON(t, writer, map[string]any{"source_map": map[string]bool{}})
		default:
			if strings.HasPrefix(request.URL.Path, "/api/scan/internal/sources/by-dataset/") {
				writeKnowledgeCatalogScanJSON(t, writer, map[string]any{"source": map[string]any{}})
				return
			}
			http.NotFound(writer, request)
		}
	}))
	t.Cleanup(func() {
		server.Close()
		mu.Lock()
		defer mu.Unlock()
		if callerChecks == 0 {
			t.Error("Scan stub did not receive a caller-aware source access request")
		}
	})
	t.Setenv("LAZYMIND_SCAN_CONTROL_PLANE_URL", server.URL)
}

func writeKnowledgeCatalogScanJSON(t *testing.T, writer http.ResponseWriter, value any) {
	t.Helper()
	writer.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(writer).Encode(value); err != nil {
		t.Errorf("encode Scan stub response: %v", err)
	}
}
