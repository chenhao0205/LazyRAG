package integration_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"gorm.io/gorm"
	"lazymind/core/acl"
	"lazymind/core/common/orm"
	"lazymind/core/common/readonlyorm"
	"lazymind/core/compat/contract"
	adaptercore "lazymind/core/compat/internal/adapters/core"
	compatknowledge "lazymind/core/compat/knowledge"
	compatruntime "lazymind/core/compat/runtime"
	"lazymind/core/doc"
)

func TestKnowledgeRuntimeWithRealPostgreSQLDocumentGet(t *testing.T) {
	if strings.TrimSpace(os.Getenv("COMPAT_INTEGRATION")) != "1" ||
		strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_DOCUMENT_INTEGRATION")) != "1" {
		t.Skip("set COMPAT_INTEGRATION=1 and COMPAT_KNOWLEDGE_DOCUMENT_INTEGRATION=1 to run knowledge document integration tests")
	}

	coreDriver, coreDSN := dbConfigFromCoreEnv(t)
	if coreDriver != orm.DriverPostgres {
		t.Fatalf("Knowledge document integration requires ACL_DB_DRIVER=%q, got %q", orm.DriverPostgres, coreDriver)
	}
	lazyDriver, lazyDSN := readonlyDBConfigFromCoreEnv(t)
	if lazyDriver != orm.DriverPostgres {
		t.Fatalf("Knowledge document integration requires readonly driver=%q, got %q", orm.DriverPostgres, lazyDriver)
	}
	if coreDriver == lazyDriver && coreDSN == lazyDSN {
		t.Fatal("Knowledge document integration requires explicit separated core and lazy DB configuration")
	}

	coreDB, err := orm.Connect(coreDriver, coreDSN)
	if err != nil {
		t.Fatalf("connect core db: %v", err)
	}
	coreSQL, err := coreDB.DB.DB()
	if err != nil {
		t.Fatalf("get core sql db: %v", err)
	}
	t.Cleanup(func() { _ = coreSQL.Close() })
	lazyDB, err := orm.Connect(lazyDriver, lazyDSN)
	if err != nil {
		t.Fatalf("connect lazy db: %v", err)
	}
	lazySQL, err := lazyDB.DB.DB()
	if err != nil {
		t.Fatalf("get lazy sql db: %v", err)
	}
	t.Cleanup(func() { _ = lazySQL.Close() })
	if coreSQL == lazySQL {
		t.Fatal("core and lazy sql.DB handles must be different")
	}

	acl.InitStore(coreDB)
	coreTx := coreDB.Begin()
	if coreTx.Error != nil {
		t.Fatalf("begin core tx: %v", coreTx.Error)
	}
	t.Cleanup(func() { _ = coreTx.Rollback().Error })
	lazyTx := lazyDB.Begin()
	if lazyTx.Error != nil {
		t.Fatalf("begin lazy tx: %v", lazyTx.Error)
	}
	t.Cleanup(func() { _ = lazyTx.Rollback().Error })

	seed := fmt.Sprintf("d%x", time.Now().UnixNano())
	userID := seed + "_user"
	tenantID := seed + "_tenant"
	var scanCalls atomic.Int32
	var invalidScanIdentity atomic.Bool
	scanServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/scan/internal/source-access/by-dataset:batch" {
			http.NotFound(w, r)
			return
		}
		if r.Header.Get("X-User-ID") != userID || r.Header.Get("X-Tenant-ID") != tenantID {
			invalidScanIdentity.Store(true)
			http.Error(w, "unexpected caller identity", http.StatusForbidden)
			return
		}
		scanCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"items":[]}`))
	}))
	t.Cleanup(scanServer.Close)
	t.Setenv("LAZYMIND_SCAN_CONTROL_PLANE_URL", scanServer.URL)

	otherUserID := seed + "_other_user"
	datasetID := seed + "_ds"
	otherDatasetID := seed + "_other_ds"
	documentID := seed + "_doc"
	otherDocumentID := seed + "_other_doc"
	lazyDocID := seed + "_lazy_doc"
	now := time.Now().UTC().Truncate(time.Microsecond)

	uploadRoot := t.TempDir()
	t.Setenv("LAZYMIND_UPLOAD_ROOT", uploadRoot)
	textPath := filepath.Join(uploadRoot, "document.txt")
	if err := os.WriteFile(textPath, []byte("hello knowledge document"), 0o600); err != nil {
		t.Fatalf("write text file: %v", err)
	}
	insertDataset(t, coreTx, datasetID, userID, tenantID, now)
	insertDataset(t, coreTx, otherDatasetID, userID, tenantID, now)
	insertDocument(t, coreTx, datasetID, documentID, lazyDocID, userID, "core display should lose to lazy", textPath, "text/plain; charset=utf-8", now)
	insertDocument(t, coreTx, otherDatasetID, otherDocumentID, seed+"_other_lazy", userID, "other doc", textPath, "text/plain; charset=utf-8", now)
	insertLazyDocument(t, lazyTx, lazyDocID, "lazy-title.md", "FILE_SYSTEM", "READY_FROM_LAZY", 321, now)

	service, err := doc.NewDocumentService(doc.DocumentServiceDeps{DB: coreTx, LazyDB: lazyTx})
	if err != nil {
		t.Fatalf("NewDocumentService: %v", err)
	}
	adapter, err := adaptercore.NewKnowledgeDocumentAdapter(service)
	if err != nil {
		t.Fatalf("NewKnowledgeDocumentAdapter: %v", err)
	}
	rt, err := compatruntime.New(compatruntime.Dependencies{KnowledgeDocument: adapter})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	if rt.Knowledge == nil {
		t.Fatal("Runtime.Knowledge is nil")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	callCtx := contract.CallContext{UserID: userID, TenantID: tenantID}

	t.Run("metadata dual db", func(t *testing.T) {
		got, err := rt.Knowledge.GetDocument(ctx, callCtx, compatknowledge.GetDocumentInput{
			KnowledgeID: datasetID,
			DocumentID:  documentID,
		})
		if err != nil {
			t.Fatalf("GetDocument metadata: %v", err)
		}
		docDetail := got.Document
		if docDetail.ID != documentID || docDetail.KnowledgeID != datasetID {
			t.Fatalf("identity = %#v, want core document/dataset IDs", docDetail)
		}
		if docDetail.Name != "core display should lose to lazy" || docDetail.Source != "FILE_SYSTEM" || docDetail.ParseStatus != "READY_FROM_LAZY" || docDetail.MIMEType != "text/plain; charset=utf-8" {
			t.Fatalf("metadata not mapped from Core/readonly DB: %#v", docDetail)
		}
		if docDetail.Content != nil || docDetail.ChunksPage != nil || len(docDetail.Chunks) != 0 {
			t.Fatalf("metadata-only should not include content/chunks: %#v", docDetail)
		}
		assertNoInternalDocumentFields(t, docDetail, uploadRoot, lazyDocID)
	})

	t.Run("permission and ownership", func(t *testing.T) {
		result, err := rt.Knowledge.GetDocument(ctx, contract.CallContext{UserID: otherUserID}, compatknowledge.GetDocumentInput{KnowledgeID: datasetID, DocumentID: documentID})
		assertCompatNotFound(t, result, err)
		result, err = rt.Knowledge.GetDocument(ctx, callCtx, compatknowledge.GetDocumentInput{KnowledgeID: datasetID, DocumentID: otherDocumentID})
		assertCompatNotFound(t, result, err)
		result, err = rt.Knowledge.GetDocument(ctx, callCtx, compatknowledge.GetDocumentInput{KnowledgeID: datasetID, DocumentID: seed + "_missing_doc"})
		assertCompatNotFound(t, result, err)
		result, err = rt.Knowledge.GetDocument(ctx, callCtx, compatknowledge.GetDocumentInput{KnowledgeID: datasetID, DocumentID: lazyDocID})
		assertCompatNotFound(t, result, err)
	})
	if invalidScanIdentity.Load() {
		t.Fatal("Scan source-access check did not receive the CallContext UserID and TenantID")
	}
	if scanCalls.Load() == 0 {
		t.Fatal("expected metadata reads to call deterministic Scan source-access check")
	}

	t.Logf("Knowledge Document integration core_driver=%s lazy_driver=%s dataset=%s document=%s", coreDriver, lazyDriver, datasetID, documentID)
}

func readonlyDBConfigFromCoreEnv(t *testing.T) (string, string) {
	t.Helper()
	driver := strings.TrimSpace(os.Getenv("LAZYMIND_READONLY_DB_DRIVER"))
	dsn := strings.TrimSpace(os.Getenv("LAZYMIND_READONLY_DB_DSN"))
	if driver == "" {
		driver = strings.TrimSpace(os.Getenv("LAZYMIND_LAZYLLM_DB_DRIVER"))
	}
	if dsn == "" {
		dsn = strings.TrimSpace(os.Getenv("LAZYMIND_LAZYLLM_DB_DSN"))
	}
	if driver == "" {
		t.Fatal("LAZYMIND_READONLY_DB_DRIVER or LAZYMIND_LAZYLLM_DB_DRIVER is required")
	}
	if dsn == "" {
		t.Fatal("LAZYMIND_READONLY_DB_DSN or LAZYMIND_LAZYLLM_DB_DSN is required")
	}
	return driver, dsn
}

func insertDataset(t *testing.T, db *gorm.DB, datasetID, userID, tenantID string, now time.Time) {
	t.Helper()
	if err := db.Create(&orm.Dataset{
		ID:           datasetID,
		KbID:         "kb-" + datasetID,
		DisplayName:  "Dataset " + datasetID,
		Desc:         "compat document integration",
		DatasetState: 0,
		ShareType:    0,
		TenantID:     tenantID,
		Type:         1,
		BaseModel: orm.BaseModel{
			CreateUserID:   userID,
			CreateUserName: userID + " name",
			CreatedAt:      now,
			UpdatedAt:      now,
		},
	}).Error; err != nil {
		t.Fatalf("insert dataset %s: %v", datasetID, err)
	}
}

func insertDocument(t *testing.T, db *gorm.DB, datasetID, documentID, lazyDocID, userID, displayName, storedPath, contentType string, now time.Time) {
	t.Helper()
	ext, err := json.Marshal(map[string]any{
		"stored_path":       storedPath,
		"stored_name":       filepath.Base(storedPath),
		"original_filename": filepath.Base(storedPath),
		"file_size":         fileSize(t, storedPath),
		"content_type":      contentType,
	})
	if err != nil {
		t.Fatalf("marshal document ext: %v", err)
	}
	if err := db.Create(&orm.Document{
		ID:           documentID,
		LazyllmDocID: lazyDocID,
		DatasetID:    datasetID,
		DisplayName:  displayName,
		Tags:         []byte(`["integration"]`),
		Ext:          ext,
		BaseModel: orm.BaseModel{
			CreateUserID:   userID,
			CreateUserName: userID + " name",
			CreatedAt:      now,
			UpdatedAt:      now,
		},
	}).Error; err != nil {
		t.Fatalf("insert document %s: %v", documentID, err)
	}
}

func insertLazyDocument(t *testing.T, db *gorm.DB, lazyDocID, filename, sourceType, uploadStatus string, size int, now time.Time) {
	t.Helper()
	if err := db.Table((readonlyorm.LazyLLMDocRow{}).TableName()).Create(&readonlyorm.LazyLLMDocRow{
		DocID:        lazyDocID,
		Filename:     filename,
		Path:         "/readonly/" + filename,
		UploadStatus: uploadStatus,
		SourceType:   sourceType,
		SizeBytes:    &size,
		CreatedAt:    now.Add(-time.Hour),
		UpdatedAt:    now.Add(-time.Minute),
	}).Error; err != nil {
		t.Fatalf("insert lazy document %s: %v", lazyDocID, err)
	}
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	return info.Size()
}

func assertCompatNotFound(t *testing.T, result compatknowledge.GetDocumentResult, err error) {
	t.Helper()
	if code, ok := contract.CodeOf(err); !ok || code != contract.NotFound {
		t.Fatalf("result=%#v code=%v ok=%v err=%v, want NOT_FOUND", result, code, ok, err)
	}
}

func assertNoInternalDocumentFields(t *testing.T, detail compatknowledge.DocumentDetail, uploadRoot, lazyDocID string) {
	t.Helper()
	raw, err := json.Marshal(detail)
	if err != nil {
		t.Fatalf("marshal document detail: %v", err)
	}
	text := string(raw)
	for _, forbidden := range []string{
		uploadRoot,
		lazyDocID,
		"lazyllm_doc_id",
		"FileSystemPath",
		"file_system_path",
		"StoredPath",
		"stored_path",
		"ParseStoredPath",
		"parse_stored_path",
		"SourceStoredPath",
		"source_stored_path",
	} {
		if strings.TrimSpace(forbidden) != "" && strings.Contains(text, forbidden) {
			t.Fatalf("document detail leaks forbidden value %q: %s", forbidden, text)
		}
	}
}
