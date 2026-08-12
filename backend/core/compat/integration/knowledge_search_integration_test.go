package integration_test

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"mime/multipart"
	"net/http"
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

	"gorm.io/gorm"
)

// TestKnowledgeRuntimeWithRealSearchBackend exercises the real PostgreSQL
// resolver/document mapper and the configured HTTP knowledge-search backend.
// The search fixture is provisioned by the explicitly selected non-production
// search environment because Core has no supported API for creating and
// deleting an indexed knowledge fixture synchronously.
func TestKnowledgeRuntimeWithRealSearchBackend(t *testing.T) {
	config, ok := loadRealSearchIntegrationConfig(t)
	if !ok {
		return
	}

	db, err := orm.Connect(orm.DriverPostgres, config.dsn)
	if err != nil {
		t.Fatalf("connect core db: %v", err)
	}
	sqlDB, err := db.DB.DB()
	if err != nil {
		t.Fatalf("get sql db: %v", err)
	}
	t.Cleanup(func() { _ = sqlDB.Close() })
	acl.InitStore(db)

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	fixture := createSearchFixture(t, ctx, db.DB, config)
	t.Cleanup(func() { cleanupSearchFixture(t, context.Background(), config, fixture) })

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
		KnowledgeIDs: []string{fixture.emptyDatasetID},
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
	userID         string
	tenantID       string
	dsn            string
	searchURL      string
	datasetID      string
	emptyDatasetID string
	documentID     string
	query          string
	marker         string
}

type realSearchIntegrationConfig struct {
	userID    string
	tenantID  string
	dsn       string
	coreURL   string
	searchURL string
	token     string
}

func loadRealSearchIntegrationConfig(t *testing.T) (realSearchIntegrationConfig, bool) {
	t.Helper()
	if strings.TrimSpace(os.Getenv("COMPAT_INTEGRATION")) != "1" ||
		strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_INTEGRATION")) != "1" {
		t.Skip("set COMPAT_INTEGRATION=1 and COMPAT_KNOWLEDGE_SEARCH_INTEGRATION=1 to run real knowledge search integration")
	}
	values := map[string]string{
		"COMPAT_TEST_USER_ID":                  strings.TrimSpace(os.Getenv("COMPAT_TEST_USER_ID")),
		"COMPAT_TEST_TENANT_ID":                strings.TrimSpace(os.Getenv("COMPAT_TEST_TENANT_ID")),
		"ACL_DB_DSN":                           strings.TrimSpace(os.Getenv("ACL_DB_DSN")),
		"COMPAT_KNOWLEDGE_SEARCH_CORE_URL":     strings.TrimSpace(os.Getenv("COMPAT_KNOWLEDGE_SEARCH_CORE_URL")),
		"LAZYMIND_CHAT_SERVICE_URL":            strings.TrimSpace(os.Getenv("LAZYMIND_CHAT_SERVICE_URL")),
		"LAZYMIND_AUTH_SERVICE_INTERNAL_TOKEN": strings.TrimSpace(os.Getenv("LAZYMIND_AUTH_SERVICE_INTERNAL_TOKEN")),
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
		t.Skipf("real knowledge search integration requires an isolated Core/Search environment and env: %s", strings.Join(missing, ", "))
	}
	return realSearchIntegrationConfig{
		userID: values["COMPAT_TEST_USER_ID"], tenantID: values["COMPAT_TEST_TENANT_ID"], dsn: values["ACL_DB_DSN"],
		coreURL: values["COMPAT_KNOWLEDGE_SEARCH_CORE_URL"], searchURL: values["LAZYMIND_CHAT_SERVICE_URL"], token: values["LAZYMIND_AUTH_SERVICE_INTERNAL_TOKEN"],
	}, true
}

// createSearchFixture uses the same Core Dataset/upload/task APIs as production.
// It deliberately does not insert a synthetic Dataset or an OpenSearch document.
func createSearchFixture(t *testing.T, ctx context.Context, db *gorm.DB, config realSearchIntegrationConfig) realSearchFixture {
	t.Helper()
	marker := "compat-knowledge-search-e2e-" + uuid.NewString()
	displayName := "compat-search-e2e-" + uuid.NewString()[:12]
	client := &http.Client{Timeout: 15 * time.Second}
	fixture := realSearchFixture{userID: config.userID, tenantID: config.tenantID, dsn: config.dsn, searchURL: config.searchURL, datasetID: createSearchFixtureDataset(t, ctx, client, config, displayName), query: marker, marker: marker}
	cleanupNeeded := true
	defer func() {
		if cleanupNeeded {
			cleanupSearchFixture(t, context.Background(), config, fixture)
		}
	}()

	// CreateDataset currently has no tenant input.  The production API creates the
	// Dataset/KB and ACL rows; this test-only update supplies the tenant metadata
	// consumed by the Catalog/Scan boundary before the document is uploaded.
	if err := db.WithContext(ctx).Model(&orm.Dataset{}).Where("id = ?", fixture.datasetID).Update("tenant_id", config.tenantID).Error; err != nil {
		t.Fatalf("set fixture tenant metadata: %v", err)
	}
	var dataset orm.Dataset
	if err := db.WithContext(ctx).Where("id = ? AND deleted_at IS NULL", fixture.datasetID).First(&dataset).Error; err != nil {
		t.Fatalf("load created dataset: %v", err)
	}
	if dataset.KbID == "" || dataset.CreateUserID != config.userID || dataset.TenantID != config.tenantID {
		t.Fatalf("created dataset identity/kb mismatch: %#v", dataset)
	}
	fixture.emptyDatasetID = createSearchFixtureDataset(t, ctx, client, config, "compat-search-empty-"+uuid.NewString()[:12])
	if err := db.WithContext(ctx).Model(&orm.Dataset{}).Where("id = ?", fixture.emptyDatasetID).Update("tenant_id", config.tenantID).Error; err != nil {
		t.Fatalf("set empty fixture tenant metadata: %v", err)
	}

	uploadID := uploadSearchFixtureDocument(t, ctx, client, config, fixture.datasetID, marker)
	var taskCreated struct {
		Tasks []struct {
			TaskID     string `json:"task_id"`
			DocumentID string `json:"document_id"`
		} `json:"tasks"`
	}
	coreJSON(t, ctx, client, config, http.MethodPost, "/datasets/"+fixture.datasetID+"/tasks", map[string]any{
		"items": []any{map[string]any{
			"upload_file_id": uploadID,
			"task":           map[string]any{"task_type": "TASK_TYPE_PARSE_UPLOADED", "data_source_type": "LOCAL_FILE", "display_name": "knowledge-search-e2e.txt"},
		}},
	}, &taskCreated)
	if len(taskCreated.Tasks) != 1 || taskCreated.Tasks[0].TaskID == "" || taskCreated.Tasks[0].DocumentID == "" {
		t.Fatalf("Core CreateTask response = %#v", taskCreated)
	}
	fixture.documentID = taskCreated.Tasks[0].DocumentID
	taskID := taskCreated.Tasks[0].TaskID
	var started struct {
		StartedCount int `json:"started_count"`
	}
	coreJSON(t, ctx, client, config, http.MethodPost, "/datasets/"+fixture.datasetID+"/tasks:start", map[string]any{"task_ids": []string{taskID}}, &started)
	if started.StartedCount != 1 {
		t.Fatalf("Core StartTask started_count = %d, want 1", started.StartedCount)
	}
	waitForSearchFixtureIndexed(t, ctx, client, config, db, fixture, taskID)
	cleanupNeeded = false
	return fixture
}

func createSearchFixtureDataset(t *testing.T, ctx context.Context, client *http.Client, config realSearchIntegrationConfig, displayName string) string {
	t.Helper()
	var created struct {
		DatasetID string `json:"dataset_id"`
	}
	coreJSON(t, ctx, client, config, http.MethodPost, "/datasets", map[string]any{
		"display_name": displayName,
		"desc":         "isolated knowledge search integration fixture",
		"tags":         []string{"compat-integration", "search-e2e"},
		"algo":         map[string]string{"algo_id": "general_algo"},
	}, &created)
	if created.DatasetID == "" {
		t.Fatal("Core CreateDataset response has empty dataset_id")
	}
	return created.DatasetID
}

func uploadSearchFixtureDocument(t *testing.T, ctx context.Context, client *http.Client, config realSearchIntegrationConfig, datasetID, marker string) string {
	t.Helper()
	var body bytes.Buffer
	form := multipart.NewWriter(&body)
	part, err := form.CreateFormFile("files", "knowledge-search-e2e.txt")
	if err != nil {
		t.Fatalf("create fixture multipart: %v", err)
	}
	if _, err := io.WriteString(part, "Knowledge Search MCP E2E Fixture\n\nUnique token: "+marker+"\n"); err != nil {
		t.Fatalf("write fixture multipart: %v", err)
	}
	if err := form.Close(); err != nil {
		t.Fatalf("close fixture multipart: %v", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(config.coreURL, "/")+"/datasets/"+datasetID+"/uploads", &body)
	if err != nil {
		t.Fatalf("build fixture upload request: %v", err)
	}
	applyFixtureIdentity(req, config)
	req.Header.Set("Content-Type", form.FormDataContentType())
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Core UploadFile: %v", err)
	}
	defer resp.Body.Close()
	var payload struct {
		Files []struct {
			UploadFileID string `json:"upload_file_id"`
		} `json:"files"`
	}
	decodeFixtureResponse(t, resp, &payload)
	if len(payload.Files) != 1 || payload.Files[0].UploadFileID == "" {
		t.Fatalf("Core UploadFile response = %#v", payload)
	}
	return payload.Files[0].UploadFileID
}

func waitForSearchFixtureIndexed(t *testing.T, ctx context.Context, client *http.Client, config realSearchIntegrationConfig, db *gorm.DB, fixture realSearchFixture, taskID string) {
	t.Helper()
	deadline := time.Now().Add(90 * time.Second)
	var lastTaskState string
	var lastSearchError string
	for time.Now().Before(deadline) {
		var task struct {
			TaskState string `json:"task_state"`
			ErrMsg    string `json:"err_msg"`
		}
		coreJSON(t, ctx, client, config, http.MethodGet, "/datasets/"+fixture.datasetID+"/tasks/"+taskID, nil, &task)
		lastTaskState = task.TaskState
		if strings.EqualFold(task.TaskState, "FAILED") || strings.EqualFold(task.TaskState, "CANCELED") {
			t.Fatalf("fixture parse task ended %s: %s", task.TaskState, task.ErrMsg)
		}
		if strings.EqualFold(task.TaskState, "SUCCESS") {
			var document orm.Document
			if err := db.WithContext(ctx).Where("id = ? AND dataset_id = ? AND deleted_at IS NULL", fixture.documentID, fixture.datasetID).First(&document).Error; err != nil {
				t.Fatalf("load indexed fixture document: %v", err)
			}
			if strings.TrimSpace(document.LazyllmDocID) == "" {
				lastSearchError = "Core document lazyllm_doc_id is empty"
			} else if searchFixtureAvailable(ctx, client, config, fixture) {
				return
			} else {
				lastSearchError = "search probe has not returned the fixture hit"
			}
		}
		select {
		case <-ctx.Done():
			t.Fatalf("fixture indexing context ended: task_state=%s search=%s: %v", lastTaskState, lastSearchError, ctx.Err())
		case <-time.After(500 * time.Millisecond):
		}
	}
	t.Fatalf("fixture indexing timed out: task_state=%s search=%s", lastTaskState, lastSearchError)
}

func searchFixtureAvailable(ctx context.Context, client *http.Client, config realSearchIntegrationConfig, fixture realSearchFixture) bool {
	body, err := json.Marshal(map[string]any{"user_id": config.userID, "query": fixture.query, "kb_ids": []string{fixture.datasetID}, "top_k": 10})
	if err != nil {
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(config.searchURL, "/")+"/internal/knowledge:search", bytes.NewReader(body))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-LazyMind-Internal-Token", config.token)
	resp, err := client.Do(req)
	if err != nil || resp == nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	var payload struct {
		Hits []struct {
			KBID  string `json:"kb_id"`
			DocID string `json:"doc_id"`
			Text  string `json:"text"`
		} `json:"hits"`
	}
	if json.NewDecoder(resp.Body).Decode(&payload) != nil {
		return false
	}
	for _, hit := range payload.Hits {
		if hit.KBID == fixture.datasetID && hit.DocID == fixture.documentID && strings.Contains(hit.Text, fixture.marker) {
			return true
		}
	}
	return false
}

func cleanupSearchFixture(t *testing.T, ctx context.Context, config realSearchIntegrationConfig, fixture realSearchFixture) {
	t.Helper()
	cleanupSearchFixtureDataset(t, ctx, config, fixture.datasetID)
	cleanupSearchFixtureDataset(t, ctx, config, fixture.emptyDatasetID)
	if fixture.datasetID != "" {
		t.Logf("search fixture cleanup completed: dataset_id=%s document_id=%s", fixture.datasetID, fixture.documentID)
	}
}

func cleanupSearchFixtureDataset(t *testing.T, ctx context.Context, config realSearchIntegrationConfig, datasetID string) {
	t.Helper()
	if datasetID == "" {
		return
	}
	client := &http.Client{Timeout: 15 * time.Second}
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, strings.TrimRight(config.coreURL, "/")+"/datasets/"+datasetID, nil)
	if err != nil {
		t.Errorf("build fixture cleanup request for %s: %v", datasetID, err)
		return
	}
	applyFixtureIdentity(req, config)
	resp, err := client.Do(req)
	if err != nil {
		t.Errorf("delete search fixture dataset %s: %v", datasetID, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		t.Errorf("delete search fixture dataset %s: status=%d body=%s", datasetID, resp.StatusCode, strings.TrimSpace(string(data)))
		return
	}
}

func coreJSON(t *testing.T, ctx context.Context, client *http.Client, config realSearchIntegrationConfig, method, path string, input any, output any) {
	t.Helper()
	var reader io.Reader
	if input != nil {
		body, err := json.Marshal(input)
		if err != nil {
			t.Fatalf("marshal Core %s %s: %v", method, path, err)
		}
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, strings.TrimRight(config.coreURL, "/")+path, reader)
	if err != nil {
		t.Fatalf("build Core %s %s: %v", method, path, err)
	}
	applyFixtureIdentity(req, config)
	if input != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Core %s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	decodeFixtureResponse(t, resp, output)
}

func applyFixtureIdentity(req *http.Request, config realSearchIntegrationConfig) {
	req.Header.Set("X-User-Id", config.userID)
	req.Header.Set("X-User-Name", "Compat Search E2E")
	req.Header.Set("X-Tenant-Id", config.tenantID)
}

func decodeFixtureResponse(t *testing.T, resp *http.Response, output any) {
	t.Helper()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		t.Fatalf("Core fixture API status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(data)))
	}
	if output != nil {
		if err := json.NewDecoder(resp.Body).Decode(output); err != nil {
			t.Fatalf("decode Core fixture API response: %v", err)
		}
	}
}

func hasExpectedSearchFixtureHit(result compatknowledge.SearchResult, fixture realSearchFixture) bool {
	for _, hit := range result.Hits {
		if hit.KnowledgeID == fixture.datasetID && hit.DocumentID == fixture.documentID && strings.Contains(hit.Text, fixture.marker) {
			return true
		}
	}
	return false
}
