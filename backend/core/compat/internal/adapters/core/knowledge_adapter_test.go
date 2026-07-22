package core

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"gorm.io/gorm"

	"lazymind/core/acl"
	"lazymind/core/common/orm"
	"lazymind/core/compat/contract"
	compatknowledge "lazymind/core/compat/knowledge"
	"lazymind/core/doc"
)

type fakeDatasetCatalogService struct {
	listReq doc.DatasetListRequest
	getReq  doc.DatasetGetRequest
	listErr error
	getErr  error
}

func (s *fakeDatasetCatalogService) ListDatasets(ctx context.Context, req doc.DatasetListRequest) (doc.DatasetListResult, error) {
	s.listReq = req
	if s.listErr != nil {
		return doc.DatasetListResult{}, s.listErr
	}
	return doc.DatasetListResult{}, nil
}

func (s *fakeDatasetCatalogService) GetDataset(ctx context.Context, req doc.DatasetGetRequest) (doc.Dataset, error) {
	s.getReq = req
	if s.getErr != nil {
		return doc.Dataset{}, s.getErr
	}
	return doc.Dataset{DatasetID: req.DatasetID}, nil
}

func TestKnowledgeAdapterPassesUserIDToDatasetService(t *testing.T) {
	service := &fakeDatasetCatalogService{}
	adapter, err := NewKnowledgeCatalogAdapter(service)
	if err != nil {
		t.Fatalf("NewKnowledgeCatalogAdapter: %v", err)
	}
	_, err = adapter.List(context.Background(), contract.CallContext{UserID: " user-1 "}, compatknowledge.ListInput{})
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}
	if service.listReq.UserID != "user-1" {
		t.Fatalf("List UserID = %q, want user-1", service.listReq.UserID)
	}
	_, err = adapter.Get(context.Background(), contract.CallContext{UserID: " user-1 "}, compatknowledge.GetInput{KnowledgeID: " ds-1 "})
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if service.getReq.UserID != "user-1" || service.getReq.DatasetID != "ds-1" {
		t.Fatalf("Get req = %#v, want user-1/ds-1", service.getReq)
	}
}

func TestKnowledgeAdapterMapsDatasetUnavailable(t *testing.T) {
	service := &fakeDatasetCatalogService{listErr: &doc.DatasetServiceError{Code: doc.DatasetServiceUnavailable, Message: "db unavailable"}}
	adapter, err := NewKnowledgeCatalogAdapter(service)
	if err != nil {
		t.Fatalf("NewKnowledgeCatalogAdapter: %v", err)
	}
	_, err = adapter.List(context.Background(), contract.CallContext{UserID: "user-1"}, compatknowledge.ListInput{})
	if code, ok := contract.CodeOf(err); !ok || code != contract.BackendUnavailable {
		t.Fatalf("unavailable code = %v, %v; want BACKEND_UNAVAILABLE", code, ok)
	}
}

func TestKnowledgeAdapterPassesUserIDAndMapsDatasetFields(t *testing.T) {
	db := newKnowledgeAdapterTestDB(t)
	now := time.Date(2026, 7, 22, 10, 0, 0, 0, time.UTC)
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{
		ID:          "ds-owned",
		KbID:        "kb-external",
		UserID:      "user-1",
		Name:        "Product Docs",
		Description: "API references",
		Tags:        []string{"api", "release"},
		UpdatedAt:   now,
	})
	seedKnowledgeDocument(t, db, "doc-file", "ds-owned", "", 12, false)
	seedKnowledgeDocument(t, db, "doc-folder", "ds-owned", "", 50, true)
	adapter := mustKnowledgeAdapterForDB(t, db)

	result, err := adapter.Get(context.Background(), contract.CallContext{UserID: " user-1 "}, compatknowledge.GetInput{KnowledgeID: "ds-owned"})
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if result.ID != "ds-owned" {
		t.Fatalf("ID = %q, want datasets.id ds-owned", result.ID)
	}
	if result.ID == "kb-external" {
		t.Fatalf("ID must not use kb_id")
	}
	if result.Name != "Product Docs" || result.Description != "API references" {
		t.Fatalf("summary = %#v, want dataset metadata", result)
	}
	if len(result.Tags) != 2 || result.Tags[0] != "api" || result.Tags[1] != "release" {
		t.Fatalf("tags = %#v, want api/release", result.Tags)
	}
	if result.DocumentCount != 1 || result.DocumentSizeBytes != 12 {
		t.Fatalf("stats count=%d size=%d, want 1/12", result.DocumentCount, result.DocumentSizeBytes)
	}
	if !result.UpdatedAt.Equal(now) {
		t.Fatalf("UpdatedAt = %v, want %v", result.UpdatedAt, now)
	}
}

func TestKnowledgeAdapterListFiltersKeywordTagsAndPaginates(t *testing.T) {
	db := newKnowledgeAdapterTestDB(t)
	base := time.Date(2026, 7, 22, 10, 0, 0, 0, time.UTC)
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{ID: "ds-new", UserID: "user-1", Name: "Alpha Docs", Description: "Runbook", Tags: []string{"api", "team"}, UpdatedAt: base})
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{ID: "ds-mid", UserID: "user-1", Name: "Beta", Description: "alpha notes", Tags: []string{"api", "team"}, UpdatedAt: base.Add(-time.Hour)})
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{ID: "ds-old", UserID: "user-1", Name: "Gamma", Description: "alpha notes", Tags: []string{"api", "team"}, UpdatedAt: base.Add(-2 * time.Hour)})
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{ID: "ds-tag-miss", UserID: "user-1", Name: "Alpha Missing Tag", Tags: []string{"api"}, UpdatedAt: base.Add(time.Hour)})
	adapter := mustKnowledgeAdapterForDB(t, db)

	first, err := adapter.List(context.Background(), contract.CallContext{UserID: "user-1"}, compatknowledge.ListInput{
		Keyword: " alpha ",
		Tags:    []string{"team"},
		Page:    contract.PageRequest{PageSize: 2},
	})
	if err != nil {
		t.Fatalf("List first returned error: %v", err)
	}
	if len(first.Items) != 2 || first.Items[0].ID != "ds-new" || first.Items[1].ID != "ds-mid" {
		t.Fatalf("first items = %#v, want ds-new/ds-mid", first.Items)
	}
	if first.Page.Total == nil || *first.Page.Total != 3 {
		t.Fatalf("total = %v, want 3", first.Page.Total)
	}
	if first.Page.NextPageToken == "" {
		t.Fatalf("NextPageToken is empty")
	}

	second, err := adapter.List(context.Background(), contract.CallContext{UserID: "user-1"}, compatknowledge.ListInput{
		Keyword: "alpha",
		Tags:    []string{"team"},
		Page:    contract.PageRequest{PageSize: 2, PageToken: first.Page.NextPageToken},
	})
	if err != nil {
		t.Fatalf("List second returned error: %v", err)
	}
	if len(second.Items) != 1 || second.Items[0].ID != "ds-old" {
		t.Fatalf("second items = %#v, want ds-old", second.Items)
	}
}

func TestKnowledgeAdapterListUsesDocumentStats(t *testing.T) {
	db := newKnowledgeAdapterTestDB(t)
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{ID: "ds-stats", UserID: "user-1", Name: "Stats", UpdatedAt: time.Now().UTC()})
	seedKnowledgeDocument(t, db, "doc-a", "ds-stats", "", 11, false)
	seedKnowledgeDocument(t, db, "doc-b", "ds-stats", "", 13, false)
	seedKnowledgeDocument(t, db, "folder", "ds-stats", "", 99, true)
	adapter := mustKnowledgeAdapterForDB(t, db)

	result, err := adapter.List(context.Background(), contract.CallContext{UserID: "user-1"}, compatknowledge.ListInput{})
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}
	if len(result.Items) != 1 {
		t.Fatalf("items = %#v, want one dataset", result.Items)
	}
	if result.Items[0].DocumentCount != 2 || result.Items[0].DocumentSizeBytes != 24 {
		t.Fatalf("stats count=%d size=%d, want 2/24", result.Items[0].DocumentCount, result.Items[0].DocumentSizeBytes)
	}
}

func TestKnowledgeAdapterNotFoundAndUserIsolation(t *testing.T) {
	db := newKnowledgeAdapterTestDB(t)
	seedKnowledgeDataset(t, db, knowledgeDatasetSeed{ID: "ds-user-1", UserID: "user-1", Name: "Private", UpdatedAt: time.Now().UTC()})
	adapter := mustKnowledgeAdapterForDB(t, db)

	_, err := adapter.Get(context.Background(), contract.CallContext{UserID: "user-1"}, compatknowledge.GetInput{KnowledgeID: "missing"})
	if code, ok := contract.CodeOf(err); !ok || code != contract.NotFound {
		t.Fatalf("missing code = %v, %v; want NOT_FOUND", code, ok)
	}
	_, err = adapter.Get(context.Background(), contract.CallContext{UserID: "user-2"}, compatknowledge.GetInput{KnowledgeID: "ds-user-1"})
	if code, ok := contract.CodeOf(err); !ok || code != contract.NotFound {
		t.Fatalf("isolated get code = %v, %v; want NOT_FOUND", code, ok)
	}
	list, err := adapter.List(context.Background(), contract.CallContext{UserID: "user-2"}, compatknowledge.ListInput{})
	if err != nil {
		t.Fatalf("isolated List returned error: %v", err)
	}
	if len(list.Items) != 0 || list.Page.Total == nil || *list.Page.Total != 0 {
		t.Fatalf("isolated list = %#v, want empty total 0", list)
	}
}

func TestKnowledgeAdapterMapsInvalidAndUnavailable(t *testing.T) {
	adapter := mustKnowledgeAdapterForDB(t, newKnowledgeAdapterTestDB(t))
	_, err := adapter.List(context.Background(), contract.CallContext{UserID: "user-1"}, compatknowledge.ListInput{
		Page: contract.PageRequest{PageSize: 20, PageToken: "not-valid"},
	})
	if code, ok := contract.CodeOf(err); !ok || code != contract.InvalidArgument {
		t.Fatalf("invalid page token code = %v, %v; want INVALID_ARGUMENT", code, ok)
	}
	_, err = NewKnowledgeCatalogAdapterForDB(nil)
	if code, ok := contract.CodeOf(err); !ok || code != contract.Internal {
		t.Fatalf("nil db code = %v, %v; want INTERNAL", code, ok)
	}
}

type knowledgeDatasetSeed struct {
	ID          string
	KbID        string
	UserID      string
	Name        string
	Description string
	Tags        []string
	UpdatedAt   time.Time
}

func newKnowledgeAdapterTestDB(t *testing.T) *gorm.DB {
	t.Helper()
	db, err := orm.Connect(orm.DriverSQLite, filepath.Join(t.TempDir(), "knowledge-adapter.db"))
	if err != nil {
		t.Fatalf("connect sqlite: %v", err)
	}
	if err := db.AutoMigrate(&orm.Dataset{}, &orm.Document{}, &orm.DefaultDataset{}, &orm.ACLModel{}, &orm.KBModel{}, &orm.VisibilityModel{}, &orm.UserGroupModel{}); err != nil {
		t.Fatalf("auto migrate: %v", err)
	}
	acl.InitStore(db)
	return db.DB
}

func mustKnowledgeAdapterForDB(t *testing.T, db *gorm.DB) *KnowledgeCatalogAdapter {
	t.Helper()
	adapter, err := NewKnowledgeCatalogAdapterForDB(db)
	if err != nil {
		t.Fatalf("NewKnowledgeCatalogAdapterForDB: %v", err)
	}
	return adapter
}

func seedKnowledgeDataset(t *testing.T, db *gorm.DB, seed knowledgeDatasetSeed) {
	t.Helper()
	if seed.KbID == "" {
		seed.KbID = seed.ID
	}
	if seed.Name == "" {
		seed.Name = seed.ID
	}
	if seed.UpdatedAt.IsZero() {
		seed.UpdatedAt = time.Now().UTC()
	}
	ext, err := json.Marshal(map[string]any{"tags": seed.Tags})
	if err != nil {
		t.Fatalf("marshal ext: %v", err)
	}
	row := orm.Dataset{
		ID:           seed.ID,
		KbID:         seed.KbID,
		DisplayName:  seed.Name,
		Desc:         seed.Description,
		DatasetState: 0,
		ShareType:    0,
		Type:         1,
		Ext:          ext,
		BaseModel: orm.BaseModel{
			CreateUserID:   seed.UserID,
			CreateUserName: seed.UserID,
			CreatedAt:      seed.UpdatedAt.Add(-time.Hour),
			UpdatedAt:      seed.UpdatedAt,
		},
	}
	if err := db.Create(&row).Error; err != nil {
		t.Fatalf("create dataset %s: %v", seed.ID, err)
	}
}

func knowledgeDocumentDisplayName(id string, folder bool) string {
	if folder {
		return id
	}
	return id + ".txt"
}

func seedKnowledgeDocument(t *testing.T, db *gorm.DB, id, datasetID, pid string, fileSize int64, folder bool) {
	t.Helper()
	ext := map[string]any{"file_size": fileSize}
	if !folder {
		ext["original_filename"] = id + ".txt"
	}
	if folder {
		ext["child_document_count"] = 1
		ext["recursive_document_count"] = 1
		ext["recursive_file_size"] = fileSize
	}
	raw, err := json.Marshal(ext)
	if err != nil {
		t.Fatalf("marshal document ext: %v", err)
	}
	now := time.Now().UTC()
	row := orm.Document{
		ID:          id,
		DatasetID:   datasetID,
		DisplayName: knowledgeDocumentDisplayName(id, folder),
		PID:         pid,
		Ext:         raw,
		BaseModel: orm.BaseModel{
			CreateUserID:   "user-1",
			CreateUserName: "user-1",
			CreatedAt:      now,
			UpdatedAt:      now,
		},
	}
	if err := db.Create(&row).Error; err != nil {
		t.Fatalf("create document %s: %v", id, err)
	}
}
