package clouddocument

import (
	"time"

	"lazymind/core/compat/contract"
)

type ListInput struct {
	Keyword string
	Status  string
	Page    contract.PageRequest
}

type SourceSummary struct {
	ID                   string
	Name                 string
	Status               string
	DatasetID            string
	ConfigVersion        int64
	BindingCount         int
	Summary              map[string]any
	AuthConnectionStatus string
	DocumentCount        *int64
	CreatedAt            time.Time
	UpdatedAt            time.Time
}

type ListResult struct {
	Sources []SourceSummary
	Page    contract.PageResult
}

type GetInput struct {
	SourceID         string
	IncludeDocuments bool
	DocumentsPage    contract.PageRequest
	BindingID        string
	DocumentKeyword  string
	StateFilter      []string
	ParseStatuses    []string
}

type SourceDetail struct {
	ID            string
	Name          string
	Status        string
	DatasetID     string
	ConfigVersion int64
	Summary       map[string]any
	DocumentCount *int64
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

type DocumentSummary struct {
	ID                   string
	SourceID             string
	BindingID            string
	ObjectKey            string
	DisplayName          string
	Name                 string
	FileType             string
	SizeBytes            int64
	SourceVersion        string
	BaselineVersion      string
	CoreDocumentID       string
	ParseStatus          string
	ParseState           string
	EffectiveParseStatus string
	SourceState          string
	SyncState            string
	PendingAction        string
	ParseQueueState      string
	HasUpdate            bool
	UpdateType           string
	SourceModifiedAt     *time.Time
	LastSyncedAt         *time.Time
}

type DocumentListResult struct {
	Documents []DocumentSummary
	Page      contract.PageResult
}

type GetResult struct {
	Source        SourceDetail
	Documents     []DocumentSummary
	DocumentsPage contract.PageResult
}

type SearchInput struct {
	SourceID          string
	Query             string
	Page              contract.PageRequest
	BindingID         string
	TreeKey           string
	StateFilter       []string
	IncludeDocuments  bool
	IncludeContainers bool
}

// SearchResult contains metadata/name/tree-object matches only. It does not
// represent content full-text search, semantic search, RAG retrieval, or QA.
type SearchResult struct {
	Hits []SearchHit
	Page contract.PageResult
}

type SearchHit struct {
	Key             string
	DisplayName     string
	SearchName      string
	SourceID        string
	BindingID       string
	TreeKey         string
	ObjectKey       string
	ParentKey       string
	ObjectType      string
	IsDocument      bool
	IsContainer     bool
	HasChildren     bool
	Selectable      bool
	SourceState     string
	SyncState       string
	PendingAction   string
	ParseQueueState string
	HasUpdate       bool
	UpdateType      string
}
