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
	BindingCount         int
	AuthConnectionStatus string
	DocumentCount        *int64
	CreatedAt            time.Time
	UpdatedAt            time.Time
}

type ListResult struct {
	Sources []SourceSummary
	Page    contract.PageResult
}

// GetInput identifies a Cloud Source. When IncludeDocuments is true, Get
// returns one page of document metadata for that source; it never reads
// document body content.
type GetInput struct {
	SourceID         string
	IncludeDocuments bool
	DocumentsPage    contract.PageRequest
}

type SourceDetail struct {
	ID            string
	Name          string
	Status        string
	DatasetID     string
	DocumentCount *int64
	CreatedAt     time.Time
	UpdatedAt     time.Time
}

// KnowledgeDocumentRef is the complete reference needed by Knowledge Document
// APIs to read document details or content later.
type KnowledgeDocumentRef struct {
	KnowledgeID string
	DocumentID  string
}

type DocumentSummary struct {
	ID                string
	SourceID          string
	ObjectKey         string
	DisplayName       string
	Name              string
	FileType          string
	SizeBytes         *int64
	SourceModifiedAt  *time.Time
	LastSyncedAt      *time.Time
	KnowledgeDocument *KnowledgeDocumentRef
}

type DocumentListResult struct {
	Documents []DocumentSummary
	Page      contract.PageResult
}

type GetResult struct {
	Source        SourceDetail
	Documents     []DocumentSummary
	DocumentsPage *contract.PageResult
}

// SearchInput searches metadata indexed by Scan for a Cloud Source. It covers
// document titles, display_name/search_name, and tree node names only.
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

// SearchResult contains title/name/tree-object matches only. It does not
// represent document body full-text search, semantic vector retrieval, RAG
// chunk retrieval, or model QA.
type SearchResult struct {
	Hits []SearchHit
	Page contract.PageResult
}

type SearchHit struct {
	Key         string
	DisplayName string
	SearchName  string
	SourceID    string
	TreeKey     string
	ObjectKey   string
	ParentKey   string
	IsDocument  bool
	IsContainer bool
	HasChildren bool
	Selectable  bool
}
