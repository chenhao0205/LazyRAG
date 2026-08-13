package mcpserver

import (
	"fmt"
	"time"

	compatcloud "lazymind/core/compat/clouddocument"
	compatknowledge "lazymind/core/compat/knowledge"
	compatskill "lazymind/core/compat/skill"
)

type toolResult struct {
	Content           []textContent `json:"content"`
	StructuredContent any           `json:"structuredContent,omitempty"`
	IsError           bool          `json:"isError,omitempty"`
}

type cloudDocumentListStructuredResult struct {
	Sources []cloudDocumentSource `json:"sources"`
	Page    skillListPage         `json:"page"`
}
type cloudDocumentSource struct {
	ID                   string    `json:"id"`
	Name                 string    `json:"name"`
	Status               string    `json:"status"`
	DatasetID            string    `json:"dataset_id"`
	BindingCount         int       `json:"binding_count"`
	AuthConnectionStatus string    `json:"auth_connection_status"`
	DocumentCount        *int64    `json:"document_count,omitempty"`
	CreatedAt            time.Time `json:"created_at"`
	UpdatedAt            time.Time `json:"updated_at"`
}
type cloudDocumentMetadata struct {
	ID                string                     `json:"id"`
	SourceID          string                     `json:"source_id"`
	ObjectKey         string                     `json:"object_key"`
	DisplayName       string                     `json:"display_name"`
	Name              string                     `json:"name"`
	FileType          string                     `json:"file_type"`
	SizeBytes         *int64                     `json:"size_bytes,omitempty"`
	SourceModifiedAt  *time.Time                 `json:"source_modified_at,omitempty"`
	LastSyncedAt      *time.Time                 `json:"last_synced_at,omitempty"`
	KnowledgeDocument *cloudKnowledgeDocumentRef `json:"knowledge_document,omitempty"`
}
type cloudKnowledgeDocumentRef struct {
	KnowledgeID string `json:"knowledge_id"`
	DocumentID  string `json:"document_id"`
}
type cloudDocumentGetStructuredResult struct {
	Source        cloudDocumentSource     `json:"source"`
	Documents     []cloudDocumentMetadata `json:"documents,omitempty"`
	DocumentsPage *skillListPage          `json:"documents_page,omitempty"`
}
type cloudDocumentSearchStructuredResult struct {
	Hits []cloudDocumentSearchHit `json:"hits"`
	Page skillListPage            `json:"page"`
}
type cloudDocumentSearchHit struct {
	Key         string `json:"key"`
	DisplayName string `json:"display_name"`
	SearchName  string `json:"search_name"`
	SourceID    string `json:"source_id"`
	TreeKey     string `json:"tree_key"`
	ObjectKey   string `json:"object_key"`
	ParentKey   string `json:"parent_key"`
	IsDocument  bool   `json:"is_document"`
	IsContainer bool   `json:"is_container"`
	HasChildren bool   `json:"has_children"`
	Selectable  bool   `json:"selectable"`
}

type textContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type skillListStructuredResult struct {
	Items []skillSummary `json:"items"`
	Page  skillListPage  `json:"page"`
}

type skillSummary struct {
	ID             string       `json:"id"`
	Name           string       `json:"name"`
	Description    string       `json:"description"`
	Category       string       `json:"category"`
	Tags           []string     `json:"tags"`
	HeadRevisionID string       `json:"head_revision_id"`
	AutoEvo        bool         `json:"auto_evo"`
	Enabled        bool         `json:"enabled"`
	Draft          draftSummary `json:"draft"`
}

type draftSummary struct {
	HasUncommittedDraft bool   `json:"has_uncommitted_draft"`
	TaskID              string `json:"task_id"`
	Version             int64  `json:"version"`
}

type skillListPage struct {
	NextPageToken string `json:"next_page_token,omitempty"`
	Total         *int64 `json:"total,omitempty"`
}

type knowledgeListStructuredResult struct {
	Items []knowledgeSummary `json:"items"`
	Page  skillListPage      `json:"page"`
}

type knowledgeSummary struct {
	ID                string    `json:"id"`
	Name              string    `json:"name"`
	Description       string    `json:"description"`
	Tags              []string  `json:"tags"`
	UpdatedAt         time.Time `json:"updated_at"`
	DocumentSizeBytes int64     `json:"document_size_bytes"`
	DocumentCount     int64     `json:"document_count"`
}

type knowledgeSearchStructuredResult struct {
	Hits []knowledgeSearchHit `json:"hits"`
}

type knowledgeSearchHit struct {
	KnowledgeID string  `json:"knowledge_id"`
	DocumentID  string  `json:"document_id"`
	ChunkID     string  `json:"chunk_id"`
	Text        string  `json:"text"`
	Score       float64 `json:"score"`
	SourceURL   string  `json:"source_url,omitempty"`
	Title       string  `json:"title,omitempty"`
}

type knowledgeDocumentMetadata struct {
	DocumentID   string                    `json:"document_id"`
	KnowledgeID  string                    `json:"knowledge_id"`
	Name         string                    `json:"name"`
	Source       string                    `json:"source"`
	Tags         []string                  `json:"tags"`
	ParseStatus  string                    `json:"parse_status"`
	MIMEType     string                    `json:"mime_type"`
	SizeBytes    int64                     `json:"size_bytes"`
	CreatedAt    time.Time                 `json:"created_at"`
	UpdatedAt    time.Time                 `json:"updated_at"`
	CreatedBy    string                    `json:"created_by"`
	OriginalFile *knowledgeDocumentFileRef `json:"original_file,omitempty"`
}

type knowledgeDocumentFileRef struct {
	FileName    string `json:"file_name"`
	DownloadURL string `json:"download_url"`
}

func skillListResult(result compatskill.ListResult) toolResult {
	items := make([]skillSummary, 0, len(result.Items))
	for _, item := range result.Items {
		items = append(items, skillSummary{
			ID: item.ID, Name: item.Name, Description: item.Description, Category: item.Category,
			Tags: append([]string(nil), item.Tags...), HeadRevisionID: item.HeadRevisionID,
			AutoEvo: item.AutoEvo, Enabled: item.Enabled,
			Draft: draftSummary{HasUncommittedDraft: item.Draft.HasUncommittedDraft, TaskID: item.Draft.TaskID, Version: item.Draft.Version},
		})
	}
	structured := skillListStructuredResult{Items: items, Page: skillListPage{NextPageToken: result.Page.NextPageToken, Total: result.Page.Total}}
	text := fmt.Sprintf("Found %d skill(s).", len(items))
	if result.Page.Total != nil {
		text = fmt.Sprintf("Found %d skill(s) in this page (%d total).", len(items), *result.Page.Total)
	}
	return toolResult{Content: []textContent{{Type: "text", Text: text}}, StructuredContent: structured}
}

func skillGetResult(result compatskill.GetResult) toolResult {
	item := skillSummary{
		ID: result.Skill.ID, Name: result.Skill.Name, Description: result.Skill.Description, Category: result.Skill.Category,
		Tags: append([]string(nil), result.Skill.Tags...), HeadRevisionID: result.Skill.HeadRevisionID,
		AutoEvo: result.Skill.AutoEvo, Enabled: result.Skill.Enabled,
		Draft: draftSummary{HasUncommittedDraft: result.Skill.Draft.HasUncommittedDraft, TaskID: result.Skill.Draft.TaskID, Version: result.Skill.Draft.Version},
	}
	return toolResult{
		Content:           []textContent{{Type: "text", Text: fmt.Sprintf("Skill %q metadata.", item.Name)}},
		StructuredContent: map[string]any{"skill": item},
	}
}

func knowledgeListResult(result compatknowledge.ListResult) toolResult {
	items := make([]knowledgeSummary, 0, len(result.Items))
	for _, item := range result.Items {
		items = append(items, knowledgeSummary{
			ID: item.ID, Name: item.Name, Description: item.Description, Tags: append([]string(nil), item.Tags...),
			UpdatedAt: item.UpdatedAt, DocumentSizeBytes: item.DocumentSizeBytes, DocumentCount: item.DocumentCount,
		})
	}
	structured := knowledgeListStructuredResult{Items: items, Page: skillListPage{NextPageToken: result.Page.NextPageToken, Total: result.Page.Total}}
	text := fmt.Sprintf("Found %d knowledge catalog(s).", len(items))
	if result.Page.Total != nil {
		text = fmt.Sprintf("Found %d knowledge catalog(s) in this page (%d total).", len(items), *result.Page.Total)
	}
	return toolResult{Content: []textContent{{Type: "text", Text: text}}, StructuredContent: structured}
}

func knowledgeGetResult(result compatknowledge.GetResult) toolResult {
	item := knowledgeSummary{
		ID: result.Knowledge.ID, Name: result.Knowledge.Name, Description: result.Knowledge.Description,
		Tags: append([]string(nil), result.Knowledge.Tags...), UpdatedAt: result.Knowledge.UpdatedAt,
		DocumentSizeBytes: result.Knowledge.DocumentSizeBytes, DocumentCount: result.Knowledge.DocumentCount,
	}
	return toolResult{
		Content:           []textContent{{Type: "text", Text: fmt.Sprintf("Knowledge catalog %q metadata.", item.Name)}},
		StructuredContent: map[string]any{"knowledge": item},
	}
}

func knowledgeSearchResult(result compatknowledge.SearchResult) toolResult {
	hits := make([]knowledgeSearchHit, 0, len(result.Hits))
	for _, hit := range result.Hits {
		hits = append(hits, knowledgeSearchHit{
			KnowledgeID: hit.KnowledgeID, DocumentID: hit.DocumentID, ChunkID: hit.ChunkID,
			Text: hit.Text, Score: hit.Score, SourceURL: hit.SourceURL, Title: hit.Title,
		})
	}
	return toolResult{
		Content:           []textContent{{Type: "text", Text: fmt.Sprintf("Found %d knowledge search hit(s).", len(hits))}},
		StructuredContent: knowledgeSearchStructuredResult{Hits: hits},
	}
}

func knowledgeDocumentGetResult(result compatknowledge.GetDocumentResult) toolResult {
	document := result.Document
	item := knowledgeDocumentMetadata{
		DocumentID: document.ID, KnowledgeID: document.KnowledgeID, Name: document.Name,
		Source: document.Source, Tags: append([]string(nil), document.Tags...), ParseStatus: document.ParseStatus,
		MIMEType: document.MIMEType, SizeBytes: document.SizeBytes, CreatedAt: document.CreatedAt,
		UpdatedAt: document.UpdatedAt, CreatedBy: document.CreatedBy,
	}
	if document.OriginalFile != nil {
		item.OriginalFile = &knowledgeDocumentFileRef{FileName: document.OriginalFile.FileName, DownloadURL: document.OriginalFile.DownloadURL}
	}
	return toolResult{
		Content:           []textContent{{Type: "text", Text: fmt.Sprintf("Knowledge document %q metadata.", item.Name)}},
		StructuredContent: map[string]any{"document": item},
	}
}

func cloudSource(item compatcloud.SourceSummary) cloudDocumentSource {
	return cloudDocumentSource{ID: item.ID, Name: item.Name, Status: item.Status, DatasetID: item.DatasetID, BindingCount: item.BindingCount, AuthConnectionStatus: item.AuthConnectionStatus, DocumentCount: item.DocumentCount, CreatedAt: item.CreatedAt, UpdatedAt: item.UpdatedAt}
}
func cloudSourceDetail(item compatcloud.SourceDetail) cloudDocumentSource {
	return cloudDocumentSource{ID: item.ID, Name: item.Name, Status: item.Status, DatasetID: item.DatasetID, DocumentCount: item.DocumentCount, CreatedAt: item.CreatedAt, UpdatedAt: item.UpdatedAt}
}
func cloudDocument(item compatcloud.DocumentSummary) cloudDocumentMetadata {
	out := cloudDocumentMetadata{ID: item.ID, SourceID: item.SourceID, ObjectKey: item.ObjectKey, DisplayName: item.DisplayName, Name: item.Name, FileType: item.FileType, SizeBytes: item.SizeBytes, SourceModifiedAt: item.SourceModifiedAt, LastSyncedAt: item.LastSyncedAt}
	if item.KnowledgeDocument != nil {
		out.KnowledgeDocument = &cloudKnowledgeDocumentRef{KnowledgeID: item.KnowledgeDocument.KnowledgeID, DocumentID: item.KnowledgeDocument.DocumentID}
	}
	return out
}

func cloudDocumentListResult(result compatcloud.ListResult) toolResult {
	items := make([]cloudDocumentSource, 0, len(result.Sources))
	for _, item := range result.Sources {
		items = append(items, cloudSource(item))
	}
	text := fmt.Sprintf("Found %d Cloud source(s).", len(items))
	if result.Page.Total != nil {
		text = fmt.Sprintf("Found %d Cloud source(s) in this page (%d total).", len(items), *result.Page.Total)
	}
	return toolResult{Content: []textContent{{Type: "text", Text: text}}, StructuredContent: cloudDocumentListStructuredResult{Sources: items, Page: skillListPage{NextPageToken: result.Page.NextPageToken, Total: result.Page.Total}}}
}
func cloudDocumentGetResult(result compatcloud.GetResult) toolResult {
	documents := make([]cloudDocumentMetadata, 0, len(result.Documents))
	for _, item := range result.Documents {
		documents = append(documents, cloudDocument(item))
	}
	structured := cloudDocumentGetStructuredResult{Source: cloudSourceDetail(result.Source), Documents: documents}
	if result.DocumentsPage != nil {
		structured.DocumentsPage = &skillListPage{NextPageToken: result.DocumentsPage.NextPageToken, Total: result.DocumentsPage.Total}
	}
	return toolResult{Content: []textContent{{Type: "text", Text: fmt.Sprintf("Cloud source %q metadata.", result.Source.Name)}}, StructuredContent: structured}
}
func cloudDocumentSearchResult(result compatcloud.SearchResult) toolResult {
	hits := make([]cloudDocumentSearchHit, 0, len(result.Hits))
	for _, hit := range result.Hits {
		hits = append(hits, cloudDocumentSearchHit{Key: hit.Key, DisplayName: hit.DisplayName, SearchName: hit.SearchName, SourceID: hit.SourceID, TreeKey: hit.TreeKey, ObjectKey: hit.ObjectKey, ParentKey: hit.ParentKey, IsDocument: hit.IsDocument, IsContainer: hit.IsContainer, HasChildren: hit.HasChildren, Selectable: hit.Selectable})
	}
	return toolResult{Content: []textContent{{Type: "text", Text: fmt.Sprintf("Found %d Cloud metadata search hit(s).", len(hits))}}, StructuredContent: cloudDocumentSearchStructuredResult{Hits: hits, Page: skillListPage{NextPageToken: result.Page.NextPageToken, Total: result.Page.Total}}}
}

func toolErrorResult(code, message string) toolResult {
	return toolResult{
		Content:           []textContent{{Type: "text", Text: message}},
		StructuredContent: map[string]any{"error": map[string]string{"code": code, "message": message}},
		IsError:           true,
	}
}
