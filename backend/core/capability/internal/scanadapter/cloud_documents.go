// Package scanadapter adapts the Scan Control Plane's cloud-source API to the
// capability boundary. Provider credentials never cross this boundary.
package scanadapter

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"lazymind/core/capability"
)

const cloudConnectors = "feishu,notion"

type HTTPClient interface {
	Do(*http.Request) (*http.Response, error)
}
type CloudDocumentReader struct {
	base    *url.URL
	client  HTTPClient
	timeout time.Duration
}

func NewCloudDocumentReader(baseURL string, client HTTPClient, timeout time.Duration) (*CloudDocumentReader, error) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil || u.Scheme == "" || u.Host == "" {
		return nil, capability.NewError(capability.InvalidArgument, "cloud_document.adapter.new", "scan base_url is invalid", false, err)
	}
	if client == nil {
		client = http.DefaultClient
	}
	if timeout <= 0 {
		timeout = 5 * time.Second
	}
	return &CloudDocumentReader{base: u, client: client, timeout: timeout}, nil
}
func (a *CloudDocumentReader) ListCloudDocuments(ctx context.Context, call capability.InvocationContext, q capability.CloudDocumentListQuery) (capability.CloudDocumentListPage, error) {
	u := a.endpoint("/api/scan/sources")
	v := u.Query()
	v.Set("connector_type", cloudConnectors)
	v.Set("page", strconv.Itoa(q.Offset/q.Limit+1))
	v.Set("page_size", strconv.Itoa(q.Limit))
	if q.Keyword != "" {
		v.Set("keyword", q.Keyword)
	}
	if q.Status != "" {
		v.Set("status", q.Status)
	}
	u.RawQuery = v.Encode()
	var out listSources
	if err := a.json(ctx, call, "cloud_document.list", http.MethodGet, u, nil, &out); err != nil {
		return capability.CloudDocumentListPage{}, err
	}
	items := make([]capability.CloudDocumentSource, 0, len(out.Items))
	for _, x := range out.Items {
		items = append(items, sourceSummary(x))
	}
	return capability.CloudDocumentListPage{Items: items, Total: int64(out.Total)}, nil
}
func (a *CloudDocumentReader) GetCloudDocument(ctx context.Context, call capability.InvocationContext, in capability.GetCloudDocumentInput) (capability.GetCloudDocumentResult, error) {
	s, err := a.cloudSource(ctx, call, "cloud_document.get", in.SourceID)
	if err != nil {
		return capability.GetCloudDocumentResult{}, err
	}
	r := capability.GetCloudDocumentResult{Source: sourceDetail(s)}
	if !in.IncludeDocuments {
		return r, nil
	}
	u := a.endpoint("/api/scan/sources/" + url.PathEscape(in.SourceID) + "/documents")
	q := u.Query()
	q.Set("page", "1")
	q.Set("page_size", strconv.Itoa(in.DocumentsPage.PageSize))
	q.Set("refresh_state", "false")
	u.RawQuery = q.Encode()
	var out listDocuments
	if err = a.json(ctx, call, "cloud_document.get", http.MethodGet, u, nil, &out); err != nil {
		return capability.GetCloudDocumentResult{}, err
	}
	docs := make([]capability.CloudDocumentMetadata, 0, len(out.Items))
	for _, d := range out.Items {
		docs = append(docs, capability.CloudDocumentMetadata{ID: d.DocumentID, SourceID: d.SourceID, ObjectKey: d.ObjectKey, DisplayName: d.DisplayName, Name: d.Name, FileType: d.FileType, SizeBytes: d.SizeBytes})
	}
	total := int64(out.Total)
	r.Documents = docs
	r.DocumentsPage = &capability.PageInfo{Total: total}
	return r, nil
}
func (a *CloudDocumentReader) SearchCloudDocuments(ctx context.Context, call capability.InvocationContext, in capability.SearchCloudDocumentsInput) (capability.SearchCloudDocumentsResult, error) {
	if _, err := a.cloudSource(ctx, call, "cloud_document.search", in.SourceID); err != nil {
		return capability.SearchCloudDocumentsResult{}, err
	}
	u := a.endpoint("/api/scan/sources/" + url.PathEscape(in.SourceID) + "/tree/search")
	refresh := false
	body := map[string]any{"keyword": in.Query, "binding_id": in.BindingID, "tree_key": in.TreeKey, "refresh_state": &refresh, "include_documents": true, "include_containers": true, "state_filter": in.StateFilter, "list_mode": "page", "page_size": in.Page.PageSize, "cursor": in.Page.PageToken, "connector_types": []string{"feishu", "notion"}}
	if in.IncludeDocuments || in.IncludeContainers {
		body["include_documents"] = in.IncludeDocuments
		body["include_containers"] = in.IncludeContainers
	}
	var out treePage
	if err := a.json(ctx, call, "cloud_document.search", http.MethodPost, u, body, &out); err != nil {
		return capability.SearchCloudDocumentsResult{}, err
	}
	hits := make([]capability.CloudDocumentSearchHit, 0, len(out.Items))
	for _, x := range out.Items {
		hits = append(hits, capability.CloudDocumentSearchHit{Key: x.Key, DisplayName: x.DisplayName, SearchName: x.SearchName, SourceID: x.SourceID, TreeKey: x.TreeKey, ObjectKey: x.ObjectKey, ParentKey: x.ParentKey, IsDocument: x.IsDocument, IsContainer: x.IsContainer, HasChildren: x.HasChildren, Selectable: x.Selectable})
	}
	return capability.SearchCloudDocumentsResult{Hits: hits, Page: capability.PageInfo{NextPageToken: out.NextCursor}}, nil
}
func (a *CloudDocumentReader) cloudSource(ctx context.Context, call capability.InvocationContext, op, id string) (getSource, error) {
	u := a.endpoint("/api/scan/sources/" + url.PathEscape(id))
	q := u.Query()
	q.Set("include_bindings", "true")
	q.Set("include_summary", "true")
	u.RawQuery = q.Encode()
	var out getSource
	if err := a.json(ctx, call, op, http.MethodGet, u, nil, &out); err != nil {
		return getSource{}, err
	}
	for _, b := range out.Bindings {
		if b.ConnectorType == "feishu" || b.ConnectorType == "notion" {
			return out, nil
		}
	}
	return getSource{}, capability.NewError(capability.NotFound, op, "cloud source not found", false, nil)
}
func (a *CloudDocumentReader) endpoint(path string) *url.URL {
	u := *a.base
	u.Path = strings.TrimRight(u.Path, "/") + path
	u.RawQuery = ""
	return &u
}
func (a *CloudDocumentReader) json(ctx context.Context, call capability.InvocationContext, op, method string, u *url.URL, body, out any) error {
	var r io.Reader
	if body != nil {
		raw, e := json.Marshal(body)
		if e != nil {
			return capability.NewError(capability.Internal, op, "request encoding failed", false, e)
		}
		r = bytes.NewReader(raw)
	}
	c, cancel := context.WithTimeout(ctx, a.timeout)
	defer cancel()
	req, e := http.NewRequestWithContext(c, method, u.String(), r)
	if e != nil {
		return capability.NewError(capability.Internal, op, "request creation failed", false, e)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-User-ID", call.Principal.UserID)
	if call.Principal.TenantID != "" {
		req.Header.Set("X-Tenant-ID", call.Principal.TenantID)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, e := a.client.Do(req)
	if e != nil {
		return capability.NewError(capability.Unavailable, op, "scan backend unavailable", true, e)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		code := capability.Internal
		if resp.StatusCode == 404 {
			code = capability.NotFound
		} else if resp.StatusCode == 400 {
			code = capability.InvalidArgument
		} else if resp.StatusCode >= 500 || resp.StatusCode == 429 {
			code = capability.Unavailable
		}
		return capability.NewError(code, op, "scan request failed", code == capability.Unavailable, nil)
	}
	if e = json.NewDecoder(resp.Body).Decode(out); e != nil {
		return capability.NewError(capability.Unavailable, op, "scan response is invalid", true, e)
	}
	return nil
}

type listSources struct {
	Items []source `json:"items"`
	Total int      `json:"total"`
}
type source struct {
	SourceID  string         `json:"source_id"`
	Name      string         `json:"name"`
	Status    string         `json:"status"`
	DatasetID string         `json:"dataset_id"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
	Summary   map[string]any `json:"summary"`
}
type binding struct {
	ConnectorType string `json:"connector_type"`
}
type getSource struct {
	Source   source         `json:"source"`
	Bindings []binding      `json:"bindings"`
	Summary  map[string]any `json:"summary"`
}
type document struct {
	DocumentID  string `json:"document_id"`
	SourceID    string `json:"source_id"`
	ObjectKey   string `json:"object_key"`
	DisplayName string `json:"display_name"`
	Name        string `json:"name"`
	FileType    string `json:"file_type"`
	SizeBytes   *int64 `json:"size_bytes"`
}
type listDocuments struct {
	Items []document `json:"items"`
	Total int        `json:"total"`
}
type treeNode struct {
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
type treePage struct {
	Items      []treeNode `json:"items"`
	NextCursor string     `json:"next_cursor"`
}

func sourceSummary(s source) capability.CloudDocumentSource {
	return capability.CloudDocumentSource{ID: s.SourceID, Name: s.Name, Status: s.Status, KnowledgeID: s.DatasetID, CreatedAt: s.CreatedAt, UpdatedAt: s.UpdatedAt}
}
func sourceDetail(s getSource) capability.CloudDocumentSource { return sourceSummary(s.Source) }
