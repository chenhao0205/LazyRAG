package scan

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"lazymind/core/compat/clouddocument"
	"lazymind/core/compat/contract"
)

const (
	defaultCloudDocumentTimeout      = 5 * time.Second
	cloudDocumentConnectorTypeFilter = "feishu,notion"
)

type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

type CloudDocumentAdapter struct {
	baseURL *url.URL
	client  HTTPClient
	timeout time.Duration
}

func NewCloudDocumentAdapter(baseURL string, client HTTPClient, timeout time.Duration) (*CloudDocumentAdapter, error) {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		return nil, contract.InvalidArgumentError("cloud_document.adapter.new", "scan base_url is required")
	}
	parsed, err := url.Parse(baseURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, contract.NewError(contract.InvalidArgument, "cloud_document.adapter.new", "scan base_url is invalid", false, err)
	}
	if client == nil {
		client = http.DefaultClient
	}
	if timeout <= 0 {
		timeout = defaultCloudDocumentTimeout
	}
	return &CloudDocumentAdapter{baseURL: parsed, client: client, timeout: timeout}, nil
}

func (a *CloudDocumentAdapter) ListSources(ctx context.Context, callCtx contract.CallContext, input clouddocument.ListInput) (clouddocument.ListResult, error) {
	const operation = "cloud_document.list"
	page := input.Page.Normalize()
	offset, err := contract.DecodeOffsetPageToken(page.PageToken)
	if err != nil {
		return clouddocument.ListResult{}, contract.NewError(contract.InvalidArgument, operation, "invalid page token", false, err)
	}
	endpoint := a.endpoint("/api/scan/sources")
	query := endpoint.Query()
	if input.Keyword != "" {
		query.Set("keyword", input.Keyword)
	}
	if input.Status != "" {
		query.Set("status", input.Status)
	}
	query.Set("connector_type", cloudDocumentConnectorTypeFilter)
	query.Set("page", strconv.Itoa(offset/page.PageSize+1))
	query.Set("page_size", strconv.Itoa(page.PageSize))
	endpoint.RawQuery = query.Encode()

	var resp scanListSourcesResponse
	if err := a.doJSON(ctx, callCtx, operation, http.MethodGet, endpoint, nil, &resp); err != nil {
		return clouddocument.ListResult{}, err
	}
	sources := make([]clouddocument.SourceSummary, 0, len(resp.Items))
	for _, item := range resp.Items {
		sources = append(sources, mapSourceSummary(item))
	}
	total := int64(resp.Total)
	result := clouddocument.ListResult{
		Sources: sources,
		Page:    contract.PageResult{Total: &total},
	}
	if offset+len(resp.Items) < resp.Total {
		result.Page.NextPageToken = contract.EncodeOffsetPageToken(offset + len(resp.Items))
	}
	return result, nil
}

func (a *CloudDocumentAdapter) GetSource(ctx context.Context, callCtx contract.CallContext, sourceID string) (clouddocument.SourceDetail, error) {
	const operation = "cloud_document.get"
	endpoint := a.endpoint("/api/scan/sources/" + url.PathEscape(sourceID))
	query := endpoint.Query()
	query.Set("include_bindings", "false")
	query.Set("include_summary", "true")
	endpoint.RawQuery = query.Encode()

	var resp scanGetSourceResponse
	if err := a.doJSON(ctx, callCtx, operation, http.MethodGet, endpoint, nil, &resp); err != nil {
		return clouddocument.SourceDetail{}, err
	}
	return mapSourceDetail(resp), nil
}

func (a *CloudDocumentAdapter) ListDocuments(ctx context.Context, callCtx contract.CallContext, source clouddocument.SourceDetail, input clouddocument.GetInput) (clouddocument.DocumentListResult, error) {
	const operation = "cloud_document.get"
	page := input.DocumentsPage.Normalize()
	offset, err := contract.DecodeOffsetPageToken(page.PageToken)
	if err != nil {
		return clouddocument.DocumentListResult{}, contract.NewError(contract.InvalidArgument, operation, "invalid documents page token", false, err)
	}
	endpoint := a.endpoint("/api/scan/sources/" + url.PathEscape(input.SourceID) + "/documents")
	query := endpoint.Query()
	query.Set("page", strconv.Itoa(offset/page.PageSize+1))
	query.Set("page_size", strconv.Itoa(page.PageSize))
	query.Set("refresh_state", "false")
	endpoint.RawQuery = query.Encode()

	var resp scanListDocumentsResponse
	if err := a.doJSON(ctx, callCtx, operation, http.MethodGet, endpoint, nil, &resp); err != nil {
		return clouddocument.DocumentListResult{}, err
	}
	docs := make([]clouddocument.DocumentSummary, 0, len(resp.Items))
	for _, item := range resp.Items {
		docs = append(docs, mapDocumentSummary(source.DatasetID, item))
	}
	total := int64(resp.Total)
	result := clouddocument.DocumentListResult{
		Documents: docs,
		Page:      contract.PageResult{Total: &total},
	}
	if offset+len(resp.Items) < resp.Total {
		result.Page.NextPageToken = contract.EncodeOffsetPageToken(offset + len(resp.Items))
	}
	return result, nil
}

func (a *CloudDocumentAdapter) Search(ctx context.Context, callCtx contract.CallContext, input clouddocument.SearchInput) (clouddocument.SearchResult, error) {
	const operation = "cloud_document.search"
	page := input.Page.Normalize()
	endpoint := a.endpoint("/api/scan/sources/" + url.PathEscape(input.SourceID) + "/tree/search")
	refreshState := false
	body := scanSearchSourceTreeRequest{
		Keyword:           input.Query,
		BindingID:         input.BindingID,
		TreeKey:           input.TreeKey,
		RefreshState:      &refreshState,
		IncludeDocuments:  true,
		IncludeContainers: true,
		StateFilter:       append([]string(nil), input.StateFilter...),
		ListMode:          "page",
		PageSize:          page.PageSize,
		Cursor:            page.PageToken,
	}
	if input.IncludeDocuments || input.IncludeContainers {
		body.IncludeDocuments = input.IncludeDocuments
		body.IncludeContainers = input.IncludeContainers
	}
	var resp scanTreeNodePage
	if err := a.doJSON(ctx, callCtx, operation, http.MethodPost, endpoint, body, &resp); err != nil {
		return clouddocument.SearchResult{}, err
	}
	hits := make([]clouddocument.SearchHit, 0, len(resp.Items))
	for _, item := range resp.Items {
		hits = append(hits, mapSearchHit(item))
	}
	return clouddocument.SearchResult{
		Hits: hits,
		Page: contract.PageResult{NextPageToken: resp.NextCursor},
	}, nil
}

func (a *CloudDocumentAdapter) endpoint(path string) *url.URL {
	endpoint := *a.baseURL
	endpoint.Path = strings.TrimRight(endpoint.Path, "/") + path
	endpoint.RawQuery = ""
	return &endpoint
}

func (a *CloudDocumentAdapter) doJSON(ctx context.Context, callCtx contract.CallContext, operation, method string, endpoint *url.URL, body any, out any) error {
	var reader io.Reader
	if body != nil {
		raw, err := json.Marshal(body)
		if err != nil {
			return contract.NewError(contract.Internal, operation, "request encoding failed", false, err)
		}
		reader = bytes.NewReader(raw)
	}
	reqCtx, cancel := context.WithTimeout(ctx, a.timeout)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, method, endpoint.String(), reader)
	if err != nil {
		return contract.NewError(contract.Internal, operation, "request creation failed", false, err)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("X-User-ID", strings.TrimSpace(callCtx.UserID))
	if tenantID := strings.TrimSpace(callCtx.TenantID); tenantID != "" {
		req.Header.Set("X-Tenant-ID", tenantID)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := a.client.Do(req)
	if err != nil {
		return mapTransportError(operation, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return mapHTTPError(operation, resp)
	}
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return contract.NewError(contract.BackendUnavailable, operation, "scan response is invalid", true, err)
	}
	return nil
}

func mapTransportError(operation string, err error) error {
	if errors.Is(err, context.Canceled) {
		return contract.NewError(contract.BackendUnavailable, operation, "scan request canceled", true, err)
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return contract.NewError(contract.BackendUnavailable, operation, "scan request timed out", true, err)
	}
	return contract.NewError(contract.BackendUnavailable, operation, "scan backend unavailable", true, err)
}

func mapHTTPError(operation string, resp *http.Response) error {
	payload := scanErrorResponse{}
	_ = json.NewDecoder(io.LimitReader(resp.Body, 4096)).Decode(&payload)
	cause := scanHTTPError{StatusCode: resp.StatusCode, Code: payload.Code, Message: payload.Message}
	message := safeHTTPErrorMessage(resp.StatusCode)
	switch resp.StatusCode {
	case http.StatusBadRequest:
		return contract.NewError(contract.InvalidArgument, operation, message, false, cause)
	case http.StatusNotFound:
		return contract.NewError(contract.NotFound, operation, message, false, cause)
	case http.StatusConflict:
		return contract.NewError(contract.Conflict, operation, message, false, cause)
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return contract.NewError(contract.BackendUnavailable, operation, message, true, cause)
	default:
		if resp.StatusCode >= http.StatusInternalServerError {
			return contract.NewError(contract.BackendUnavailable, operation, message, true, cause)
		}
		return contract.NewError(contract.Internal, operation, message, false, cause)
	}
}

type scanHTTPError struct {
	StatusCode int
	Code       string
	Message    string
}

func (e scanHTTPError) Error() string {
	return fmt.Sprintf("scan request failed: status=%d code=%s message=%s", e.StatusCode, e.Code, e.Message)
}

func safeHTTPErrorMessage(statusCode int) string {
	switch statusCode {
	case http.StatusBadRequest:
		return "scan request is invalid"
	case http.StatusUnauthorized, http.StatusForbidden:
		return "scan access denied"
	case http.StatusNotFound:
		return "scan resource not found"
	case http.StatusConflict:
		return "scan resource conflict"
	case http.StatusRequestTimeout, http.StatusTooManyRequests, http.StatusBadGateway, http.StatusServiceUnavailable, http.StatusGatewayTimeout:
		return "scan backend unavailable"
	default:
		if statusCode >= http.StatusInternalServerError {
			return "scan backend unavailable"
		}
	}
	return "scan request failed"
}

func mapSourceSummary(item scanSourceListItem) clouddocument.SourceSummary {
	return clouddocument.SourceSummary{
		ID:                   item.SourceID,
		Name:                 item.Name,
		Status:               item.Status,
		DatasetID:            item.DatasetID,
		BindingCount:         item.BindingCount,
		AuthConnectionStatus: authConnectionStatus(item.AuthConnectionStatus),
		DocumentCount:        documentCount(item.Summary),
		CreatedAt:            item.CreatedAt,
		UpdatedAt:            item.UpdatedAt,
	}
}

func mapSourceDetail(resp scanGetSourceResponse) clouddocument.SourceDetail {
	return clouddocument.SourceDetail{
		ID:            resp.Source.SourceID,
		Name:          resp.Source.Name,
		Status:        resp.Source.Status,
		DatasetID:     resp.Source.DatasetID,
		DocumentCount: documentCount(resp.Summary),
		CreatedAt:     resp.Source.CreatedAt,
		UpdatedAt:     resp.Source.UpdatedAt,
	}
}

func mapDocumentSummary(knowledgeID string, item scanDocumentItem) clouddocument.DocumentSummary {
	return clouddocument.DocumentSummary{
		ID:                item.DocumentID,
		SourceID:          item.SourceID,
		ObjectKey:         item.ObjectKey,
		DisplayName:       item.DisplayName,
		Name:              item.Name,
		FileType:          item.FileType,
		SizeBytes:         item.SizeBytes,
		SourceModifiedAt:  item.SourceModifiedAt,
		LastSyncedAt:      item.LastSyncedAt,
		KnowledgeDocument: knowledgeDocumentRef(knowledgeID, item),
	}
}

func mapSearchHit(item scanTreeNode) clouddocument.SearchHit {
	return clouddocument.SearchHit{
		Key:         item.Key,
		DisplayName: item.DisplayName,
		SearchName:  item.SearchName,
		SourceID:    item.SourceID,
		TreeKey:     item.TreeKey,
		ObjectKey:   item.ObjectKey,
		ParentKey:   item.ParentKey,
		IsDocument:  item.IsDocument,
		IsContainer: item.IsContainer,
		HasChildren: item.HasChildren,
		Selectable:  item.Selectable,
	}
}

func knowledgeDocumentRef(knowledgeID string, item scanDocumentItem) *clouddocument.KnowledgeDocumentRef {
	knowledgeID = strings.TrimSpace(knowledgeID)
	documentID := strings.TrimSpace(item.CoreDocumentID)
	if knowledgeID == "" || documentID == "" {
		return nil
	}
	if strings.EqualFold(strings.TrimSpace(item.PendingAction), "DELETE") {
		return nil
	}
	switch strings.ToUpper(strings.TrimSpace(item.SourceState)) {
	case "DELETED", "OUT_OF_SCOPE":
		return nil
	}
	if !strings.EqualFold(strings.TrimSpace(item.ParseStatus), "SUCCEEDED") &&
		!strings.EqualFold(strings.TrimSpace(item.EffectiveParseStatus), "PARSED") {
		return nil
	}
	return &clouddocument.KnowledgeDocumentRef{KnowledgeID: knowledgeID, DocumentID: documentID}
}

func authConnectionStatus(status *scanAuthConnectionStatus) string {
	if status == nil {
		return ""
	}
	return status.Status
}

func documentCount(summary map[string]any) *int64 {
	for _, key := range []string{"total_document_count", "document_objects"} {
		if value, ok := summary[key]; ok {
			if n, ok := int64Value(value); ok {
				return &n
			}
		}
	}
	return nil
}

func int64Value(value any) (int64, bool) {
	switch v := value.(type) {
	case int:
		return int64(v), true
	case int64:
		return v, true
	case float64:
		return int64(v), true
	case json.Number:
		n, err := v.Int64()
		return n, err == nil
	default:
		return 0, false
	}
}

type scanErrorResponse struct {
	Code    string         `json:"code"`
	Message string         `json:"message"`
	Details map[string]any `json:"details"`
}

type scanListSourcesResponse struct {
	Items []scanSourceListItem `json:"items"`
	Total int                  `json:"total"`
}

type scanSourceListItem struct {
	SourceID             string                    `json:"source_id"`
	Name                 string                    `json:"name"`
	DatasetID            string                    `json:"dataset_id"`
	Status               string                    `json:"status"`
	BindingCount         int                       `json:"binding_count"`
	AuthConnectionStatus *scanAuthConnectionStatus `json:"auth_connection_status,omitempty"`
	Summary              map[string]any            `json:"summary,omitempty"`
	CreatedAt            time.Time                 `json:"created_at"`
	UpdatedAt            time.Time                 `json:"updated_at"`
}

type scanAuthConnectionStatus struct {
	Status        string   `json:"status"`
	ConnectionIDs []string `json:"connection_ids"`
}

type scanGetSourceResponse struct {
	Source  scanSource     `json:"source"`
	Summary map[string]any `json:"summary,omitempty"`
}

type scanSource struct {
	SourceID  string    `json:"source_id"`
	Name      string    `json:"name"`
	DatasetID string    `json:"dataset_id"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type scanListDocumentsResponse struct {
	Items    []scanDocumentItem `json:"items"`
	Total    int                `json:"total"`
	Page     int                `json:"page"`
	PageSize int                `json:"page_size"`
}

type scanDocumentItem struct {
	DocumentID           string     `json:"document_id,omitempty"`
	SourceID             string     `json:"source_id"`
	BindingID            string     `json:"binding_id"`
	ObjectKey            string     `json:"object_key"`
	DisplayName          string     `json:"display_name"`
	Name                 string     `json:"name,omitempty"`
	FileType             string     `json:"file_type,omitempty"`
	SizeBytes            *int64     `json:"size_bytes,omitempty"`
	CoreDocumentID       string     `json:"core_document_id,omitempty"`
	ParseStatus          string     `json:"parse_status,omitempty"`
	EffectiveParseStatus string     `json:"effective_parse_status,omitempty"`
	SourceState          string     `json:"source_state,omitempty"`
	PendingAction        string     `json:"pending_action,omitempty"`
	SourceModifiedAt     *time.Time `json:"source_modified_at,omitempty"`
	LastSyncedAt         *time.Time `json:"last_synced_at,omitempty"`
}

type scanSearchSourceTreeRequest struct {
	Keyword           string   `json:"keyword"`
	BindingID         string   `json:"binding_id,omitempty"`
	TreeKey           string   `json:"tree_key,omitempty"`
	RefreshState      *bool    `json:"refresh_state,omitempty"`
	IncludeDocuments  bool     `json:"include_documents"`
	IncludeContainers bool     `json:"include_containers"`
	StateFilter       []string `json:"state_filter,omitempty"`
	ListMode          string   `json:"list_mode,omitempty"`
	PageSize          int      `json:"page_size,omitempty"`
	Cursor            string   `json:"cursor,omitempty"`
}

type scanTreeNodePage struct {
	Items      []scanTreeNode `json:"items"`
	NextCursor string         `json:"next_cursor,omitempty"`
	HasMore    bool           `json:"has_more"`
	SearchMode string         `json:"search_mode,omitempty"`
}

type scanTreeNode struct {
	Key         string `json:"key"`
	DisplayName string `json:"display_name"`
	SearchName  string `json:"search_name,omitempty"`
	SourceID    string `json:"source_id,omitempty"`
	BindingID   string `json:"binding_id,omitempty"`
	TreeKey     string `json:"tree_key,omitempty"`
	ObjectKey   string `json:"object_key,omitempty"`
	ParentKey   string `json:"parent_key,omitempty"`
	IsDocument  bool   `json:"is_document"`
	IsContainer bool   `json:"is_container"`
	HasChildren bool   `json:"has_children"`
	Selectable  bool   `json:"selectable"`
}
