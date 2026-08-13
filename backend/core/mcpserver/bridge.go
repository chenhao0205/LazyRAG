package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	compatcloud "lazymind/core/compat/clouddocument"
	"lazymind/core/compat/contract"
	compatknowledge "lazymind/core/compat/knowledge"
	compatskill "lazymind/core/compat/skill"
)

const (
	skillListToolName            = "lazymind_skill_list"
	skillGetToolName             = "lazymind_skill_get"
	knowledgeListToolName        = "lazymind_knowledge_list"
	knowledgeGetToolName         = "lazymind_knowledge_get"
	knowledgeDocumentGetToolName = "lazymind_knowledge_document_get"
	knowledgeSearchToolName      = "lazymind_knowledge_search"
	cloudDocumentListToolName    = "lazymind_cloud_document_list"
	cloudDocumentGetToolName     = "lazymind_cloud_document_get"
	cloudDocumentSearchToolName  = "lazymind_cloud_document_search"
)

type toolCallParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

type skillListArguments struct {
	Keyword   string   `json:"keyword"`
	Category  string   `json:"category"`
	Tags      []string `json:"tags"`
	PageSize  int      `json:"page_size"`
	PageToken string   `json:"page_token"`
}

type skillGetArguments struct {
	SkillID string `json:"skill_id"`
}

type knowledgeListArguments struct {
	Keyword   string   `json:"keyword"`
	Tags      []string `json:"tags"`
	PageSize  int      `json:"page_size"`
	PageToken string   `json:"page_token"`
}

type knowledgeGetArguments struct {
	KnowledgeID string `json:"knowledge_id"`
}

type knowledgeDocumentGetArguments struct {
	KnowledgeID string `json:"knowledge_id"`
	DocumentID  string `json:"document_id"`
}

type knowledgeSearchArguments struct {
	Query        string   `json:"query"`
	KnowledgeIDs []string `json:"knowledge_ids"`
	TopK         int      `json:"top_k"`
}

type cloudDocumentListArguments struct {
	Keyword   string `json:"keyword"`
	Status    string `json:"status"`
	PageSize  int    `json:"page_size"`
	PageToken string `json:"page_token"`
}

type cloudDocumentGetArguments struct {
	SourceID         string `json:"source_id"`
	IncludeDocuments bool   `json:"include_documents"`
	PageSize         int    `json:"page_size"`
	PageToken        string `json:"page_token"`
}

type cloudDocumentSearchArguments struct {
	SourceID          string   `json:"source_id"`
	Query             string   `json:"query"`
	PageSize          int      `json:"page_size"`
	PageToken         string   `json:"page_token"`
	BindingID         string   `json:"binding_id"`
	TreeKey           string   `json:"tree_key"`
	StateFilter       []string `json:"state_filter"`
	IncludeDocuments  bool     `json:"include_documents"`
	IncludeContainers bool     `json:"include_containers"`
}

func skillListTool() ToolDefinition {
	return ToolDefinition{
		Name:        skillListToolName,
		Description: "List the authenticated user's LazyMind skills with optional metadata filters and pagination.",
		ReadOnly:    true,
		Annotations: ToolAnnotations{ReadOnlyHint: true},
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"keyword":    map[string]any{"type": "string", "description": "Optional metadata keyword."},
				"category":   map[string]any{"type": "string", "description": "Optional exact skill category."},
				"tags":       map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional tags; all must match."},
				"page_size":  map[string]any{"type": "integer", "minimum": contract.MinPageSize, "maximum": contract.MaxPageSize},
				"page_token": map[string]any{"type": "string", "description": "Opaque pagination token."},
			},
			"additionalProperties": false,
		},
	}
}

func skillGetTool() ToolDefinition {
	return ToolDefinition{
		Name:        skillGetToolName,
		Description: "Get metadata for one authenticated user's LazyMind skill. Skill file content is not included.",
		ReadOnly:    true,
		Annotations: ToolAnnotations{ReadOnlyHint: true},
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"skill_id": map[string]any{"type": "string", "description": "Stable LazyMind skill ID."},
			},
			"required":             []string{"skill_id"},
			"additionalProperties": false,
		},
	}
}

func knowledgeListTool() ToolDefinition {
	return ToolDefinition{
		Name:        knowledgeListToolName,
		Description: "List the authenticated user's LazyMind knowledge catalogs with optional metadata filters and pagination.",
		ReadOnly:    true,
		Annotations: ToolAnnotations{ReadOnlyHint: true},
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"keyword":    map[string]any{"type": "string", "description": "Optional catalog metadata keyword."},
				"tags":       map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional tags; all must match."},
				"page_size":  map[string]any{"type": "integer", "minimum": contract.MinPageSize, "maximum": contract.MaxPageSize},
				"page_token": map[string]any{"type": "string", "description": "Opaque pagination token."},
			},
			"additionalProperties": false,
		},
	}
}

func knowledgeGetTool() ToolDefinition {
	return ToolDefinition{
		Name:        knowledgeGetToolName,
		Description: "Get catalog metadata for one authenticated user's LazyMind knowledge base. Document content is not included.",
		ReadOnly:    true,
		Annotations: ToolAnnotations{ReadOnlyHint: true},
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"knowledge_id": map[string]any{"type": "string", "description": "Stable LazyMind knowledge catalog ID."},
			},
			"required":             []string{"knowledge_id"},
			"additionalProperties": false,
		},
	}
}

func knowledgeDocumentGetTool() ToolDefinition {
	return ToolDefinition{
		Name:        knowledgeDocumentGetToolName,
		Description: "Get metadata for one document in an authenticated LazyMind knowledge catalog. Content and chunks are not included.",
		ReadOnly:    true,
		Annotations: ToolAnnotations{ReadOnlyHint: true},
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"knowledge_id": map[string]any{"type": "string", "description": "Stable LazyMind knowledge catalog ID."},
				"document_id":  map[string]any{"type": "string", "description": "Stable LazyMind Core document ID."},
			},
			"required":             []string{"knowledge_id", "document_id"},
			"additionalProperties": false,
		},
	}
}

func knowledgeSearchTool() ToolDefinition {
	return ToolDefinition{
		Name:        knowledgeSearchToolName,
		Description: "Search the authenticated user's specified LazyMind knowledge catalogs and return mapped document chunks.",
		ReadOnly:    true,
		Annotations: ToolAnnotations{ReadOnlyHint: true},
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query":         map[string]any{"type": "string", "description": "Search query."},
				"knowledge_ids": map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "One or more stable LazyMind knowledge catalog IDs."},
				"top_k":         map[string]any{"type": "integer", "minimum": 1, "maximum": compatknowledge.MaxSearchTopK, "description": "Maximum result count."},
			},
			"required":             []string{"query", "knowledge_ids"},
			"additionalProperties": false,
		},
	}
}

func cloudDocumentListTool() ToolDefinition {
	return ToolDefinition{Name: cloudDocumentListToolName, Description: "List Cloud document sources visible to the authenticated caller.", ReadOnly: true, Annotations: ToolAnnotations{ReadOnlyHint: true}, InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{
			"keyword":    map[string]any{"type": "string", "description": "Optional Cloud source metadata keyword."},
			"status":     map[string]any{"type": "string", "description": "Optional source status."},
			"page_size":  map[string]any{"type": "integer", "minimum": contract.MinPageSize, "maximum": contract.MaxPageSize},
			"page_token": map[string]any{"type": "string", "description": "Opaque pagination token."},
		}, "additionalProperties": false,
	}}
}

func cloudDocumentGetTool() ToolDefinition {
	return ToolDefinition{Name: cloudDocumentGetToolName, Description: "Get Cloud source metadata and, optionally, one page of document metadata. Document body content is not included.", ReadOnly: true, Annotations: ToolAnnotations{ReadOnlyHint: true}, InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{
			"source_id":         map[string]any{"type": "string", "description": "Stable Cloud source ID."},
			"include_documents": map[string]any{"type": "boolean", "description": "Include one page of document metadata."},
			"page_size":         map[string]any{"type": "integer", "minimum": contract.MinPageSize, "maximum": contract.MaxPageSize},
			"page_token":        map[string]any{"type": "string", "description": "Opaque document pagination token."},
		}, "required": []string{"source_id"}, "additionalProperties": false,
	}}
}

func cloudDocumentSearchTool() ToolDefinition {
	return ToolDefinition{Name: cloudDocumentSearchToolName, Description: "Search titles, display names, and tree metadata within one Cloud source. This does not search document body content.", ReadOnly: true, Annotations: ToolAnnotations{ReadOnlyHint: true}, InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{
			"source_id":          map[string]any{"type": "string", "description": "Stable Cloud source ID."},
			"query":              map[string]any{"type": "string", "description": "Cloud metadata search query."},
			"page_size":          map[string]any{"type": "integer", "minimum": contract.MinPageSize, "maximum": contract.MaxPageSize},
			"page_token":         map[string]any{"type": "string", "description": "Opaque pagination token."},
			"binding_id":         map[string]any{"type": "string", "description": "Optional source binding filter."},
			"tree_key":           map[string]any{"type": "string", "description": "Optional tree root filter."},
			"state_filter":       map[string]any{"type": "array", "items": map[string]any{"type": "string"}, "description": "Optional source state filters."},
			"include_documents":  map[string]any{"type": "boolean"},
			"include_containers": map[string]any{"type": "boolean"},
		}, "required": []string{"source_id", "query"}, "additionalProperties": false,
	}}
}

func (s *Server) callTool(ctx context.Context, request rpcRequest) rpcResponse {
	var params toolCallParams
	if err := json.Unmarshal(request.Params, &params); err != nil {
		return s.errorResponse(request.ID, newRPCError(rpcInvalidParams, "invalid tools/call parameters"))
	}
	if _, ok := s.registry.Get(params.Name); !ok {
		return s.resultResponse(request.ID, toolErrorResult("NOT_FOUND", "Unknown tool."))
	}
	requestHTTP, ok := requestFromContext(ctx)
	if !ok {
		return s.resultResponse(request.ID, toolErrorResult("UNAUTHENTICATED", "Authenticated principal is required."))
	}
	principal, err := s.identity.Principal(ctx, requestHTTP)
	if err != nil {
		return s.resultResponse(request.ID, toolErrorResult("UNAUTHENTICATED", "Authenticated principal is required."))
	}
	callCtx, err := callContext(principal, requestHTTP.Header.Get("X-Request-Id"))
	if err != nil {
		return s.resultResponse(request.ID, toolErrorResult("UNAUTHENTICATED", "Authenticated principal is required."))
	}

	switch params.Name {
	case skillListToolName:
		return s.callSkillList(ctx, request.ID, params.Arguments, callCtx)
	case skillGetToolName:
		return s.callSkillGet(ctx, request.ID, params.Arguments, callCtx)
	case knowledgeListToolName:
		return s.callKnowledgeList(ctx, request.ID, params.Arguments, callCtx)
	case knowledgeGetToolName:
		return s.callKnowledgeGet(ctx, request.ID, params.Arguments, callCtx)
	case knowledgeDocumentGetToolName:
		return s.callKnowledgeDocumentGet(ctx, request.ID, params.Arguments, callCtx)
	case knowledgeSearchToolName:
		return s.callKnowledgeSearch(ctx, request.ID, params.Arguments, callCtx)
	case cloudDocumentListToolName:
		return s.callCloudDocumentList(ctx, request.ID, params.Arguments, callCtx)
	case cloudDocumentGetToolName:
		return s.callCloudDocumentGet(ctx, request.ID, params.Arguments, callCtx)
	case cloudDocumentSearchToolName:
		return s.callCloudDocumentSearch(ctx, request.ID, params.Arguments, callCtx)
	default:
		return s.resultResponse(request.ID, toolErrorResult("NOT_FOUND", "Unknown tool."))
	}
}

func (s *Server) callCloudDocumentList(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.CloudDocument == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Cloud document tools are not configured."))
	}
	args, err := decodeCloudDocumentListArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.CloudDocument.List(ctx, callCtx, compatcloud.ListInput{Keyword: args.Keyword, Status: args.Status, Page: contract.PageRequest{PageSize: args.PageSize, PageToken: args.PageToken}})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, cloudDocumentListResult(result))
}

func (s *Server) callCloudDocumentGet(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.CloudDocument == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Cloud document tools are not configured."))
	}
	args, err := decodeCloudDocumentGetArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.CloudDocument.Get(ctx, callCtx, compatcloud.GetInput{SourceID: args.SourceID, IncludeDocuments: args.IncludeDocuments, DocumentsPage: contract.PageRequest{PageSize: args.PageSize, PageToken: args.PageToken}})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, cloudDocumentGetResult(result))
}

func (s *Server) callCloudDocumentSearch(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.CloudDocument == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Cloud document tools are not configured."))
	}
	args, err := decodeCloudDocumentSearchArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.CloudDocument.Search(ctx, callCtx, compatcloud.SearchInput{SourceID: args.SourceID, Query: args.Query, Page: contract.PageRequest{PageSize: args.PageSize, PageToken: args.PageToken}, BindingID: args.BindingID, TreeKey: args.TreeKey, StateFilter: args.StateFilter, IncludeDocuments: args.IncludeDocuments, IncludeContainers: args.IncludeContainers})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, cloudDocumentSearchResult(result))
}

func (s *Server) callKnowledgeDocumentGet(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.Knowledge == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Knowledge tools are not configured."))
	}
	args, err := decodeKnowledgeDocumentGetArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.Knowledge.GetDocument(ctx, callCtx, compatknowledge.GetDocumentInput{
		KnowledgeID: args.KnowledgeID, DocumentID: args.DocumentID,
		IncludeContent: false, IncludeChunks: false,
	})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, knowledgeDocumentGetResult(result))
}

func (s *Server) callKnowledgeSearch(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.Knowledge == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Knowledge tools are not configured."))
	}
	args, err := decodeKnowledgeSearchArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.Knowledge.Search(ctx, callCtx, compatknowledge.SearchInput{
		Query:        args.Query,
		KnowledgeIDs: args.KnowledgeIDs,
		TopK:         args.TopK,
	})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, knowledgeSearchResult(result))
}

func (s *Server) callKnowledgeList(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.Knowledge == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Knowledge tools are not configured."))
	}
	args, err := decodeKnowledgeListArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.Knowledge.List(ctx, callCtx, compatknowledge.ListInput{
		Keyword: args.Keyword,
		Tags:    args.Tags,
		Page:    contract.PageRequest{PageSize: args.PageSize, PageToken: args.PageToken},
	})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, knowledgeListResult(result))
}

func (s *Server) callKnowledgeGet(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.Knowledge == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Knowledge tools are not configured."))
	}
	args, err := decodeKnowledgeGetArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.Knowledge.Get(ctx, callCtx, compatknowledge.GetInput{KnowledgeID: args.KnowledgeID})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, knowledgeGetResult(result))
}

func (s *Server) callSkillGet(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.Skill == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Skill tools are not configured."))
	}
	args, err := decodeSkillGetArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.Skill.Get(ctx, callCtx, compatskill.GetInput{SkillID: args.SkillID, IncludeContent: false})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, skillGetResult(result))
}

func (s *Server) callSkillList(ctx context.Context, requestID json.RawMessage, raw json.RawMessage, callCtx contract.CallContext) rpcResponse {
	if s.runtime.Skill == nil {
		return s.resultResponse(requestID, toolErrorResult("UNSUPPORTED", "Skill tools are not configured."))
	}
	args, err := decodeSkillListArguments(raw)
	if err != nil {
		return s.resultResponse(requestID, toolErrorResult("INVALID_ARGUMENT", "Invalid tool arguments."))
	}
	result, err := s.runtime.Skill.List(ctx, callCtx, compatskill.ListInput{
		Keyword:  args.Keyword,
		Category: args.Category,
		Tags:     args.Tags,
		Page:     contract.PageRequest{PageSize: args.PageSize, PageToken: args.PageToken},
	})
	if err != nil {
		return s.resultResponse(requestID, toolErrorFromCompat(err))
	}
	return s.resultResponse(requestID, skillListResult(result))
}

func decodeSkillListArguments(raw json.RawMessage) (skillListArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return skillListArguments{}, nil
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	var args skillListArguments
	if err := decoder.Decode(&args); err != nil {
		return skillListArguments{}, fmt.Errorf("decode skill list arguments: %w", err)
	}
	if decoder.More() {
		return skillListArguments{}, fmt.Errorf("multiple JSON values")
	}
	return args, nil
}

func decodeSkillGetArguments(raw json.RawMessage) (skillGetArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return skillGetArguments{}, fmt.Errorf("skill_id is required")
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	var args skillGetArguments
	if err := decoder.Decode(&args); err != nil {
		return skillGetArguments{}, fmt.Errorf("decode skill get arguments: %w", err)
	}
	if strings.TrimSpace(args.SkillID) == "" {
		return skillGetArguments{}, fmt.Errorf("skill_id is required")
	}
	return args, nil
}

func decodeKnowledgeListArguments(raw json.RawMessage) (knowledgeListArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return knowledgeListArguments{}, nil
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	var args knowledgeListArguments
	if err := decoder.Decode(&args); err != nil {
		return knowledgeListArguments{}, fmt.Errorf("decode knowledge list arguments: %w", err)
	}
	if decoder.More() {
		return knowledgeListArguments{}, fmt.Errorf("multiple JSON values")
	}
	return args, nil
}

func decodeKnowledgeGetArguments(raw json.RawMessage) (knowledgeGetArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return knowledgeGetArguments{}, fmt.Errorf("knowledge_id is required")
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	var args knowledgeGetArguments
	if err := decoder.Decode(&args); err != nil {
		return knowledgeGetArguments{}, fmt.Errorf("decode knowledge get arguments: %w", err)
	}
	if strings.TrimSpace(args.KnowledgeID) == "" {
		return knowledgeGetArguments{}, fmt.Errorf("knowledge_id is required")
	}
	return args, nil
}

func decodeKnowledgeDocumentGetArguments(raw json.RawMessage) (knowledgeDocumentGetArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return knowledgeDocumentGetArguments{}, fmt.Errorf("knowledge_id and document_id are required")
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	var args knowledgeDocumentGetArguments
	if err := decoder.Decode(&args); err != nil {
		return knowledgeDocumentGetArguments{}, fmt.Errorf("decode knowledge document get arguments: %w", err)
	}
	if strings.TrimSpace(args.KnowledgeID) == "" || strings.TrimSpace(args.DocumentID) == "" {
		return knowledgeDocumentGetArguments{}, fmt.Errorf("knowledge_id and document_id are required")
	}
	return args, nil
}

func decodeKnowledgeSearchArguments(raw json.RawMessage) (knowledgeSearchArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return knowledgeSearchArguments{}, fmt.Errorf("query and knowledge_ids are required")
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	var args knowledgeSearchArguments
	if err := decoder.Decode(&args); err != nil {
		return knowledgeSearchArguments{}, fmt.Errorf("decode knowledge search arguments: %w", err)
	}
	if strings.TrimSpace(args.Query) == "" || len(args.KnowledgeIDs) == 0 {
		return knowledgeSearchArguments{}, fmt.Errorf("query and knowledge_ids are required")
	}
	return args, nil
}

func decodeCloudDocumentListArguments(raw json.RawMessage) (cloudDocumentListArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return cloudDocumentListArguments{}, nil
	}
	var args cloudDocumentListArguments
	if err := decodeStrictArguments(raw, &args); err != nil {
		return cloudDocumentListArguments{}, err
	}
	return args, nil
}

func decodeCloudDocumentGetArguments(raw json.RawMessage) (cloudDocumentGetArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return cloudDocumentGetArguments{}, fmt.Errorf("source_id is required")
	}
	var args cloudDocumentGetArguments
	if err := decodeStrictArguments(raw, &args); err != nil {
		return cloudDocumentGetArguments{}, err
	}
	if strings.TrimSpace(args.SourceID) == "" {
		return cloudDocumentGetArguments{}, fmt.Errorf("source_id is required")
	}
	return args, nil
}

func decodeCloudDocumentSearchArguments(raw json.RawMessage) (cloudDocumentSearchArguments, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return cloudDocumentSearchArguments{}, fmt.Errorf("source_id and query are required")
	}
	var args cloudDocumentSearchArguments
	if err := decodeStrictArguments(raw, &args); err != nil {
		return cloudDocumentSearchArguments{}, err
	}
	if strings.TrimSpace(args.SourceID) == "" || strings.TrimSpace(args.Query) == "" {
		return cloudDocumentSearchArguments{}, fmt.Errorf("source_id and query are required")
	}
	return args, nil
}

func decodeStrictArguments(raw json.RawMessage, target any) error {
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if decoder.More() {
		return fmt.Errorf("multiple JSON values")
	}
	return nil
}

type requestContextKey struct{}

func requestFromContext(ctx context.Context) (*http.Request, bool) {
	request, ok := ctx.Value(requestContextKey{}).(*http.Request)
	return request, ok && request != nil
}
