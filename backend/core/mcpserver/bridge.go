package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"lazymind/core/compat/contract"
	compatknowledge "lazymind/core/compat/knowledge"
	compatskill "lazymind/core/compat/skill"
)

const (
	skillListToolName     = "lazymind_skill_list"
	skillGetToolName      = "lazymind_skill_get"
	knowledgeListToolName = "lazymind_knowledge_list"
	knowledgeGetToolName  = "lazymind_knowledge_get"
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
	default:
		return s.resultResponse(request.ID, toolErrorResult("NOT_FOUND", "Unknown tool."))
	}
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

type requestContextKey struct{}

func requestFromContext(ctx context.Context) (*http.Request, bool) {
	request, ok := ctx.Value(requestContextKey{}).(*http.Request)
	return request, ok && request != nil
}
