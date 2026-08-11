package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"lazymind/core/compat/contract"
	compatskill "lazymind/core/compat/skill"
)

const skillListToolName = "lazymind_skill_list"

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

func skillListTool() ToolDefinition {
	return ToolDefinition{
		Name:        skillListToolName,
		Description: "List the authenticated user's LazyMind skills with optional metadata filters and pagination.",
		ReadOnly:    true,
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
	default:
		return s.resultResponse(request.ID, toolErrorResult("NOT_FOUND", "Unknown tool."))
	}
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

type requestContextKey struct{}

func requestFromContext(ctx context.Context) (*http.Request, bool) {
	request, ok := ctx.Value(requestContextKey{}).(*http.Request)
	return request, ok && request != nil
}
