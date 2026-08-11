package mcpserver

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"lazymind/core/compat/contract"
	compatruntime "lazymind/core/compat/runtime"
	compatskill "lazymind/core/compat/skill"
)

type fakeSkillPort struct {
	listResult compatskill.ListResult
	listErr    error
	callCtx    contract.CallContext
	input      compatskill.ListInput
}

func (p *fakeSkillPort) List(_ context.Context, callCtx contract.CallContext, input compatskill.ListInput) (compatskill.ListResult, error) {
	p.callCtx, p.input = callCtx, input
	return p.listResult, p.listErr
}

func (p *fakeSkillPort) GetMetadata(context.Context, contract.CallContext, string) (compatskill.Summary, error) {
	return compatskill.Summary{}, errors.New("not implemented")
}

func (p *fakeSkillPort) ReadContent(context.Context, contract.CallContext, string, string) (compatskill.Content, error) {
	return compatskill.Content{}, errors.New("not implemented")
}

func testServer(t *testing.T, port *fakeSkillPort) *Server {
	t.Helper()
	rt, err := compatruntime.New(compatruntime.Dependencies{SkillPort: port})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	server, err := New(rt, HeaderIdentityProvider{}, Options{ServerName: "test-server", ServerVersion: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return server
}

func TestInitializeCapabilities(t *testing.T) {
	server := testServer(t, &fakeSkillPort{})
	response := server.Handle(context.Background(), rpcRequest{JSONRPC: "2.0", ID: json.RawMessage("1"), Method: "initialize"})
	if response.Error != nil {
		t.Fatalf("initialize error: %#v", response.Error)
	}
	result := response.Result.(map[string]any)
	if result["protocolVersion"] != protocolVersion {
		t.Fatalf("protocolVersion = %v", result["protocolVersion"])
	}
	capabilities := result["capabilities"].(map[string]any)
	if _, ok := capabilities["tools"]; !ok {
		t.Fatalf("tools capability missing: %#v", capabilities)
	}
}

func TestToolsListPublishesSkillListSchemaWithoutIdentityFields(t *testing.T) {
	server := testServer(t, &fakeSkillPort{})
	response := server.Handle(context.Background(), rpcRequest{JSONRPC: "2.0", ID: json.RawMessage("1"), Method: "tools/list"})
	tools := response.Result.(map[string]any)["tools"].([]ToolDefinition)
	if len(tools) != 1 || tools[0].Name != skillListToolName {
		t.Fatalf("tools = %#v", tools)
	}
	if tools[0].Description == "" || !tools[0].ReadOnly {
		t.Fatalf("tool metadata = %#v", tools[0])
	}
	properties := tools[0].InputSchema["properties"].(map[string]any)
	for _, field := range []string{"keyword", "category", "tags", "page_size", "page_token"} {
		if _, ok := properties[field]; !ok {
			t.Fatalf("schema missing %q: %#v", field, properties)
		}
	}
	for _, forbidden := range []string{"user_id", "user_name"} {
		if _, ok := properties[forbidden]; ok {
			t.Fatalf("schema exposes forbidden identity field %q", forbidden)
		}
	}
	if tools[0].InputSchema["additionalProperties"] != false {
		t.Fatalf("additionalProperties must be false")
	}
}

func TestRegistryRejectsDuplicateTool(t *testing.T) {
	registry := NewRegistry()
	tool := skillListTool()
	if err := registry.Register(tool); err != nil {
		t.Fatalf("first registration: %v", err)
	}
	if err := registry.Register(tool); err == nil {
		t.Fatal("duplicate registration succeeded")
	}
}

func TestToolsCallSkillListUsesPrincipalNotArguments(t *testing.T) {
	total := int64(1)
	port := &fakeSkillPort{listResult: compatskill.ListResult{Items: []compatskill.Summary{{ID: "skill-1", Name: "One"}}, Page: contract.PageResult{Total: &total}}}
	server := testServer(t, port)
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_skill_list","arguments":{"keyword":" hello ","category":"dev","tags":["go"],"page_size":5,"page_token":"offset:5","user_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-User-Name": "Trusted", "X-Request-Id": "req-1"})
	result := response.Result.(map[string]any)
	if result["isError"] != true {
		t.Fatalf("user_id argument must be rejected, got %#v", result)
	}

	response = callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_skill_list","arguments":{"keyword":" hello ","category":"dev","tags":["go"],"page_size":5,"page_token":"offset:5"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-User-Name": "Trusted", "X-Request-Id": "req-1"})
	result = response.Result.(map[string]any)
	if result["isError"] == true {
		t.Fatalf("tools/call error: %#v", result)
	}
	if port.callCtx.UserID != "trusted-user" || port.callCtx.UserName != "Trusted" || port.callCtx.RequestID != "req-1" {
		t.Fatalf("call context = %#v", port.callCtx)
	}
	if port.input.Keyword != "hello" || port.input.Category != "dev" || port.input.Page.PageSize != 5 || port.input.Page.PageToken != "offset:5" {
		t.Fatalf("input = %#v", port.input)
	}
	encoded, err := json.Marshal(result["structuredContent"])
	if err != nil || !strings.Contains(string(encoded), `"skill-1"`) || !strings.Contains(string(encoded), `"total":1`) {
		t.Fatalf("structured result = %s, err=%v", encoded, err)
	}
}

func TestToolsCallRejectsUnknownMalformedInvalidAndUnauthenticated(t *testing.T) {
	server := testServer(t, &fakeSkillPort{})
	unknown := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"unknown","arguments":{}}}`, map[string]string{"X-User-Id": "user"})
	if !unknown.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("unknown tool result = %#v", unknown.Result)
	}
	invalid := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_skill_list","arguments":{"page_size":"wrong"}}}`, map[string]string{"X-User-Id": "user"})
	if !invalid.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("invalid arguments result = %#v", invalid.Result)
	}
	unauthenticated := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_skill_list","arguments":{}}}`, nil)
	if !unauthenticated.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("missing principal result = %#v", unauthenticated.Result)
	}

	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(`{"jsonrpc":`))
	recorder := httptest.NewRecorder()
	server.StreamableHTTPHandler(TransportOptions{}).ServeHTTP(recorder, req)
	var malformed rpcResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &malformed); err != nil {
		t.Fatalf("decode malformed response: %v", err)
	}
	if malformed.Error == nil || malformed.Error.Code != rpcParseError {
		t.Fatalf("malformed response = %#v", malformed)
	}
}

func TestCallContextRequiresPrincipal(t *testing.T) {
	if _, err := callContext(Principal{}, "request"); err == nil {
		t.Fatal("empty principal was accepted")
	}
	ctx, err := callContext(Principal{UserID: " user ", UserName: " name "}, " request ")
	if err != nil || ctx.UserID != "user" || ctx.UserName != "name" || ctx.RequestID != "request" {
		t.Fatalf("CallContext = %#v, err=%v", ctx, err)
	}
}

func TestCompatErrorsAreSafeAndStable(t *testing.T) {
	for _, code := range []contract.ErrorCode{contract.InvalidArgument, contract.NotFound, contract.Conflict, contract.BackendUnavailable, contract.Unsupported, contract.Internal} {
		t.Run(string(code), func(t *testing.T) {
			result := toolErrorFromCompat(contract.NewError(code, "internal.operation", "postgres /private/path stacktrace", false, errors.New("SQL password=secret")))
			if !result.IsError {
				t.Fatal("expected tool error")
			}
			raw, err := json.Marshal(result)
			if err != nil {
				t.Fatalf("marshal: %v", err)
			}
			if strings.Contains(string(raw), "secret") || strings.Contains(string(raw), "postgres") || strings.Contains(string(raw), "stacktrace") {
				t.Fatalf("unsafe error leaked: %s", raw)
			}
			if !strings.Contains(string(raw), string(code)) {
				t.Fatalf("error code missing: %s", raw)
			}
		})
	}
}

func TestSkillListResultIsStructuredAndStableForEmptyList(t *testing.T) {
	total := int64(0)
	result := skillListResult(compatskill.ListResult{Page: contract.PageResult{Total: &total}})
	if result.IsError || len(result.Content) != 1 || result.Content[0].Text == "" {
		t.Fatalf("result = %#v", result)
	}
	structured, ok := result.StructuredContent.(skillListStructuredResult)
	if !ok || structured.Items == nil || len(structured.Items) != 0 || structured.Page.Total == nil || *structured.Page.Total != 0 {
		t.Fatalf("structured = %#v", result.StructuredContent)
	}
}

func callHTTP(t *testing.T, server *Server, body string, headers map[string]string) rpcResponse {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/mcp", strings.NewReader(body))
	for key, value := range headers {
		req.Header.Set(key, value)
	}
	recorder := httptest.NewRecorder()
	server.StreamableHTTPHandler(TransportOptions{}).ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, body=%s", recorder.Code, recorder.Body.String())
	}
	var response rpcResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response: %v (%s)", err, recorder.Body.String())
	}
	if response.Error != nil {
		t.Fatalf("rpc error = %#v", response.Error)
	}
	return response
}
