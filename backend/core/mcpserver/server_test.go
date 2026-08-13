package mcpserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	compatcloud "lazymind/core/compat/clouddocument"
	"lazymind/core/compat/contract"
	compatknowledge "lazymind/core/compat/knowledge"
	compatruntime "lazymind/core/compat/runtime"
	compatskill "lazymind/core/compat/skill"
)

type fakeSkillPort struct {
	listResult compatskill.ListResult
	listErr    error
	getResult  compatskill.Summary
	getErr     error
	callCtx    contract.CallContext
	input      compatskill.ListInput
	getCallCtx contract.CallContext
	getSkillID string
}

type fakeKnowledgeCatalogPort struct {
	listResult  compatknowledge.ListResult
	listErr     error
	getResult   compatknowledge.GetResult
	getErr      error
	listCallCtx contract.CallContext
	listInput   compatknowledge.ListInput
	getCallCtx  contract.CallContext
	getInput    compatknowledge.GetInput
}

type fakeKnowledgeSearchPort struct {
	result  compatknowledge.SearchResult
	err     error
	callCtx contract.CallContext
	input   compatknowledge.SearchInput
}

type fakeKnowledgeDocumentPort struct {
	result  compatknowledge.GetDocumentResult
	err     error
	callCtx contract.CallContext
	input   compatknowledge.GetDocumentInput
}

type fakeCloudDocumentPort struct {
	listResult                                              compatcloud.ListResult
	getResult                                               compatcloud.SourceDetail
	docResult                                               compatcloud.DocumentListResult
	searchResult                                            compatcloud.SearchResult
	listErr, getErr, docErr, searchErr                      error
	listCallCtx, getCallCtx, documentCallCtx, searchCallCtx contract.CallContext
	listInput                                               compatcloud.ListInput
	getSourceID                                             string
	documentInput                                           compatcloud.GetInput
	searchInput                                             compatcloud.SearchInput
}

func (p *fakeCloudDocumentPort) ListSources(_ context.Context, callCtx contract.CallContext, input compatcloud.ListInput) (compatcloud.ListResult, error) {
	p.listCallCtx, p.listInput = callCtx, input
	return p.listResult, p.listErr
}
func (p *fakeCloudDocumentPort) GetSource(_ context.Context, callCtx contract.CallContext, sourceID string) (compatcloud.SourceDetail, error) {
	p.getCallCtx, p.getSourceID = callCtx, sourceID
	return p.getResult, p.getErr
}
func (p *fakeCloudDocumentPort) ListDocuments(_ context.Context, callCtx contract.CallContext, _ compatcloud.SourceDetail, input compatcloud.GetInput) (compatcloud.DocumentListResult, error) {
	p.documentCallCtx, p.documentInput = callCtx, input
	return p.docResult, p.docErr
}
func (p *fakeCloudDocumentPort) Search(_ context.Context, callCtx contract.CallContext, input compatcloud.SearchInput) (compatcloud.SearchResult, error) {
	p.searchCallCtx, p.searchInput = callCtx, input
	return p.searchResult, p.searchErr
}

func (p *fakeKnowledgeDocumentPort) GetDocument(_ context.Context, callCtx contract.CallContext, input compatknowledge.GetDocumentInput) (compatknowledge.GetDocumentResult, error) {
	p.callCtx, p.input = callCtx, input
	return p.result, p.err
}

func (p *fakeKnowledgeSearchPort) Search(_ context.Context, callCtx contract.CallContext, input compatknowledge.SearchInput) (compatknowledge.SearchResult, error) {
	p.callCtx, p.input = callCtx, input
	return p.result, p.err
}

func (p *fakeKnowledgeCatalogPort) List(_ context.Context, callCtx contract.CallContext, input compatknowledge.ListInput) (compatknowledge.ListResult, error) {
	p.listCallCtx, p.listInput = callCtx, input
	return p.listResult, p.listErr
}

func (p *fakeKnowledgeCatalogPort) Get(_ context.Context, callCtx contract.CallContext, input compatknowledge.GetInput) (compatknowledge.GetResult, error) {
	p.getCallCtx, p.getInput = callCtx, input
	return p.getResult, p.getErr
}

func (p *fakeSkillPort) List(_ context.Context, callCtx contract.CallContext, input compatskill.ListInput) (compatskill.ListResult, error) {
	p.callCtx, p.input = callCtx, input
	return p.listResult, p.listErr
}

func (p *fakeSkillPort) GetMetadata(_ context.Context, callCtx contract.CallContext, skillID string) (compatskill.Summary, error) {
	p.getCallCtx, p.getSkillID = callCtx, skillID
	return p.getResult, p.getErr
}

func (p *fakeSkillPort) ReadContent(context.Context, contract.CallContext, string, string) (compatskill.Content, error) {
	return compatskill.Content{}, errors.New("not implemented")
}

func testServer(t *testing.T, port *fakeSkillPort) *Server {
	return testServerWithKnowledgeSearch(t, port, &fakeKnowledgeCatalogPort{}, &fakeKnowledgeSearchPort{})
}

func testServerWithKnowledge(t *testing.T, port *fakeSkillPort, catalog *fakeKnowledgeCatalogPort) *Server {
	return testServerWithKnowledgeSearch(t, port, catalog, &fakeKnowledgeSearchPort{})
}

func testServerWithKnowledgeSearch(t *testing.T, port *fakeSkillPort, catalog *fakeKnowledgeCatalogPort, search *fakeKnowledgeSearchPort) *Server {
	t.Helper()
	rt, err := compatruntime.New(compatruntime.Dependencies{SkillPort: port, KnowledgeCatalog: catalog, KnowledgeSearch: search})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	server, err := New(rt, HeaderIdentityProvider{}, Options{ServerName: "test-server", ServerVersion: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return server
}

func testServerWithKnowledgeDocument(t *testing.T, document *fakeKnowledgeDocumentPort) *Server {
	t.Helper()
	rt, err := compatruntime.New(compatruntime.Dependencies{SkillPort: &fakeSkillPort{}, KnowledgeDocument: document})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	server, err := New(rt, HeaderIdentityProvider{}, Options{ServerName: "test-server", ServerVersion: "test"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return server
}

func testServerWithCloudDocument(t *testing.T, cloud *fakeCloudDocumentPort) *Server {
	t.Helper()
	rt, err := compatruntime.New(compatruntime.Dependencies{SkillPort: &fakeSkillPort{}, CloudDocumentPort: cloud})
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
	if len(tools) != 9 {
		t.Fatalf("tools = %#v", tools)
	}
	byName := toolDefinitionsByName(tools)
	listTool := byName[skillListToolName]
	if listTool.Description == "" || !listTool.ReadOnly || !listTool.Annotations.ReadOnlyHint {
		t.Fatalf("tool metadata = %#v", listTool)
	}
	properties := listTool.InputSchema["properties"].(map[string]any)
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
	if listTool.InputSchema["additionalProperties"] != false {
		t.Fatalf("additionalProperties must be false")
	}
	getTool := byName[skillGetToolName]
	getSchema := getTool.InputSchema
	if getSchema["additionalProperties"] != false || !getTool.Annotations.ReadOnlyHint {
		t.Fatalf("skill get schema = %#v", getSchema)
	}
	if _, ok := getSchema["properties"].(map[string]any)["skill_id"]; !ok {
		t.Fatalf("skill get schema missing skill_id: %#v", getSchema)
	}
}

func TestToolsListPublishesKnowledgeSchemasAndReadOnlyAnnotations(t *testing.T) {
	server := testServer(t, &fakeSkillPort{})
	response := server.Handle(context.Background(), rpcRequest{JSONRPC: "2.0", ID: json.RawMessage("1"), Method: "tools/list"})
	tools := response.Result.(map[string]any)["tools"].([]ToolDefinition)
	byName := toolDefinitionsByName(tools)
	for _, name := range []string{knowledgeDocumentGetToolName, knowledgeGetToolName, knowledgeListToolName, knowledgeSearchToolName} {
		tool := byName[name]
		if !tool.Annotations.ReadOnlyHint || tool.InputSchema["additionalProperties"] != false {
			t.Fatalf("knowledge tool metadata = %#v", tool)
		}
		properties := tool.InputSchema["properties"].(map[string]any)
		for _, forbidden := range []string{"user_id", "user_name", "tenant_id"} {
			if _, ok := properties[forbidden]; ok {
				t.Fatalf("schema exposes %q", forbidden)
			}
		}
	}
	if _, ok := byName[knowledgeListToolName].InputSchema["properties"].(map[string]any)["keyword"]; !ok {
		t.Fatalf("knowledge list schema = %#v", byName[knowledgeListToolName].InputSchema)
	}
	if _, ok := byName[knowledgeListToolName].InputSchema["properties"].(map[string]any)["tags"]; !ok {
		t.Fatalf("knowledge list schema = %#v", byName[knowledgeListToolName].InputSchema)
	}
	if _, ok := byName[knowledgeGetToolName].InputSchema["properties"].(map[string]any)["knowledge_id"]; !ok {
		t.Fatalf("knowledge get schema = %#v", byName[knowledgeGetToolName].InputSchema)
	}
	searchProperties := byName[knowledgeSearchToolName].InputSchema["properties"].(map[string]any)
	for _, field := range []string{"query", "knowledge_ids", "top_k"} {
		if _, ok := searchProperties[field]; !ok {
			t.Fatalf("knowledge search schema missing %q: %#v", field, searchProperties)
		}
	}
	raw, err := json.Marshal(response.Result)
	if err != nil || !strings.Contains(string(raw), `"annotations":{"readOnlyHint":true}`) {
		t.Fatalf("tools/list wire annotations = %s, err=%v", raw, err)
	}
}

func TestToolsCallKnowledgeDocumentGetMetadataOnlyUsesPrincipal(t *testing.T) {
	document := &fakeKnowledgeDocumentPort{result: compatknowledge.GetDocumentResult{Document: compatknowledge.DocumentDetail{
		ID: "doc-1", KnowledgeID: "knowledge-1", Name: "Readme", MIMEType: "text/plain",
		Content: &compatknowledge.DocumentContent{Text: "must not leak"}, Chunks: []compatknowledge.DocumentChunk{{ID: "chunk-1", Text: "must not leak"}},
	}}}
	server := testServerWithKnowledgeDocument(t, document)
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_knowledge_document_get","arguments":{"knowledge_id":"knowledge-1","document_id":"doc-1"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || document.callCtx.UserID != "trusted-user" || document.callCtx.TenantID != "tenant-a" || document.input.IncludeContent || document.input.IncludeChunks {
		t.Fatalf("result=%#v callCtx=%#v input=%#v", result, document.callCtx, document.input)
	}
	raw, err := json.Marshal(result["structuredContent"])
	if err != nil || !strings.Contains(string(raw), `"document_id":"doc-1"`) || strings.Contains(string(raw), "must not leak") || strings.Contains(string(raw), "chunks") {
		t.Fatalf("metadata-only structured result=%s err=%v", raw, err)
	}

	for _, arguments := range []string{
		`{"knowledge_id":"knowledge-1"}`,
		`{"document_id":"doc-1"}`,
		`{"knowledge_id":"knowledge-1","document_id":"doc-1","tenant_id":"attacker"}`,
	} {
		response = callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_knowledge_document_get","arguments":`+arguments+`}}`, map[string]string{"X-User-Id": "trusted-user"})
		if response.Result.(map[string]any)["isError"] != true {
			t.Fatalf("arguments %s accepted: %#v", arguments, response.Result)
		}
	}
	response = callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_knowledge_document_get","arguments":{"knowledge_id":"knowledge-1","document_id":"missing"}}}`, nil)
	if response.Result.(map[string]any)["isError"] != true {
		t.Fatalf("missing principal accepted: %#v", response.Result)
	}

	document.err = contract.NewError(contract.NotFound, "knowledge.document.get", "SQL /private/path secret", false, errors.New("postgres password=secret"))
	response = callHTTP(t, server, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"lazymind_knowledge_document_get","arguments":{"knowledge_id":"knowledge-1","document_id":"missing"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safe, _ := json.Marshal(response.Result)
	if !strings.Contains(string(safe), "NOT_FOUND") || strings.Contains(string(safe), "secret") || strings.Contains(string(safe), "private") || strings.Contains(string(safe), "SQL") {
		t.Fatalf("unsafe document error=%s", safe)
	}
	document.err = contract.NewError(contract.BackendUnavailable, "knowledge.document.get", "readonly database /private/path", true, errors.New("postgres password=secret"))
	response = callHTTP(t, server, `{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"lazymind_knowledge_document_get","arguments":{"knowledge_id":"knowledge-1","document_id":"doc-1"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safe, _ = json.Marshal(response.Result)
	if !strings.Contains(string(safe), "BACKEND_UNAVAILABLE") || strings.Contains(string(safe), "secret") || strings.Contains(string(safe), "private") || strings.Contains(string(safe), "database") {
		t.Fatalf("unsafe document unavailable error=%s", safe)
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

func TestToolsCallSkillGetMetadataOnlyUsesPrincipal(t *testing.T) {
	port := &fakeSkillPort{getResult: compatskill.Summary{ID: "skill-1", Name: "One", HeadRevisionID: "rev-1", Enabled: true}}
	server := testServer(t, port)
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_skill_get","arguments":{"skill_id":"skill-1"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || port.getCallCtx.UserID != "trusted-user" || port.getSkillID != "skill-1" {
		t.Fatalf("result=%#v callCtx=%#v skillID=%q", result, port.getCallCtx, port.getSkillID)
	}
	raw, err := json.Marshal(result["structuredContent"])
	if err != nil || !strings.Contains(string(raw), `"skill-1"`) || strings.Contains(string(raw), "content") {
		t.Fatalf("structured result=%s err=%v", raw, err)
	}

	invalid := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_skill_get","arguments":{"skill_id":"skill-1","user_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	if !invalid.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("identity override was accepted: %#v", invalid.Result)
	}
}

func TestToolsCallSkillGetMapsNotFoundAndInvalid(t *testing.T) {
	server := testServer(t, &fakeSkillPort{getErr: contract.NewError(contract.NotFound, "skill.get", "sql details", false, errors.New("postgres secret"))})
	notFound := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_skill_get","arguments":{"skill_id":"missing"}}}`, map[string]string{"X-User-Id": "user"})
	raw, _ := json.Marshal(notFound.Result)
	if !strings.Contains(string(raw), "NOT_FOUND") || strings.Contains(string(raw), "secret") {
		t.Fatalf("not found result=%s", raw)
	}
	invalid := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_skill_get","arguments":{}}}`, map[string]string{"X-User-Id": "user"})
	if !invalid.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("invalid result=%#v", invalid.Result)
	}
}

func TestToolsCallKnowledgeListUsesPrincipalAndTenantNotArguments(t *testing.T) {
	total := int64(1)
	catalog := &fakeKnowledgeCatalogPort{listResult: compatknowledge.ListResult{
		Items: []compatknowledge.Summary{{ID: "knowledge-1", Name: "Catalog", Tags: []string{"go"}, DocumentCount: 2}},
		Page:  contract.PageResult{NextPageToken: "offset:1", Total: &total},
	}}
	server := testServerWithKnowledge(t, &fakeSkillPort{}, catalog)
	bad := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_knowledge_list","arguments":{"user_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	if !bad.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("identity override accepted: %#v", bad.Result)
	}
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_knowledge_list","arguments":{"keyword":" catalog ","tags":["go"],"page_size":5,"page_token":"offset:5"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true {
		t.Fatalf("tools/call error: %#v", result)
	}
	if catalog.listCallCtx.UserID != "trusted-user" || catalog.listCallCtx.TenantID != "tenant-a" || catalog.listInput.Keyword != "catalog" || len(catalog.listInput.Tags) != 1 || catalog.listInput.Tags[0] != "go" || catalog.listInput.Page.PageSize != 5 || catalog.listInput.Page.PageToken != "offset:5" {
		t.Fatalf("callCtx=%#v input=%#v", catalog.listCallCtx, catalog.listInput)
	}
	raw, err := json.Marshal(result["structuredContent"])
	if err != nil || !strings.Contains(string(raw), `"knowledge-1"`) || !strings.Contains(string(raw), `"next_page_token":"offset:1"`) {
		t.Fatalf("structured result=%s err=%v", raw, err)
	}
	catalog.listErr = contract.NewError(contract.BackendUnavailable, "knowledge.list", "database /private/path", true, errors.New("postgres secret"))
	safe := callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_knowledge_list","arguments":{}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safeRaw, _ := json.Marshal(safe.Result)
	if !strings.Contains(string(safeRaw), "BACKEND_UNAVAILABLE") || strings.Contains(string(safeRaw), "secret") || strings.Contains(string(safeRaw), "private") {
		t.Fatalf("unsafe list error=%s", safeRaw)
	}
}

func TestToolsCallKnowledgeGetSafeMetadataAndErrors(t *testing.T) {
	catalog := &fakeKnowledgeCatalogPort{getResult: compatknowledge.GetResult{Knowledge: compatknowledge.Summary{ID: "knowledge-1", Name: "Catalog", DocumentCount: 3}}}
	server := testServerWithKnowledge(t, &fakeSkillPort{}, catalog)
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_knowledge_get","arguments":{"knowledge_id":"knowledge-1"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || catalog.getCallCtx.UserID != "trusted-user" || catalog.getCallCtx.TenantID != "tenant-a" || catalog.getInput.KnowledgeID != "knowledge-1" {
		t.Fatalf("result=%#v callCtx=%#v input=%#v", result, catalog.getCallCtx, catalog.getInput)
	}
	raw, err := json.Marshal(result["structuredContent"])
	if err != nil || !strings.Contains(string(raw), `"knowledge-1"`) || strings.Contains(string(raw), "content") {
		t.Fatalf("unsafe structured result=%s err=%v", raw, err)
	}
	invalid := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_knowledge_get","arguments":{"knowledge_id":"knowledge-1","tenant_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	if !invalid.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("tenant override accepted: %#v", invalid.Result)
	}
	empty := callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_knowledge_get","arguments":{}}}`, map[string]string{"X-User-Id": "trusted-user"})
	if !empty.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("empty knowledge id accepted: %#v", empty.Result)
	}

	catalog.getErr = contract.NewError(contract.NotFound, "knowledge.get", "sql details", false, errors.New("postgres secret"))
	notFound := callHTTP(t, server, `{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"lazymind_knowledge_get","arguments":{"knowledge_id":"missing"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safe, _ := json.Marshal(notFound.Result)
	if !strings.Contains(string(safe), "NOT_FOUND") || strings.Contains(string(safe), "secret") || strings.Contains(string(safe), "sql details") {
		t.Fatalf("unsafe error=%s", safe)
	}
}

func TestKnowledgeToolsRequirePrincipal(t *testing.T) {
	server := testServer(t, &fakeSkillPort{})
	for _, name := range []string{knowledgeListToolName, knowledgeGetToolName, knowledgeSearchToolName} {
		t.Run(name, func(t *testing.T) {
			response := callHTTP(t, server, fmt.Sprintf(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":%q,"arguments":{}}}`, name), nil)
			result := response.Result.(map[string]any)
			if result["isError"] != true {
				t.Fatalf("missing principal result=%#v", result)
			}
		})
	}
}

func TestToolsCallKnowledgeSearchUsesPrincipalAndSafeStructuredResult(t *testing.T) {
	search := &fakeKnowledgeSearchPort{result: compatknowledge.SearchResult{Hits: []compatknowledge.SearchHit{
		{KnowledgeID: "knowledge-1", DocumentID: "document-1", ChunkID: "chunk-1", Text: "useful text", Score: 0.9, Title: "Guide", SourceURL: "https://files.test/guide"},
		{KnowledgeID: "knowledge-1", DocumentID: "document-2", ChunkID: "chunk-2", Text: "other text", Score: 0.5},
	}}}
	server := testServerWithKnowledgeSearch(t, &fakeSkillPort{}, &fakeKnowledgeCatalogPort{}, search)
	bad := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_knowledge_search","arguments":{"query":"q","knowledge_ids":["knowledge-1"],"tenant_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	if !bad.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("identity override accepted: %#v", bad.Result)
	}
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_knowledge_search","arguments":{"query":" q ","knowledge_ids":["knowledge-1","knowledge-1"],"top_k":2}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || search.callCtx.UserID != "trusted-user" || search.callCtx.TenantID != "tenant-a" {
		t.Fatalf("result=%#v callCtx=%#v", result, search.callCtx)
	}
	if search.input.Query != "q" || len(search.input.KnowledgeIDs) != 1 || search.input.KnowledgeIDs[0] != "knowledge-1" || search.input.TopK != 2 {
		t.Fatalf("search input=%#v", search.input)
	}
	raw, err := json.Marshal(result["structuredContent"])
	if err != nil || !strings.Contains(string(raw), `"document-1"`) || !strings.Contains(string(raw), `"score":0.9`) || strings.Contains(string(raw), "internal_token") {
		t.Fatalf("structured result=%s err=%v", raw, err)
	}

	search.err = contract.NewError(contract.BackendUnavailable, "knowledge.search", "http://backend/token secret", true, errors.New("timeout secret"))
	safe := callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_knowledge_search","arguments":{"query":"q","knowledge_ids":["knowledge-1"]}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safeRaw, _ := json.Marshal(safe.Result)
	if !strings.Contains(string(safeRaw), "BACKEND_UNAVAILABLE") || strings.Contains(string(safeRaw), "http://") || strings.Contains(string(safeRaw), "secret") {
		t.Fatalf("unsafe search error=%s", safeRaw)
	}
}

func TestKnowledgeSearchArgumentsAndEmptyResult(t *testing.T) {
	server := testServerWithKnowledgeSearch(t, &fakeSkillPort{}, &fakeKnowledgeCatalogPort{}, &fakeKnowledgeSearchPort{})
	invalid := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_knowledge_search","arguments":{"query":"","knowledge_ids":[]}}}`, map[string]string{"X-User-Id": "user"})
	if !invalid.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("invalid search accepted: %#v", invalid.Result)
	}
	malformed := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_knowledge_search","arguments":{"query":"q","knowledge_ids":["id"],"unexpected":true}}}`, map[string]string{"X-User-Id": "user"})
	if !malformed.Result.(map[string]any)["isError"].(bool) {
		t.Fatalf("unknown argument accepted: %#v", malformed.Result)
	}
	result := knowledgeSearchResult(compatknowledge.SearchResult{})
	structured, ok := result.StructuredContent.(knowledgeSearchStructuredResult)
	if result.IsError || !ok || structured.Hits == nil || len(structured.Hits) != 0 {
		t.Fatalf("empty search result=%#v", result)
	}
}

func TestKnowledgeSearchReportsUnsupportedWhenRuntimeHasNoSearchPort(t *testing.T) {
	rt, err := compatruntime.New(compatruntime.Dependencies{SkillPort: &fakeSkillPort{}, KnowledgeCatalog: &fakeKnowledgeCatalogPort{}})
	if err != nil {
		t.Fatalf("Runtime.New: %v", err)
	}
	server, err := New(rt, HeaderIdentityProvider{}, Options{})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_knowledge_search","arguments":{"query":"q","knowledge_ids":["knowledge-1"]}}}`, map[string]string{"X-User-Id": "user"})
	raw, _ := json.Marshal(response.Result)
	if !strings.Contains(string(raw), "UNSUPPORTED") || strings.Contains(string(raw), "token") {
		t.Fatalf("unconfigured search result=%s", raw)
	}
}

func TestKnowledgeListResultIsStructuredAndStableForEmptyList(t *testing.T) {
	total := int64(0)
	result := knowledgeListResult(compatknowledge.ListResult{Page: contract.PageResult{Total: &total}})
	structured, ok := result.StructuredContent.(knowledgeListStructuredResult)
	if result.IsError || !ok || structured.Items == nil || len(structured.Items) != 0 || structured.Page.Total == nil || *structured.Page.Total != 0 {
		t.Fatalf("structured result=%#v", result)
	}
}

func TestCallContextRequiresPrincipal(t *testing.T) {
	if _, err := callContext(Principal{}, "request"); err == nil {
		t.Fatal("empty principal was accepted")
	}
	ctx, err := callContext(Principal{UserID: " user ", UserName: " name ", TenantID: " tenant "}, " request ")
	if err != nil || ctx.UserID != "user" || ctx.UserName != "name" || ctx.TenantID != "tenant" || ctx.RequestID != "request" {
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

func TestToolsListPublishesCloudDocumentSchemas(t *testing.T) {
	server := testServerWithCloudDocument(t, &fakeCloudDocumentPort{})
	response := server.Handle(context.Background(), rpcRequest{JSONRPC: "2.0", ID: json.RawMessage("1"), Method: "tools/list"})
	tools := response.Result.(map[string]any)["tools"].([]ToolDefinition)
	want := map[string][]string{
		cloudDocumentListToolName:   {"keyword", "status", "page_size", "page_token"},
		cloudDocumentGetToolName:    {"source_id", "include_documents", "page_size", "page_token"},
		cloudDocumentSearchToolName: {"source_id", "query", "page_size", "page_token", "binding_id", "tree_key", "state_filter", "include_documents", "include_containers"},
	}
	for _, tool := range tools {
		fields, ok := want[tool.Name]
		if !ok {
			continue
		}
		if !tool.Annotations.ReadOnlyHint || tool.InputSchema["additionalProperties"] != false {
			t.Fatalf("tool metadata=%#v", tool)
		}
		properties := tool.InputSchema["properties"].(map[string]any)
		for _, field := range fields {
			if _, ok := properties[field]; !ok {
				t.Fatalf("%s missing %s", tool.Name, field)
			}
		}
		for _, forbidden := range []string{"user_id", "user_name", "tenant_id", "access_token", "credential", "provider_secret", "feishu_token", "notion_token"} {
			if _, ok := properties[forbidden]; ok {
				t.Fatalf("%s exposes %s", tool.Name, forbidden)
			}
		}
		delete(want, tool.Name)
	}
	if len(want) != 0 {
		t.Fatalf("missing cloud tools=%#v", want)
	}
	raw, err := json.Marshal(response.Result)
	if err != nil || !strings.Contains(string(raw), `"annotations":{"readOnlyHint":true}`) {
		t.Fatalf("wire tools=%s err=%v", raw, err)
	}
}

func TestCloudDocumentListUsesPrincipalTenantAndStructuredResult(t *testing.T) {
	total := int64(2)
	cloud := &fakeCloudDocumentPort{listResult: compatcloud.ListResult{Sources: []compatcloud.SourceSummary{{ID: "source-1", Name: "Team Docs", Status: "ACTIVE", DatasetID: "dataset-1"}}, Page: contract.PageResult{NextPageToken: "offset:1", Total: &total}}}
	server := testServerWithCloudDocument(t, cloud)
	bad := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_cloud_document_list","arguments":{"user_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	if bad.Result.(map[string]any)["isError"] != true {
		t.Fatalf("identity override accepted=%#v", bad.Result)
	}
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_cloud_document_list","arguments":{"keyword":" docs ","status":" ACTIVE ","page_size":5,"page_token":"offset:5"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || cloud.listCallCtx.UserID != "trusted-user" || cloud.listCallCtx.TenantID != "tenant-a" || cloud.listInput.Keyword != "docs" || cloud.listInput.Status != "ACTIVE" || cloud.listInput.Page.PageSize != 5 || cloud.listInput.Page.PageToken != "offset:5" {
		t.Fatalf("result=%#v ctx=%#v input=%#v", result, cloud.listCallCtx, cloud.listInput)
	}
	raw, _ := json.Marshal(result["structuredContent"])
	if !strings.Contains(string(raw), `"source-1"`) || !strings.Contains(string(raw), `"next_page_token":"offset:1"`) {
		t.Fatalf("structured=%s", raw)
	}
	cloud.listErr = contract.NewError(contract.BackendUnavailable, "cloud_document.list", "http://scan/private secret", true, errors.New("token secret"))
	safe := callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_cloud_document_list","arguments":{}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safeRaw, _ := json.Marshal(safe.Result)
	if !strings.Contains(string(safeRaw), "BACKEND_UNAVAILABLE") || strings.Contains(string(safeRaw), "secret") || strings.Contains(string(safeRaw), "http://") {
		t.Fatalf("unsafe=%s", safeRaw)
	}
}

func TestCloudDocumentGetMapsMetadataOnlyAndRejectsInvalidArguments(t *testing.T) {
	total := int64(1)
	cloud := &fakeCloudDocumentPort{getResult: compatcloud.SourceDetail{ID: "source-1", Name: "Team Docs", DatasetID: "dataset-1"}, docResult: compatcloud.DocumentListResult{Documents: []compatcloud.DocumentSummary{{ID: "cloud-doc-1", SourceID: "source-1", ObjectKey: "object-1", DisplayName: "Guide", Name: "Guide.md", KnowledgeDocument: &compatcloud.KnowledgeDocumentRef{KnowledgeID: "dataset-1", DocumentID: "core-doc-1"}}}, Page: contract.PageResult{Total: &total}}}
	server := testServerWithCloudDocument(t, cloud)
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_cloud_document_get","arguments":{"source_id":"source-1","include_documents":true,"page_size":5}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || cloud.getCallCtx.UserID != "trusted-user" || cloud.getCallCtx.TenantID != "tenant-a" || cloud.getSourceID != "source-1" || cloud.documentInput.DocumentsPage.PageSize != 5 {
		t.Fatalf("result=%#v get=%#v doc=%#v", result, cloud.getCallCtx, cloud.documentInput)
	}
	raw, _ := json.Marshal(result["structuredContent"])
	for _, want := range []string{`"source-1"`, `"cloud-doc-1"`, `"core-doc-1"`} {
		if !strings.Contains(string(raw), want) {
			t.Fatalf("structured=%s missing=%s", raw, want)
		}
	}
	for _, arguments := range []string{`{}`, `{"source_id":"source-1","tenant_id":"attacker"}`, `{"source_id":"source-1","raw":true}`} {
		invalid := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_cloud_document_get","arguments":`+arguments+`}}`, map[string]string{"X-User-Id": "trusted-user"})
		if invalid.Result.(map[string]any)["isError"] != true {
			t.Fatalf("arguments accepted=%s result=%#v", arguments, invalid.Result)
		}
	}
	cloud.getErr = contract.NewError(contract.NotFound, "cloud_document.get", "scan /private/path secret", false, errors.New("token secret"))
	notFound := callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_cloud_document_get","arguments":{"source_id":"missing"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safe, _ := json.Marshal(notFound.Result)
	if !strings.Contains(string(safe), "NOT_FOUND") || strings.Contains(string(safe), "secret") || strings.Contains(string(safe), "private") {
		t.Fatalf("unsafe=%s", safe)
	}
}

func TestCloudDocumentSearchUsesPrincipalAndSafeStructuredResult(t *testing.T) {
	total := int64(1)
	cloud := &fakeCloudDocumentPort{searchResult: compatcloud.SearchResult{Hits: []compatcloud.SearchHit{{Key: "hit-1", DisplayName: "Guide", SearchName: "Guide.md", SourceID: "source-1", TreeKey: "root", ObjectKey: "object-1", IsDocument: true, Selectable: true}}, Page: contract.PageResult{Total: &total}}}
	server := testServerWithCloudDocument(t, cloud)
	bad := callHTTP(t, server, `{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"lazymind_cloud_document_search","arguments":{"source_id":"source-1","query":"guide","tenant_id":"attacker"}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	if bad.Result.(map[string]any)["isError"] != true {
		t.Fatalf("identity override accepted=%#v", bad.Result)
	}
	response := callHTTP(t, server, `{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"lazymind_cloud_document_search","arguments":{"source_id":"source-1","query":" guide ","binding_id":"binding-1","tree_key":"root","state_filter":[" ACTIVE "],"include_documents":true,"page_size":2}}}`, map[string]string{"X-User-Id": "trusted-user", "X-Tenant-Id": "tenant-a"})
	result := response.Result.(map[string]any)
	if result["isError"] == true || cloud.searchCallCtx.UserID != "trusted-user" || cloud.searchCallCtx.TenantID != "tenant-a" || cloud.searchInput.Query != "guide" || cloud.searchInput.BindingID != "binding-1" || len(cloud.searchInput.StateFilter) != 1 || cloud.searchInput.StateFilter[0] != "ACTIVE" {
		t.Fatalf("result=%#v ctx=%#v input=%#v", result, cloud.searchCallCtx, cloud.searchInput)
	}
	raw, _ := json.Marshal(result["structuredContent"])
	if !strings.Contains(string(raw), `"hit-1"`) || strings.Contains(string(raw), "token") {
		t.Fatalf("structured=%s", raw)
	}
	cloud.searchErr = contract.NewError(contract.BackendUnavailable, "cloud_document.search", "http://scan/private secret", true, errors.New("token secret"))
	safe := callHTTP(t, server, `{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"lazymind_cloud_document_search","arguments":{"source_id":"source-1","query":"guide"}}}`, map[string]string{"X-User-Id": "trusted-user"})
	safeRaw, _ := json.Marshal(safe.Result)
	if !strings.Contains(string(safeRaw), "BACKEND_UNAVAILABLE") || strings.Contains(string(safeRaw), "secret") || strings.Contains(string(safeRaw), "http://") {
		t.Fatalf("unsafe=%s", safeRaw)
	}
}

func TestCloudDocumentToolsRequirePrincipalAndHaveStableEmptyResults(t *testing.T) {
	server := testServerWithCloudDocument(t, &fakeCloudDocumentPort{})
	for _, name := range []string{cloudDocumentListToolName, cloudDocumentGetToolName, cloudDocumentSearchToolName} {
		response := callHTTP(t, server, fmt.Sprintf(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":%q,"arguments":{}}}`, name), nil)
		if response.Result.(map[string]any)["isError"] != true {
			t.Fatalf("missing principal accepted for %s", name)
		}
	}
	list := cloudDocumentListResult(compatcloud.ListResult{})
	search := cloudDocumentSearchResult(compatcloud.SearchResult{})
	if list.StructuredContent.(cloudDocumentListStructuredResult).Sources == nil || search.StructuredContent.(cloudDocumentSearchStructuredResult).Hits == nil {
		t.Fatalf("empty results are not stable list=%#v search=%#v", list, search)
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

func toolDefinitionsByName(tools []ToolDefinition) map[string]ToolDefinition {
	byName := make(map[string]ToolDefinition, len(tools))
	for _, tool := range tools {
		byName[tool.Name] = tool
	}
	return byName
}
