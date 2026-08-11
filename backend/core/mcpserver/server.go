// Package mcpserver implements LazyMind's inbound MCP server.
//
// It deliberately depends on Compat only. Application wiring owns database,
// service, adapter, and Runtime construction.
package mcpserver

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"lazymind/core/compat/runtime"
)

const protocolVersion = "2025-03-26"

type Options struct {
	ServerName    string
	ServerVersion string
}

type Server struct {
	runtime  *runtime.Runtime
	identity IdentityProvider
	registry *Registry
	name     string
	version  string
}

type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcResponse struct {
	JSONRPC string    `json:"jsonrpc"`
	ID      any       `json:"id,omitempty"`
	Result  any       `json:"result,omitempty"`
	Error   *RPCError `json:"error,omitempty"`
}

// New constructs an inbound server around an application-owned Compat Runtime.
func New(rt *runtime.Runtime, identity IdentityProvider, options Options) (*Server, error) {
	if rt == nil {
		return nil, fmt.Errorf("mcpserver runtime is required")
	}
	if identity == nil {
		return nil, fmt.Errorf("mcpserver identity provider is required")
	}
	registry := NewRegistry()
	if err := registry.Register(skillListTool()); err != nil {
		return nil, err
	}
	if err := registry.Register(skillGetTool()); err != nil {
		return nil, err
	}
	name := strings.TrimSpace(options.ServerName)
	if name == "" {
		name = "lazymind"
	}
	version := strings.TrimSpace(options.ServerVersion)
	if version == "" {
		version = "0.1.0"
	}
	return &Server{runtime: rt, identity: identity, registry: registry, name: name, version: version}, nil
}

func (s *Server) Handle(ctx context.Context, request rpcRequest) rpcResponse {
	if request.JSONRPC != "2.0" {
		return s.errorResponse(request.ID, newRPCError(rpcInvalidRequest, "invalid JSON-RPC version"))
	}
	switch request.Method {
	case "initialize":
		return s.resultResponse(request.ID, s.initializeResult())
	case "notifications/initialized":
		return rpcResponse{}
	case "tools/list":
		return s.resultResponse(request.ID, map[string]any{"tools": s.registry.List()})
	case "tools/call":
		return s.callTool(ctx, request)
	default:
		return s.errorResponse(request.ID, newRPCError(rpcMethodNotFound, "method not found"))
	}
}

func (s *Server) initializeResult() map[string]any {
	return map[string]any{
		"protocolVersion": protocolVersion,
		"capabilities":    map[string]any{"tools": map[string]any{}},
		"serverInfo":      map[string]string{"name": s.name, "version": s.version},
	}
}

func (s *Server) resultResponse(id json.RawMessage, result any) rpcResponse {
	return rpcResponse{JSONRPC: "2.0", ID: responseID(id), Result: result}
}

func (s *Server) errorResponse(id json.RawMessage, rpcErr *RPCError) rpcResponse {
	return rpcResponse{JSONRPC: "2.0", ID: responseID(id), Error: rpcErr}
}

func responseID(raw json.RawMessage) any {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	var id any
	if err := json.Unmarshal(raw, &id); err != nil {
		return nil
	}
	return id
}
