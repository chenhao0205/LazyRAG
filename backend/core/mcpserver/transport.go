package mcpserver

import (
	"context"
	"encoding/json"
	"net/http"
	"time"
)

const (
	defaultRequestTimeout = 15 * time.Second
	defaultBodyLimit      = 1 << 20
)

type TransportOptions struct {
	RequestTimeout time.Duration
	MaxBodyBytes   int64
}

// StreamableHTTPHandler provides the minimal POST JSON-RPC transport needed
// for remote Streamable HTTP clients. Session resumption is intentionally out
// of scope for this first read-only tool slice.
func (s *Server) StreamableHTTPHandler(options TransportOptions) http.Handler {
	timeout := options.RequestTimeout
	if timeout <= 0 {
		timeout = defaultRequestTimeout
	}
	limit := options.MaxBodyBytes
	if limit <= 0 {
		limit = defaultBodyLimit
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			writeRPCResponse(w, http.StatusMethodNotAllowed, rpcResponse{JSONRPC: "2.0", Error: newRPCError(rpcInvalidRequest, "POST is required")})
			return
		}
		defer r.Body.Close()
		r.Body = http.MaxBytesReader(w, r.Body, limit)
		var request rpcRequest
		decoder := json.NewDecoder(r.Body)
		if err := decoder.Decode(&request); err != nil {
			writeRPCResponse(w, http.StatusBadRequest, rpcResponse{JSONRPC: "2.0", Error: newRPCError(rpcParseError, "invalid JSON-RPC request")})
			return
		}
		ctx, cancel := context.WithTimeout(context.WithValue(r.Context(), requestContextKey{}, r), timeout)
		defer cancel()
		response := s.Handle(ctx, request)
		if response.JSONRPC == "" { // notification
			w.WriteHeader(http.StatusAccepted)
			return
		}
		writeRPCResponse(w, http.StatusOK, response)
	})
}

func writeRPCResponse(w http.ResponseWriter, status int, response rpcResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(response)
}
