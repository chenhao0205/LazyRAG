package mcpserver

import (
	"errors"

	"lazymind/core/compat/contract"
)

const (
	rpcParseError     = -32700
	rpcInvalidRequest = -32600
	rpcMethodNotFound = -32601
	rpcInvalidParams  = -32602
)

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func newRPCError(code int, message string) *RPCError { return &RPCError{Code: code, Message: message} }

// toolErrorFromCompat purposefully exposes only stable codes and generic safe
// messages. Compat causes may hold SQL, paths, or backend response details.
func toolErrorFromCompat(err error) toolResult {
	var compatErr *contract.Error
	if !errors.As(err, &compatErr) {
		return toolErrorResult("INTERNAL", "The tool could not complete.")
	}
	switch compatErr.Code {
	case contract.InvalidArgument:
		return toolErrorResult(string(contract.InvalidArgument), "Invalid tool arguments.")
	case contract.NotFound:
		return toolErrorResult(string(contract.NotFound), "Requested resource was not found.")
	case contract.Conflict:
		return toolErrorResult(string(contract.Conflict), "The request conflicts with current state.")
	case contract.BackendUnavailable:
		return toolErrorResult(string(contract.BackendUnavailable), "The backend is temporarily unavailable.")
	case contract.Unsupported:
		return toolErrorResult(string(contract.Unsupported), "This operation is not supported.")
	default:
		return toolErrorResult(string(contract.Internal), "The tool could not complete.")
	}
}
