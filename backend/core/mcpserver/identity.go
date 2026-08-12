package mcpserver

import (
	"context"
	"fmt"
	"net/http"
	"strings"

	"lazymind/core/compat/contract"
)

type Principal struct {
	UserID   string
	UserName string
	TenantID string
}

// IdentityProvider resolves an already authenticated request to a Principal.
// It must not read tool arguments.
type IdentityProvider interface {
	Principal(context.Context, *http.Request) (Principal, error)
}

// HeaderIdentityProvider adapts identity headers installed by a trusted
// authenticated gateway. It must not be exposed directly to untrusted clients.
type HeaderIdentityProvider struct{}

func (HeaderIdentityProvider) Principal(_ context.Context, request *http.Request) (Principal, error) {
	if request == nil {
		return Principal{}, fmt.Errorf("missing request")
	}
	principal := Principal{
		UserID:   strings.TrimSpace(request.Header.Get("X-User-Id")),
		UserName: strings.TrimSpace(request.Header.Get("X-User-Name")),
		TenantID: strings.TrimSpace(request.Header.Get("X-Tenant-Id")),
	}
	if principal.UserID == "" {
		return Principal{}, fmt.Errorf("authenticated principal is required")
	}
	return principal, nil
}

func callContext(principal Principal, requestID string) (contract.CallContext, error) {
	userID := strings.TrimSpace(principal.UserID)
	if userID == "" {
		return contract.CallContext{}, fmt.Errorf("authenticated principal is required")
	}
	return contract.CallContext{
		UserID:    userID,
		UserName:  strings.TrimSpace(principal.UserName),
		TenantID:  strings.TrimSpace(principal.TenantID),
		RequestID: strings.TrimSpace(requestID),
	}, nil
}
