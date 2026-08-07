package clouddocument

import (
	"context"

	"lazymind/core/compat/contract"
)

type Port interface {
	// ListSources lists Cloud Sources visible to the caller.
	ListSources(ctx context.Context, callCtx contract.CallContext, input ListInput) (ListResult, error)
	// GetSource returns Cloud Source metadata only.
	GetSource(ctx context.Context, callCtx contract.CallContext, sourceID string) (SourceDetail, error)
	// ListDocuments returns one page of document metadata for a Cloud Source.
	// It does not read document body content.
	ListDocuments(ctx context.Context, callCtx contract.CallContext, input GetInput) (DocumentListResult, error)
	// Search searches document titles, display_name/search_name, and tree node
	// names indexed by Scan. It does not search document body content.
	Search(ctx context.Context, callCtx contract.CallContext, input SearchInput) (SearchResult, error)
}
