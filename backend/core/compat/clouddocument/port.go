package clouddocument

import (
	"context"

	"lazymind/core/compat/contract"
)

type Port interface {
	ListSources(ctx context.Context, callCtx contract.CallContext, input ListInput) (ListResult, error)
	GetSource(ctx context.Context, callCtx contract.CallContext, sourceID string) (SourceDetail, error)
	ListDocuments(ctx context.Context, callCtx contract.CallContext, input GetInput) (DocumentListResult, error)
	Search(ctx context.Context, callCtx contract.CallContext, input SearchInput) (SearchResult, error)
}
