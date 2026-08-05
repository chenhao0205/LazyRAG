package clouddocument

import (
	"context"
	"strings"

	"lazymind/core/compat/contract"
)

type Facade struct {
	port Port
}

func NewFacade(port Port) (*Facade, error) {
	if port == nil {
		return nil, contract.NewError(contract.Internal, "cloud_document.facade.new", "cloud document port is required", false, nil)
	}
	return &Facade{port: port}, nil
}

func (f *Facade) List(ctx context.Context, callCtx contract.CallContext, input ListInput) (ListResult, error) {
	callCtx.UserID = strings.TrimSpace(callCtx.UserID)
	if callCtx.UserID == "" {
		return ListResult{}, contract.InvalidArgumentError("cloud_document.list", "user_id is required")
	}
	input.Keyword = strings.TrimSpace(input.Keyword)
	input.Status = strings.TrimSpace(input.Status)
	input.Page = input.Page.Normalize()
	return f.port.ListSources(ctx, callCtx, input)
}

func (f *Facade) Get(ctx context.Context, callCtx contract.CallContext, input GetInput) (GetResult, error) {
	callCtx.UserID = strings.TrimSpace(callCtx.UserID)
	if callCtx.UserID == "" {
		return GetResult{}, contract.InvalidArgumentError("cloud_document.get", "user_id is required")
	}
	input.SourceID = strings.TrimSpace(input.SourceID)
	if input.SourceID == "" {
		return GetResult{}, contract.InvalidArgumentError("cloud_document.get", "source_id is required")
	}
	input.BindingID = strings.TrimSpace(input.BindingID)
	input.DocumentKeyword = strings.TrimSpace(input.DocumentKeyword)
	input.DocumentsPage = input.DocumentsPage.Normalize()
	input.StateFilter = trimStrings(input.StateFilter)
	input.ParseStatuses = trimStrings(input.ParseStatuses)
	source, err := f.port.GetSource(ctx, callCtx, input.SourceID)
	if err != nil {
		return GetResult{}, err
	}
	result := GetResult{Source: source}
	if !input.IncludeDocuments {
		return result, nil
	}
	docs, err := f.port.ListDocuments(ctx, callCtx, input)
	if err != nil {
		return GetResult{}, err
	}
	result.Documents = docs.Documents
	result.DocumentsPage = docs.Page
	return result, nil
}

func (f *Facade) Search(ctx context.Context, callCtx contract.CallContext, input SearchInput) (SearchResult, error) {
	callCtx.UserID = strings.TrimSpace(callCtx.UserID)
	if callCtx.UserID == "" {
		return SearchResult{}, contract.InvalidArgumentError("cloud_document.search", "user_id is required")
	}
	input.SourceID = strings.TrimSpace(input.SourceID)
	if input.SourceID == "" {
		return SearchResult{}, contract.InvalidArgumentError("cloud_document.search", "source_id is required")
	}
	input.Query = strings.TrimSpace(input.Query)
	if input.Query == "" {
		return SearchResult{}, contract.InvalidArgumentError("cloud_document.search", "query is required")
	}
	input.BindingID = strings.TrimSpace(input.BindingID)
	input.TreeKey = strings.TrimSpace(input.TreeKey)
	input.StateFilter = trimStrings(input.StateFilter)
	input.Page = input.Page.Normalize()
	return f.port.Search(ctx, callCtx, input)
}

func trimStrings(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}
