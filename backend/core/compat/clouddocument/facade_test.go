package clouddocument

import (
	"context"
	"errors"
	"testing"

	"lazymind/core/compat/contract"
)

type fakePort struct {
	listCalls []listCall
	getCalls  []getCall
	docCalls  []GetInput
	searches  []searchCall
	listErr   error
	getErr    error
	docErr    error
	searchErr error
}

type listCall struct {
	callCtx contract.CallContext
	input   ListInput
}

type getCall struct {
	callCtx  contract.CallContext
	sourceID string
}

type searchCall struct {
	callCtx contract.CallContext
	input   SearchInput
}

func (p *fakePort) ListSources(ctx context.Context, callCtx contract.CallContext, input ListInput) (ListResult, error) {
	p.listCalls = append(p.listCalls, listCall{callCtx: callCtx, input: input})
	if p.listErr != nil {
		return ListResult{}, p.listErr
	}
	total := int64(1)
	return ListResult{Sources: []SourceSummary{{ID: "source-1"}}, Page: contract.PageResult{Total: &total}}, nil
}

func (p *fakePort) GetSource(ctx context.Context, callCtx contract.CallContext, sourceID string) (SourceDetail, error) {
	p.getCalls = append(p.getCalls, getCall{callCtx: callCtx, sourceID: sourceID})
	if p.getErr != nil {
		return SourceDetail{}, p.getErr
	}
	return SourceDetail{ID: sourceID, Name: "Docs"}, nil
}

func (p *fakePort) ListDocuments(ctx context.Context, callCtx contract.CallContext, input GetInput) (DocumentListResult, error) {
	p.docCalls = append(p.docCalls, input)
	if p.docErr != nil {
		return DocumentListResult{}, p.docErr
	}
	total := int64(1)
	return DocumentListResult{
		Documents: []DocumentSummary{{ID: "doc-1", ObjectKey: "obj-1"}},
		Page:      contract.PageResult{Total: &total},
	}, nil
}

func (p *fakePort) Search(ctx context.Context, callCtx contract.CallContext, input SearchInput) (SearchResult, error) {
	p.searches = append(p.searches, searchCall{callCtx: callCtx, input: input})
	if p.searchErr != nil {
		return SearchResult{}, p.searchErr
	}
	return SearchResult{Hits: []SearchHit{{Key: "hit-1"}}}, nil
}

func TestFacadeListReturnsSourcesAndPassesUserID(t *testing.T) {
	port := &fakePort{}
	facade := mustFacade(t, port)
	result, err := facade.List(context.Background(), contract.CallContext{UserID: " user-1 "}, ListInput{
		Keyword: " docs ",
		Status:  " ACTIVE ",
	})
	if err != nil {
		t.Fatalf("List returned error: %v", err)
	}
	if len(result.Sources) != 1 || result.Sources[0].ID != "source-1" {
		t.Fatalf("sources = %#v, want source-1", result.Sources)
	}
	if len(port.listCalls) != 1 || port.listCalls[0].callCtx.UserID != "user-1" {
		t.Fatalf("list calls = %#v, want trimmed user", port.listCalls)
	}
	if got := port.listCalls[0].input; got.Keyword != "docs" || got.Status != "ACTIVE" {
		t.Fatalf("input = %#v, want trimmed filters", got)
	}
}

func TestFacadeGetWithoutDocuments(t *testing.T) {
	port := &fakePort{}
	facade := mustFacade(t, port)
	result, err := facade.Get(context.Background(), contract.CallContext{UserID: "user-1"}, GetInput{SourceID: " source-1 "})
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if result.Source.ID != "source-1" || len(result.Documents) != 0 || len(port.docCalls) != 0 {
		t.Fatalf("result=%#v docCalls=%d, want source only", result, len(port.docCalls))
	}
}

func TestFacadeGetWithDocuments(t *testing.T) {
	port := &fakePort{}
	facade := mustFacade(t, port)
	result, err := facade.Get(context.Background(), contract.CallContext{UserID: "user-1"}, GetInput{
		SourceID:         "source-1",
		IncludeDocuments: true,
		DocumentKeyword:  " spec ",
		StateFilter:      []string{" NEW ", ""},
		ParseStatuses:    []string{" SUCCEEDED "},
	})
	if err != nil {
		t.Fatalf("Get returned error: %v", err)
	}
	if len(result.Documents) != 1 || result.Documents[0].ID != "doc-1" {
		t.Fatalf("documents = %#v, want doc-1", result.Documents)
	}
	if len(port.docCalls) != 1 || port.docCalls[0].DocumentKeyword != "spec" {
		t.Fatalf("doc calls = %#v, want trimmed input", port.docCalls)
	}
	if got := port.docCalls[0].StateFilter; len(got) != 1 || got[0] != "NEW" {
		t.Fatalf("state filter = %#v, want NEW", got)
	}
}

func TestFacadeSearchReturnsHitsAndPassesUserID(t *testing.T) {
	port := &fakePort{}
	facade := mustFacade(t, port)
	result, err := facade.Search(context.Background(), contract.CallContext{UserID: " user-1 "}, SearchInput{SourceID: " source-1 ", Query: " handbook "})
	if err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if len(result.Hits) != 1 || result.Hits[0].Key != "hit-1" {
		t.Fatalf("hits = %#v, want hit-1", result.Hits)
	}
	if len(port.searches) != 1 || port.searches[0].callCtx.UserID != "user-1" || port.searches[0].input.Query != "handbook" {
		t.Fatalf("search calls = %#v, want trimmed user/query", port.searches)
	}
}

func TestFacadeValidatesRequiredInputs(t *testing.T) {
	facade := mustFacade(t, &fakePort{})
	tests := []struct {
		name string
		run  func() error
	}{
		{name: "list user", run: func() error {
			_, err := facade.List(context.Background(), contract.CallContext{UserID: " "}, ListInput{})
			return err
		}},
		{name: "get user", run: func() error {
			_, err := facade.Get(context.Background(), contract.CallContext{UserID: " "}, GetInput{SourceID: "source-1"})
			return err
		}},
		{name: "get source", run: func() error {
			_, err := facade.Get(context.Background(), contract.CallContext{UserID: "user-1"}, GetInput{SourceID: " "})
			return err
		}},
		{name: "search user", run: func() error {
			_, err := facade.Search(context.Background(), contract.CallContext{UserID: " "}, SearchInput{SourceID: "source-1", Query: "doc"})
			return err
		}},
		{name: "search source", run: func() error {
			_, err := facade.Search(context.Background(), contract.CallContext{UserID: "user-1"}, SearchInput{SourceID: " ", Query: "doc"})
			return err
		}},
		{name: "search query", run: func() error {
			_, err := facade.Search(context.Background(), contract.CallContext{UserID: "user-1"}, SearchInput{SourceID: "source-1", Query: " "})
			return err
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if code, ok := contract.CodeOf(tc.run()); !ok || code != contract.InvalidArgument {
				t.Fatalf("code = %s, %v; want INVALID_ARGUMENT", code, ok)
			}
		})
	}
}

func TestFacadeNormalizesPaging(t *testing.T) {
	port := &fakePort{}
	facade := mustFacade(t, port)
	if _, err := facade.List(context.Background(), contract.CallContext{UserID: "user-1"}, ListInput{Page: contract.PageRequest{PageSize: 0}}); err != nil {
		t.Fatalf("List returned error: %v", err)
	}
	if got := port.listCalls[0].input.Page.PageSize; got != contract.DefaultPageSize {
		t.Fatalf("list page size = %d, want default", got)
	}
	if _, err := facade.Search(context.Background(), contract.CallContext{UserID: "user-1"}, SearchInput{SourceID: "source-1", Query: "doc", Page: contract.PageRequest{PageSize: contract.MaxPageSize + 1}}); err != nil {
		t.Fatalf("Search returned error: %v", err)
	}
	if got := port.searches[0].input.Page.PageSize; got != contract.MaxPageSize {
		t.Fatalf("search page size = %d, want max", got)
	}
}

func TestFacadePassesPortErrorsThrough(t *testing.T) {
	notFound := contract.NewError(contract.NotFound, "cloud_document.get", "source not found", false, errors.New("missing"))
	unavailable := contract.NewError(contract.BackendUnavailable, "cloud_document.search", "backend unavailable", true, errors.New("down"))
	facade := mustFacade(t, &fakePort{getErr: notFound})
	if _, err := facade.Get(context.Background(), contract.CallContext{UserID: "user-1"}, GetInput{SourceID: "source-1"}); !errors.Is(err, notFound) {
		t.Fatalf("Get err = %v, want passthrough not found", err)
	}
	facade = mustFacade(t, &fakePort{searchErr: unavailable})
	if _, err := facade.Search(context.Background(), contract.CallContext{UserID: "user-1"}, SearchInput{SourceID: "source-1", Query: "doc"}); !errors.Is(err, unavailable) {
		t.Fatalf("Search err = %v, want passthrough unavailable", err)
	}
}

func mustFacade(t *testing.T, port Port) *Facade {
	t.Helper()
	facade, err := NewFacade(port)
	if err != nil {
		t.Fatalf("NewFacade returned error: %v", err)
	}
	return facade
}
