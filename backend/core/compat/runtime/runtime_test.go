package runtime

import (
	"context"
	"reflect"
	"testing"

	"lazymind/core/compat/clouddocument"
	"lazymind/core/compat/contract"
	"lazymind/core/compat/skill"
)

type stubSkillPort struct{}

func (stubSkillPort) List(context.Context, contract.CallContext, skill.ListInput) (skill.ListResult, error) {
	return skill.ListResult{}, nil
}

func (stubSkillPort) GetMetadata(context.Context, contract.CallContext, string) (skill.Summary, error) {
	return skill.Summary{}, nil
}

func (stubSkillPort) ReadContent(context.Context, contract.CallContext, string, string) (skill.Content, error) {
	return skill.Content{}, nil
}

type stubCloudDocumentPort struct{}

func (stubCloudDocumentPort) ListSources(context.Context, contract.CallContext, clouddocument.ListInput) (clouddocument.ListResult, error) {
	return clouddocument.ListResult{}, nil
}

func (stubCloudDocumentPort) GetSource(context.Context, contract.CallContext, string) (clouddocument.SourceDetail, error) {
	return clouddocument.SourceDetail{}, nil
}

func (stubCloudDocumentPort) ListDocuments(context.Context, contract.CallContext, clouddocument.SourceDetail, clouddocument.GetInput) (clouddocument.DocumentListResult, error) {
	return clouddocument.DocumentListResult{}, nil
}

func (stubCloudDocumentPort) Search(context.Context, contract.CallContext, clouddocument.SearchInput) (clouddocument.SearchResult, error) {
	return clouddocument.SearchResult{}, nil
}

func TestNewCreatesSkillFacadeWhenPortProvided(t *testing.T) {
	rt, err := New(Dependencies{SkillPort: stubSkillPort{}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if rt.Skill == nil {
		t.Fatalf("Skill facade is nil")
	}
}

func TestNewAllowsNilSkillPort(t *testing.T) {
	rt, err := New(Dependencies{})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if rt == nil {
		t.Fatalf("Runtime is nil")
	}
	if rt.Skill != nil {
		t.Fatalf("Skill facade = %#v, want nil", rt.Skill)
	}
}

func TestNewCreatesCloudDocumentFacadeWhenPortProvided(t *testing.T) {
	rt, err := New(Dependencies{CloudDocumentPort: stubCloudDocumentPort{}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if rt.CloudDocument == nil {
		t.Fatalf("CloudDocument facade is nil")
	}
}

func TestNewAllowsNilCloudDocumentPortWithoutAffectingSkill(t *testing.T) {
	rt, err := New(Dependencies{SkillPort: stubSkillPort{}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if rt.Skill == nil {
		t.Fatalf("Skill facade is nil")
	}
	if rt.CloudDocument != nil {
		t.Fatalf("CloudDocument facade = %#v, want nil", rt.CloudDocument)
	}
}

func TestRuntimeDoesNotContainRequestState(t *testing.T) {
	typ := reflect.TypeOf(Runtime{})
	for _, name := range []string{"UserID", "UserName", "RequestID", "PageRequest"} {
		if _, ok := typ.FieldByName(name); ok {
			t.Fatalf("Runtime contains request field %s", name)
		}
	}
	if typ.NumField() != 2 {
		t.Fatalf("Runtime field count = %d, want 2", typ.NumField())
	}
}
