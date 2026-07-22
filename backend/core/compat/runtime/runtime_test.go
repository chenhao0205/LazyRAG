package runtime

import (
	"context"
	"reflect"
	"testing"

	"lazymind/core/compat/contract"
	"lazymind/core/compat/knowledge"
	"lazymind/core/compat/skill"
)

type stubSkillPort struct{}

func (stubSkillPort) List(context.Context, contract.CallContext, skill.ListInput) (skill.ListResult, error) {
	return skill.ListResult{}, nil
}

func (stubSkillPort) Get(context.Context, contract.CallContext, string) (skill.GetResult, error) {
	return skill.GetResult{}, nil
}

func (stubSkillPort) ReadContent(context.Context, contract.CallContext, string) (skill.Content, error) {
	return skill.Content{}, nil
}

type stubKnowledgeCatalog struct{}

func (stubKnowledgeCatalog) List(context.Context, contract.CallContext, knowledge.ListInput) (knowledge.ListResult, error) {
	return knowledge.ListResult{}, nil
}

func (stubKnowledgeCatalog) Get(context.Context, contract.CallContext, knowledge.GetInput) (knowledge.Summary, error) {
	return knowledge.Summary{}, nil
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

func TestNewCreatesKnowledgeFacadeWhenCatalogProvided(t *testing.T) {
	rt, err := New(Dependencies{KnowledgeCatalog: stubKnowledgeCatalog{}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if rt.Knowledge == nil {
		t.Fatalf("Knowledge facade is nil")
	}
}

func TestNewAllowsNilKnowledgeCatalog(t *testing.T) {
	rt, err := New(Dependencies{})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if rt == nil {
		t.Fatalf("Runtime is nil")
	}
	if rt.Knowledge != nil {
		t.Fatalf("Knowledge facade = %#v, want nil", rt.Knowledge)
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
