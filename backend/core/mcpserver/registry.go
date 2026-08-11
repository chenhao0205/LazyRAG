package mcpserver

import (
	"fmt"
	"sort"
	"strings"
	"sync"
)

type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema map[string]any  `json:"inputSchema"`
	Annotations ToolAnnotations `json:"annotations,omitempty"`
	ReadOnly    bool            `json:"-"`
}

type ToolAnnotations struct {
	ReadOnlyHint bool `json:"readOnlyHint,omitempty"`
}

type Registry struct {
	mu    sync.RWMutex
	tools map[string]ToolDefinition
}

func NewRegistry() *Registry { return &Registry{tools: make(map[string]ToolDefinition)} }

func (r *Registry) Register(tool ToolDefinition) error {
	name := strings.TrimSpace(tool.Name)
	if name == "" {
		return fmt.Errorf("tool name is required")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.tools[name]; exists {
		return fmt.Errorf("tool %q is already registered", name)
	}
	tool.Name = name
	r.tools[name] = tool
	return nil
}

func (r *Registry) Get(name string) (ToolDefinition, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tool, ok := r.tools[name]
	return tool, ok
}

func (r *Registry) List() []ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()
	tools := make([]ToolDefinition, 0, len(r.tools))
	for _, tool := range r.tools {
		tools = append(tools, tool)
	}
	sort.Slice(tools, func(i, j int) bool { return tools[i].Name < tools[j].Name })
	return tools
}
