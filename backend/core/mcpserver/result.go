package mcpserver

import (
	"fmt"

	compatskill "lazymind/core/compat/skill"
)

type toolResult struct {
	Content           []textContent `json:"content"`
	StructuredContent any           `json:"structuredContent,omitempty"`
	IsError           bool          `json:"isError,omitempty"`
}

type textContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type skillListStructuredResult struct {
	Items []skillSummary `json:"items"`
	Page  skillListPage  `json:"page"`
}

type skillSummary struct {
	ID             string       `json:"id"`
	Name           string       `json:"name"`
	Description    string       `json:"description"`
	Category       string       `json:"category"`
	Tags           []string     `json:"tags"`
	HeadRevisionID string       `json:"head_revision_id"`
	AutoEvo        bool         `json:"auto_evo"`
	Enabled        bool         `json:"enabled"`
	Draft          draftSummary `json:"draft"`
}

type draftSummary struct {
	HasUncommittedDraft bool   `json:"has_uncommitted_draft"`
	TaskID              string `json:"task_id"`
	Version             int64  `json:"version"`
}

type skillListPage struct {
	NextPageToken string `json:"next_page_token,omitempty"`
	Total         *int64 `json:"total,omitempty"`
}

func skillListResult(result compatskill.ListResult) toolResult {
	items := make([]skillSummary, 0, len(result.Items))
	for _, item := range result.Items {
		items = append(items, skillSummary{
			ID: item.ID, Name: item.Name, Description: item.Description, Category: item.Category,
			Tags: append([]string(nil), item.Tags...), HeadRevisionID: item.HeadRevisionID,
			AutoEvo: item.AutoEvo, Enabled: item.Enabled,
			Draft: draftSummary{HasUncommittedDraft: item.Draft.HasUncommittedDraft, TaskID: item.Draft.TaskID, Version: item.Draft.Version},
		})
	}
	structured := skillListStructuredResult{Items: items, Page: skillListPage{NextPageToken: result.Page.NextPageToken, Total: result.Page.Total}}
	text := fmt.Sprintf("Found %d skill(s).", len(items))
	if result.Page.Total != nil {
		text = fmt.Sprintf("Found %d skill(s) in this page (%d total).", len(items), *result.Page.Total)
	}
	return toolResult{Content: []textContent{{Type: "text", Text: text}}, StructuredContent: structured}
}

func skillGetResult(result compatskill.GetResult) toolResult {
	item := skillSummary{
		ID: result.Skill.ID, Name: result.Skill.Name, Description: result.Skill.Description, Category: result.Skill.Category,
		Tags: append([]string(nil), result.Skill.Tags...), HeadRevisionID: result.Skill.HeadRevisionID,
		AutoEvo: result.Skill.AutoEvo, Enabled: result.Skill.Enabled,
		Draft: draftSummary{HasUncommittedDraft: result.Skill.Draft.HasUncommittedDraft, TaskID: result.Skill.Draft.TaskID, Version: result.Skill.Draft.Version},
	}
	return toolResult{
		Content:           []textContent{{Type: "text", Text: fmt.Sprintf("Skill %q metadata.", item.Name)}},
		StructuredContent: map[string]any{"skill": item},
	}
}

func toolErrorResult(code, message string) toolResult {
	return toolResult{
		Content:           []textContent{{Type: "text", Text: message}},
		StructuredContent: map[string]any{"error": map[string]string{"code": code, "message": message}},
		IsError:           true,
	}
}
