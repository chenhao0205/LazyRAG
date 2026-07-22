package core

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strconv"
	"strings"

	"gorm.io/gorm"

	"lazymind/core/compat/contract"
	compatskill "lazymind/core/compat/skill"
	skillsearch "lazymind/core/skillv2/search"
	skillservice "lazymind/core/skillv2/service"
)

const skillMDPath = "SKILL.md"

type SkillService interface {
	ListSkills(ctx context.Context, req skillservice.ListSkillsRequest) (skillservice.ListSkillsResponse, error)
	GetSkill(ctx context.Context, req skillservice.GetSkillRequest) (skillservice.SkillDetail, error)
	ReadFile(ctx context.Context, ref skillservice.FileRef) (skillservice.FileContent, error)
}

type HeadTextSearcher interface {
	Contains(ctx context.Context, skillID, keyword string) (bool, error)
}

type SkillAdapter struct {
	service  SkillService
	searcher HeadTextSearcher
}

func NewSkillAdapter(service SkillService, searcher HeadTextSearcher) (*SkillAdapter, error) {
	if service == nil {
		return nil, contract.NewError(contract.Internal, "skill.adapter.new", "skill service is required", false, nil)
	}
	if searcher == nil {
		return nil, contract.NewError(contract.Internal, "skill.adapter.new", "head text searcher is required", false, nil)
	}
	return &SkillAdapter{service: service, searcher: searcher}, nil
}

func NewSkillAdapterForDB(service *skillservice.SkillService, db *gorm.DB) (*SkillAdapter, error) {
	if service == nil {
		return nil, contract.NewError(contract.Internal, "skill.adapter.new", "skill service is required", false, nil)
	}
	if db == nil {
		return nil, contract.NewError(contract.Internal, "skill.adapter.new", "gorm db is required", false, nil)
	}
	return NewSkillAdapter(service, skillsearch.NewService(skillsearch.ServiceDeps{DB: db}))
}

func (a *SkillAdapter) List(ctx context.Context, callCtx contract.CallContext, input compatskill.ListInput) (compatskill.ListResult, error) {
	userID := strings.TrimSpace(callCtx.UserID)
	if userID == "" {
		return compatskill.ListResult{}, contract.InvalidArgumentError("skill.list", "user_id is required")
	}
	page := input.Page.Normalize()
	offset, err := decodePageToken(page.PageToken)
	if err != nil {
		return compatskill.ListResult{}, contract.NewError(contract.InvalidArgument, "skill.list", "invalid page token", false, err)
	}
	resp, err := a.service.ListSkills(ctx, skillservice.ListSkillsRequest{UserID: userID})
	if err != nil {
		return compatskill.ListResult{}, mapServiceError("skill.list", err)
	}
	filtered, err := a.filter(ctx, resp.Items, input)
	if err != nil {
		return compatskill.ListResult{}, mapServiceError("skill.list", err)
	}
	total := int64(len(filtered))
	if offset > len(filtered) {
		offset = len(filtered)
	}
	end := offset + page.PageSize
	if end > len(filtered) {
		end = len(filtered)
	}
	items := make([]compatskill.Summary, 0, end-offset)
	for _, item := range filtered[offset:end] {
		items = append(items, mapSummary(item))
	}
	result := compatskill.ListResult{
		Items: items,
		Page:  contract.PageResult{Total: &total},
	}
	if end < len(filtered) {
		result.Page.NextPageToken = encodePageToken(end)
	}
	return result, nil
}

func (a *SkillAdapter) Get(ctx context.Context, callCtx contract.CallContext, skillID string) (compatskill.GetResult, error) {
	userID := strings.TrimSpace(callCtx.UserID)
	if userID == "" {
		return compatskill.GetResult{}, contract.InvalidArgumentError("skill.get", "user_id is required")
	}
	detail, err := a.service.GetSkill(ctx, skillservice.GetSkillRequest{SkillID: skillID, UserID: userID})
	if err != nil {
		return compatskill.GetResult{}, mapServiceError("skill.get", err)
	}
	return compatskill.GetResult{Skill: mapSummary(detail.SkillSummary)}, nil
}

func (a *SkillAdapter) ReadContent(ctx context.Context, callCtx contract.CallContext, skillID string) (compatskill.Content, error) {
	userID := strings.TrimSpace(callCtx.UserID)
	if userID == "" {
		return compatskill.Content{}, contract.InvalidArgumentError("skill.read_content", "user_id is required")
	}
	if _, err := a.service.GetSkill(ctx, skillservice.GetSkillRequest{SkillID: skillID, UserID: userID}); err != nil {
		return compatskill.Content{}, mapServiceError("skill.read_content", err)
	}
	file, err := a.service.ReadFile(ctx, skillservice.FileRef{SkillID: skillID, RefType: "head", Path: skillMDPath})
	if err != nil {
		return compatskill.Content{}, mapServiceError("skill.read_content", err)
	}
	if file.Binary {
		return compatskill.Content{}, contract.NewError(contract.Unsupported, "skill.read_content", "SKILL.md is binary", false, nil)
	}
	return compatskill.Content{Path: file.Path, Text: file.Content}, nil
}

func (a *SkillAdapter) filter(ctx context.Context, items []skillservice.SkillSummary, input compatskill.ListInput) ([]skillservice.SkillSummary, error) {
	keyword := strings.ToLower(strings.TrimSpace(input.Keyword))
	category := strings.TrimSpace(input.Category)
	tags := compactStrings(input.Tags)
	out := make([]skillservice.SkillSummary, 0, len(items))
	for _, item := range items {
		if category != "" && item.Category != category {
			continue
		}
		if len(tags) > 0 && !hasAllTags(item.Tags, tags) {
			continue
		}
		if keyword != "" && !strings.Contains(strings.ToLower(item.Name+" "+item.SkillName+" "+item.Description), keyword) {
			if a.searcher == nil {
				continue
			}
			matched, err := a.searcher.Contains(ctx, item.ID, keyword)
			if err != nil {
				return nil, err
			}
			if !matched {
				continue
			}
		}
		out = append(out, item)
	}
	return out, nil
}

func mapSummary(item skillservice.SkillSummary) compatskill.Summary {
	return compatskill.Summary{
		ID:             item.ID,
		Name:           item.Name,
		Description:    item.Description,
		Category:       item.Category,
		Tags:           append([]string(nil), item.Tags...),
		HeadRevisionID: item.HeadRevisionID,
		AutoEvo:        item.AutoEvo,
		Enabled:        item.IsEnabled,
		Draft: compatskill.DraftSummary{
			HasUncommittedDraft: item.Draft.HasUncommittedDraft,
			TaskID:              item.Draft.TaskID,
			Version:             item.Draft.Version,
		},
	}
}

func compactStrings(values []string) []string {
	out := make([]string, 0, len(values))
	seen := map[string]struct{}{}
	for _, value := range values {
		trimmed := strings.TrimSpace(value)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		out = append(out, trimmed)
	}
	return out
}

func hasAllTags(haystack, needles []string) bool {
	set := map[string]struct{}{}
	for _, tag := range haystack {
		set[tag] = struct{}{}
	}
	for _, tag := range needles {
		if _, ok := set[tag]; !ok {
			return false
		}
	}
	return true
}

func encodePageToken(offset int) string {
	return base64.RawURLEncoding.EncodeToString([]byte(fmt.Sprintf("offset:%d", offset)))
}

func decodePageToken(token string) (int, error) {
	if token == "" {
		return 0, nil
	}
	raw, err := base64.RawURLEncoding.DecodeString(token)
	if err != nil {
		return 0, err
	}
	value := string(raw)
	if !strings.HasPrefix(value, "offset:") {
		return 0, fmt.Errorf("unsupported token")
	}
	offset, err := strconv.Atoi(strings.TrimPrefix(value, "offset:"))
	if err != nil {
		return 0, err
	}
	if offset < 0 {
		return 0, fmt.Errorf("negative offset")
	}
	return offset, nil
}

func mapServiceError(operation string, err error) error {
	if err == nil {
		return nil
	}
	var compatErr *contract.Error
	if errors.As(err, &compatErr) {
		return err
	}
	msg := strings.ToLower(strings.TrimSpace(err.Error()))
	switch {
	case errors.Is(err, gorm.ErrRecordNotFound), strings.Contains(msg, "not found"):
		return contract.NewError(contract.NotFound, operation, err.Error(), false, err)
	case strings.Contains(msg, "stale"), strings.Contains(msg, "conflict"), strings.Contains(msg, "already exists"), strings.Contains(msg, "duplicate"):
		return contract.NewError(contract.Conflict, operation, err.Error(), false, err)
	case strings.Contains(msg, "unsupported"):
		return contract.NewError(contract.Unsupported, operation, err.Error(), false, err)
	case strings.Contains(msg, "db is not configured"), strings.Contains(msg, "connection refused"), strings.Contains(msg, "timeout"):
		return contract.NewError(contract.BackendUnavailable, operation, "backend unavailable", true, err)
	default:
		return contract.NewError(contract.Internal, operation, "internal error", false, err)
	}
}
