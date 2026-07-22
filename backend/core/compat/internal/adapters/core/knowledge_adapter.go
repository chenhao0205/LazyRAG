package core

import (
	"context"
	"errors"
	"strings"

	"gorm.io/gorm"

	"lazymind/core/compat/contract"
	"lazymind/core/compat/knowledge"
	"lazymind/core/doc"
)

type DatasetCatalogService interface {
	ListDatasets(ctx context.Context, req doc.DatasetListRequest) (doc.DatasetListResult, error)
	GetDataset(ctx context.Context, req doc.DatasetGetRequest) (doc.Dataset, error)
}

type KnowledgeCatalogAdapter struct {
	service DatasetCatalogService
}

func NewKnowledgeCatalogAdapter(service DatasetCatalogService) (*KnowledgeCatalogAdapter, error) {
	if service == nil {
		return nil, contract.NewError(contract.Internal, "knowledge.adapter.new", "dataset service is required", false, nil)
	}
	return &KnowledgeCatalogAdapter{service: service}, nil
}

func NewKnowledgeCatalogAdapterForDB(db *gorm.DB) (*KnowledgeCatalogAdapter, error) {
	if db == nil {
		return nil, contract.NewError(contract.Internal, "knowledge.adapter.new", "gorm db is required", false, nil)
	}
	return NewKnowledgeCatalogAdapter(doc.NewDatasetService(doc.DatasetServiceDeps{DB: db}))
}

func (a *KnowledgeCatalogAdapter) List(ctx context.Context, callCtx contract.CallContext, input knowledge.ListInput) (knowledge.ListResult, error) {
	userID := strings.TrimSpace(callCtx.UserID)
	if userID == "" {
		return knowledge.ListResult{}, contract.InvalidArgumentError("knowledge.list", "user_id is required")
	}
	page := input.Page.Normalize()
	resp, err := a.service.ListDatasets(ctx, doc.DatasetListRequest{
		UserID:          userID,
		PageToken:       page.PageToken,
		PageSize:        page.PageSize,
		Keyword:         input.Keyword,
		Tags:            input.Tags,
		StrictPageToken: true,
		IncludeHTTPMeta: false,
	})
	if err != nil {
		return knowledge.ListResult{}, mapDatasetServiceError("knowledge.list", err)
	}
	total := resp.TotalSize
	return knowledge.ListResult{
		Items: mapKnowledgeSummaries(resp.Datasets),
		Page: contract.PageResult{
			NextPageToken: resp.NextPageToken,
			Total:         &total,
		},
	}, nil
}

func (a *KnowledgeCatalogAdapter) Get(ctx context.Context, callCtx contract.CallContext, input knowledge.GetInput) (knowledge.Summary, error) {
	userID := strings.TrimSpace(callCtx.UserID)
	if userID == "" {
		return knowledge.Summary{}, contract.InvalidArgumentError("knowledge.get", "user_id is required")
	}
	datasetID := strings.TrimSpace(input.KnowledgeID)
	if datasetID == "" {
		return knowledge.Summary{}, contract.InvalidArgumentError("knowledge.get", "knowledge_id is required")
	}
	resp, err := a.service.GetDataset(ctx, doc.DatasetGetRequest{
		UserID:          userID,
		DatasetID:       datasetID,
		IncludeHTTPMeta: false,
	})
	if err != nil {
		return knowledge.Summary{}, mapDatasetServiceError("knowledge.get", err)
	}
	return mapKnowledgeSummary(resp), nil
}

func mapKnowledgeSummaries(items []doc.Dataset) []knowledge.Summary {
	out := make([]knowledge.Summary, 0, len(items))
	for _, item := range items {
		out = append(out, mapKnowledgeSummary(item))
	}
	return out
}

func mapKnowledgeSummary(item doc.Dataset) knowledge.Summary {
	return knowledge.Summary{
		ID:                item.DatasetID,
		Name:              item.DisplayName,
		Description:       item.Desc,
		Tags:              append([]string(nil), item.Tags...),
		UpdatedAt:         item.UpdateTime,
		DocumentSizeBytes: item.DocumentSize,
		DocumentCount:     item.DocumentCount,
	}
}

func mapDatasetServiceError(operation string, err error) error {
	if err == nil {
		return nil
	}
	var compatErr *contract.Error
	if errors.As(err, &compatErr) {
		return err
	}
	var svcErr *doc.DatasetServiceError
	if errors.As(err, &svcErr) {
		switch svcErr.Code {
		case doc.DatasetServiceInvalidArgument:
			return contract.NewError(contract.InvalidArgument, operation, svcErr.Message, false, err)
		case doc.DatasetServiceNotFound, doc.DatasetServiceForbidden:
			return contract.NewError(contract.NotFound, operation, "knowledge not found", false, err)
		case doc.DatasetServiceUnavailable:
			return contract.NewError(contract.BackendUnavailable, operation, "backend unavailable", true, err)
		default:
			return contract.NewError(contract.Internal, operation, "internal error", false, err)
		}
	}
	if errors.Is(err, gorm.ErrRecordNotFound) {
		return contract.NewError(contract.NotFound, operation, "knowledge not found", false, err)
	}
	msg := strings.ToLower(strings.TrimSpace(err.Error()))
	switch {
	case strings.Contains(msg, "not found"):
		return contract.NewError(contract.NotFound, operation, "knowledge not found", false, err)
	case strings.Contains(msg, "connection refused"), strings.Contains(msg, "timeout"):
		return contract.NewError(contract.BackendUnavailable, operation, "backend unavailable", true, err)
	default:
		return contract.NewError(contract.Internal, operation, "internal error", false, err)
	}
}
