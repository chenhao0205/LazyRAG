// Package bootstrap contains application-facing Compat composition helpers.
// It is intentionally outside mcpserver because it owns service and adapter
// construction, while transports only receive an already-built Runtime.
package bootstrap

import (
	"os"
	"strings"

	"gorm.io/gorm"

	"lazymind/core/common"
	adaptercore "lazymind/core/compat/internal/adapters/core"
	adapterscan "lazymind/core/compat/internal/adapters/scan"
	compatruntime "lazymind/core/compat/runtime"
	skillservice "lazymind/core/skillv2/service"
)

// NewSkillRuntime constructs the application-owned Compat Runtime. The name is
// retained for compatibility with the existing bootstrap call site; the runtime
// now also owns the Knowledge Catalog facade when its dependencies are present.
func NewSkillRuntime(db, readonlyDB *gorm.DB, objectRoot string) (*compatruntime.Runtime, error) {
	skillService := skillservice.NewSkillService(skillservice.SkillServiceDeps{
		DB:        db,
		BlobStore: skillservice.NewBlobStore(db, skillservice.NewLocalObjectStore(objectRoot)),
	})
	skillAdapter, err := adaptercore.NewSkillAdapterForDB(skillService, db)
	if err != nil {
		return nil, err
	}
	knowledgeAdapter, err := adaptercore.NewKnowledgeCatalogAdapterForDB(db)
	if err != nil {
		return nil, err
	}
	documentAdapter, err := adaptercore.NewKnowledgeDocumentAdapterForDBs(db, readonlyDB)
	if err != nil {
		return nil, err
	}
	deps := compatruntime.Dependencies{
		SkillPort:         skillAdapter,
		KnowledgeCatalog:  knowledgeAdapter,
		KnowledgeDocument: documentAdapter,
	}
	cloudDocumentAdapter, err := adapterscan.NewCloudDocumentAdapter(common.ScanControlPlaneEndpoint(), nil, 0)
	if err != nil {
		return nil, err
	}
	deps.CloudDocumentPort = cloudDocumentAdapter
	// Search requires an internal service credential. Leave its facade port
	// unconfigured until application wiring supplies one, rather than making
	// all Core startup depend on an optional search backend.
	if strings.TrimSpace(os.Getenv("LAZYMIND_AUTH_SERVICE_INTERNAL_TOKEN")) != "" {
		searchAdapter, err := adaptercore.NewKnowledgeSearchAdapterForDB(db, common.ChatServiceEndpoint())
		if err != nil {
			return nil, err
		}
		deps.KnowledgeSearch = searchAdapter
	}
	return compatruntime.New(deps)
}
