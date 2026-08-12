// Package bootstrap contains application-facing Compat composition helpers.
// It is intentionally outside mcpserver because it owns service and adapter
// construction, while transports only receive an already-built Runtime.
package bootstrap

import (
	"gorm.io/gorm"

	adaptercore "lazymind/core/compat/internal/adapters/core"
	compatruntime "lazymind/core/compat/runtime"
	skillservice "lazymind/core/skillv2/service"
)

// NewSkillRuntime constructs the application-owned Compat Runtime. The name is
// retained for compatibility with the existing bootstrap call site; the runtime
// now also owns the Knowledge Catalog facade when its dependencies are present.
func NewSkillRuntime(db *gorm.DB, objectRoot string) (*compatruntime.Runtime, error) {
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
	return compatruntime.New(compatruntime.Dependencies{
		SkillPort:        skillAdapter,
		KnowledgeCatalog: knowledgeAdapter,
	})
}
