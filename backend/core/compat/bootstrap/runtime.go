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

func NewSkillRuntime(db *gorm.DB, objectRoot string) (*compatruntime.Runtime, error) {
	skillService := skillservice.NewSkillService(skillservice.SkillServiceDeps{
		DB:        db,
		BlobStore: skillservice.NewBlobStore(db, skillservice.NewLocalObjectStore(objectRoot)),
	})
	skillAdapter, err := adaptercore.NewSkillAdapterForDB(skillService, db)
	if err != nil {
		return nil, err
	}
	return compatruntime.New(compatruntime.Dependencies{SkillPort: skillAdapter})
}
