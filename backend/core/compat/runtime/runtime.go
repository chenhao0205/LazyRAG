package runtime

import (
	"lazymind/core/compat/clouddocument"
	"lazymind/core/compat/skill"
)

type Runtime struct {
	Skill         *skill.Facade
	CloudDocument *clouddocument.Facade
}

type Dependencies struct {
	SkillPort         skill.Port
	CloudDocumentPort clouddocument.Port
}

func New(deps Dependencies) (*Runtime, error) {
	rt := &Runtime{}
	if deps.SkillPort != nil {
		facade, err := skill.NewFacade(deps.SkillPort)
		if err != nil {
			return nil, err
		}
		rt.Skill = facade
	}
	if deps.CloudDocumentPort != nil {
		facade, err := clouddocument.NewFacade(deps.CloudDocumentPort)
		if err != nil {
			return nil, err
		}
		rt.CloudDocument = facade
	}
	return rt, nil
}
