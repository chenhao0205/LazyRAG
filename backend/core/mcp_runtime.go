package main

import (
	"os"
	"path/filepath"
	"strings"
)

func inboundMCPObjectRoot() string {
	if value := strings.TrimSpace(os.Getenv("LAZYMIND_SKILL_OBJECT_ROOT")); value != "" {
		return strings.TrimRight(value, "/")
	}
	if value := strings.TrimSpace(os.Getenv("LAZYMIND_UPLOAD_ROOT")); value != "" {
		return filepath.Join(strings.TrimRight(value, "/"), "skill-objects")
	}
	return "/var/lib/lazymind/uploads/skill-objects"
}
