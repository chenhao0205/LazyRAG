package modelprovider

import (
	"strings"
)

var opencodeModelNames = opencodeModelSet(
	"claude-haiku-4-5", "claude-opus-4-7", "claude-sonnet-4-6",
	"deepseek-v4-flash", "deepseek-v4-pro",
	"GLM-5", "GLM-5.1",
	"kimi-k2.5", "kimi-k2.6",
	"MiniMax-M2.5", "MiniMax-M2.7",
	"gpt-5", "gpt-5-nano", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini",
	"gpt-5.4-nano", "gpt-5.4-pro", "gpt-5.5", "gpt-5.5-pro",
	"qwen3.5-plus", "qwen3.6-plus",
)

func opencodeModelSet(names ...string) map[string]struct{} {
	models := make(map[string]struct{}, len(names))
	for _, name := range names {
		models[normalizeOpenCodeModelName(name)] = struct{}{}
	}
	return models
}

func normalizeOpenCodeModelName(value string) string {
	modelName := strings.TrimSpace(value)
	if index := strings.LastIndex(modelName, "/"); index >= 0 {
		modelName = modelName[index+1:]
	}
	return strings.ToLower(strings.TrimSpace(modelName))
}

func isOpenCodeCompatibleModel(modelName string) bool {
	_, ok := opencodeModelNames[normalizeOpenCodeModelName(modelName)]
	return ok
}
