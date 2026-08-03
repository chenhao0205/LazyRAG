package modelprovider

import (
	"regexp"
	"strings"
)

var opencodeProviderAliases = map[string]string{
	"alibaba":    "qwen",
	"alibabacn":  "qwen",
	"anthropic":  "claude",
	"claude":     "claude",
	"dashscope":  "qwen",
	"deepseek":   "deepseek",
	"glm":        "glm",
	"kimi":       "kimi",
	"minimax":    "minimax",
	"moonshot":   "kimi",
	"moonshotai": "kimi",
	"openai":     "openai",
	"qwen":       "qwen",
	"zhipu":      "glm",
	"zhipuai":    "glm",
}

var opencodeModelsByProvider = map[string]map[string]struct{}{
	"claude":   opencodeModelSet("claude-haiku-4-5", "claude-opus-4-7", "claude-sonnet-4-6"),
	"deepseek": opencodeModelSet("deepseek-v4-flash", "deepseek-v4-pro"),
	"glm":      opencodeModelSet("GLM-5", "GLM-5.1"),
	"kimi":     opencodeModelSet("kimi-k2.5", "kimi-k2.6"),
	"minimax":  opencodeModelSet("MiniMax-M2.5", "MiniMax-M2.7"),
	"openai": opencodeModelSet(
		"gpt-5", "gpt-5-nano", "gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini",
		"gpt-5.4-nano", "gpt-5.4-pro", "gpt-5.5", "gpt-5.5-pro",
	),
	"qwen": opencodeModelSet("qwen3.5-plus", "qwen3.6-plus"),
}

var opencodeProviderKeyPattern = regexp.MustCompile(`[\s_.-]+`)

func opencodeModelSet(names ...string) map[string]struct{} {
	models := make(map[string]struct{}, len(names))
	for _, name := range names {
		models[normalizeOpenCodeModelName(name)] = struct{}{}
	}
	return models
}

func normalizeOpenCodeProviderName(value string) string {
	return opencodeProviderKeyPattern.ReplaceAllString(strings.ToLower(strings.TrimSpace(value)), "")
}

func normalizeOpenCodeModelName(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

func isOpenCodeCompatibleModel(providerName, modelName string) bool {
	provider := opencodeProviderAliases[normalizeOpenCodeProviderName(providerName)]
	models, ok := opencodeModelsByProvider[provider]
	if !ok {
		return false
	}
	_, ok = models[normalizeOpenCodeModelName(modelName)]
	return ok
}
