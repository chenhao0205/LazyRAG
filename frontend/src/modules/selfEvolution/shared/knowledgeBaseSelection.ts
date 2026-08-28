export type KnowledgeBaseSelectionOption = {
  value: string;
  label: string;
};

export function pruneKnowledgeBaseSelection(
  selected: readonly string[],
  options: readonly KnowledgeBaseSelectionOption[],
): string[] {
  const chosen = new Set(selected);
  return options.map((option) => option.value).filter((value) => chosen.has(value));
}

export function knowledgeBaseNamesFor(
  selected: readonly string[],
  options: readonly KnowledgeBaseSelectionOption[],
): Record<string, string> {
  const chosen = new Set(selected);
  return Object.fromEntries(
    options
      .filter((option) => chosen.has(option.value))
      .map((option) => [option.value, option.label]),
  );
}

export function selectionSummary(
  selected: readonly string[],
  options: readonly KnowledgeBaseSelectionOption[],
  placeholder: string,
): string {
  const resolved = pruneKnowledgeBaseSelection(selected, options);
  if (resolved.length === 0) return placeholder;
  if (resolved.length === 1) {
    return options.find((option) => option.value === resolved[0])?.label || placeholder;
  }
  return `已选择 ${resolved.length} 个知识库`;
}
