import type { AdjustmentOptions, CapabilityEntry } from "./types";

export type CapabilityKind = "split_rules" | "layout_types";

export type CapabilityDescription = {
  /** Display name from the capability catalog, falling back to the raw id. */
  label: string;
  /** False when the catalog does not list the id at all. */
  known: boolean;
  supported: boolean;
  enabled: boolean;
};

/**
 * Resolves a raw `split_rule` / `layout_type` identifier — as returned by
 * document details and chunk payloads — against the capability catalog from
 * `materials/adjustment-options`, which is the single source of truth for
 * display names, source support and current participation.
 *
 * Chunk payloads and the catalog share one identifier namespace: the parsing
 * pipeline canonicalises layout types before writing candidates.
 *
 * When the catalog itself is unavailable the raw id is shown as-is and no state
 * is claimed, so a failed option request never mislabels existing data.
 */
export function describeCapability(
  options: AdjustmentOptions | undefined,
  kind: CapabilityKind,
  id: string,
): CapabilityDescription {
  const catalog = options?.[kind];
  if (!catalog) {
    return { label: id || "—", known: true, supported: true, enabled: true };
  }
  const entry = catalog.find((item) => item.id === id);
  if (!entry) {
    return { label: id || "—", known: false, supported: false, enabled: false };
  }
  return {
    label: entry.name || id,
    known: true,
    supported: entry.supported,
    enabled: entry.enabled,
  };
}

/** Short reason to show next to a type that is not currently contributing. */
export function capabilityNote(description: CapabilityDescription): string | undefined {
  if (!description.known) return "不在当前能力目录";
  if (!description.supported) return "当前来源不支持";
  if (!description.enabled) return "未启用";
  return undefined;
}

/**
 * Tags for a chunk card: catalog display names, each annotated when the type no
 * longer participates in the current candidate configuration.
 */
export function chunkTags(
  options: AdjustmentOptions | undefined,
  chunk: { split_rule: string; layout_type: string },
): string[] {
  return (
    [
      ["split_rules", chunk.split_rule],
      ["layout_types", chunk.layout_type],
    ] as Array<[CapabilityKind, string]>
  )
    .filter(([, id]) => Boolean(id))
    .map(([kind, id]) => {
      const described = describeCapability(options, kind, id);
      const note = capabilityNote(described);
      return note ? `${described.label} · ${note}` : described.label;
    });
}

/**
 * Filter options for a stage list: the catalog order, restricted to the ids
 * that can actually appear, plus any id already seen in loaded data so a value
 * missing from the catalog never becomes unfilterable.
 */
export function capabilityFilterOptions(
  options: AdjustmentOptions | undefined,
  kind: CapabilityKind,
  seen: string[],
): Array<{ value: string; label: string }> {
  const catalog: CapabilityEntry[] = options?.[kind] || [];
  const ids = [
    ...catalog.map((item) => item.id),
    ...seen.filter((id) => id && !catalog.some((item) => item.id === id)),
  ];
  return ids.map((id) => {
    const described = describeCapability(options, kind, id);
    const note = capabilityNote(described);
    return { value: id, label: note ? `${described.label}（${note}）` : described.label };
  });
}
