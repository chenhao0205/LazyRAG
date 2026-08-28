import { type EvolutionMode, type ThreadRestorePayload } from "./types";
import {
  getNestedArrayField,
  getNestedRecordField,
  getNestedStringField,
  isRecord,
} from "./fields";

export function getThreadTitleFromPayload(payload: ThreadRestorePayload) {
  if (!isRecord(payload)) {
    return undefined;
  }

  return (
    getNestedStringField(payload, ["title", "name", "thread_name"]) ||
    getNestedStringField(getNestedRecordField(payload, ["thread", "upstream", "data"]), [
      "title",
      "name",
      "thread_name",
    ])
  );
}

export function getThreadKnowledgeBaseIds(payload: ThreadRestorePayload): string[] {
  if (!isRecord(payload)) {
    return [];
  }

  const threadPayload = getThreadPayloadFromRestorePayload(payload);
  const inputs =
    getNestedRecordField(threadPayload, ["inputs", "input", "config"]) ||
    getNestedRecordField(payload, ["inputs", "input", "config"]);
  const keys = ["kb_id", "kb_ids", "knowledge_base_id", "knowledge_base_ids", "dataset_id"];
  const sources = [threadPayload, payload, inputs];

  for (const source of sources) {
    const ids = getNestedArrayField(source, keys)
      .filter((value): value is string => typeof value === "string")
      .map((value) => value.trim())
      .filter(Boolean);
    if (ids.length) {
      return [...new Set(ids)];
    }
  }

  for (const source of sources) {
    const id = getNestedStringField(source, keys);
    if (id) {
      return [id];
    }
  }

  return [];
}

export function getThreadPayloadFromRestorePayload(payload: ThreadRestorePayload) {
  if (!isRecord(payload)) {
    return undefined;
  }

  const threadRecord = getNestedRecordField(payload, ["thread"]);
  return (
    getNestedRecordField(threadRecord, ["thread_payload", "threadPayload", "payload"]) ||
    getNestedRecordField(payload, ["thread_payload", "threadPayload", "payload"])
  );
}

export function getThreadModeFromPayload(payload: ThreadRestorePayload): EvolutionMode | undefined {
  if (!isRecord(payload)) {
    return undefined;
  }

  const threadPayload = getThreadPayloadFromRestorePayload(payload);
  const inputs =
    getNestedRecordField(threadPayload, ["inputs", "input", "config"]) ||
    getNestedRecordField(payload, ["inputs", "input", "config"]);
  const modeValue =
    getNestedStringField(threadPayload, ["mode", "evolution_mode", "interaction_mode"]) ||
    getNestedStringField(payload, ["mode", "evolution_mode", "interaction_mode"]) ||
    getNestedStringField(inputs, ["mode", "evolution_mode", "interaction_mode"]);

  return modeValue === "auto" || modeValue === "interactive" ? modeValue : undefined;
}
