import { createHash } from "crypto";
import type { ChatMessage } from "../types.js";

export function hashConversation(messages: ChatMessage[]): string {
  const serialized = serializeConversation(messages);
  return createHash("sha256").update(serialized).digest("hex");
}

export function serializeConversation(messages: ChatMessage[]): string {
  return messages.map((m) => `${m.role}:${m.content}`).join("\n---\n");
}
