import type { LLMProvider, ChatMessage } from "../../types.js";

export class AnthropicChat implements LLMProvider {
  private client: import("@anthropic-ai/sdk").Anthropic;
  private model: string;

  constructor(apiKey: string, model = "claude-sonnet-4-6") {
    this.model = model;
    const { Anthropic } = require("@anthropic-ai/sdk") as typeof import("@anthropic-ai/sdk");
    this.client = new Anthropic({ apiKey });
  }

  async complete(messages: ChatMessage[]): Promise<string> {
    const systemMessage = messages.find((m) => m.role === "system")?.content;
    const userMessages = messages.filter((m) => m.role !== "system");

    const res = await this.client.messages.create({
      model: this.model,
      max_tokens: 4096,
      ...(systemMessage ? { system: systemMessage } : {}),
      messages: userMessages as Array<{ role: "user" | "assistant"; content: string }>,
    });

    const block = res.content[0];
    if (!block || block.type !== "text") throw new Error("Anthropic returned no text content");
    return block.text;
  }
}
