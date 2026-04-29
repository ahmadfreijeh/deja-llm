import type { LLMProvider, ChatMessage } from "../../types.js";

export class OpenAIChat implements LLMProvider {
  private client: import("openai").OpenAI;
  private model: string;

  constructor(apiKey: string, model = "gpt-4o-mini") {
    this.model = model;
    const { OpenAI } = require("openai") as typeof import("openai");
    this.client = new OpenAI({ apiKey });
  }

  async complete(messages: ChatMessage[]): Promise<string> {
    const res = await this.client.chat.completions.create({
      model: this.model,
      messages,
    });

    const content = res.choices[0]?.message?.content;
    if (!content) throw new Error("OpenAI returned no content");
    return content;
  }
}
