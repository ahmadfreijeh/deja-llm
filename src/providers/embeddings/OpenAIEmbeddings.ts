import OpenAI from "openai";
import type { EmbeddingProvider } from "../../types.js";

export class OpenAIEmbeddings implements EmbeddingProvider {
  readonly model: string;
  readonly dimensions: number;

  private client: OpenAI;

  constructor(apiKey: string, model = "text-embedding-3-small") {
    this.model = model;
    this.dimensions = dimensionsForModel(model);
    this.client = new OpenAI({ apiKey });
  }

  async embed(text: string): Promise<number[]> {
    const res = await this.client.embeddings.create({
      model: this.model,
      input: text,
    });

    const embedding = res.data[0]?.embedding;
    if (!embedding) throw new Error("OpenAI returned no embedding");
    return embedding;
  }
}

function dimensionsForModel(model: string): number {
  const map: Record<string, number> = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
  };
  return map[model] ?? 1536;
}
