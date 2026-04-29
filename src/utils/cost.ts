// Rough token estimation: ~4 chars per token
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

// Pricing per 1M tokens (USD) — update as providers change rates
const EMBEDDING_COST_PER_1M: Record<string, number> = {
  "text-embedding-3-small": 0.02,
  "text-embedding-3-large": 0.13,
  "text-embedding-ada-002": 0.1,
};

const LLM_INPUT_COST_PER_1M: Record<string, number> = {
  "gpt-4o": 5.0,
  "gpt-4o-mini": 0.15,
  "gpt-4-turbo": 10.0,
  "claude-3-5-sonnet-20241022": 3.0,
  "claude-3-5-haiku-20241022": 0.8,
  "claude-opus-4-7": 15.0,
};

const LLM_OUTPUT_COST_PER_1M: Record<string, number> = {
  "gpt-4o": 15.0,
  "gpt-4o-mini": 0.6,
  "gpt-4-turbo": 30.0,
  "claude-3-5-sonnet-20241022": 15.0,
  "claude-3-5-haiku-20241022": 4.0,
  "claude-opus-4-7": 75.0,
};

export function estimateSavingsUSD(opts: {
  embeddingSkipped: boolean;
  llmSkipped: boolean;
  embeddingModel: string;
  llmModel: string | undefined;
  queryText: string | undefined;
  responseText: string | undefined;
}): number | null {
  const embRate = EMBEDDING_COST_PER_1M[opts.embeddingModel];
  const inputRate = opts.llmModel ? LLM_INPUT_COST_PER_1M[opts.llmModel] : undefined;
  const outputRate = opts.llmModel ? LLM_OUTPUT_COST_PER_1M[opts.llmModel] : undefined;

  if (!embRate && !inputRate) return null;

  let saved = 0;

  if (opts.embeddingSkipped && embRate && opts.queryText) {
    saved += (estimateTokens(opts.queryText) / 1_000_000) * embRate;
  }

  if (opts.llmSkipped && inputRate && outputRate && opts.queryText && opts.responseText) {
    saved += (estimateTokens(opts.queryText) / 1_000_000) * inputRate;
    saved += (estimateTokens(opts.responseText) / 1_000_000) * outputRate;
  }

  return saved;
}
