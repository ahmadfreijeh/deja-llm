// Rough token estimation: ~4 chars per token
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

const EMBEDDING_COST_PER_1M: Record<string, number> = {
  "text-embedding-3-small": 0.02,
  "text-embedding-3-large": 0.13,
  "text-embedding-ada-002": 0.1,
};

export function estimateSavingsUSD(opts: {
  embeddingSkipped: boolean;
  embeddingModel: string;
  queryText: string | undefined;
}): number | null {
  const rate = EMBEDDING_COST_PER_1M[opts.embeddingModel];
  if (!rate || !opts.embeddingSkipped || !opts.queryText) return null;
  return (estimateTokens(opts.queryText) / 1_000_000) * rate;
}
