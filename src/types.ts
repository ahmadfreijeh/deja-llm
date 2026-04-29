export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface EmbeddingProvider {
  embed(text: string): Promise<number[]>;
  readonly dimensions: number;
  readonly model: string;
}

export interface VectorHit {
  id: string;
  score: number;
  payload: VectorPayload;
}

export interface VectorPayload extends Record<string, unknown> {
  response: string;
  query: string;
  cachedAt: number;
  expiresAt: number | null;
}

export interface VectorStore {
  search(vector: number[], threshold: number): Promise<VectorHit[]>;
  upsert(id: string, vector: number[], payload: VectorPayload): Promise<void>;
  ensureCollection(dimensions: number): Promise<void>;
  deleteExpired(): Promise<number>;
}

export interface ExactCache {
  get(key: string): Promise<string | null>;
  set(key: string, value: string, ttlSeconds?: number): Promise<void>;
}

export interface CacheResult {
  response: string;
  layer: "exact" | "semantic" | false;
  similarity?: number;
  match?: {
    cachedAt: Date;
  };
  latency: {
    exactLookup: number;
    embeddingCacheLookup: number;
    embedding: number | null;
    semanticSearch: number;
    writeBack: number | null;
    total: number;
  };
  savings: {
    embeddingSkipped: boolean;
    estimatedUSD: number | null;
  };
}

export interface Logger {
  debug(msg: string, meta?: object): void;
  warn(msg: string, meta?: object): void;
  error(msg: string, meta?: object): void;
}

export interface OpenAIEmbeddingConfig {
  provider: "openai";
  apiKey: string;
  model?: string;
}

export interface DejaLLMConfig {
  redis: {
    url?: string;
    ttl?: number;
    keyPrefix?: string;
  };
  qdrant: {
    url: string;
    apiKey?: string;
    collectionName?: string;
    ttl?: number;
  };
  embedding: EmbeddingProvider | OpenAIEmbeddingConfig;
  threshold?: number;
  failSilently?: boolean;
  // Optional overrides for testing or custom implementations
  _exactCache?: ExactCache;
  _vectorStore?: VectorStore;
  _embeddingCache?: { get(key: string): Promise<number[] | null>; set(key: string, vector: number[]): Promise<void> };
  logger?: Logger;
}

export class DejaValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DejaValidationError";
  }
}
