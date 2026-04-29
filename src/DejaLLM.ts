import { randomUUID } from "crypto";

import { RedisExactCache } from "./cache/RedisExactCache.js";
import { RedisEmbeddingCache } from "./cache/RedisEmbeddingCache.js";
import { OpenAIEmbeddings } from "./providers/embeddings/OpenAIEmbeddings.js";
import { OpenAIChat } from "./providers/llm/OpenAIChat.js";
import { AnthropicChat } from "./providers/llm/AnthropicChat.js";
import { QdrantStore } from "./vector/QdrantStore.js";
import { hashConversation, serializeConversation } from "./utils/hash.js";
import { estimateSavingsUSD } from "./utils/cost.js";

import type {
  ChatMessage,
  CacheResult,
  DejaLLMConfig,
  EmbeddingProvider,
  LLMProvider,
  Logger,
} from "./types.js";
import { DejaValidationError, DejaConfigError } from "./types.js";

const DEFAULT_THRESHOLD = 0.92;
const DEFAULT_REDIS_URL = "redis://localhost:6379";
const DEFAULT_KEY_PREFIX = "deja:";

export class DejaLLM {
  private exactCache: RedisExactCache;
  private embeddingCache: RedisEmbeddingCache;
  private vectorStore: QdrantStore;
  private embedding: EmbeddingProvider;
  private llm: LLMProvider | undefined;
  private threshold: number;
  private failSilently: boolean;
  private logger: Logger | undefined;
  private config: DejaLLMConfig;
  private ready: Promise<void>;

  constructor(config: DejaLLMConfig) {
    this.config = config;
    this.threshold = config.threshold ?? DEFAULT_THRESHOLD;
    this.failSilently = config.failSilently ?? true;
    this.logger = config.logger;

    // Build embedding provider
    this.embedding = isEmbeddingProvider(config.embedding)
      ? config.embedding
      : new OpenAIEmbeddings(config.embedding.apiKey, config.embedding.model);

    // Build LLM provider (optional — only needed for mode 1)
    if (config.llm) {
      if (isLLMProvider(config.llm)) {
        this.llm = config.llm;
      } else if (config.llm.provider === "openai") {
        this.llm = new OpenAIChat(config.llm.apiKey, config.llm.model);
      } else if (config.llm.provider === "anthropic") {
        this.llm = new AnthropicChat(config.llm.apiKey, config.llm.model);
      }
    }

    // Build Redis clients
    const redisUrl = config.redis.url ?? DEFAULT_REDIS_URL;
    const keyPrefix = config.redis.keyPrefix ?? DEFAULT_KEY_PREFIX;

    this.exactCache = new RedisExactCache(redisUrl, keyPrefix, config.redis.ttl);

    // Reuse the same underlying Redis connection for embedding cache
    const { Redis } = require("ioredis") as typeof import("ioredis");
    const redisClient = new Redis(redisUrl, { lazyConnect: true });
    this.embeddingCache = new RedisEmbeddingCache(
      redisClient,
      `${keyPrefix}emb:`,
      config.redis.ttl,
    );

    // Build Qdrant store — collection name encodes model + dimensions to catch model swaps
    const collectionName =
      config.qdrant.collectionName ??
      `deja__${this.embedding.model}__${this.embedding.dimensions}`.replace(/[^a-z0-9_]/gi, "_");

    this.vectorStore = new QdrantStore({
      url: config.qdrant.url,
      ...(config.qdrant.apiKey ? { apiKey: config.qdrant.apiKey } : {}),
      collectionName,
      ...(config.qdrant.ttl !== undefined ? { ttl: config.qdrant.ttl } : {}),
    });

    // Ensure Qdrant collection exists before any request comes in
    this.ready = this.vectorStore.ensureCollection(this.embedding.dimensions).catch((err) => {
      this.logger?.warn("Failed to ensure Qdrant collection on startup", { err });
    });
  }

  // Mode 1: library handles the full flow including the LLM call
  async query(messages: ChatMessage[]): Promise<CacheResult> {
    validate(messages);

    if (!this.llm) {
      throw new DejaConfigError(
        "No LLM provider configured. Pass `llm` in config to use query(), or use check()/store() instead.",
      );
    }

    const hit = await this.check(messages);
    if (hit) return hit;

    const startLLM = Date.now();
    const response = await this.llm.complete(messages);
    const llmCall = Date.now() - startLLM;

    return this._store(messages, response, { llmCall });
  }

  // Mode 2a: just check the cache, return null on miss
  async check(messages: ChatMessage[]): Promise<CacheResult | null> {
    validate(messages);
    await this.ready;

    const start = Date.now();
    const serialized = serializeConversation(messages);
    const hash = hashConversation(messages);
    const timings = makeTimings();

    // --- Layer 1: Redis exact match ---
    const t1 = Date.now();
    const exact = await this.safeRun(() => this.exactCache.get(hash));
    timings.exactLookup = Date.now() - t1;

    if (exact !== null && exact !== undefined) {
      this.logger?.debug("Cache hit: exact", { hash });
      return buildResult({
        response: exact,
        layer: "exact",
        similarity: undefined,
        cachedAt: undefined,
        timings: { ...timings, total: Date.now() - start },
        embeddingSkipped: true,
        llmSkipped: true,
        embeddingModel: this.embedding.model,
        llmModel: llmModel(this.config.llm),
        queryText: serialized,
        responseText: exact,
      });
    }

    // --- Embedding cache check ---
    const t2 = Date.now();
    let vector = await this.safeRun(() => this.embeddingCache.get(hash));
    timings.embeddingCacheLookup = Date.now() - t2;

    const embeddingSkipped = vector !== null && vector !== undefined;

    // --- Embed if not cached ---
    if (!embeddingSkipped) {
      const t3 = Date.now();
      vector = await this.safeRun(() => this.embedding.embed(serialized));
      timings.embedding = Date.now() - t3;

      if (vector) {
        await this.safeRun(() => this.embeddingCache.set(hash, vector!));
      }
    }

    // --- Layer 2: Qdrant semantic search ---
    const t4 = Date.now();
    const hits = vector
      ? await this.safeRun(() => this.vectorStore.search(vector!, this.threshold))
      : null;
    timings.semanticSearch = Date.now() - t4;

    const topHit = hits?.[0];
    if (topHit) {
      this.logger?.debug("Cache hit: semantic", { score: topHit.score });
      return buildResult({
        response: topHit.payload.response,
        layer: "semantic",
        similarity: topHit.score,
        cachedAt: new Date(topHit.payload.cachedAt),
        timings: { ...timings, total: Date.now() - start },
        embeddingSkipped,
        llmSkipped: true,
        embeddingModel: this.embedding.model,
        llmModel: llmModel(this.config.llm) ?? undefined,
        queryText: serialized,
        responseText: topHit.payload.response,
      });
    }

    return null;
  }

  // Store a response after the user has called the LLM themselves
  async store(messages: ChatMessage[], response: string): Promise<CacheResult> {
    validate(messages);
    return this._store(messages, response, { llmCall: null });
  }

  // Delete all expired Qdrant points
  async vacuum(): Promise<number> {
    return this.vectorStore.deleteExpired();
  }

  // Internal: write response back to all cache layers and return a result
  private async _store(
    messages: ChatMessage[],
    response: string,
    opts: { llmCall: number | null },
  ): Promise<CacheResult> {
    const start = Date.now();
    const serialized = serializeConversation(messages);
    const hash = hashConversation(messages);

    // Get or compute the embedding for write-back
    let vector = await this.safeRun(() => this.embeddingCache.get(hash));
    if (!vector) {
      vector = await this.safeRun(() => this.embedding.embed(serialized));
      if (vector) {
        await this.safeRun(() => this.embeddingCache.set(hash, vector!));
      }
    }

    const startWrite = Date.now();

    await Promise.all([
      this.safeRun(() => this.exactCache.set(hash, response, this.config.redis.ttl)),
      vector
        ? this.safeRun(() =>
            this.vectorStore.upsert(randomUUID(), vector!, {
              response,
              query: serialized,
              cachedAt: Date.now(),
              expiresAt: this.vectorStore.expiresAt(),
            }),
          )
        : Promise.resolve(),
    ]);

    const writeBack = Date.now() - startWrite;

    this.logger?.debug("Stored response in cache", { hash });

    return buildResult({
      response,
      layer: false,
      similarity: undefined,
      cachedAt: undefined,
      timings: {
        exactLookup: 0,
        embeddingCacheLookup: 0,
        embedding: null,
        semanticSearch: 0,
        llmCall: opts.llmCall,
        writeBack,
        total: Date.now() - start,
      },
      embeddingSkipped: false,
      llmSkipped: false,
      embeddingModel: this.embedding.model,
      llmModel: llmModel(this.config.llm) ?? undefined,
      queryText: serialized,
      responseText: response,
    });
  }

  private async safeRun<T>(fn: () => Promise<T>): Promise<T | null> {
    try {
      return await fn();
    } catch (err) {
      if (this.failSilently) {
        this.logger?.warn("Cache operation failed", { err });
        return null;
      }
      throw err;
    }
  }
}

// ---- Helpers ----

function validate(messages: ChatMessage[]): void {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new DejaValidationError("messages must be a non-empty array");
  }
  for (const m of messages) {
    if (!m.role || !m.content || m.content.trim() === "") {
      throw new DejaValidationError("Each message must have a role and non-empty content");
    }
  }
}

function isEmbeddingProvider(
  v: DejaLLMConfig["embedding"],
): v is import("./types.js").EmbeddingProvider {
  return typeof (v as import("./types.js").EmbeddingProvider).embed === "function";
}

function isLLMProvider(
  v: NonNullable<DejaLLMConfig["llm"]>,
): v is import("./types.js").LLMProvider {
  return typeof (v as import("./types.js").LLMProvider).complete === "function";
}

function llmModel(llm: DejaLLMConfig["llm"]): string | undefined {
  if (!llm) return undefined;
  if (isLLMProvider(llm)) return undefined;
  return llm.model;
}

interface Timings {
  exactLookup: number;
  embeddingCacheLookup: number;
  embedding: number | null;
  semanticSearch: number;
  llmCall: number | null;
  writeBack: number | null;
  total: number;
}

function makeTimings(): Omit<Timings, "total"> {
  return {
    exactLookup: 0,
    embeddingCacheLookup: 0,
    embedding: null,
    semanticSearch: 0,
    llmCall: null,
    writeBack: null,
  };
}

function buildResult(opts: {
  response: string;
  layer: CacheResult["layer"];
  similarity: number | undefined;
  cachedAt: Date | undefined;
  timings: Timings;
  embeddingSkipped: boolean;
  llmSkipped: boolean;
  embeddingModel: string;
  llmModel: string | undefined;
  queryText: string;
  responseText: string;
}): CacheResult {
  return {
    response: opts.response,
    layer: opts.layer,
    ...(opts.similarity !== undefined ? { similarity: opts.similarity } : {}),
    ...(opts.cachedAt ? { match: { cachedAt: opts.cachedAt } } : {}),
    latency: opts.timings,
    savings: {
      embeddingSkipped: opts.embeddingSkipped,
      llmSkipped: opts.llmSkipped,
      estimatedUSD: estimateSavingsUSD({
        embeddingSkipped: opts.embeddingSkipped,
        llmSkipped: opts.llmSkipped,
        embeddingModel: opts.embeddingModel,
        llmModel: opts.llmModel,
        queryText: opts.queryText,
        responseText: opts.responseText,
      }),
    },
  };
}
