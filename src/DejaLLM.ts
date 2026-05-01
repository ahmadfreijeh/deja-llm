import { randomUUID } from "crypto";

import { RedisExactCache } from "./cache/RedisExactCache.js";
import { RedisEmbeddingCache } from "./cache/RedisEmbeddingCache.js";
import { OpenAIEmbeddings } from "./providers/embeddings/OpenAIEmbeddings.js";
import { QdrantStore } from "./vector/QdrantStore.js";
import { hashConversation, serializeConversation } from "./utils/hash.js";
import { estimateSavingsUSD } from "./utils/cost.js";
import { Stats } from "./Stats.js";

import type {
  ChatMessage,
  CacheResult,
  DejaLLMConfig,
  EmbeddingProvider,
  ExactCache,
  VectorStore,
  Logger,
  Hooks,
} from "./types.js";
import { DejaValidationError } from "./types.js";
import type { StatsSnapshot } from "./Stats.js";

const DEFAULT_THRESHOLD = 0.92;
const DEFAULT_REDIS_URL = "redis://localhost:6379";
const DEFAULT_KEY_PREFIX = "deja:";

export class DejaLLM {
  private exactCache: ExactCache;
  private embeddingCache: {
    get(key: string): Promise<number[] | null>;
    set(key: string, vector: number[]): Promise<void>;
  };
  private vectorStore: VectorStore;
  private embedding: EmbeddingProvider;
  private threshold: number;
  private failSilently: boolean;
  private logger: Logger | undefined;
  private hooks: Hooks | undefined;
  private config: DejaLLMConfig;
  private qdrantTTL: number | undefined;
  private ready: Promise<void>;
  private _stats: Stats;

  constructor(config: DejaLLMConfig) {
    this.config = config;
    this.threshold = config.threshold ?? DEFAULT_THRESHOLD;
    this.failSilently = config.failSilently ?? true;
    this.logger = config.logger;
    this.hooks = config.hooks;
    this.qdrantTTL = config.qdrant.ttl;
    this._stats = new Stats();

    this.embedding = isEmbeddingProvider(config.embedding)
      ? config.embedding
      : new OpenAIEmbeddings(config.embedding.apiKey, config.embedding.model);

    const redisUrl = config.redis.url ?? DEFAULT_REDIS_URL;
    const keyPrefix = config.redis.keyPrefix ?? DEFAULT_KEY_PREFIX;

    this.exactCache =
      config._exactCache ?? new RedisExactCache(redisUrl, keyPrefix, config.redis.ttl);

    if (config._embeddingCache) {
      this.embeddingCache = config._embeddingCache;
    } else {
      const { Redis } = require("ioredis") as typeof import("ioredis");
      const redisClient = new Redis(redisUrl, { lazyConnect: true });
      this.embeddingCache = new RedisEmbeddingCache(
        redisClient,
        `${keyPrefix}emb:`,
        config.redis.ttl,
      );
    }

    if (config._vectorStore) {
      this.vectorStore = config._vectorStore;
      this.ready = Promise.resolve();
    } else {
      const collectionName =
        config.qdrant.collectionName ??
        `deja__${this.embedding.model}__${this.embedding.dimensions}`.replace(/[^a-z0-9_]/gi, "_");

      const qdrantStore = new QdrantStore({
        url: config.qdrant.url,
        ...(config.qdrant.apiKey ? { apiKey: config.qdrant.apiKey } : {}),
        collectionName,
        ...(config.qdrant.ttl !== undefined ? { ttl: config.qdrant.ttl } : {}),
      });

      this.vectorStore = qdrantStore;
      this.ready = qdrantStore.ensureCollection(this.embedding.dimensions).catch((err) => {
        this.logger?.warn("Failed to ensure Qdrant collection on startup", { err });
      });
    }
  }

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
      const result = buildResult({
        response: exact,
        layer: "exact",
        similarity: undefined,
        cachedAt: undefined,
        timings: { ...timings, writeBack: null, total: Date.now() - start },
        embeddingSkipped: true,
        embeddingModel: this.embedding.model,
        queryText: serialized,
        responseText: exact,
      });
      this._stats.recordExactHit(result.savings.estimatedUSD);
      this.hooks?.onHit?.(result);
      return result;
    }

    // --- Embedding cache check ---
    const t2 = Date.now();
    let vector = await this.safeRun(() => this.embeddingCache.get(hash));
    timings.embeddingCacheLookup = Date.now() - t2;

    const embeddingSkipped = vector !== null && vector !== undefined;

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
      const result = buildResult({
        response: topHit.payload.response,
        layer: "semantic",
        similarity: topHit.score,
        cachedAt: new Date(topHit.payload.cachedAt),
        timings: { ...timings, writeBack: null, total: Date.now() - start },
        embeddingSkipped,
        embeddingModel: this.embedding.model,
        queryText: serialized,
        responseText: topHit.payload.response,
      });
      this._stats.recordSemanticHit(result.savings.estimatedUSD);
      this.hooks?.onHit?.(result);
      return result;
    }

    this._stats.recordMiss();
    this.hooks?.onMiss?.();
    return null;
  }

  async store(messages: ChatMessage[], response: string): Promise<CacheResult> {
    validate(messages);

    const start = Date.now();
    const serialized = serializeConversation(messages);
    const hash = hashConversation(messages);

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
              query: hash,
              cachedAt: Date.now(),
              expiresAt: this.qdrantTTL
                ? Date.now() + this.qdrantTTL * 1000
                : Number.MAX_SAFE_INTEGER,
            }),
          )
        : Promise.resolve(),
    ]);

    const writeBack = Date.now() - startWrite;

    this.logger?.debug("Stored response in cache", { hash });

    const result = buildResult({
      response,
      layer: false,
      similarity: undefined,
      cachedAt: undefined,
      timings: {
        exactLookup: 0,
        embeddingCacheLookup: 0,
        embedding: null,
        semanticSearch: 0,
        writeBack,
        total: Date.now() - start,
      },
      embeddingSkipped: false,
      embeddingModel: this.embedding.model,
      queryText: serialized,
      responseText: response,
    });

    this.hooks?.onStore?.(result);
    return result;
  }

  stats(): StatsSnapshot {
    return this._stats.snapshot();
  }

  resetStats(): void {
    this._stats.reset();
  }

  async vacuum(): Promise<number> {
    return this.vectorStore.deleteExpired();
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

function isEmbeddingProvider(v: DejaLLMConfig["embedding"]): v is EmbeddingProvider {
  return typeof (v as EmbeddingProvider).embed === "function";
}

interface Timings {
  exactLookup: number;
  embeddingCacheLookup: number;
  embedding: number | null;
  semanticSearch: number;
  writeBack: number | null;
  total: number;
}

function makeTimings(): Omit<Timings, "writeBack" | "total"> {
  return {
    exactLookup: 0,
    embeddingCacheLookup: 0,
    embedding: null,
    semanticSearch: 0,
  };
}

function buildResult(opts: {
  response: string;
  layer: CacheResult["layer"];
  similarity: number | undefined;
  cachedAt: Date | undefined;
  timings: Timings;
  embeddingSkipped: boolean;
  embeddingModel: string;
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
      estimatedUSD: estimateSavingsUSD({
        embeddingSkipped: opts.embeddingSkipped,
        embeddingModel: opts.embeddingModel,
        queryText: opts.queryText,
      }),
    },
  };
}
