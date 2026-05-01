import { describe, it, expect, vi, beforeEach } from "vitest";
import { DejaLLM } from "../DejaLLM.js";
import { DejaValidationError } from "../types.js";
import type { EmbeddingProvider, ExactCache, VectorStore, VectorHit, VectorPayload } from "../types.js";

// ---- Mock factories ----

const MOCK_VECTOR = Array.from({ length: 8 }, (_, i) => i * 0.1);

function mockEmbeddingProvider(): EmbeddingProvider {
  return {
    model: "test-model",
    dimensions: 8,
    embed: vi.fn(async () => [...MOCK_VECTOR]),
  };
}

function mockExactCache(): ExactCache & { _store: Map<string, string> } {
  const store = new Map<string, string>();
  return {
    _store: store,
    get: vi.fn(async (key: string) => store.get(key) ?? null),
    set: vi.fn(async (key: string, value: string) => { store.set(key, value); }),
  };
}

function mockEmbeddingCache(): { get(key: string): Promise<number[] | null>; set(key: string, vector: number[]): Promise<void>; _store: Map<string, number[]> } {
  const store = new Map<string, number[]>();
  return {
    _store: store,
    get: vi.fn(async (key: string) => store.get(key) ?? null),
    set: vi.fn(async (key: string, vector: number[]) => { store.set(key, vector); }),
  };
}

function mockVectorStore(
  hits: VectorHit[] = [],
): VectorStore & { _upserted: Array<{ id: string; vector: number[]; payload: VectorPayload }> } {
  const upserted: Array<{ id: string; vector: number[]; payload: VectorPayload }> = [];
  return {
    _upserted: upserted,
    ensureCollection: vi.fn(async () => {}),
    search: vi.fn(async () => hits),
    upsert: vi.fn(async (id, vector, payload) => { upserted.push({ id, vector, payload }); }),
    deleteExpired: vi.fn(async () => 1),
  };
}

function makeConfig(
  embedding: EmbeddingProvider,
  exactCache: ExactCache,
  vectorStore: VectorStore,
  embeddingCache: ReturnType<typeof mockEmbeddingCache>,
) {
  return {
    redis: { url: "redis://localhost:6379" },
    qdrant: { url: "http://localhost:6333" },
    embedding,
    _exactCache: exactCache,
    _vectorStore: vectorStore,
    _embeddingCache: embeddingCache,
  };
}

// ---- Tests ----

describe("DejaLLM", () => {
  let embedding: EmbeddingProvider;
  let exactCache: ReturnType<typeof mockExactCache>;
  let embeddingCache: ReturnType<typeof mockEmbeddingCache>;
  let vectorStore: ReturnType<typeof mockVectorStore>;

  beforeEach(() => {
    vi.clearAllMocks();
    embedding = mockEmbeddingProvider();
    exactCache = mockExactCache();
    embeddingCache = mockEmbeddingCache();
    vectorStore = mockVectorStore();
  });

  // --- Validation ---

  describe("validation", () => {
    it("throws DejaValidationError on empty messages array", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await expect(deja.check([])).rejects.toThrow(DejaValidationError);
    });

    it("throws DejaValidationError on message with empty content", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await expect(
        deja.check([{ role: "user", content: "   " }]),
      ).rejects.toThrow(DejaValidationError);
    });

    it("throws DejaValidationError from store() on empty messages", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await expect(deja.store([], "response")).rejects.toThrow(DejaValidationError);
    });
  });

  // --- check(): full miss ---

  describe("check() — full miss", () => {
    it("returns null when nothing is cached", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const result = await deja.check([{ role: "user", content: "What is 2+2?" }]);
      expect(result).toBeNull();
    });

    it("calls the embedding provider once on a miss", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await deja.check([{ role: "user", content: "What is 2+2?" }]);
      expect(embedding.embed).toHaveBeenCalledTimes(1);
    });
  });

  // --- check(): exact hit ---

  describe("check() — exact hit", () => {
    it("returns layer: exact after storing the same conversation", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];

      await deja.store(messages, "4");
      const result = await deja.check(messages);

      expect(result).not.toBeNull();
      expect(result!.layer).toBe("exact");
      expect(result!.response).toBe("4");
    });

    it("does not call the embedding provider on an exact hit", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];

      await deja.store(messages, "4");
      vi.clearAllMocks();

      await deja.check(messages);
      expect(embedding.embed).not.toHaveBeenCalled();
    });

    it("reports embeddingSkipped: true on an exact hit", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];

      await deja.store(messages, "4");
      const result = await deja.check(messages);

      expect(result!.savings.embeddingSkipped).toBe(true);
    });

    it("includes all latency fields", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "Hello" }];
      await deja.store(messages, "Hi");
      const result = await deja.check(messages);
      expect(result!.latency).toMatchObject({
        exactLookup: expect.any(Number),
        embeddingCacheLookup: expect.any(Number),
        semanticSearch: expect.any(Number),
        total: expect.any(Number),
      });
    });
  });

  // --- check(): semantic hit ---

  describe("check() — semantic hit", () => {
    it("returns layer: semantic when vector store returns a hit", async () => {
      const semanticHits: VectorHit[] = [
        {
          id: "abc",
          score: 0.95,
          payload: { response: "4", query: "2+2", cachedAt: Date.now(), expiresAt: Number.MAX_SAFE_INTEGER },
        },
      ];
      const deja = new DejaLLM(makeConfig(embedding, exactCache, mockVectorStore(semanticHits), embeddingCache));

      const result = await deja.check([{ role: "user", content: "What is two plus two?" }]);

      expect(result).not.toBeNull();
      expect(result!.layer).toBe("semantic");
      expect(result!.response).toBe("4");
      expect(result!.similarity).toBe(0.95);
    });

    it("includes cachedAt in match on a semantic hit", async () => {
      const cachedAt = Date.now() - 5000;
      const semanticHits: VectorHit[] = [
        {
          id: "abc",
          score: 0.97,
          payload: { response: "Berlin", query: "capital of Germany", cachedAt, expiresAt: Number.MAX_SAFE_INTEGER },
        },
      ];
      const deja = new DejaLLM(makeConfig(embedding, exactCache, mockVectorStore(semanticHits), embeddingCache));

      const result = await deja.check([{ role: "user", content: "Germany capital?" }]);
      expect(result!.match?.cachedAt).toEqual(new Date(cachedAt));
    });
  });

  // --- store() ---

  describe("store()", () => {
    it("returns layer: false", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const result = await deja.store([{ role: "user", content: "What is 2+2?" }], "4");
      expect(result.layer).toBe(false);
      expect(result.response).toBe("4");
    });

    it("writes to the exact cache", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await deja.store([{ role: "user", content: "What is 2+2?" }], "4");
      expect(exactCache.set).toHaveBeenCalledTimes(1);
    });

    it("upserts to the vector store", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await deja.store([{ role: "user", content: "What is 2+2?" }], "4");
      expect(vectorStore.upsert).toHaveBeenCalledTimes(1);
      expect(vectorStore._upserted[0]?.payload.response).toBe("4");
    });

    it("embeds the conversation when called without a prior check()", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await deja.store([{ role: "user", content: "What is 2+2?" }], "4");
      expect(embedding.embed).toHaveBeenCalledTimes(1);
    });

    it("reuses the cached embedding when called after check()", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];

      await deja.check(messages);
      vi.clearAllMocks();

      await deja.store(messages, "4");
      expect(embedding.embed).not.toHaveBeenCalled();
    });

    it("includes writeBack latency in the result", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const result = await deja.store([{ role: "user", content: "What is 2+2?" }], "4");
      expect(result.latency.writeBack).not.toBeNull();
      expect(result.latency.total).toBeGreaterThanOrEqual(0);
    });
  });

  // --- Embedding cache ---

  describe("embedding cache", () => {
    it("only embeds once across multiple checks of the same conversation", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];

      await deja.check(messages);
      await deja.check(messages);
      await deja.check(messages);

      expect(embedding.embed).toHaveBeenCalledTimes(1);
    });
  });

  // --- failSilently ---

  describe("failSilently", () => {
    it("returns null instead of throwing when embedding fails and failSilently: true", async () => {
      const brokenEmbedding: EmbeddingProvider = {
        model: "test-model",
        dimensions: 8,
        embed: vi.fn(async () => { throw new Error("embed failed"); }),
      };

      const deja = new DejaLLM({
        ...makeConfig(brokenEmbedding, exactCache, vectorStore, embeddingCache),
        failSilently: true,
      });

      const result = await deja.check([{ role: "user", content: "Hello" }]);
      expect(result).toBeNull();
    });

    it("propagates errors when failSilently: false", async () => {
      const brokenEmbedding: EmbeddingProvider = {
        model: "test-model",
        dimensions: 8,
        embed: vi.fn(async () => { throw new Error("embed failed"); }),
      };

      const deja = new DejaLLM({
        ...makeConfig(brokenEmbedding, exactCache, vectorStore, embeddingCache),
        failSilently: false,
      });

      await expect(
        deja.check([{ role: "user", content: "Hello" }]),
      ).rejects.toThrow("embed failed");
    });
  });

  // --- vacuum() ---

  describe("vacuum()", () => {
    it("calls deleteExpired on the vector store", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await deja.vacuum();
      expect(vectorStore.deleteExpired).toHaveBeenCalledTimes(1);
    });
  });

  // --- stats() ---

  describe("stats()", () => {
    it("starts with zero requests", () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const snap = deja.stats();
      expect(snap.requests).toBe(0);
      expect(snap.hits.exact).toBe(0);
      expect(snap.hits.semantic).toBe(0);
      expect(snap.hits.miss).toBe(0);
      expect(snap.hitRate).toBe(0);
      expect(snap.estimatedUSDSaved).toBe(0);
    });

    it("records a miss on a cache miss", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      await deja.check([{ role: "user", content: "What is 2+2?" }]);
      const snap = deja.stats();
      expect(snap.requests).toBe(1);
      expect(snap.hits.miss).toBe(1);
      expect(snap.hits.exact).toBe(0);
      expect(snap.hits.semantic).toBe(0);
      expect(snap.hitRate).toBe(0);
    });

    it("records an exact hit after store + check", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];
      await deja.store(messages, "4");
      await deja.check(messages);
      const snap = deja.stats();
      expect(snap.requests).toBe(1);
      expect(snap.hits.exact).toBe(1);
      expect(snap.hits.miss).toBe(0);
      expect(snap.hitRate).toBe(100);
    });

    it("records a semantic hit", async () => {
      const semanticHits: VectorHit[] = [
        {
          id: "abc",
          score: 0.95,
          payload: { response: "4", query: "hash", cachedAt: Date.now(), expiresAt: Number.MAX_SAFE_INTEGER },
        },
      ];
      const deja = new DejaLLM(makeConfig(embedding, exactCache, mockVectorStore(semanticHits), embeddingCache));
      await deja.check([{ role: "user", content: "What is two plus two?" }]);
      const snap = deja.stats();
      expect(snap.requests).toBe(1);
      expect(snap.hits.semantic).toBe(1);
      expect(snap.hits.exact).toBe(0);
      expect(snap.hitRate).toBe(100);
    });

    it("calculates hitRate correctly across mixed results", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "Cached question" }];
      await deja.store(messages, "Cached answer");
      await deja.check(messages); // exact hit
      await deja.check([{ role: "user", content: "Unknown question" }]); // miss
      const snap = deja.stats();
      expect(snap.requests).toBe(2);
      expect(snap.hits.exact).toBe(1);
      expect(snap.hits.miss).toBe(1);
      expect(snap.hitRate).toBe(50);
    });
  });

  // --- resetStats() ---

  describe("resetStats()", () => {
    it("resets all counters to zero", async () => {
      const deja = new DejaLLM(makeConfig(embedding, exactCache, vectorStore, embeddingCache));
      const messages = [{ role: "user" as const, content: "Hello" }];
      await deja.store(messages, "Hi");
      await deja.check(messages);
      await deja.check([{ role: "user", content: "Unknown" }]);
      deja.resetStats();
      const snap = deja.stats();
      expect(snap.requests).toBe(0);
      expect(snap.hits.exact).toBe(0);
      expect(snap.hits.semantic).toBe(0);
      expect(snap.hits.miss).toBe(0);
      expect(snap.hitRate).toBe(0);
      expect(snap.estimatedUSDSaved).toBe(0);
    });
  });

  // --- hooks ---

  describe("hooks", () => {
    it("calls onHit with the result on an exact hit", async () => {
      const onHit = vi.fn();
      const deja = new DejaLLM({ ...makeConfig(embedding, exactCache, vectorStore, embeddingCache), hooks: { onHit } });
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];
      await deja.store(messages, "4");
      await deja.check(messages);
      expect(onHit).toHaveBeenCalledTimes(1);
      expect(onHit.mock.calls[0][0].layer).toBe("exact");
    });

    it("calls onHit with the result on a semantic hit", async () => {
      const onHit = vi.fn();
      const semanticHits: VectorHit[] = [
        {
          id: "abc",
          score: 0.95,
          payload: { response: "4", query: "hash", cachedAt: Date.now(), expiresAt: Number.MAX_SAFE_INTEGER },
        },
      ];
      const deja = new DejaLLM({
        ...makeConfig(embedding, exactCache, mockVectorStore(semanticHits), embeddingCache),
        hooks: { onHit },
      });
      await deja.check([{ role: "user", content: "Two plus two?" }]);
      expect(onHit).toHaveBeenCalledTimes(1);
      expect(onHit.mock.calls[0][0].layer).toBe("semantic");
    });

    it("calls onMiss on a cache miss", async () => {
      const onMiss = vi.fn();
      const deja = new DejaLLM({ ...makeConfig(embedding, exactCache, vectorStore, embeddingCache), hooks: { onMiss } });
      await deja.check([{ role: "user", content: "Unknown?" }]);
      expect(onMiss).toHaveBeenCalledTimes(1);
    });

    it("calls onStore after storing a response", async () => {
      const onStore = vi.fn();
      const deja = new DejaLLM({ ...makeConfig(embedding, exactCache, vectorStore, embeddingCache), hooks: { onStore } });
      await deja.store([{ role: "user", content: "What is 2+2?" }], "4");
      expect(onStore).toHaveBeenCalledTimes(1);
      expect(onStore.mock.calls[0][0].layer).toBe(false);
    });

    it("does not call onHit on a miss", async () => {
      const onHit = vi.fn();
      const deja = new DejaLLM({ ...makeConfig(embedding, exactCache, vectorStore, embeddingCache), hooks: { onHit } });
      await deja.check([{ role: "user", content: "Unknown?" }]);
      expect(onHit).not.toHaveBeenCalled();
    });

    it("does not call onMiss on a hit", async () => {
      const onMiss = vi.fn();
      const deja = new DejaLLM({ ...makeConfig(embedding, exactCache, vectorStore, embeddingCache), hooks: { onMiss } });
      const messages = [{ role: "user" as const, content: "What is 2+2?" }];
      await deja.store(messages, "4");
      await deja.check(messages);
      expect(onMiss).not.toHaveBeenCalled();
    });
  });
});
