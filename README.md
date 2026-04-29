# deja-llm

A self-hostable, multi-layer semantic caching library for Node.js LLM applications.

The name is a pun on *déjà vu* — the library recognizes questions it has seen before and answers instantly without calling the LLM again.

Think of it as GPTCache for Node.js. No vendor lock-in, fully self-hostable, built for production.

**Why does this exist?** The Node.js ecosystem has no proper solution for this. The closest is [@upstash/semantic-cache](https://github.com/upstash/semantic-cache) but it is a fundamentally different concept — it only does semantic similarity matching and is locked to Upstash's hosted infrastructure. `deja-llm` adds an exact-match layer before the semantic search (so repeated identical queries cost nothing), caches the embeddings themselves to avoid re-embedding, is fully self-hostable with your own Redis and Qdrant instances, and returns full observability on every result including latency breakdown and estimated cost saved.

---

## How it works

Every query passes through two cache layers before falling through to your LLM. You own the LLM call — the library is purely a caching layer.

```
Query
  │
  ▼
Layer 1 — Redis exact match
  If the exact same conversation was seen before → return instantly, zero cost
  │ miss
  ▼
Layer 2 — Qdrant semantic search
  Embed the conversation, find similar past queries by cosine similarity
  If similarity >= threshold → return cached response
  │ miss
  ▼
Your LLM call
  Call the LLM however you want, then store the response back into the cache
```

Embeddings are also cached in Redis so the same conversation is never embedded twice.

Every result includes which layer it hit on, similarity score, full latency breakdown, and estimated cost saved.

---

## Install

```bash
npm install deja-llm ioredis @qdrant/js-client-rest openai
```

You also need a running Redis and Qdrant instance. The quickest way to get both locally:

```bash
docker run -d -p 6379:6379 redis
docker run -d -p 6333:6333 qdrant/qdrant
```

---

## Usage

```ts
import { DejaLLM } from "deja-llm";
import Anthropic from "@anthropic-ai/sdk";

const deja = new DejaLLM({
  redis: { url: "redis://localhost:6379" },
  qdrant: { url: "http://localhost:6333" },
  embedding: { provider: "openai", apiKey: process.env.OPENAI_API_KEY },
});

const anthropic = new Anthropic();
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is the capital of France?" },
];

// Check cache first
const hit = await deja.check(messages);
if (hit) {
  console.log(hit.response); // served from cache
  console.log(hit.layer);    // "exact" | "semantic"
  return;
}

// Cache miss — call the LLM yourself however you want
const res = await anthropic.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 1024,
  messages,
});
const response = res.content[0].text;

// Store in cache for next time
await deja.store(messages, response);
```

Works with any LLM — OpenAI, Anthropic, Mistral, local models, anything that returns a string.

### Streaming

`check()` first, stream on a miss, then `store()` once the stream is complete:

```ts
const hit = await deja.check(messages);
if (hit) return hit.response;

let response = "";
const stream = await anthropic.messages.stream({ model: "claude-sonnet-4-6", max_tokens: 1024, messages });
for await (const chunk of stream) {
  if (chunk.type === "content_block_delta") response += chunk.delta.text;
}

await deja.store(messages, response);
```

---

## Result object

Both `check()` and `store()` return a `CacheResult`:

```ts
{
  response: string;

  // Which layer answered. false means it was a miss (returned by store()).
  layer: "exact" | "semantic" | false;

  // Only present on a semantic hit
  similarity?: number;

  // Only present on a semantic hit
  match?: { cachedAt: Date };

  latency: {
    exactLookup: number;          // ms
    embeddingCacheLookup: number; // ms
    embedding: number | null;     // null if served from embedding cache
    semanticSearch: number;       // ms
    writeBack: number | null;     // null on cache hit
    total: number;                // ms
  };

  savings: {
    embeddingSkipped: boolean;
    estimatedUSD: number | null;  // embedding cost saved; null if model unknown
  };
}
```

---

## Configuration

> **Production warning:** Always set `redis.ttl` in production. The Redis exact cache stores the full serialized conversation as the value — without a TTL, the cache grows unbounded as unique conversations accumulate. A recommended pattern is a short TTL on Redis (e.g. 3600s) to catch hot repeated queries, and a longer TTL on Qdrant (e.g. 86400s) for semantic matching over a longer window.

```ts
const deja = new DejaLLM({
  redis: {
    url: "redis://localhost:6379",   // default
    ttl: 3600,                       // seconds; strongly recommended in production
    keyPrefix: "deja:",              // default
  },

  qdrant: {
    url: "http://localhost:6333",
    apiKey: "...",                   // for Qdrant Cloud
    collectionName: "my_cache",      // auto-generated from model name if omitted
    ttl: 86400,                      // seconds; omit for no expiry
  },

  embedding: {
    provider: "openai",
    apiKey: "...",
    model: "text-embedding-3-small", // default
  },

  threshold: 0.92,     // semantic similarity threshold, default 0.92
  failSilently: true,  // on cache errors, fall through silently — default true
  logger: console,     // any object with debug/warn/error methods
});
```

### Bring your own embedding provider

`embedding` accepts a custom provider instance directly, as long as it implements the interface:

```ts
import type { EmbeddingProvider } from "deja-llm";

class MyEmbeddings implements EmbeddingProvider {
  readonly model = "my-model";
  readonly dimensions = 1536;
  async embed(text: string): Promise<number[]> { ... }
}

const deja = new DejaLLM({
  embedding: new MyEmbeddings(),
  // ...
});
```

---

## Maintenance

### Vacuum expired Qdrant points

Qdrant does not expire vectors automatically. Call `vacuum()` periodically to delete expired points:

```ts
const deleted = await deja.vacuum();
```

---

## Design decisions

**Why full conversation is used for caching**

Both the exact hash and the semantic embedding are computed from the entire message array — system prompt, conversation history, and the latest user message. This prevents returning a cached response that was generated under a different system prompt or different context. The trade-off is fewer cache hits compared to embedding only the last user message, but no risk of returning wrong answers for context-dependent follow-up questions.

**Why embeddings are cached in Redis**

Embedding the same conversation twice wastes money. The embedding vector is stored in Redis alongside the exact-match cache, keyed by the same conversation hash. On a Redis hit, the Qdrant search runs without an embedding API call.

**Why the Qdrant collection name encodes the model**

If you switch embedding models, the existing vectors become incompatible. Encoding the model name and dimensions in the collection name (`deja__text_embedding_3_small__1536`) means a model change automatically creates a new collection rather than silently searching with mismatched vectors.

**Known limitation: ambiguous follow-up questions**

Semantic caching works best for self-contained questions. An ambiguous follow-up like *"And Germany?"* will only match another cached conversation where the full context is semantically similar. This is correct behavior — returning a cached answer from a different context would be wrong. The similarity threshold is the primary safety net.

---

## License

MIT
