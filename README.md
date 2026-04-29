# deja-llm

A self-hostable, multi-layer semantic caching library for Node.js LLM applications.

The name is a pun on *déjà vu* — the library recognizes questions it has seen before and answers instantly without calling the LLM again.

Think of it as GPTCache for Node.js. No vendor lock-in, fully self-hostable, built for production.

**Why does this exist?** The Node.js ecosystem has no proper solution for this. The closest is [@upstash/semantic-cache](https://github.com/upstash/semantic-cache) but it is a fundamentally different concept — it only does semantic similarity matching and is locked to Upstash's hosted infrastructure. `deja-llm` adds an exact-match layer before the semantic search (so repeated identical queries cost nothing), caches the embeddings themselves to avoid re-embedding, is fully self-hostable with your own Redis and Qdrant instances, and returns full observability on every result including latency breakdown and estimated cost saved.

---

## How it works

Every query passes through three layers in order. Each layer is cheaper and faster than the next — the LLM is only called on a full miss.

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
Layer 3 — LLM call
  Call the LLM, save the response back to Redis and Qdrant
```

Embeddings are also cached in Redis so the same conversation is never embedded twice.

Every result includes which layer it hit on, similarity score, full latency breakdown, and estimated cost saved.

---

## Install

```bash
npm install deja-llm
```

Install the providers you need as peer dependencies:

```bash
# If using OpenAI (embeddings + LLM)
npm install openai

# If using Anthropic (LLM only)
npm install @anthropic-ai/sdk

# Always required
npm install ioredis @qdrant/js-client-rest
```

You also need a running Redis and Qdrant instance. The quickest way to get both locally:

```bash
docker run -d -p 6379:6379 redis
docker run -d -p 6333:6333 qdrant/qdrant
```

---

## Usage

### Mode 1 — Library handles everything

Pass your messages in and get a response back. The library checks the cache, calls the LLM on a miss, and stores the result automatically.

```ts
import { DejaLLM } from "deja-llm";

const deja = new DejaLLM({
  redis: { url: "redis://localhost:6379" },
  qdrant: { url: "http://localhost:6333" },
  embedding: { provider: "openai", apiKey: process.env.OPENAI_API_KEY },
  llm: { provider: "anthropic", apiKey: process.env.ANTHROPIC_API_KEY },
});

const result = await deja.query([
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is the capital of France?" },
]);

console.log(result.response);  // "Paris."
console.log(result.layer);     // "exact" | "semantic" | false
```

### Mode 2 — You handle the LLM call

Use this when you need full control over the LLM call — custom parameters, streaming after a miss, your own SDK setup, etc.

```ts
import { DejaLLM } from "deja-llm";
import Anthropic from "@anthropic-ai/sdk";

const deja = new DejaLLM({
  redis: { url: "redis://localhost:6379" },
  qdrant: { url: "http://localhost:6333" },
  embedding: { provider: "openai", apiKey: process.env.OPENAI_API_KEY },
  // no llm: here
});

const anthropic = new Anthropic();
const messages = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is the capital of France?" },
];

// Check cache first
const hit = await deja.check(messages);
if (hit) {
  console.log(hit.response);  // served from cache
  return;
}

// Cache miss — call the LLM yourself
const res = await anthropic.messages.create({
  model: "claude-sonnet-4-6",
  max_tokens: 1024,
  messages,
});
const response = res.content[0].text;

// Store in cache for next time
await deja.store(messages, response);
```

---

## Result object

Every method returns a `CacheResult`:

```ts
{
  response: string;

  // Which layer answered. false means the LLM was called.
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
    llmCall: number | null;       // null on cache hit
    writeBack: number | null;     // null on cache hit
    total: number;                // ms
  };

  savings: {
    embeddingSkipped: boolean;
    llmSkipped: boolean;
    estimatedUSD: number | null;  // null if model pricing unknown
  };
}
```

---

## Configuration

```ts
const deja = new DejaLLM({
  redis: {
    url: "redis://localhost:6379",  // default
    ttl: 3600,                      // seconds; omit for no expiry
    keyPrefix: "deja:",             // default
  },

  qdrant: {
    url: "http://localhost:6333",
    apiKey: "...",                  // for Qdrant Cloud
    collectionName: "my_cache",     // auto-generated from model name if omitted
    ttl: 86400,                     // seconds; omit for no expiry
  },

  embedding: {
    provider: "openai",
    apiKey: "...",
    model: "text-embedding-3-small", // default
  },

  llm: {
    provider: "anthropic",           // or "openai"
    apiKey: "...",
    model: "claude-sonnet-4-6",
  },

  threshold: 0.92,      // semantic similarity threshold, default 0.92
  failSilently: true,   // on cache errors, fall through to LLM — default true
  logger: console,      // any object with debug/warn/error methods
});
```

### Bring your own providers

Both `embedding` and `llm` accept a custom provider instance directly, as long as it implements the interface:

```ts
import type { EmbeddingProvider, LLMProvider } from "deja-llm";

class MyEmbeddings implements EmbeddingProvider {
  readonly model = "my-model";
  readonly dimensions = 1536;
  async embed(text: string): Promise<number[]> { ... }
}

const deja = new DejaLLM({
  embedding: new MyEmbeddings(),
  llm: new MyLLM(),
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

**Streaming**

Mode 2 supports streaming — call `check()` first, and if it returns `null`, stream from the LLM yourself, then call `store()` once the stream is complete.

**Known limitation: ambiguous follow-up questions**

Semantic caching works best for self-contained questions. An ambiguous follow-up like *"And Germany?"* will only match another cached conversation where the full context is semantically similar. This is correct behavior — returning a cached answer from a different context would be wrong. The similarity threshold is the primary safety net.

---

## License

MIT
