import { QdrantClient } from "@qdrant/js-client-rest";
import type { VectorStore, VectorHit, VectorPayload } from "../types.js";

// Sentinel value used when no TTL is set — far enough in the future to never expire
const NO_EXPIRY = Number.MAX_SAFE_INTEGER;

export class QdrantStore implements VectorStore {
  private client: QdrantClient;
  private collectionName: string;
  private ttl: number | undefined;

  constructor(opts: {
    url: string;
    apiKey?: string;
    collectionName: string;
    ttl?: number;
  }) {
    this.collectionName = opts.collectionName;
    this.ttl = opts.ttl;
    this.client = new QdrantClient(
      opts.apiKey ? { url: opts.url, apiKey: opts.apiKey } : { url: opts.url },
    );
  }

  async ensureCollection(dimensions: number): Promise<void> {
    const collections = await this.client.getCollections();
    const exists = collections.collections.some((c) => c.name === this.collectionName);
    if (!exists) {
      await this.client.createCollection(this.collectionName, {
        vectors: { size: dimensions, distance: "Cosine" },
      });
    }
    await this.client.createPayloadIndex(this.collectionName, {
      field_name: "expiresAt",
      field_schema: "integer",
    });
  }

  async search(vector: number[], threshold: number): Promise<VectorHit[]> {
    const now = Date.now();

    const results = await this.client.search(this.collectionName, {
      vector,
      limit: 1,
      score_threshold: threshold,
      filter: {
        must: [{ key: "expiresAt", range: { gt: now } }],
      },
      with_payload: true,
    });

    return results.map((r) => ({
      id: String(r.id),
      score: r.score,
      payload: r.payload as unknown as VectorPayload,
    }));
  }

  async upsert(id: string, vector: number[], payload: VectorPayload): Promise<void> {
    await this.client.upsert(this.collectionName, {
      points: [{ id, vector, payload: payload as Record<string, unknown> }],
    });
  }

  async deleteExpired(): Promise<number> {
    const now = Date.now();
    const result = await this.client.delete(this.collectionName, {
      filter: {
        must: [{ key: "expiresAt", range: { lt: now } }],
      },
    });
    return result.status === "completed" ? 1 : 0;
  }

  expiresAt(): number {
    return this.ttl ? Date.now() + this.ttl * 1000 : NO_EXPIRY;
  }
}
