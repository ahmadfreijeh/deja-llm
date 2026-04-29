import type { Redis } from "ioredis";

export class RedisEmbeddingCache {
  private client: Redis;
  private prefix: string;
  private defaultTTL: number | undefined;

  constructor(client: Redis, prefix = "deja:emb:", ttl?: number) {
    this.client = client;
    this.prefix = prefix;
    this.defaultTTL = ttl;
  }

  async get(key: string): Promise<number[] | null> {
    const raw = await this.client.get(this.prefix + key);
    if (!raw) return null;
    return JSON.parse(raw) as number[];
  }

  async set(key: string, vector: number[]): Promise<void> {
    const serialized = JSON.stringify(vector);
    if (this.defaultTTL) {
      await this.client.set(this.prefix + key, serialized, "EX", this.defaultTTL);
    } else {
      await this.client.set(this.prefix + key, serialized);
    }
  }
}
