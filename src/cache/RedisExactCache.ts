import type { ExactCache } from "../types.js";

export class RedisExactCache implements ExactCache {
  private client: import("ioredis").Redis;
  private prefix: string;
  private defaultTTL: number | undefined;

  constructor(url = "redis://localhost:6379", prefix = "deja:", ttl?: number) {
    this.prefix = prefix;
    this.defaultTTL = ttl;
    const { Redis } = require("ioredis") as typeof import("ioredis");
    this.client = new Redis(url, { lazyConnect: true });
  }

  async get(key: string): Promise<string | null> {
    return this.client.get(this.prefix + key);
  }

  async set(key: string, value: string, ttlSeconds?: number): Promise<void> {
    const ttl = ttlSeconds ?? this.defaultTTL;
    if (ttl) {
      await this.client.set(this.prefix + key, value, "EX", ttl);
    } else {
      await this.client.set(this.prefix + key, value);
    }
  }

  async quit(): Promise<void> {
    await this.client.quit();
  }
}
