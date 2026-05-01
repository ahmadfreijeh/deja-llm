export interface StatsSnapshot {
  requests: number;
  hits: {
    exact: number;
    semantic: number;
    miss: number;
  };
  hitRate: number;
  estimatedUSDSaved: number;
}

export class Stats {
  private requests = 0;
  private exactHits = 0;
  private semanticHits = 0;
  private misses = 0;
  private usdSaved = 0;

  recordExactHit(usdSaved: number | null): void {
    this.requests++;
    this.exactHits++;
    this.usdSaved += usdSaved ?? 0;
  }

  recordSemanticHit(usdSaved: number | null): void {
    this.requests++;
    this.semanticHits++;
    this.usdSaved += usdSaved ?? 0;
  }

  recordMiss(): void {
    this.requests++;
    this.misses++;
  }

  snapshot(): StatsSnapshot {
    const totalHits = this.exactHits + this.semanticHits;
    return {
      requests: this.requests,
      hits: {
        exact: this.exactHits,
        semantic: this.semanticHits,
        miss: this.misses,
      },
      hitRate: this.requests === 0 ? 0 : Math.round((totalHits / this.requests) * 100),
      estimatedUSDSaved: Math.round(this.usdSaved * 1_000_000) / 1_000_000,
    };
  }

  reset(): void {
    this.requests = 0;
    this.exactHits = 0;
    this.semanticHits = 0;
    this.misses = 0;
    this.usdSaved = 0;
  }
}
