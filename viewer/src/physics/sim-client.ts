/** Worker への計算依頼。古い依頼の結果は捨てる（スライダー連打時に最新だけ反映）。 */
import type { FlightDocument } from '../domain/flight-schema';
import type { FlightSummary } from './to-flight-document';

export interface SimResponse {
  doc: FlightDocument & { computedBy: string };
  summary: FlightSummary;
  elapsedMs: number;
}

export class SimClient {
  private readonly worker = new Worker(new URL('./sim.worker.ts', import.meta.url), { type: 'module' });
  private latest = 0;
  private pending = new Map<number, { resolve: (r: SimResponse | null) => void; reject: (e: Error) => void }>();

  constructor() {
    this.worker.onmessage = (event) => {
      const { id, ok, error, ...rest } = event.data;
      const entry = this.pending.get(id);
      if (!entry) return;
      this.pending.delete(id);
      if (id !== this.latest) entry.resolve(null); // 追い越された依頼
      else if (ok) entry.resolve(rest as SimResponse);
      else entry.reject(new Error(error));
    };
  }

  run(raw: Record<string, unknown>, name: string): Promise<SimResponse | null> {
    const id = ++this.latest;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, raw, name });
    });
  }
}
