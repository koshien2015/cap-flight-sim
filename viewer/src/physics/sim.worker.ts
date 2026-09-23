/// <reference lib="webworker" />
/** 物理計算を UI スレッドから分離する Web Worker。 */
import { configFromRaw } from './config';
import { runSimulation } from './simulate';
import { summarize, toFlightDocument } from './to-flight-document';

export interface SimRequest {
  id: number;
  raw: Record<string, unknown>;
  name: string;
}

self.onmessage = (event: MessageEvent<SimRequest>) => {
  const { id, raw, name } = event.data;
  try {
    const t0 = performance.now();
    const result = runSimulation(configFromRaw(raw, name));
    self.postMessage({ id, ok: true, doc: toFlightDocument(result), summary: summarize(result), elapsedMs: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, ok: false, error: (err as Error).message });
  }
};
