/**
 * 積分とイベント（cap_flight_sim/integrators.py, events.py, simulation.py の移植）。
 * ブラウザ版は固定刻み RK4 のみ。Python 側も solver: rk4 で同じ刻みを使えば ~1e-8 m で一致する。
 */
import { buildSimpleModelParams, type AeroResult, type SimpleModelParams } from './aerodynamics';
import type { SimConfig } from './config';
import { aeroAt, makeDerivative, NonFiniteStateError, type Derivative } from './dynamics';
import { qNormalize, qToMatrix, matVec, type Q4, type V3 } from './vector';

export const EVENT_GROUND = 'ground';
export const EVENT_CATCHER = 'catcher_plane';

interface Segment {
  t0: number;
  t1: number;
  interp: (t: number) => number[];
}

function hermite(t0: number, t1: number, y0: number[], y1: number[], f0: number[], f1: number[]) {
  const h = t1 - t0;
  return (t: number): number[] => {
    const s = (t - t0) / h;
    const h00 = 2 * s ** 3 - 3 * s ** 2 + 1;
    const h10 = s ** 3 - 2 * s ** 2 + s;
    const h01 = -2 * s ** 3 + 3 * s ** 2;
    const h11 = s ** 3 - s ** 2;
    return y0.map((v, i) => h00 * v + h10 * h * f0[i] + h01 * y1[i] + h11 * h * f1[i]);
  };
}

function* rk4Segments(fun: Derivative, t0: number, y0: number[], tEnd: number, dt: number): Generator<Segment> {
  let t = t0;
  let y = y0;
  let f = fun(t, y);
  while (t < tEnd - 1e-15) {
    const h = Math.min(dt, tEnd - t);
    const k1 = f;
    const k2 = fun(t + 0.5 * h, y.map((v, i) => v + 0.5 * h * k1[i]));
    const k3 = fun(t + 0.5 * h, y.map((v, i) => v + 0.5 * h * k2[i]));
    const k4 = fun(t + h, y.map((v, i) => v + h * k3[i]));
    const y1 = y.map((v, i) => v + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
    const f1 = fun(t + h, y1);
    yield { t0: t, t1: t + h, interp: hermite(t, t + h, y, y1, f, f1) };
    t += h;
    y = y1;
    f = f1;
  }
}

interface EventSpec {
  name: string;
  g: (y: number[]) => number;
  direction: 1 | -1;
  terminal: boolean;
}

function buildEvents(cfg: SimConfig): EventSpec[] {
  const s = cfg.settings;
  const n3 = (a: number, b: number, c: number) => Math.sqrt(a * a + b * b + c * c);
  return [
    { name: EVENT_GROUND, g: (y) => y[2] - cfg.groundZM, direction: -1, terminal: s.stopAtGround },
    { name: EVENT_CATCHER, g: (y) => y[0] - cfg.pitchDistanceM, direction: 1, terminal: s.stopAtCatcherPlane },
    { name: 'abnormal_speed', g: (y) => s.maxSpeed - n3(y[3], y[4], y[5]), direction: -1, terminal: true },
    { name: 'abnormal_angular_speed', g: (y) => s.maxAngularSpeed - n3(y[10], y[11], y[12]), direction: -1, terminal: true },
  ];
}

/** Brent 法（scipy.optimize.brentq 相当、xtol=1e-12）。 */
function brent(f: (t: number) => number, a0: number, b0: number, xtol = 1e-12, rtol = 1e-10, maxIter = 100): number {
  let a = a0, b = b0, fa = f(a), fb = f(b);
  if (fa === 0) return a;
  if (fb === 0) return b;
  let c = a, fc = fa, d = b - a, e = d;
  for (let i = 0; i < maxIter; i++) {
    if (fb * fc > 0) { c = a; fc = fa; d = b - a; e = d; }
    if (Math.abs(fc) < Math.abs(fb)) { a = b; b = c; c = a; fa = fb; fb = fc; fc = fa; }
    const tol = 2 * Number.EPSILON * Math.abs(b) + 0.5 * (xtol + rtol * Math.abs(b));
    const m = 0.5 * (c - b);
    if (Math.abs(m) <= tol || fb === 0) return b;
    if (Math.abs(e) >= tol && Math.abs(fa) > Math.abs(fb)) {
      const s = fb / fa;
      let p: number, q: number;
      if (a === c) { p = 2 * m * s; q = 1 - s; }
      else {
        const qq = fa / fc, r = fb / fc;
        p = s * (2 * m * qq * (qq - r) - (b - a) * (r - 1));
        q = (qq - 1) * (r - 1) * (s - 1);
      }
      if (p > 0) q = -q; else p = -p;
      if (2 * p < Math.min(3 * m * q - Math.abs(tol * q), Math.abs(e * q))) { e = d; d = p / q; }
      else { d = m; e = m; }
    } else { d = m; e = m; }
    a = b; fa = fb;
    b += Math.abs(d) > tol ? d : m > 0 ? tol : -tol;
    fb = f(b);
  }
  return b;
}

function locate(spec: EventSpec, seg: Segment): number | null {
  const g0 = spec.g(seg.interp(seg.t0));
  const g1 = spec.g(seg.interp(seg.t1));
  const crossed = spec.direction > 0 ? g0 < 0 && g1 >= 0 : g0 > 0 && g1 <= 0;
  if (!crossed) return null;
  if (g1 === 0) return seg.t1;
  return brent((t) => spec.g(seg.interp(t)), seg.t0, seg.t1);
}

export interface SimFrame {
  t: number;
  y: number[]; // q は正規化済み
  omegaWorld: V3;
  normalWorld: V3;
  aero: AeroResult;
}

export interface SimResult {
  config: SimConfig;
  frames: SimFrame[];
  events: { type: string; timeSec: number; frame?: SimFrame }[];
  terminationReason: string;
  warnings: string[];
  model: SimpleModelParams;
}

function makeFrame(t: number, yRaw: number[], cfg: SimConfig, params: SimpleModelParams): SimFrame {
  const q = qNormalize([yRaw[6], yRaw[7], yRaw[8], yRaw[9]] as unknown as Q4);
  const y = [...yRaw.slice(0, 6), ...q, ...yRaw.slice(10, 13)];
  const R = qToMatrix(q);
  return {
    t,
    y,
    omegaWorld: matVec(R, [y[10], y[11], y[12]]),
    normalWorld: [R[0][2], R[1][2], R[2][2]],
    aero: aeroAt(cfg, params, y),
  };
}

function outputTimes(tFinal: number, hz: number): number[] {
  const times: number[] = [];
  // numpy.arange と同じく start + i * step で生成（累積加算の誤差を避ける）
  const step = 1 / hz;
  for (let i = 0; i * step < tFinal; i++) times.push(i * step);
  if (times.length === 0 || tFinal - times[times.length - 1] > 1e-12) times.push(tFinal);
  return times;
}

function sampleFrames(segments: Segment[], y0: number[], times: number[], cfg: SimConfig, params: SimpleModelParams): SimFrame[] {
  if (segments.length === 0) return [makeFrame(0, y0, cfg, params)];
  const ends = segments.map((s) => s.t1);
  let idx = 0;
  return times.map((t) => {
    while (idx < ends.length - 1 && ends[idx] < t - 1e-15) idx++; // searchsorted(side='left') 相当（times は昇順）
    return makeFrame(t, t === 0 ? y0 : segments[idx].interp(t), cfg, params);
  });
}

export function runSimulation(cfg: SimConfig): SimResult {
  const params = buildSimpleModelParams(cfg.aerodynamics);
  const fun = makeDerivative(cfg, params);
  const s = cfg.settings;
  const warnings = [...cfg.warnings];
  if (s.solver !== 'rk4') warnings.push('ブラウザ版は固定刻み RK4 で計算（Python の solve_ivp 結果とは ~1e-4 m 程度の差があり得る）');
  const specs = buildEvents(cfg);
  const segments: Segment[] = [];
  const hits: { name: string; t: number; seg: Segment }[] = [];
  let reason = 'max_time';

  try {
    for (const seg of rk4Segments(fun, 0, cfg.y0, s.durationSec, s.fixedStepSec)) {
      const found = specs
        .map((spec) => ({ spec, t: locate(spec, seg) }))
        .filter((h): h is { spec: EventSpec; t: number } => h.t !== null)
        .sort((a, b) => a.t - b.t);
      let terminalTime: number | null = null;
      for (const h of found) {
        if (terminalTime !== null && h.t > terminalTime) break;
        hits.push({ name: h.spec.name, t: h.t, seg });
        if (h.spec.terminal) {
          terminalTime = h.t;
          reason = h.spec.name;
        }
      }
      if (terminalTime !== null) {
        segments.push({ ...seg, t1: terminalTime });
        break;
      }
      segments.push(seg);
    }
  } catch (err) {
    if (!(err instanceof NonFiniteStateError)) throw err;
    reason = 'abnormal_non_finite';
    warnings.push(`異常終了: ${err.message}`);
  }

  const tFinal = segments.length ? segments[segments.length - 1].t1 : 0;
  const frames = sampleFrames(segments, cfg.y0, outputTimes(tFinal, s.outputHz), cfg, params);
  const events: SimResult['events'] = [
    { type: 'release', timeSec: 0 },
    ...hits.map((h) => ({ type: h.name, timeSec: h.t, frame: makeFrame(h.t, h.seg.interp(h.t), cfg, params) })),
  ];
  if (reason === 'max_time' || reason === 'abnormal_non_finite') events.push({ type: reason, timeSec: tFinal });
  if (!hits.some((h) => h.name === EVENT_CATCHER)) warnings.push(`捕手面(X=${cfg.pitchDistanceM}m)に到達せず終了: ${reason}`);
  if (frames.some((f) => f.aero.diagnostics.lift_degenerate_face_on)) warnings.push('面が相対風に正対（揚力方向が退化）したフレームあり: 揚力 0 として処理');
  return { config: cfg, frames, events, terminationReason: reason, warnings, model: params };
}
