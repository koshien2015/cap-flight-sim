/**
 * 後流の非定常性による時間変動の横力・モーメント（cap_flight_sim/aerodynamics/wake.py の移植）。
 * Issue #7 の実験的モデルで、既定 OFF。係数・周波数帯・回転への依存はすべて仮定値。式は Python 側 docstring を参照。
 */
import type { CapParams } from './config';
import { ConfigError } from './config';
import { add, cross, safeUnit, scale, type V3 } from './vector';

const BASIS_EPS = 1e-9;
const SIGNAL_COUNT = 4; // 横力 2 軸 + モーメント 2 軸
const WORLD_UP: V3 = [0, 0, 1];
const WORLD_LEFT: V3 = [0, 1, 0];

export interface WakeParams {
  /** C_w（RMS、投影面積基準）。0 = 横力なし */
  force_coefficient: number;
  /** C_mw（RMS、投影面積 × 外径基準）。0 = モーメントなし */
  moment_coefficient: number;
  seed: number;
  /** k = f D / |u| */
  reduced_frequencies: number[];
  weights: number[];
  /** S0。null = 回転に依存しない */
  spin_parameter_scale: number | null;
}

export const WAKE_DEFAULTS: WakeParams = {
  force_coefficient: 0,
  moment_coefficient: 0,
  seed: 1,
  reduced_frequencies: [0.006, 0.011, 0.02],
  weights: [1.0, 0.8, 0.5],
  spin_parameter_scale: 0.05,
};

export function validateWakeParams(p: WakeParams): void {
  // 有限の数値だけを受け付ける（NaN・Infinity・真偽値・文字列は拒否。Python 側の検証と同じ）
  const positive = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v > 0;
  const nonNegative = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0;
  if (!nonNegative(p.force_coefficient) || !nonNegative(p.moment_coefficient)) {
    throw new ConfigError('wake.force_coefficient / moment_coefficient must be finite numbers >= 0');
  }
  if (!Number.isInteger(p.seed) || p.seed < 0 || p.seed > 0xffffffff) throw new ConfigError('wake.seed must be an integer in [0, 2^32)');
  if (!Array.isArray(p.reduced_frequencies) || !Array.isArray(p.weights) || p.reduced_frequencies.length === 0 || p.reduced_frequencies.length !== p.weights.length) {
    throw new ConfigError('wake.reduced_frequencies and wake.weights must be non-empty and the same length');
  }
  if (!p.reduced_frequencies.every(positive) || !p.weights.every(positive)) {
    throw new ConfigError('wake.reduced_frequencies and wake.weights must be positive');
  }
  if (p.spin_parameter_scale !== null && !positive(p.spin_parameter_scale)) throw new ConfigError('wake.spin_parameter_scale must be positive or null');
}

export const wakeEnabled = (p: WakeParams): boolean => p.force_coefficient > 0 || p.moment_coefficient > 0;

/** シード付き擬似乱数 [0, 1)（mulberry32）。Python 側と同じ列を返す。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 信号ごと（横力 e1・e2、モーメント e1・e2）× 正弦波ごとの位相のずれ [rad]。 */
export function phaseOffsets(seed: number, bandCount: number): number[][] {
  const rng = mulberry32(seed);
  return Array.from({ length: SIGNAL_COUNT }, () => Array.from({ length: bandCount }, () => rng() * 2 * Math.PI));
}

export function bandSignal(offsets: readonly number[], reducedFrequencies: readonly number[], weights: readonly number[], phase: number): number {
  let total = 0;
  let power = 0;
  for (let i = 0; i < offsets.length; i++) {
    total += weights[i] * Math.sin(reducedFrequencies[i] * phase + offsets[i]);
    power += weights[i] * weights[i];
  }
  return total / Math.sqrt(0.5 * power);
}

export const spinAttenuation = (spinParameter: number, s0: number | null): number => (s0 === null ? 1 : 1 / (1 + (spinParameter / s0) ** 2));

/** 相対風に垂直な world 固定の基底。水平飛行では e1 = 左（+Y）、e2 = 上（+Z）。 */
export function wakeBasis(uHat: V3): [V3, V3] {
  let e1 = safeUnit(cross(WORLD_UP, uHat), BASIS_EPS);
  if (e1[0] === 0 && e1[1] === 0 && e1[2] === 0) e1 = WORLD_LEFT; // 真上・真下へ飛ぶとき
  return [e1, cross(uHat, e1)];
}

export interface WakeLoads {
  force: V3;
  momentWorld: V3;
  phaseRate: number;
  spinAttenuation: number;
}

/** 位相のずれは設定ごとに一度だけ引く（力の評価中に乱数を引かない）。 */
export function makeWakeEvaluator(p: WakeParams) {
  const offsets = phaseOffsets(p.seed, p.reduced_frequencies.length);
  return (cap: CapParams, uHat: V3, speed: number, qDyn: number, projectedArea: number, angularSpeed: number, phase: number): WakeLoads => {
    const s = offsets.map((o) => bandSignal(o, p.reduced_frequencies, p.weights, phase));
    const gain = spinAttenuation((cap.radiusM * angularSpeed) / speed, p.spin_parameter_scale);
    const [e1, e2] = wakeBasis(uHat);
    const base = qDyn * projectedArea * gain;
    return {
      force: scale(add(scale(e1, s[0]), scale(e2, s[1])), base * p.force_coefficient),
      momentWorld: scale(add(scale(e1, s[2]), scale(e2, s[3])), base * cap.diameterM * p.moment_coefficient),
      phaseRate: (2 * Math.PI * speed) / cap.diameterM,
      spinAttenuation: gain,
    };
  };
}
