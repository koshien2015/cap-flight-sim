/**
 * 複数投球の同期再生。共通の「進行パラメータ」p を各投球の時刻へ写像する。
 *
 * absolute      : p = 絶対時刻 [s]
 * release       : p = リリースからの経過時間 [s]（release イベント時刻を 0 とする）
 * x_progress    : p = X 方向の位置 [m]。各投球で X(t) = p となる時刻
 * catcher_norm  : p ∈ [0, 1]。捕手面到達（未到達なら計算終了）を 1 とした正規化時刻
 */
import type { FlightDocument } from '../domain/flight-schema';

export type SyncMode = 'absolute' | 'release' | 'x_progress' | 'catcher_norm';

export const SYNC_MODE_LABELS: Record<SyncMode, string> = {
  absolute: '絶対時刻で同期',
  release: 'リリースを0秒として同期',
  x_progress: 'X方向の進行率で同期',
  catcher_norm: '捕手面到達を100%として正規化',
};

export function releaseTime(doc: FlightDocument): number {
  return doc.events.find((e) => e.type === 'release')?.timeSec ?? doc.frames[0].timeSec;
}

export function endTime(doc: FlightDocument): number {
  const catcher = doc.events.find((e) => e.type === 'catcher_plane');
  return catcher?.timeSec ?? doc.frames[doc.frames.length - 1].timeSec;
}

/** X(t) = x となる時刻（X は区間内で単調と仮定、範囲外はクランプ）。 */
export function timeAtX(doc: FlightDocument, x: number): number {
  const f = doc.frames;
  if (x <= f[0].positionM[0]) return f[0].timeSec;
  for (let i = 1; i < f.length; i++) {
    const x0 = f[i - 1].positionM[0];
    const x1 = f[i].positionM[0];
    if (x1 >= x && x1 > x0) {
      const s = (x - x0) / (x1 - x0);
      return f[i - 1].timeSec + s * (f[i].timeSec - f[i - 1].timeSec);
    }
  }
  return f[f.length - 1].timeSec;
}

export function parameterDomain(docs: FlightDocument[], mode: SyncMode): [number, number] {
  if (docs.length === 0) return [0, 1];
  switch (mode) {
    case 'absolute':
      return [Math.min(...docs.map((d) => d.frames[0].timeSec)), Math.max(...docs.map(endTime))];
    case 'release':
      return [0, Math.max(...docs.map((d) => endTime(d) - releaseTime(d)))];
    case 'x_progress':
      return [0, Math.max(...docs.map((d) => Math.max(...d.frames.map((f) => f.positionM[0]))))];
    case 'catcher_norm':
      return [0, 1];
  }
}

export function timeForParameter(doc: FlightDocument, mode: SyncMode, p: number): number {
  switch (mode) {
    case 'absolute':
      return p;
    case 'release':
      return releaseTime(doc) + p;
    case 'x_progress':
      return timeAtX(doc, p);
    case 'catcher_norm': {
      const t0 = releaseTime(doc);
      return t0 + p * (endTime(doc) - t0);
    }
  }
}

export function parameterUnit(mode: SyncMode): string {
  return mode === 'x_progress' ? 'm' : mode === 'catcher_norm' ? '%' : 's';
}

export interface PlaybackState {
  readonly parameter: number;
  readonly playing: boolean;
  readonly speed: number;
}

/** 実時間 dt [s] だけ再生を進めた新しい状態（p の単位はモードに依存するため速度は領域長で換算）。 */
export function advance(state: PlaybackState, dtSec: number, domain: [number, number], referenceDurationSec: number): PlaybackState {
  if (!state.playing) return state;
  const rate = (domain[1] - domain[0]) / Math.max(referenceDurationSec, 1e-6);
  const next = state.parameter + dtSec * state.speed * rate;
  return next >= domain[1] ? { ...state, parameter: domain[1], playing: false } : { ...state, parameter: next };
}
