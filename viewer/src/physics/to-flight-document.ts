/** ブラウザ計算結果 → ビューア用 FlightDocument（Python の viewer.json と同じスキーマ）と要約。 */
import type { FlightDocument } from '../domain/flight-schema';
import type { SimResult } from './simulate';
import { angleBetweenDeg, radSToRpm, norm, type V3 } from './vector';

export const BROWSER_ENGINE = 'browser-ts-rk4 0.1.0';

const v3 = (a: readonly number[]): [number, number, number] => [a[0], a[1], a[2]];

export function toFlightDocument(result: SimResult): FlightDocument & { computedBy: string } {
  const cfg = result.config;
  return {
    schemaVersion: '1.0.0',
    name: cfg.name,
    label: cfg.label,
    computedBy: BROWSER_ENGINE,
    coordinateSystem: { handedness: 'right', forwardAxis: '+X', upAxis: '+Z', unit: 'meter' },
    field: { pitchDistanceM: cfg.pitchDistanceM, groundZM: cfg.groundZM },
    cap: {
      diameterM: cfg.cap.diameterM,
      heightM: cfg.cap.heightM,
      modelUri: null,
      bodyNormalAxis: '+Z',
      surfaceOrientation: cfg.cap.surfaceOrientation,
    },
    frames: result.frames.map((f) => ({
      timeSec: f.t,
      positionM: v3(f.y),
      quaternionWxyz: [f.y[6], f.y[7], f.y[8], f.y[9]],
      velocityMS: v3(f.y.slice(3, 6)),
      angularVelocityWorldRadS: v3(f.omegaWorld),
      faceNormalWorld: v3(f.normalWorld),
      rpm: radSToRpm(norm(f.omegaWorld)),
      angleOfAttackDeg: f.aero.diagnostics.angle_of_attack_deg,
      sideslipDeg: f.aero.diagnostics.sideslip_deg,
      forcesN: { drag: v3(f.aero.drag), lift: v3(f.aero.lift), magnus: v3(f.aero.magnus), total: v3(f.aero.force) },
    })),
    events: result.events.map((e) => ({ type: e.type, timeSec: e.timeSec })),
    terminationReason: result.terminationReason,
    warnings: result.warnings,
    visualExaggeration: null,
  };
}

export interface FlightSummary {
  terminationReason: string;
  flightTimeSec: number;
  catcher: { yM: number; zM: number; speedKmh: number; rpm: number } | null;
  final: V3;
  lateralBreakM: number;
  dropM: number;
  faceNormalChangeDeg: number;
}

/** Python の compute_summary の主要指標（初速方向の直線からの横ずれ・落下量など）。 */
export function summarize(result: SimResult): FlightSummary {
  const f0 = result.frames[0];
  const last = result.frames[result.frames.length - 1];
  const catcherEvent = result.events.find((e) => e.type === 'catcher_plane');
  const lineAt = (t: number, k: number) => f0.y[k] + f0.y[3 + k] * t;
  const cf = catcherEvent?.frame;
  return {
    terminationReason: result.terminationReason,
    flightTimeSec: last.t,
    catcher: cf
      ? { yM: cf.y[1], zM: cf.y[2], speedKmh: norm(v3(cf.y.slice(3, 6))) * 3.6, rpm: radSToRpm(norm(cf.omegaWorld)) }
      : null,
    final: v3(last.y),
    lateralBreakM: last.y[1] - lineAt(last.t, 1),
    dropM: lineAt(last.t, 2) - last.y[2],
    faceNormalChangeDeg: angleBetweenDeg(f0.normalWorld, last.normalWorld),
  };
}
