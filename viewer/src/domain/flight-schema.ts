/**
 * cap_flight_sim が出力する viewer.json（schemaVersion 1.x）の型と検証。
 * 座標はすべて物理座標（右手系: +X 捕手方向, +Y 投手から見て左, +Z 上, メートル）。
 */

export type Vec3 = [number, number, number];
/** [w, x, y, z]（Python 側と同じ並び）。three.js 変換は coordinate-transform.ts のみで行う。 */
export type QuatWxyz = [number, number, number, number];

export type PitchLabel = 'good' | 'bad' | 'virtual' | string;

export interface ForceSet {
  drag: Vec3;
  lift: Vec3;
  magnus: Vec3;
  total: Vec3;
}

export interface FlightFrame {
  timeSec: number;
  positionM: Vec3;
  quaternionWxyz: QuatWxyz;
  velocityMS: Vec3;
  angularVelocityWorldRadS: Vec3;
  faceNormalWorld?: Vec3;
  rpm?: number;
  angleOfAttackDeg?: number | null;
  sideslipDeg?: number | null;
  forcesN: ForceSet;
}

export interface FlightEvent {
  type: string;
  timeSec: number;
}

/** 将来: 実測軌道を同一空間へ重ねるための型（MVP では未使用）。 */
export interface ObservedPoint {
  timeSec: number;
  positionM: Vec3;
  confidence: number;
}

export interface FlightDocument {
  schemaVersion: string;
  name: string;
  label: PitchLabel;
  coordinateSystem: { handedness: string; forwardAxis: string; upAxis: string; unit: string };
  field: { pitchDistanceM: number; groundZM: number };
  cap: { diameterM: number; heightM: number; modelUri: string | null; bodyNormalAxis: string; surfaceOrientation?: string };
  frames: FlightFrame[];
  events: FlightEvent[];
  terminationReason?: string;
  warnings: string[];
  /** 見栄えのための誇張があればその内容（Python 側は常に null） */
  visualExaggeration?: string | null;
  /** 計算エンジン（ブラウザ計算なら 'browser-ts-rk4 x.y.z'。Python 出力には無い） */
  computedBy?: string;
  observed?: ObservedPoint[];
}

export class FlightSchemaError extends Error {}

function isVec(v: unknown, n: number): boolean {
  return Array.isArray(v) && v.length === n && v.every((x) => typeof x === 'number' && Number.isFinite(x));
}

export function parseFlightDocument(input: unknown): FlightDocument {
  if (typeof input !== 'object' || input === null) throw new FlightSchemaError('JSON object expected');
  const doc = input as Partial<FlightDocument>;
  if (typeof doc.schemaVersion !== 'string' || !doc.schemaVersion.startsWith('1.')) {
    throw new FlightSchemaError(`unsupported schemaVersion: ${String(doc.schemaVersion)}`);
  }
  if (doc.coordinateSystem?.handedness !== 'right' || doc.coordinateSystem?.upAxis !== '+Z') {
    throw new FlightSchemaError('coordinateSystem must be right-handed Z-up physics frame');
  }
  if (!Array.isArray(doc.frames) || doc.frames.length === 0) throw new FlightSchemaError('frames must be non-empty');
  doc.frames.forEach((f, i) => {
    if (!isVec(f.positionM, 3) || !isVec(f.quaternionWxyz, 4) || !isVec(f.velocityMS, 3)) {
      throw new FlightSchemaError(`frame ${i}: invalid position/quaternion/velocity`);
    }
    if (i > 0 && f.timeSec < doc.frames![i - 1].timeSec) throw new FlightSchemaError(`frame ${i}: time not monotonic`);
  });
  return {
    ...(doc as FlightDocument),
    name: doc.name ?? 'unnamed',
    label: doc.label ?? 'virtual',
    events: doc.events ?? [],
    warnings: doc.warnings ?? [],
  };
}
