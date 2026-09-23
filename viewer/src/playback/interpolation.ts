/**
 * フレーム間補間（位置・速度・力は線形、姿勢は SLERP）。
 * 表示用の補間のみで、補間値を物理結果として保存しない。
 */
import { Quaternion } from 'three';
import type { FlightFrame, QuatWxyz, Vec3 } from '../domain/flight-schema';

export interface InterpolatedFrame extends Omit<FlightFrame, 'quaternionWxyz'> {
  quaternionWxyz: QuatWxyz;
  index: number;
}

const lerp = (a: number, b: number, s: number) => a + (b - a) * s;
const lerpVec = (a: Vec3, b: Vec3, s: number): Vec3 => [lerp(a[0], b[0], s), lerp(a[1], b[1], s), lerp(a[2], b[2], s)];
const lerpOpt = (a: number | null | undefined, b: number | null | undefined, s: number) =>
  a == null || b == null ? (a ?? b ?? null) : lerp(a, b, s);

/** 時刻 t 以下で最大のフレーム番号（範囲外はクランプ）。 */
export function findFrameIndex(frames: FlightFrame[], t: number): number {
  if (t <= frames[0].timeSec) return 0;
  const last = frames.length - 1;
  if (t >= frames[last].timeSec) return last;
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].timeSec <= t) lo = mid;
    else hi = mid;
  }
  return lo;
}

export function slerpWxyz(a: QuatWxyz, b: QuatWxyz, s: number): QuatWxyz {
  const qa = new Quaternion(a[1], a[2], a[3], a[0]);
  const qb = new Quaternion(b[1], b[2], b[3], b[0]);
  qa.slerp(qb, s); // 最短経路（内積が負なら反転）を three.js が処理する
  return [qa.w, qa.x, qa.y, qa.z];
}

export function interpolateFrame(frames: FlightFrame[], t: number): InterpolatedFrame {
  const i = findFrameIndex(frames, t);
  const a = frames[i];
  const b = frames[Math.min(i + 1, frames.length - 1)];
  const span = b.timeSec - a.timeSec;
  const s = span > 0 ? Math.min(Math.max((t - a.timeSec) / span, 0), 1) : 0;
  return {
    index: i,
    timeSec: lerp(a.timeSec, b.timeSec, s),
    positionM: lerpVec(a.positionM, b.positionM, s),
    velocityMS: lerpVec(a.velocityMS, b.velocityMS, s),
    angularVelocityWorldRadS: lerpVec(a.angularVelocityWorldRadS, b.angularVelocityWorldRadS, s),
    faceNormalWorld: a.faceNormalWorld && b.faceNormalWorld ? lerpVec(a.faceNormalWorld, b.faceNormalWorld, s) : undefined,
    rpm: lerpOpt(a.rpm, b.rpm, s) ?? undefined,
    angleOfAttackDeg: lerpOpt(a.angleOfAttackDeg, b.angleOfAttackDeg, s),
    sideslipDeg: s < 0.5 ? a.sideslipDeg : b.sideslipDeg, // ±180° で折り返すため線形補間しない
    quaternionWxyz: slerpWxyz(a.quaternionWxyz, b.quaternionWxyz, s),
    forcesN: {
      drag: lerpVec(a.forcesN.drag, b.forcesN.drag, s),
      lift: lerpVec(a.forcesN.lift, b.forcesN.lift, s),
      magnus: lerpVec(a.forcesN.magnus, b.forcesN.magnus, s),
      total: lerpVec(a.forcesN.total, b.forcesN.total, s),
    },
  };
}
