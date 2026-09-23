/**
 * 3次元ベクトル・クォータニオン・3x3 行列の純関数（cap_flight_sim/coordinates.py の移植）。
 * 物理座標（右手系 +X 捕手, +Y 左, +Z 上）で扱う。three.js の型には依存しない。
 */
export type V3 = readonly [number, number, number];
export type Q4 = readonly [number, number, number, number]; // [w, x, y, z]
export type M3 = readonly [V3, V3, V3]; // 行優先

export const EPS = 1e-12;
export const ZERO3: V3 = [0, 0, 0];

export const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const scale = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: V3, b: V3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const cross = (a: V3, b: V3): V3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
/** numpy.linalg.norm と同じく sqrt(Σx²)。 */
export const norm = (a: V3): number => Math.sqrt(dot(a, a));
export const isZero = (a: V3): boolean => a[0] === 0 && a[1] === 0 && a[2] === 0;

/** 単位ベクトル。ほぼゼロなら零ベクトル（NaN を出さない）。 */
export function safeUnit(a: V3, eps = EPS): V3 {
  const n = norm(a);
  return n < eps ? ZERO3 : scale(a, 1 / n);
}

export const perpendicular = (v: V3, unitDir: V3): V3 => sub(v, scale(unitDir, dot(v, unitDir)));

export const matVec = (m: M3, v: V3): V3 => [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
export const matTVec = (m: M3, v: V3): V3 => [
  m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
  m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
  m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];
export const column = (m: M3, j: 0 | 1 | 2): V3 => [m[0][j], m[1][j], m[2][j]];

export function invert3(m: M3): M3 {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-300) throw new Error('singular matrix');
  const s = 1 / det;
  return [
    [A * s, -(b * i - c * h) * s, (b * f - c * e) * s],
    [B * s, (a * i - c * g) * s, -(a * f - c * d) * s],
    [C * s, -(a * h - b * g) * s, (a * e - b * d) * s],
  ];
}

// ------------------------------------------------------------ クォータニオン

export const qNorm = (q: Q4): number => Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);

export function qNormalize(q: Q4): Q4 {
  const n = qNorm(q);
  if (n < EPS || !Number.isFinite(n)) throw new Error(`quaternion cannot be normalized: ${q.join(',')}`);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

export function qMultiply(a: Q4, b: Q4): Q4 {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

/** ボディ→ワールド回転行列（q は正規化済みを想定）。 */
export function qToMatrix(q: Q4): M3 {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/** 回転行列 → [w,x,y,z]（w >= 0 に正規化。Python と同じ分岐）。 */
export function matrixToQuat(m: M3): Q4 {
  const tr = m[0][0] + m[1][1] + m[2][2];
  let q: Q4;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    q = [0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s];
  } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
    const s = Math.sqrt(1 + m[0][0] - m[1][1] - m[2][2]) * 2;
    q = [(m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s];
  } else if (m[1][1] > m[2][2]) {
    const s = Math.sqrt(1 + m[1][1] - m[0][0] - m[2][2]) * 2;
    q = [(m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s];
  } else {
    const s = Math.sqrt(1 + m[2][2] - m[0][0] - m[1][1]) * 2;
    q = [(m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s];
  }
  const n = qNormalize(q);
  return n[0] >= 0 ? n : [-n[0], -n[1], -n[2], -n[3]];
}

export function qFromAxisAngle(axis: V3, angleRad: number): Q4 {
  const u = safeUnit(axis);
  if (isZero(u)) return [1, 0, 0, 0];
  const h = 0.5 * angleRad;
  const s = Math.sin(h);
  return [Math.cos(h), s * u[0], s * u[1], s * u[2]];
}

/** dq/dt = ½ q ⊗ [0, ω] + k (1 − |q|²) q */
export function qDerivative(q: Q4, omegaBody: V3, driftGain: number): Q4 {
  const d = qMultiply(q, [0, omegaBody[0], omegaBody[1], omegaBody[2]]);
  const k = driftGain ? driftGain * (1 - (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])) : 0;
  return [0.5 * d[0] + k * q[0], 0.5 * d[1] + k * q[1], 0.5 * d[2] + k * q[2], 0.5 * d[3] + k * q[3]];
}

const deg2rad = (d: number) => (d * Math.PI) / 180;
const rad2deg = (r: number) => (r * 180) / Math.PI;

/** [roll, pitch, yaw] deg（内在的 Z-Y'-X''）→ [w,x,y,z] */
export function eulerDegToQuat(rpy: V3): Q4 {
  const qz = qFromAxisAngle([0, 0, 1], deg2rad(rpy[2]));
  const qy = qFromAxisAngle([0, 1, 0], deg2rad(rpy[1]));
  const qx = qFromAxisAngle([1, 0, 0], deg2rad(rpy[0]));
  return qNormalize(qMultiply(qMultiply(qz, qy), qx));
}

export function quatToEulerDeg(q: Q4): V3 {
  const m = qToMatrix(qNormalize(q));
  const pitch = Math.asin(Math.min(1, Math.max(-1, -m[2][0])));
  if (Math.abs(m[2][0]) < 1 - 1e-9) {
    return [rad2deg(Math.atan2(m[2][1], m[2][2])), rad2deg(pitch), rad2deg(Math.atan2(m[1][0], m[0][0]))];
  }
  return [0, rad2deg(pitch), rad2deg(Math.atan2(-m[0][1], m[1][1]))];
}

/** body +Z を normal に合わせる姿勢。body +X は hint を面内へ射影した方向（Python と同じフォールバック）。 */
export function quatFromFaceNormal(normal: V3, bodyXHint?: V3): Q4 {
  const z = safeUnit(normal);
  if (isZero(z)) throw new Error('face_normal_world must be non-zero');
  let x = perpendicular(bodyXHint ?? [1, 0, 0], z);
  if (norm(x) < 1e-6) {
    const fallback: V3 = Math.abs(z[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0];
    x = perpendicular(fallback, z);
  }
  x = safeUnit(x);
  const y = cross(z, x);
  return matrixToQuat([
    [x[0], y[0], z[0]],
    [x[1], y[1], z[1]],
    [x[2], y[2], z[2]],
  ]);
}

export const rpmToRadS = (rpm: number) => (rpm * 2 * Math.PI) / 60;
export const radSToRpm = (w: number) => (w * 60) / (2 * Math.PI);
export const kmhToMS = (v: number) => v / 3.6;
export { deg2rad, rad2deg };

export function angleBetweenDeg(a: V3, b: V3): number {
  const ua = safeUnit(a);
  const ub = safeUnit(b);
  if (isZero(ua) || isZero(ub)) return Number.NaN;
  return rad2deg(Math.acos(Math.min(1, Math.max(-1, dot(ua, ub)))));
}
