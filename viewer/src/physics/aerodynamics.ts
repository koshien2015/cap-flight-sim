/**
 * 流れ場の幾何量と簡易係数モデル（cap_flight_sim/aerodynamics/base.py, simple.py の移植）。
 * 符号規約・式は Python 側 docstring と README §3 を参照。
 */
import type { CapParams, EnvironmentParams } from './config';
import { ConfigError } from './config';
import {
  add,
  cross,
  dot,
  matTVec,
  matVec,
  norm,
  qToMatrix,
  safeUnit,
  scale,
  sub,
  ZERO3,
  type Q4,
  type V3,
} from './vector';

const SPEED_EPS = 1e-6;
const DEGENERACY_EPS = 1e-6;

export interface AeroResult {
  force: V3;
  momentBody: V3;
  drag: V3;
  lift: V3;
  magnus: V3;
  coefficients: { cd: number; cl: number; cy: number; cm_pitch: number };
  diagnostics: {
    speed_m_s: number;
    dynamic_pressure_pa: number;
    angle_of_attack_deg: number;
    sideslip_deg: number;
    normal_to_wind_deg: number;
    projected_area_m2: number;
    spin_parameter: number;
    lift_degenerate_face_on: boolean;
    stationary: boolean;
  };
}

export interface BodyState {
  velocity: V3;
  q: Q4; // 正規化済み
  omegaBody: V3;
}

interface Flow {
  speed: number;
  qDyn: number;
  u: V3;
  uHat: V3;
  alpha: number;
  beta: number;
  normalToWindDeg: number;
  liftDir: V3;
  inplaneBody: V3;
  pitchAxisBody: V3;
  omegaPerp: V3;
  spinParameter: number;
  faceOn: boolean;
}

function flowGeometry(s: BodyState, cap: CapParams, env: EnvironmentParams): Flow | null {
  const R = qToMatrix(s.q);
  const normal: V3 = [R[0][2], R[1][2], R[2][2]];
  const omegaWorld = matVec(R, s.omegaBody);
  const u = sub(s.velocity, env.windWorld); // = -(wind - v)
  const speed = norm(u);
  if (speed < SPEED_EPS) return null;
  const qDyn = 0.5 * env.airDensity * speed ** 2;
  const uHat = scale(u, 1 / speed);
  const uBody = matTVec(R, u);
  const inplane = Math.hypot(uBody[0], uBody[1]);
  const alpha = Math.atan2(-uBody[2], inplane);
  const beta = inplane > DEGENERACY_EPS * speed ? Math.atan2(uBody[1], uBody[0]) : 0;
  const cosN = Math.min(1, Math.max(-1, dot(normal, scale(uHat, -1))));
  const liftRaw = sub(normal, scale(uHat, dot(normal, uHat)));
  const faceOn = norm(liftRaw) < DEGENERACY_EPS;
  const liftDir = faceOn ? ZERO3 : scale(liftRaw, 1 / norm(liftRaw));
  const inplaneBody = safeUnit([uBody[0], uBody[1], 0], DEGENERACY_EPS * speed);
  const pitchAxisBody = cross(inplaneBody, [0, 0, 1]);
  const omegaPerp = sub(omegaWorld, scale(uHat, dot(omegaWorld, uHat)));
  return {
    speed, qDyn, u, uHat, alpha, beta,
    normalToWindDeg: (Math.acos(cosN) * 180) / Math.PI,
    liftDir, inplaneBody, pitchAxisBody, omegaPerp,
    spinParameter: (cap.radiusM * norm(omegaPerp)) / speed,
    faceOn,
  };
}

// ------------------------------------------------------------ 係数パラメータ

export interface SimpleModelParams {
  drag: { model: string; cd_base: number; cd_projected: number; cd_alpha2_per_rad2: number };
  lift: { cl0: number; cl_alpha_per_rad: number; stall_angle_deg: number; post_stall: string };
  magnus: { model: string; coefficient: number; max_coefficient: number };
  moments: {
    center_of_pressure_body_m: V3 | null;
    cm_pitch0: number;
    cm_pitch_alpha_per_rad: number;
    spin_damping: number;
    tumble_damping: number;
  };
  enable: { drag: boolean; lift: boolean; magnus: boolean; moments: boolean };
}

const DEFAULTS: SimpleModelParams = {
  drag: { model: 'projected_area', cd_base: 1.0, cd_projected: 1.0, cd_alpha2_per_rad2: 0.0 },
  lift: { cl0: 0.0, cl_alpha_per_rad: 0.0, stall_angle_deg: 45.0, post_stall: 'decay' },
  magnus: { model: 'linear_spin_parameter', coefficient: 0.0, max_coefficient: 1.0 },
  moments: { center_of_pressure_body_m: null, cm_pitch0: 0, cm_pitch_alpha_per_rad: 0, spin_damping: 0, tumble_damping: 0 },
  enable: { drag: true, lift: true, magnus: true, moments: true },
};

function merged<T extends object>(defaults: T, given: unknown, section: string): T {
  const g = (typeof given === 'object' && given !== null ? given : {}) as Record<string, unknown>;
  for (const k of Object.keys(g)) {
    if (!(k in defaults)) throw new ConfigError(`unknown key aerodynamics.${section}.${k}`);
  }
  return { ...defaults, ...g } as T;
}

export function buildSimpleModelParams(aero: Record<string, unknown>): SimpleModelParams {
  const p: SimpleModelParams = {
    drag: merged(DEFAULTS.drag, aero.drag, 'drag'),
    lift: merged(DEFAULTS.lift, aero.lift, 'lift'),
    magnus: merged(DEFAULTS.magnus, aero.magnus, 'magnus'),
    moments: merged(DEFAULTS.moments, aero.moments, 'moments'),
    enable: merged(DEFAULTS.enable, aero.enable, 'enable'),
  };
  if (!['projected_area', 'constant', 'quadratic_aoa'].includes(p.drag.model)) throw new ConfigError('invalid drag.model');
  if (!['decay', 'hold'].includes(p.lift.post_stall)) throw new ConfigError('invalid lift.post_stall');
  if (!['linear_spin_parameter', 'constant'].includes(p.magnus.model)) throw new ConfigError('invalid magnus.model');
  if (!(p.lift.stall_angle_deg > 0 && p.lift.stall_angle_deg < 90)) throw new ConfigError('lift.stall_angle_deg must be in (0, 90)');
  return p;
}

export function projectedArea(alpha: number, cap: CapParams): number {
  return Math.PI * cap.radiusM ** 2 * Math.abs(Math.sin(alpha)) + cap.diameterM * cap.heightM * Math.abs(Math.cos(alpha));
}

export function dragCoefficient(p: SimpleModelParams, alpha: number, cap: CapParams): [number, number] {
  const area = projectedArea(alpha, cap);
  if (p.drag.model === 'constant') return [p.drag.cd_base, area];
  if (p.drag.model === 'quadratic_aoa') return [p.drag.cd_base + p.drag.cd_alpha2_per_rad2 * alpha ** 2, area];
  return [(p.drag.cd_projected * area) / cap.referenceAreaM2, area];
}

export function liftCoefficient(p: SimpleModelParams, alpha: number): number {
  const stall = (p.lift.stall_angle_deg * Math.PI) / 180;
  const a = Math.abs(alpha);
  let odd: number;
  if (a <= stall) odd = p.lift.cl_alpha_per_rad * a;
  else if (p.lift.post_stall === 'hold') odd = p.lift.cl_alpha_per_rad * stall;
  else odd = (p.lift.cl_alpha_per_rad * stall * (0.5 * Math.PI - a)) / (0.5 * Math.PI - stall);
  return p.lift.cl0 + Math.sign(alpha) * odd;
}

export function magnusCoefficient(p: SimpleModelParams, spinParameter: number): number {
  if (spinParameter <= 0) return 0;
  const value = p.magnus.model === 'constant' ? p.magnus.coefficient : p.magnus.coefficient * spinParameter;
  return Math.min(p.magnus.max_coefficient, Math.max(-p.magnus.max_coefficient, value));
}

const STATIONARY: AeroResult = {
  force: ZERO3, momentBody: ZERO3, drag: ZERO3, lift: ZERO3, magnus: ZERO3,
  coefficients: { cd: 0, cl: 0, cy: 0, cm_pitch: 0 },
  diagnostics: {
    speed_m_s: 0, dynamic_pressure_pa: 0, angle_of_attack_deg: 0, sideslip_deg: 0, normal_to_wind_deg: Number.NaN,
    projected_area_m2: 0, spin_parameter: 0, lift_degenerate_face_on: false, stationary: true,
  },
};

export function evaluateAero(p: SimpleModelParams, s: BodyState, cap: CapParams, env: EnvironmentParams): AeroResult {
  const flow = flowGeometry(s, cap, env);
  if (!flow) return STATIONARY;
  const qa = flow.qDyn * cap.referenceAreaM2;
  const [cd, area] = dragCoefficient(p, flow.alpha, cap);
  const drag = p.enable.drag ? scale(flow.uHat, -qa * cd) : ZERO3;
  const cl = liftCoefficient(p, flow.alpha);
  const lift = p.enable.lift ? scale(flow.liftDir, qa * cl) : ZERO3;
  const cy = magnusCoefficient(p, flow.spinParameter);
  const magnusDir = safeUnit(cross(flow.omegaPerp, flow.u));
  const magnus = p.enable.magnus ? scale(magnusDir, qa * cy) : ZERO3;
  const force = add(add(drag, lift), magnus);

  let moment: V3 = ZERO3;
  let cmPitch = 0;
  if (p.enable.moments) {
    const m = p.moments;
    if (m.center_of_pressure_body_m) {
      const arm = sub(m.center_of_pressure_body_m, cap.centerOfMassBodyM);
      moment = add(moment, cross(arm, matTVec(qToMatrix(s.q), force)));
    }
    cmPitch = m.cm_pitch0 + m.cm_pitch_alpha_per_rad * flow.alpha;
    moment = add(moment, scale(flow.pitchAxisBody, flow.qDyn * cap.referenceAreaM2 * cap.diameterM * cmPitch));
    const damp = 0.5 * env.airDensity * flow.speed * cap.referenceAreaM2 * cap.diameterM * cap.radiusM;
    const w = s.omegaBody;
    moment = sub(moment, scale([m.tumble_damping * w[0], m.tumble_damping * w[1], m.spin_damping * w[2]], damp));
  }

  return {
    force, momentBody: moment, drag, lift, magnus,
    coefficients: { cd, cl, cy, cm_pitch: cmPitch },
    diagnostics: {
      speed_m_s: flow.speed,
      dynamic_pressure_pa: flow.qDyn,
      angle_of_attack_deg: (flow.alpha * 180) / Math.PI,
      sideslip_deg: (flow.beta * 180) / Math.PI,
      normal_to_wind_deg: flow.normalToWindDeg,
      projected_area_m2: area,
      spin_parameter: flow.spinParameter,
      lift_degenerate_face_on: flow.faceOn,
      stationary: false,
    },
  };
}

