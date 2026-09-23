/**
 * raw 設定（Python の YAML と同じ構造の JSON）→ SI の計算設定（cap_flight_sim/config.py の移植）。
 * ブラウザ版は aerodynamics.model = simple のみ対応。
 */
import {
  eulerDegToQuat,
  kmhToMS,
  matTVec,
  qNormalize,
  qToMatrix,
  quatFromFaceNormal,
  rpmToRadS,
  safeUnit,
  isZero,
  scale,
  deg2rad,
  type M3,
  type Q4,
  type V3,
} from './vector';

export type RawConfig = Record<string, unknown>;
export const PARAMETER_SOURCES = ['measured', 'fitted', 'literature', 'assumed'] as const;

export class ConfigError extends Error {}

export interface CapParams {
  massKg: number;
  diameterM: number;
  heightM: number;
  radiusM: number;
  referenceAreaM2: number;
  centerOfMassBodyM: V3;
  inertiaBody: M3;
  surfaceOrientation: 'body_z_is_top' | 'body_z_is_cavity';
}

export interface EnvironmentParams {
  airDensity: number;
  windWorld: V3;
  gravity: number;
}

export interface Settings {
  durationSec: number;
  outputHz: number;
  maxStepSec: number;
  solver: 'rk4' | 'solve_ivp';
  fixedStepSec: number;
  driftGain: number;
  stopAtGround: boolean;
  stopAtCatcherPlane: boolean;
  maxSpeed: number;
  maxAngularSpeed: number;
}

export interface SimConfig {
  name: string;
  label: string;
  cap: CapParams;
  environment: EnvironmentParams;
  pitchDistanceM: number;
  groundZM: number;
  settings: Settings;
  /** 13 要素状態 [r, v, q(wxyz), ω_body] */
  y0: number[];
  aerodynamics: Record<string, unknown>;
  raw: RawConfig;
  warnings: string[];
}

// ------------------------------------------------------------ パラメータ展開

const isParamSpec = (v: unknown): v is { value: unknown; source?: string } =>
  typeof v === 'object' && v !== null && !Array.isArray(v) && 'value' in v &&
  Object.keys(v).every((k) => k === 'value' || k === 'source' || k === 'note');

/** {value, source, note} を素の値へ展開した新しいオブジェクト。 */
export function resolveParameters(node: unknown): unknown {
  if (isParamSpec(node)) {
    if (node.source !== undefined && !PARAMETER_SOURCES.includes(node.source as never)) {
      throw new ConfigError(`invalid source '${node.source}'`);
    }
    return resolveParameters(node.value);
  }
  if (Array.isArray(node)) return node.map(resolveParameters);
  if (typeof node === 'object' && node !== null) {
    return Object.fromEntries(Object.entries(node).map(([k, v]) => [k, resolveParameters(v)]));
  }
  return node;
}

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (typeof v === 'object' && v !== null && !Array.isArray(v) ? (v as Obj) : {});
const num = (v: unknown, fallback?: number): number => {
  if (v === undefined || v === null) {
    if (fallback === undefined) throw new ConfigError('missing numeric value');
    return fallback;
  }
  const n = Number(v);
  if (!Number.isFinite(n)) throw new ConfigError(`not a number: ${String(v)}`);
  return n;
};
function vec(v: unknown, name: string, size = 3): number[] {
  if (!Array.isArray(v) || v.length !== size) throw new ConfigError(`${name} must have ${size} elements`);
  return v.map((x) => num(x));
}
const v3 = (v: unknown, name: string): V3 => vec(v, name) as unknown as V3;

// ------------------------------------------------------------ 質量特性

export function approximateMassProperties(mass: number, diameter: number, height: number, method: string, bodyZIsTop: boolean): [V3, M3] {
  const r = 0.5 * diameter;
  const h = height;
  const diag = (ixx: number, izz: number): M3 => [[ixx, 0, 0], [0, ixx, 0], [0, 0, izz]];
  if (method === 'solid_cylinder') return [[0, 0, 0], diag((mass * (3 * r ** 2 + h ** 2)) / 12, 0.5 * mass * r ** 2)];
  if (method === 'thin_disk') return [[0, 0, 0], diag(0.25 * mass * r ** 2, 0.5 * mass * r ** 2)];
  if (method === 'thin_walled_cup') {
    const areaTop = Math.PI * r ** 2;
    const areaWall = 2 * Math.PI * r * h;
    const mTop = (mass * areaTop) / (areaTop + areaWall);
    const mWall = mass - mTop;
    const zTop = bodyZIsTop ? 0.5 * h : -0.5 * h;
    const zc = (mTop * zTop) / mass;
    const izz = 0.5 * mTop * r ** 2 + mWall * r ** 2;
    const ixx = 0.25 * mTop * r ** 2 + mTop * (zTop - zc) ** 2 + (mWall * (0.5 * r ** 2 + h ** 2 / 12) + mWall * zc ** 2);
    return [[0, 0, zc], diag(ixx, izz)];
  }
  throw new ConfigError(`unknown inertia method: ${method}`);
}

function buildCap(s: Obj): CapParams {
  const mass = num(s.mass_kg);
  const diameter = num(s.outer_diameter_m);
  const height = num(s.height_m);
  if (mass <= 0 || diameter <= 0 || height <= 0) throw new ConfigError('cap mass/diameter/height must be positive');
  const surface = (s.surface_orientation ?? 'body_z_is_top') as CapParams['surfaceOrientation'];
  if (surface !== 'body_z_is_top' && surface !== 'body_z_is_cavity') throw new ConfigError('invalid surface_orientation');
  const area = s.reference_area_m2 == null ? Math.PI * (0.5 * diameter) ** 2 : num(s.reference_area_m2);
  const [comApprox, inertiaApprox] = approximateMassProperties(
    mass, diameter, height, String(s.inertia_method ?? 'thin_walled_cup'), surface === 'body_z_is_top',
  );
  let inertia = inertiaApprox;
  if (s.inertia_tensor_body_kg_m2 != null) {
    const raw = s.inertia_tensor_body_kg_m2 as unknown[];
    if (raw.length === 3 && !Array.isArray(raw[0])) {
      const [a, b, c] = vec(raw, 'inertia');
      inertia = [[a, 0, 0], [0, b, 0], [0, 0, c]];
    } else {
      inertia = raw.map((row, i) => v3(row, `inertia[${i}]`)) as unknown as M3;
    }
  }
  return {
    massKg: mass,
    diameterM: diameter,
    heightM: height,
    radiusM: 0.5 * diameter,
    referenceAreaM2: area,
    centerOfMassBodyM: s.center_of_mass_body_m == null ? comApprox : v3(s.center_of_mass_body_m, 'center_of_mass_body_m'),
    inertiaBody: inertia,
    surfaceOrientation: surface,
  };
}

function buildVelocity(s: Obj): V3 {
  if (s.velocity_world_m_s !== undefined) return v3(s.velocity_world_m_s, 'velocity_world_m_s');
  let speed: number;
  if (s.speed_kmh !== undefined) speed = kmhToMS(num(s.speed_kmh));
  else if (s.speed_m_s !== undefined) speed = num(s.speed_m_s);
  else throw new ConfigError('initial_state needs speed_kmh, speed_m_s or velocity_world_m_s');
  const az = deg2rad(num(s.launch_azimuth_deg, 0));
  const el = deg2rad(num(s.launch_elevation_deg, 0));
  return [speed * Math.cos(el) * Math.cos(az), speed * Math.cos(el) * Math.sin(az), speed * Math.sin(el)];
}

function buildOrientation(s: Obj): Q4 {
  const keys = ['orientation_quaternion_wxyz', 'orientation_euler_deg', 'face_normal_world'].filter((k) => k in s);
  if (keys.length !== 1) throw new ConfigError('initial_state needs exactly one orientation form');
  const key = keys[0];
  if (key === 'orientation_quaternion_wxyz') return qNormalize(vec(s[key], key, 4) as unknown as Q4);
  if (key === 'orientation_euler_deg') return eulerDegToQuat(v3(s[key], key));
  return quatFromFaceNormal(v3(s[key], key), s.body_x_hint_world == null ? undefined : v3(s.body_x_hint_world, 'body_x_hint_world'));
}

function buildOmegaBody(spec: unknown, q: Q4): V3 {
  if (spec == null) return [0, 0, 0];
  const s = obj(spec);
  const forms = ['omega_world_rad_s', 'omega_body_rad_s', 'rotation_sense', 'axis_world', 'axis_body'].filter((k) => k in s);
  if (forms.length > 1) throw new ConfigError(`angular_velocity has conflicting forms ${forms.join(', ')}`);
  const R = qToMatrix(q);
  if ('omega_world_rad_s' in s) return matTVec(R, v3(s.omega_world_rad_s, 'omega_world_rad_s'));
  if ('omega_body_rad_s' in s) return v3(s.omega_body_rad_s, 'omega_body_rad_s');
  let rate: number;
  if ('rpm' in s) rate = rpmToRadS(num(s.rpm));
  else if ('rad_s' in s) rate = num(s.rad_s);
  else throw new ConfigError('angular_velocity needs omega_world_rad_s, omega_body_rad_s, rpm or rad_s');

  if ('rotation_sense' in s) {
    if (s.rotation_sense !== 'clockwise' && s.rotation_sense !== 'counterclockwise') throw new ConfigError('invalid rotation_sense');
    if (!('viewed_from_world' in s)) throw new ConfigError('rotation_sense requires viewed_from_world');
    const toward = safeUnit(v3(s.viewed_from_world, 'viewed_from_world'));
    if (isZero(toward)) throw new ConfigError('viewed_from_world must be non-zero');
    const axis = s.rotation_sense === 'counterclockwise' ? toward : scale(toward, -1);
    return matTVec(R, scale(axis, rate));
  }
  if ('axis_world' in s) {
    const axis = safeUnit(v3(s.axis_world, 'axis_world'));
    if (isZero(axis)) throw new ConfigError('axis_world must be non-zero');
    return matTVec(R, scale(axis, rate));
  }
  if ('axis_body' in s) {
    const axis = safeUnit(v3(s.axis_body, 'axis_body'));
    if (isZero(axis)) throw new ConfigError('axis_body must be non-zero');
    return scale(axis, rate);
  }
  throw new ConfigError('angular_velocity with rpm/rad_s needs axis_world, axis_body or rotation_sense');
}

function buildSettings(s: Obj, omegaBody: V3, warnings: string[]): Settings {
  const fps = num(s.video_fps, 240);
  const spin = Math.sqrt(omegaBody[0] ** 2 + omegaBody[1] ** 2 + omegaBody[2] ** 2);
  const period = spin > 0 ? (2 * Math.PI) / spin : Infinity;
  const auto = Math.min(period / 20, 1 / (4 * fps), 0.002);
  const maxStep = s.max_step_sec == null ? auto : num(s.max_step_sec);
  if (s.max_step_sec != null && maxStep > period / 10) warnings.push(`max_step_sec=${maxStep} は回転周期の1/10より大きい`);
  const solver = (s.solver ?? 'solve_ivp') as Settings['solver'];
  if (solver !== 'rk4' && solver !== 'solve_ivp') throw new ConfigError('simulation.solver must be solve_ivp or rk4');
  const term = obj(s.termination);
  return {
    durationSec: num(s.duration_sec, 2),
    outputHz: num(s.output_hz, 240),
    maxStepSec: maxStep,
    solver,
    fixedStepSec: s.fixed_step_sec == null ? Math.min(maxStep, 1e-4) : num(s.fixed_step_sec),
    driftGain: num(s.quaternion_drift_gain, 10),
    stopAtGround: term.stop_at_ground === undefined ? true : Boolean(term.stop_at_ground),
    stopAtCatcherPlane: term.stop_at_catcher_plane === undefined ? true : Boolean(term.stop_at_catcher_plane),
    maxSpeed: num(term.max_speed_m_s, 100),
    maxAngularSpeed: num(term.max_angular_speed_rad_s, 5000),
  };
}

export function configFromRaw(rawInput: RawConfig, name = 'run'): SimConfig {
  const raw = resolveParameters(rawInput) as Obj;
  if (!('cap' in raw)) throw new ConfigError('missing section: cap');
  if (!('initial_state' in raw)) throw new ConfigError('missing section: initial_state');
  const warnings: string[] = [];
  const cap = buildCap(obj(raw.cap));
  const env = obj(raw.environment);
  const environment: EnvironmentParams = {
    airDensity: num(env.air_density_kg_m3, 1.204),
    windWorld: env.wind_velocity_world_m_s == null ? [0, 0, 0] : v3(env.wind_velocity_world_m_s, 'wind'),
    gravity: num(env.gravity_m_s2, 9.80665),
  };
  const field = obj(raw.field);
  const init = obj(raw.initial_state);
  const q0 = buildOrientation(init);
  const omegaBody = buildOmegaBody(init.angular_velocity, q0);
  const position = init.position_world_m == null ? ([0, 0, 1.45] as V3) : v3(init.position_world_m, 'position_world_m');
  const aerodynamics = obj(raw.aerodynamics);
  if ((aerodynamics.model ?? 'simple') !== 'simple') {
    throw new ConfigError(`ブラウザ版は aerodynamics.model=simple のみ対応（${String(aerodynamics.model)} は Python CLI で計算してください）`);
  }
  return {
    name: String(raw.name ?? name),
    label: String(raw.label ?? 'virtual'),
    cap,
    environment,
    pitchDistanceM: num(field.pitch_distance_m, 9.22),
    groundZM: num(field.ground_z_m, 0),
    settings: buildSettings(obj(raw.simulation), omegaBody, warnings),
    y0: [...position, ...buildVelocity(init), ...q0, ...omegaBody],
    aerodynamics,
    raw: rawInput,
    warnings,
  };
}

/** 検証用: 設定から求めた初期面法線（world）。 */
export const initialNormal = (cfg: SimConfig): V3 => {
  const R = qToMatrix(cfg.y0.slice(6, 10) as unknown as Q4);
  return [R[0][2], R[1][2], R[2][2]];
};
