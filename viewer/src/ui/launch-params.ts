/**
 * UI の投球パラメータ → raw 設定（Python の YAML と同じ構造）。純関数のみ。
 *
 * 傾きの定義（実験 B と同じ）:
 *   右傾き bankRightDeg : world X 軸まわり +θ。面水平なら右の縁が下がり、立てたキャップなら上端が右へ倒れる
 *   前縁上げ noseUpDeg   : world Y 軸まわり −θ。面水平なら前縁（捕手側）が上がる
 *   向き yawLeftDeg      : world Z 軸まわり +θ。面が左（+Y）方向へ回る
 *   面法線 = Rz(yaw) · Ry(−noseUp) · Rx(bankRight) · 基準法線（flat: +Z, edge_on: +Y）
 */
import { resolveParameters } from '../physics/config';
import { deg2rad, qFromAxisAngle, qMultiply, qToMatrix, matVec, type V3 } from '../physics/vector';

export type BaseAttitude = 'flat' | 'edge_on';
export type SpinAxisMode = 'cap' | 'world_x' | 'world_y' | 'world_z';

export interface LaunchParams {
  presetName: string;
  speedKmh: number;
  elevationDeg: number;
  azimuthLeftDeg: number;
  releaseHeightM: number;
  baseAttitude: BaseAttitude;
  bankRightDeg: number;
  noseUpDeg: number;
  yawLeftDeg: number;
  rpm: number;
  spinAxis: SpinAxisMode;
  /** +1: 軸の正方向（右手の法則）/ -1: 逆回転 */
  spinSign: 1 | -1;
  enable: { drag: boolean; lift: boolean; magnus: boolean; moments: boolean };
  cdProjected: number;
  clAlphaPerRad: number;
  cl0: number;
  magnusCoefficient: number;
  cmPitch0: number;
  spinDamping: number;
  tumbleDamping: number;
}

export const DEFAULT_LAUNCH: LaunchParams = {
  presetName: 'overhand_drop',
  speedKmh: 58,
  elevationDeg: 2,
  azimuthLeftDeg: 0,
  releaseHeightM: 1.45,
  baseAttitude: 'edge_on',
  bankRightDeg: 0,
  noseUpDeg: 0,
  yawLeftDeg: 0,
  rpm: 1200,
  spinAxis: 'cap',
  spinSign: 1,
  enable: { drag: true, lift: true, magnus: true, moments: true },
  cdProjected: 1.0,
  clAlphaPerRad: 1.4,
  cl0: 0,
  magnusCoefficient: 1.0,
  cmPitch0: 0,
  spinDamping: 0,
  tumbleDamping: 0,
};

const BASE_NORMAL: Record<BaseAttitude, V3> = { flat: [0, 0, 1], edge_on: [0, 1, 0] };

export function faceNormalFor(p: Pick<LaunchParams, 'baseAttitude' | 'bankRightDeg' | 'noseUpDeg' | 'yawLeftDeg'>): V3 {
  const q = qMultiply(
    qMultiply(qFromAxisAngle([0, 0, 1], deg2rad(p.yawLeftDeg)), qFromAxisAngle([0, 1, 0], -deg2rad(p.noseUpDeg))),
    qFromAxisAngle([1, 0, 0], deg2rad(p.bankRightDeg)),
  );
  return matVec(qToMatrix(q), BASE_NORMAL[p.baseAttitude]);
}

const WORLD_AXES: Record<Exclude<SpinAxisMode, 'cap'>, V3> = { world_x: [1, 0, 0], world_y: [0, 1, 0], world_z: [0, 0, 1] };

function angularVelocitySpec(p: LaunchParams): Record<string, unknown> {
  if (p.spinAxis === 'cap') return { rpm: p.rpm, axis_body: [0, 0, p.spinSign] };
  const a = WORLD_AXES[p.spinAxis];
  return { rpm: p.rpm, axis_world: [a[0] * p.spinSign, a[1] * p.spinSign, a[2] * p.spinSign] };
}

const clone = <T>(v: T): T => JSON.parse(JSON.stringify(v)) as T;
const section = (raw: Record<string, unknown>, key: string): Record<string, unknown> =>
  (typeof raw[key] === 'object' && raw[key] !== null ? raw[key] : {}) as Record<string, unknown>;
const ORIENTATION_KEYS = ['orientation_quaternion_wxyz', 'orientation_euler_deg', 'face_normal_world', 'body_x_hint_world'];

/** プリセット raw を土台に、UI の値で上書きした新しい raw 設定。solver は rk4（Python CLI で同じ結果を再現できる）。 */
export function buildRawConfig(preset: Record<string, unknown>, p: LaunchParams, name: string): Record<string, unknown> {
  const raw = clone(preset);
  const init = Object.fromEntries(Object.entries(section(raw, 'initial_state')).filter(([k]) => !ORIENTATION_KEYS.includes(k)));
  delete init.speed_m_s;
  delete init.velocity_world_m_s;
  const aero = section(raw, 'aerodynamics');
  const assumed = (value: number, note?: string) => ({ value, source: 'assumed', ...(note ? { note } : {}) });
  return {
    ...raw,
    name,
    label: 'virtual',
    initial_state: {
      ...init,
      position_world_m: [0, 0, p.releaseHeightM],
      speed_kmh: p.speedKmh,
      launch_azimuth_deg: p.azimuthLeftDeg,
      launch_elevation_deg: p.elevationDeg,
      face_normal_world: [...faceNormalFor(p)],
      angular_velocity: angularVelocitySpec(p),
    },
    simulation: { ...section(raw, 'simulation'), solver: 'rk4' },
    aerodynamics: {
      ...aero,
      model: 'simple',
      enable: { ...p.enable },
      drag: { ...section(aero, 'drag'), cd_projected: assumed(p.cdProjected) },
      lift: { ...section(aero, 'lift'), cl_alpha_per_rad: assumed(p.clAlphaPerRad), cl0: assumed(p.cl0) },
      magnus: { ...section(aero, 'magnus'), coefficient: assumed(p.magnusCoefficient) },
      moments: {
        ...section(aero, 'moments'),
        cm_pitch0: assumed(p.cmPitch0),
        spin_damping: assumed(p.spinDamping),
        tumble_damping: assumed(p.tumbleDamping),
      },
    },
  };
}

/** プリセットに合わせた UI 初期値（プリセットの姿勢・係数から推定する）。 */
export function launchParamsFromPreset(name: string, preset: Record<string, unknown>, current: LaunchParams): LaunchParams {
  const resolved = resolveParameters(preset) as Record<string, unknown>;
  const init = section(resolved, 'initial_state');
  const aero = section(resolved, 'aerodynamics');
  const num = (v: unknown, fallback: number) => (typeof v === 'number' ? v : fallback);
  const normal = init.face_normal_world as number[] | undefined;
  const isEdgeOn = normal ? Math.abs(normal[1]) > Math.abs(normal[2]) : false;
  const euler = init.orientation_euler_deg as number[] | undefined;
  const pos = init.position_world_m as number[] | undefined;
  const spin = section(init, 'angular_velocity');
  const enable = section(aero, 'enable');
  const flag = (k: keyof LaunchParams['enable']) => (enable[k] === undefined ? true : Boolean(enable[k]));
  return {
    ...current,
    presetName: name,
    speedKmh: num(init.speed_kmh, current.speedKmh),
    elevationDeg: num(init.launch_elevation_deg, 0),
    azimuthLeftDeg: num(init.launch_azimuth_deg, 0),
    releaseHeightM: pos ? pos[2] : current.releaseHeightM,
    baseAttitude: isEdgeOn ? 'edge_on' : 'flat',
    bankRightDeg: 0,
    noseUpDeg: euler ? -euler[1] : 0,
    yawLeftDeg: 0,
    rpm: num(spin.rpm, current.rpm),
    spinAxis: 'cap',
    spinSign: spin.rotation_sense === 'clockwise' ? -1 : 1,
    enable: { drag: flag('drag'), lift: flag('lift'), magnus: flag('magnus'), moments: flag('moments') },
    cdProjected: num(section(aero, 'drag').cd_projected, current.cdProjected),
    clAlphaPerRad: num(section(aero, 'lift').cl_alpha_per_rad, current.clAlphaPerRad),
    cl0: num(section(aero, 'lift').cl0, 0),
    magnusCoefficient: num(section(aero, 'magnus').coefficient, current.magnusCoefficient),
    cmPitch0: num(section(aero, 'moments').cm_pitch0, 0),
    spinDamping: num(section(aero, 'moments').spin_damping, 0),
    tumbleDamping: num(section(aero, 'moments').tumble_damping, 0),
  };
}

export function cliCommand(fileName: string): string {
  return `uv run python -m cap_flight_sim simulate --config ${fileName} --output output/${fileName.replace(/\.json$/, '')}`;
}
