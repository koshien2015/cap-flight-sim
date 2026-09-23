/**
 * 物理座標 → three.js 表示座標の変換（このファイルだけで行う）。
 *
 * 物理: 右手系 +X 捕手方向, +Y 投手から見て左, +Z 上
 * three: 右手系 Y-up
 *
 *   three.x =  phys.x
 *   three.y =  phys.z
 *   three.z = -phys.y
 *
 * 行列 M = [[1,0,0],[0,0,1],[0,-1,0]] は X 軸まわり -90° の回転（det = +1、鏡映ではない）。
 * 姿勢は R_three = M R_phys Mᵀ、クォータニオンでは q_three = q_M ⊗ q_phys ⊗ q_M⁻¹。
 * キャップのメッシュは three の CylinderGeometry（軸 = ローカル Y）で、M·(body +Z) = ローカル Y と一致する。
 *
 * クォータニオン並び: Python/JSON は [w, x, y, z]、three.js の Quaternion は (x, y, z, w)。
 */
import { Matrix3, Quaternion, Vector3 } from 'three';
import type { QuatWxyz, Vec3 } from './flight-schema';

const HALF_SQRT2 = Math.SQRT1_2;
/** X 軸まわり -90° */
const Q_M = new Quaternion(-HALF_SQRT2, 0, 0, HALF_SQRT2);
const Q_M_INV = Q_M.clone().invert();

export const PHYSICS_TO_THREE_MATRIX = new Matrix3().set(1, 0, 0, 0, 0, 1, 0, -1, 0);

export function physicsToThreeVector(v: Vec3, target = new Vector3()): Vector3 {
  return target.set(v[0], v[2], -v[1]);
}

export function threeToPhysicsVector(v: Vector3): Vec3 {
  return [v.x, -v.z, v.y];
}

/** [w,x,y,z] を three.js Quaternion(x,y,z,w) に並べ替えるだけ（座標変換はしない）。 */
export function wxyzToThreeOrder(q: QuatWxyz, target = new Quaternion()): Quaternion {
  return target.set(q[1], q[2], q[3], q[0]);
}

export function physicsToThreeQuaternion(q: QuatWxyz, target = new Quaternion()): Quaternion {
  const qp = wxyzToThreeOrder(q);
  return target.copy(Q_M).multiply(qp).multiply(Q_M_INV).normalize();
}
