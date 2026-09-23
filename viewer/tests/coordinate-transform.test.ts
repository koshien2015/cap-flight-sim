import { describe, expect, it } from 'vitest';
import { Quaternion, Vector3 } from 'three';
import {
  PHYSICS_TO_THREE_MATRIX,
  physicsToThreeQuaternion,
  physicsToThreeVector,
  threeToPhysicsVector,
  wxyzToThreeOrder,
} from '../src/domain/coordinate-transform';
import type { QuatWxyz, Vec3 } from '../src/domain/flight-schema';

/** 物理座標系でのクォータニオン回転（ハミルトン積）を独立に実装した参照。 */
function rotatePhysics(q: QuatWxyz, v: Vec3): Vec3 {
  const [w, x, y, z] = q;
  const m = [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
  return m.map((row) => row[0] * v[0] + row[1] * v[1] + row[2] * v[2]) as Vec3;
}

const normalize = (q: number[]): QuatWxyz => {
  const n = Math.hypot(...q);
  return q.map((c) => c / n) as QuatWxyz;
};

describe('coordinate-transform', () => {
  it('is a proper rotation (det = +1), not a reflection', () => {
    expect(PHYSICS_TO_THREE_MATRIX.determinant()).toBeCloseTo(1, 12);
  });

  it('maps physics axes to three axes', () => {
    expect(physicsToThreeVector([1, 0, 0]).toArray()).toEqual([1, 0, -0]);
    expect(physicsToThreeVector([0, 0, 1]).toArray()).toEqual([0, 1, -0]);
    expect(physicsToThreeVector([0, 1, 0]).toArray()).toEqual([0, 0, -1]);
  });

  it('round-trips vectors', () => {
    const v: Vec3 = [1.2, -3.4, 5.6];
    threeToPhysicsVector(physicsToThreeVector(v)).forEach((c, i) => expect(c).toBeCloseTo(v[i], 12));
  });

  it('reorders [w,x,y,z] to three (x,y,z,w)', () => {
    const q = wxyzToThreeOrder([0.1, 0.2, 0.3, 0.4]);
    expect([q.x, q.y, q.z, q.w]).toEqual([0.2, 0.3, 0.4, 0.1]);
  });

  it('rotate-then-transform equals transform-then-rotate', () => {
    const samples: QuatWxyz[] = [normalize([0.9, 0.1, -0.3, 0.2]), normalize([0.2, -0.7, 0.5, 0.4]), [1, 0, 0, 0]];
    const vectors: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.3, -0.8, 0.5]];
    for (const q of samples) {
      const qThree = physicsToThreeQuaternion(q);
      for (const v of vectors) {
        const expected = physicsToThreeVector(rotatePhysics(q, v));
        const actual = physicsToThreeVector(v).applyQuaternion(qThree);
        expect(actual.distanceTo(expected)).toBeLessThan(1e-12);
      }
    }
  });

  it('body +Z (face normal) maps to cylinder local +Y', () => {
    const q = normalize([0.8, 0.3, -0.4, 0.2]);
    const normalPhysics = rotatePhysics(q, [0, 0, 1]);
    const meshAxis = new Vector3(0, 1, 0).applyQuaternion(physicsToThreeQuaternion(q));
    expect(meshAxis.distanceTo(physicsToThreeVector(normalPhysics))).toBeLessThan(1e-12);
  });

  it('merely reordering components is not the conversion', () => {
    const q: QuatWxyz = normalize([0.9, 0.3, 0, 0]);
    const naive = wxyzToThreeOrder(q);
    expect(naive.equals(physicsToThreeQuaternion(q, new Quaternion()))).toBe(false);
  });
});
