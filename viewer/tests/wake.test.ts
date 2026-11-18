/** 後流の非定常項（Issue #7）の入力検証と乱数列。式の照合は physics-parity.test.ts（Python のゴールデン）で行う。 */
import { describe, expect, it } from 'vitest';
import { buildSimpleModelParams } from '../src/physics/aerodynamics';
import { mulberry32, wakeBasis } from '../src/physics/wake';

describe('wake model', () => {
  it('draws the same mulberry32 sequence as the Python implementation', () => {
    // tests/test_wake.py の test_mulberry32_matches_the_javascript_reference と同じ期待値
    const cases: [number, number[]][] = [
      [1, [0.6270739405881613, 0.002735721180215478, 0.5274470399599522]],
      [42, [0.6011037519201636, 0.44829055899754167, 0.8524657934904099]],
      [4294967295, [0.8964226141106337, 0.189478256739676, 0.7156526781618595]],
    ];
    for (const [seed, expected] of cases) {
      const rng = mulberry32(seed);
      expect([rng(), rng(), rng()]).toEqual(expected);
    }
  });

  it('is off by default and rejects the same invalid inputs as Python', () => {
    expect(buildSimpleModelParams({}).evaluateWake).toBeNull();
    expect(buildSimpleModelParams({ wake: { force_coefficient: 0, moment_coefficient: 0, seed: 5 } }).evaluateWake).toBeNull();
    expect(buildSimpleModelParams({ wake: { force_coefficient: 0.15 } }).evaluateWake).not.toBeNull();
    const bad: unknown[] = [
      { reduced_frequencies: [0.01, 0.02], weights: [1.0] },
      { reduced_frequencies: [], weights: [] },
      { reduced_frequencies: [-0.01, 0.02, 0.03] },
      { reduced_frequencies: 0.01 },
      { weights: ['1', '0.8', '0.5'] },
      { spin_parameter_scale: 0 },
      { spin_parameter_scale: Number.NaN },
      { seed: 1.5 },
      { seed: -1 },
      { force_coefficient: -0.1 },
      { force_coefficient: Number.NaN },
      { force_coefficient: Number.NaN, moment_coefficient: 0.001 },
      { moment_coefficient: Number.POSITIVE_INFINITY },
      { force_coefficient: true },
      { force_coefficient: '0.1' },
      { force_coeficient: 0.1 },
      true,
      [0.15],
    ];
    for (const wake of bad) expect(() => buildSimpleModelParams({ wake }), JSON.stringify(wake)).toThrow();
  });

  it('falls back to a fixed basis when flying straight up or down', () => {
    for (const uHat of [[0, 0, 1], [0, 0, -1]] as [number, number, number][]) {
      const [e1, e2] = wakeBasis(uHat);
      expect(e1).toEqual([0, 1, 0]);
      expect(Math.hypot(...e2)).toBeCloseTo(1, 12);
      expect(e2[0] * uHat[0] + e2[1] * uHat[1] + e2[2] * uHat[2]).toBeCloseTo(0, 12);
    }
  });
});
