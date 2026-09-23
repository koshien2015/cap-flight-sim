/**
 * UI が書き出す条件 JSON を Python CLI でも同じ結果にするための照合。
 * tests/fixtures/exported-config.json は下の生成関数の出力で、Python 側 tests/test_ts_goldens.py がこれを読み、
 * 同じ終了位置になることを確認する。UI の既定値や buildRawConfig を変えたら UPDATE_FIXTURES=1 pnpm test で更新する。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import presets from '../src/presets/presets.generated.json';
import { configFromRaw } from '../src/physics/config';
import { runSimulation } from '../src/physics/simulate';
import { buildRawConfig, DEFAULT_LAUNCH, faceNormalFor, launchParamsFromPreset } from '../src/ui/launch-params';

const FIXTURE = new URL('./fixtures/exported-config.json', import.meta.url);
const PRESETS = presets as Record<string, Record<string, unknown>>;

function buildFixture() {
  const base = launchParamsFromPreset('overhand_drop', PRESETS.overhand_drop, DEFAULT_LAUNCH);
  const params = { ...base, bankRightDeg: 20, noseUpDeg: 5, yawLeftDeg: -10, rpm: 1800, spinSign: -1 as const, cmPitch0: 0.01 };
  const raw = buildRawConfig(PRESETS.overhand_drop, params, 'ts_exported_fixture');
  const result = runSimulation(configFromRaw(raw));
  const last = result.frames[result.frames.length - 1];
  return { config: raw, expected: { terminationReason: result.terminationReason, finalPosition: last.y.slice(0, 3), finalTime: last.t } };
}

describe('export round trip', () => {
  it('fixture matches the current UI export', () => {
    const fresh = buildFixture();
    if (process.env.UPDATE_FIXTURES) writeFileSync(FIXTURE, JSON.stringify(fresh, null, 1) + '\n');
    const committed = JSON.parse(readFileSync(FIXTURE, 'utf-8'));
    expect(fresh.config).toEqual(committed.config);
    fresh.expected.finalPosition.forEach((v, i) => expect(Math.abs(v - committed.expected.finalPosition[i])).toBeLessThan(1e-12));
  });

  it('tilt sliders match experiment B definitions', () => {
    const r = (d: number) => (d * Math.PI) / 180;
    const close = (a: readonly number[], b: number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 12));
    close(faceNormalFor({ baseAttitude: 'edge_on', bankRightDeg: 10, noseUpDeg: 0, yawLeftDeg: 0 }), [0, Math.cos(r(10)), Math.sin(r(10))]);
    close(faceNormalFor({ baseAttitude: 'flat', bankRightDeg: 0, noseUpDeg: 10, yawLeftDeg: 0 }), [-Math.sin(r(10)), 0, Math.cos(r(10))]);
    close(faceNormalFor({ baseAttitude: 'flat', bankRightDeg: 10, noseUpDeg: 0, yawLeftDeg: 0 }), [0, -Math.sin(r(10)), Math.cos(r(10))]);
  });

  it('every generated preset runs in the browser engine', () => {
    for (const [name, raw] of Object.entries(PRESETS)) {
      const res = runSimulation(configFromRaw(raw, name));
      expect(res.frames.length, name).toBeGreaterThan(10);
    }
  });
});
