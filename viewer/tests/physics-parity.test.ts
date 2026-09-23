/**
 * Python（正本）との照合。ゴールデンは scripts/generate_ts_goldens.py が生成する。
 * Python の物理を変えたらゴールデンを再生成し、このテストが通るよう TS 側を追従させること。
 */
import { describe, expect, it } from 'vitest';
import derivatives from './golden/derivatives.json';
import trajectories from './golden/trajectories.json';
import { buildSimpleModelParams, evaluateAero } from '../src/physics/aerodynamics';
import { configFromRaw } from '../src/physics/config';
import { makeDerivative } from '../src/physics/dynamics';
import { runSimulation } from '../src/physics/simulate';
import { qNormalize, type Q4, type V3 } from '../src/physics/vector';

const REL = 1e-10;

function expectClose(actual: readonly number[], expected: readonly number[], label: string, rel = REL, abs = 1e-14) {
  expect(actual.length, label).toBe(expected.length);
  const scale = Math.max(1, ...expected.map(Math.abs));
  actual.forEach((a, i) => {
    const tol = Math.max(abs, rel * scale);
    if (!(Math.abs(a - expected[i]) <= tol)) {
      throw new Error(`${label}[${i}]: TS ${a} vs Python ${expected[i]} (|Δ|=${Math.abs(a - expected[i])} > ${tol})`);
    }
  });
}

describe('config parity', () => {
  for (const c of [...derivatives.cases, ...trajectories.cases]) {
    const name = 'variant' in c ? c.variant : c.name;
    it(`derives the same SI config: ${name}`, () => {
      const cfg = configFromRaw(c.raw as Record<string, unknown>);
      expectClose(cfg.y0, c.config.initialStateVector, 'y0');
      expectClose(cfg.cap.centerOfMassBodyM, c.config.centerOfMassBodyM, 'com');
      expectClose(cfg.cap.inertiaBody.flat(), c.config.inertiaBodyKgM2, 'inertia');
      expect(cfg.settings.maxStepSec).toBeCloseTo(c.config.maxStepSec, 15);
      expect(cfg.settings.fixedStepSec).toBeCloseTo(c.config.fixedStepSec, 15);
    });
  }
});

describe('derivative & aero parity', () => {
  for (const c of derivatives.cases) {
    const cfg = configFromRaw(c.raw as Record<string, unknown>);
    const params = buildSimpleModelParams(cfg.aerodynamics);
    const fun = makeDerivative(cfg, params);
    for (const st of c.states) {
      it(`${c.variant} / ${st.name}`, () => {
        expectClose(fun(0, st.y), st.dydt, 'dydt');
        const q = qNormalize(st.y.slice(6, 10) as unknown as Q4);
        const aero = evaluateAero(params, { velocity: st.y.slice(3, 6) as unknown as V3, q, omegaBody: st.y.slice(10, 13) as unknown as V3 }, cfg.cap, cfg.environment);
        expectClose(aero.force, st.force, 'force');
        expectClose(aero.momentBody, st.moment, 'moment', REL, 1e-18);
        expectClose(aero.drag, st.drag, 'drag');
        expectClose(aero.lift, st.lift, 'lift');
        expectClose(aero.magnus, st.magnus, 'magnus');
        for (const [k, v] of Object.entries(st.coefficients)) expect(aero.coefficients[k as keyof typeof aero.coefficients], k).toBeCloseTo(v, 10);
        const d = aero.diagnostics as Record<string, number | boolean>;
        for (const [k, v] of Object.entries(st.diagnostics)) {
          if (typeof v === 'number') expect(d[k] as number, k).toBeCloseTo(v, 9);
          else if (v === null) expect(Number.isNaN(d[k] as number), k).toBe(true);
          else expect(d[k], k).toBe(v);
        }
      });
    }
  }
});

describe('RK4 trajectory parity', () => {
  for (const c of trajectories.cases) {
    it(`${c.name}`, () => {
      const result = runSimulation(configFromRaw(c.raw as Record<string, unknown>, c.name));
      expect(result.terminationReason).toBe(c.terminationReason);
      expect(result.events.map((e) => e.type)).toEqual(c.events.map((e) => e.type));
      result.events.forEach((e, i) => expect(Math.abs(e.timeSec - c.events[i].timeSec), `event ${e.type}`).toBeLessThan(1e-9));
      expect(result.frames.length).toBe(c.frames.length);
      result.frames.forEach((f, i) => {
        const g = c.frames[i];
        expect(Math.abs(f.t - g.t)).toBeLessThan(1e-12);
        expectClose(f.y.slice(0, 3), g.position, `pos@${i}`, 1e-8, 1e-8);
        expectClose(f.y.slice(3, 6), g.velocity, `vel@${i}`, 1e-8, 1e-8);
        expectClose(f.y.slice(6, 10), g.quaternion, `quat@${i}`, 1e-7, 1e-7);
        expectClose(f.y.slice(10, 13), g.omegaBody, `omega@${i}`, 1e-7, 1e-7);
        expectClose(f.aero.force, g.force, `force@${i}`, 1e-7, 1e-10);
      });
    });
  }
});
