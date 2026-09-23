/** 状態微分（cap_flight_sim/rigid_body.py の移植）。y = [r(3), v(3), q(4), ω_body(3)] */
import { evaluateAero, type AeroResult, type SimpleModelParams } from './aerodynamics';
import type { SimConfig } from './config';
import { cross, invert3, matVec, qDerivative, qNormalize, type M3, type Q4, type V3 } from './vector';

export class NonFiniteStateError extends Error {}

export type Derivative = (t: number, y: readonly number[]) => number[];

export const stateVelocity = (y: readonly number[]): V3 => [y[3], y[4], y[5]];
export const stateQuat = (y: readonly number[]): Q4 => [y[6], y[7], y[8], y[9]];
export const stateOmega = (y: readonly number[]): V3 => [y[10], y[11], y[12]];

export function aeroAt(cfg: SimConfig, params: SimpleModelParams, y: readonly number[]): AeroResult {
  return evaluateAero(params, { velocity: stateVelocity(y), q: qNormalize(stateQuat(y)), omegaBody: stateOmega(y) }, cfg.cap, cfg.environment);
}

export function makeDerivative(cfg: SimConfig, params: SimpleModelParams): Derivative {
  const inertia: M3 = cfg.cap.inertiaBody;
  const inertiaInv = invert3(inertia);
  const mass = cfg.cap.massKg;
  const g = cfg.environment.gravity;
  const drift = cfg.settings.driftGain;
  return (t, y) => {
    if (!y.every(Number.isFinite)) throw new NonFiniteStateError(`non-finite state at t=${t.toFixed(6)}`);
    const qRaw = stateQuat(y);
    const omega = stateOmega(y);
    const aero = evaluateAero(params, { velocity: stateVelocity(y), q: qNormalize(qRaw), omegaBody: omega }, cfg.cap, cfg.environment);
    const iw = matVec(inertia, omega);
    const c = cross(omega, iw);
    const alpha = matVec(inertiaInv, [aero.momentBody[0] - c[0], aero.momentBody[1] - c[1], aero.momentBody[2] - c[2]]);
    const dq = qDerivative(qRaw, omega, drift);
    const dy = [
      y[3], y[4], y[5],
      aero.force[0] / mass, aero.force[1] / mass, -g + aero.force[2] / mass,
      dq[0], dq[1], dq[2], dq[3],
      alpha[0], alpha[1], alpha[2],
    ];
    if (!dy.every(Number.isFinite)) throw new NonFiniteStateError(`non-finite derivative at t=${t.toFixed(6)}`);
    return dy;
  };
}
