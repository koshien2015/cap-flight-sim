import { describe, expect, it } from 'vitest';
import type { FlightFrame, QuatWxyz } from '../src/domain/flight-schema';
import { findFrameIndex, interpolateFrame, slerpWxyz } from '../src/playback/interpolation';

const zero = { drag: [0, 0, 0], lift: [0, 0, 0], magnus: [0, 0, 0], total: [0, 0, 0] } as FlightFrame['forcesN'];
const frame = (t: number, x: number, q: QuatWxyz): FlightFrame => ({
  timeSec: t, positionM: [x, 0, 1], quaternionWxyz: q, velocityMS: [10, 0, 0], angularVelocityWorldRadS: [0, 0, 1], forcesN: zero,
});

describe('interpolation', () => {
  const half = Math.SQRT1_2;
  const frames = [frame(0, 0, [1, 0, 0, 0]), frame(0.1, 1, [half, 0, 0, half]), frame(0.2, 2, [0, 0, 0, 1])];

  it('finds frame index with clamping', () => {
    expect(findFrameIndex(frames, -1)).toBe(0);
    expect(findFrameIndex(frames, 0.15)).toBe(1);
    expect(findFrameIndex(frames, 5)).toBe(2);
  });

  it('linearly interpolates position', () => {
    expect(interpolateFrame(frames, 0.05).positionM[0]).toBeCloseTo(0.5, 12);
  });

  it('uses SLERP for orientation (constant angular rate, unit norm)', () => {
    const q = interpolateFrame(frames, 0.05).quaternionWxyz; // 0°→90° の中間 = 45° about Z
    expect(Math.hypot(...q)).toBeCloseTo(1, 12);
    expect(q[0]).toBeCloseTo(Math.cos(Math.PI / 8), 12);
    expect(q[3]).toBeCloseTo(Math.sin(Math.PI / 8), 12);
  });

  it('takes the shortest path when quaternions have opposite sign', () => {
    const q = slerpWxyz([1, 0, 0, 0], [-half, 0, 0, -half], 0.5); // -q は同じ姿勢
    expect(Math.abs(q[0])).toBeCloseTo(Math.cos(Math.PI / 8), 12);
  });
});
