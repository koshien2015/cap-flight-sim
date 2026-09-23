import { describe, expect, it } from 'vitest';
import type { FlightDocument } from '../src/domain/flight-schema';
import { parseFlightDocument } from '../src/domain/flight-schema';
import { advance, parameterDomain, timeAtX, timeForParameter } from '../src/playback/timeline';

function doc(name: string, speed: number, catcherT: number | null): FlightDocument {
  const frames = Array.from({ length: 11 }, (_, i) => ({
    timeSec: i * 0.1,
    positionM: [speed * i * 0.1, 0, 1] as [number, number, number],
    quaternionWxyz: [1, 0, 0, 0] as [number, number, number, number],
    velocityMS: [speed, 0, 0] as [number, number, number],
    angularVelocityWorldRadS: [0, 0, 0] as [number, number, number],
    forcesN: { drag: [0, 0, 0], lift: [0, 0, 0], magnus: [0, 0, 0], total: [0, 0, 0] } as FlightDocument['frames'][0]['forcesN'],
  }));
  return parseFlightDocument({
    schemaVersion: '1.0.0', name, label: 'virtual',
    coordinateSystem: { handedness: 'right', forwardAxis: '+X', upAxis: '+Z', unit: 'meter' },
    field: { pitchDistanceM: 9.22, groundZM: 0 },
    cap: { diameterM: 0.03, heightM: 0.015, modelUri: null, bodyNormalAxis: '+Z' },
    frames,
    events: [{ type: 'release', timeSec: 0 }, ...(catcherT ? [{ type: 'catcher_plane', timeSec: catcherT }] : [])],
    warnings: [],
  });
}

describe('timeline sync modes', () => {
  const fast = doc('fast', 10, 0.9);
  const slow = doc('slow', 5, null);

  it('absolute/release domains span the longest run', () => {
    expect(parameterDomain([fast, slow], 'release')).toEqual([0, 1.0]);
  });

  it('x_progress maps the same X to different times', () => {
    expect(timeAtX(fast, 3)).toBeCloseTo(0.3, 12);
    expect(timeForParameter(slow, 'x_progress', 3)).toBeCloseTo(0.6, 12);
  });

  it('catcher_norm uses catcher arrival (or end) as 100%', () => {
    expect(timeForParameter(fast, 'catcher_norm', 1)).toBeCloseTo(0.9, 12);
    expect(timeForParameter(slow, 'catcher_norm', 0.5)).toBeCloseTo(0.5, 12);
  });

  it('advance stops at domain end and does not mutate', () => {
    const s = { parameter: 0.95, playing: true, speed: 1 };
    const next = advance(s, 0.2, [0, 1], 1);
    expect(next).toEqual({ parameter: 1, playing: false, speed: 1 });
    expect(s.parameter).toBe(0.95);
  });

  it('rejects left-handed or unknown schema', () => {
    expect(() => parseFlightDocument({ ...fast, coordinateSystem: { ...fast.coordinateSystem, handedness: 'left' } })).toThrow();
    expect(() => parseFlightDocument({ ...fast, schemaVersion: '2.0.0' })).toThrow();
  });
});
