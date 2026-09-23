/** 軌道線（計算済みの全フレームを結ぶ）。表示専用で軌道補正はしない。 */
import { BufferGeometry, Line, LineBasicMaterial, LineDashedMaterial, Vector3 } from 'three';
import { physicsToThreeVector } from '../domain/coordinate-transform';
import type { FlightDocument } from '../domain/flight-schema';

export function createTrajectoryLine(doc: FlightDocument, color: number): Line {
  const points = doc.frames.map((f) => physicsToThreeVector(f.positionM, new Vector3()));
  return new Line(new BufferGeometry().setFromPoints(points), new LineBasicMaterial({ color }));
}

/** 実測軌道の重ね描き用（将来拡張）。観測点を破線で結ぶ。 */
export function createObservedLine(doc: FlightDocument, color: number): Line | null {
  if (!doc.observed?.length) return null;
  const points = doc.observed.map((p) => physicsToThreeVector(p.positionM, new Vector3()));
  const line = new Line(new BufferGeometry().setFromPoints(points), new LineDashedMaterial({ color, dashSize: 0.05, gapSize: 0.03 }));
  line.computeLineDistances();
  return line;
}
