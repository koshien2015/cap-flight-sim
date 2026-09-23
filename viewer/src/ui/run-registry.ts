/** 読み込んだ投球（ラン）の保持と、表示オブジェクトの生成。 */
import type { Line } from 'three';
import type { FlightDocument } from '../domain/flight-schema';
import { CapObject } from '../scene/cap-object';
import { ForceArrows } from '../scene/force-arrows';
import { createObservedLine, createTrajectoryLine } from '../scene/trajectory-line';

export const RUN_COLORS = [0xff9f1c, 0x2ec4b6, 0xe71d36, 0x9b5de5, 0xf15bb5, 0x00bbf9, 0xfee440, 0x8ac926];

export interface Run {
  id: number;
  doc: FlightDocument;
  color: number;
  visible: boolean;
  cap: CapObject;
  arrows: ForceArrows;
  line: Line;
  observedLine: Line | null;
}

let nextId = 0;

export const LIVE_COLOR = 0xffffff;

export function createRun(doc: FlightDocument, index: number, colorOverride?: number): Run {
  const color = colorOverride ?? RUN_COLORS[index % RUN_COLORS.length];
  const bodyZIsTop = doc.cap.surfaceOrientation !== 'body_z_is_cavity';
  return {
    id: nextId++,
    doc,
    color,
    visible: true,
    cap: new CapObject(doc.cap.diameterM, doc.cap.heightM, color, bodyZIsTop),
    arrows: new ForceArrows(),
    line: createTrajectoryLine(doc, color),
    observedLine: createObservedLine(doc, color),
  };
}

export function hexColor(color: number): string {
  return `#${color.toString(16).padStart(6, '0')}`;
}
