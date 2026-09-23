/**
 * 速度・空気力の矢印。長さは表示倍率（UI で変更）を掛けた見かけの長さで、物理値は変更しない。
 */
import { ArrowHelper, Group, Vector3 } from 'three';
import { physicsToThreeVector } from '../domain/coordinate-transform';
import type { InterpolatedFrame } from '../playback/interpolation';

export const VECTOR_COLORS = {
  velocity: 0xffd400,
  drag: 0xff3b30,
  lift: 0x2f7bff,
  magnus: 0xb04dff,
} as const;

export type VectorKind = keyof typeof VECTOR_COLORS;

export interface VectorScales {
  /** 表示長 [m] / 速度 [m/s] */
  velocityMPerMS: number;
  /** 表示長 [m] / 力 [N] */
  forceMPerN: number;
}

const MIN_VISIBLE = 1e-6;

export class ForceArrows {
  readonly group = new Group();
  private readonly arrows: Record<VectorKind, ArrowHelper>;

  constructor() {
    const make = (color: number) => new ArrowHelper(new Vector3(1, 0, 0), new Vector3(), 0.1, color, 0.03, 0.015);
    this.arrows = {
      velocity: make(VECTOR_COLORS.velocity),
      drag: make(VECTOR_COLORS.drag),
      lift: make(VECTOR_COLORS.lift),
      magnus: make(VECTOR_COLORS.magnus),
    };
    Object.values(this.arrows).forEach((a) => this.group.add(a));
  }

  update(frame: InterpolatedFrame, scales: VectorScales, visible: Record<VectorKind, boolean>): void {
    physicsToThreeVector(frame.positionM, this.group.position);
    const vectors: Record<VectorKind, [Vector3, number]> = {
      velocity: [physicsToThreeVector(frame.velocityMS), scales.velocityMPerMS],
      drag: [physicsToThreeVector(frame.forcesN.drag), scales.forceMPerN],
      lift: [physicsToThreeVector(frame.forcesN.lift), scales.forceMPerN],
      magnus: [physicsToThreeVector(frame.forcesN.magnus), scales.forceMPerN],
    };
    (Object.keys(vectors) as VectorKind[]).forEach((kind) => {
      const [v, scale] = vectors[kind];
      const length = v.length() * scale;
      const arrow = this.arrows[kind];
      arrow.visible = visible[kind] && length > MIN_VISIBLE;
      if (!arrow.visible) return;
      arrow.setDirection(v.normalize());
      arrow.setLength(length, Math.min(0.04, length * 0.3), Math.min(0.02, length * 0.15));
    });
  }
}
