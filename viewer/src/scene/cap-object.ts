/** キャップ（薄い円柱）と、その面法線・回転軸の表示。 */
import { ArrowHelper, CylinderGeometry, DoubleSide, Group, Mesh, MeshStandardMaterial, Vector3 } from 'three';
import { physicsToThreeQuaternion, physicsToThreeVector } from '../domain/coordinate-transform';
import type { InterpolatedFrame } from '../playback/interpolation';

export const AXIS_COLORS = { normal: 0x00e5ff, spinAxis: 0x33dd55 } as const;
const AXIS_LENGTH_M = 0.12;

export class CapObject {
  readonly group = new Group();
  private readonly body = new Group();
  private readonly normalArrow: ArrowHelper;
  private readonly spinArrow: ArrowHelper;

  constructor(diameterM: number, heightM: number, color: number, bodyZIsTop = true) {
    const radius = diameterM / 2;
    const side = new Mesh(
      new CylinderGeometry(radius, radius, heightM, 40, 1, true),
      new MeshStandardMaterial({ color, roughness: 0.5, side: DoubleSide }),
    );
    // 天面（閉じた側）を白で区別する。body +Z = three ローカル +Y
    const top = new Mesh(new CylinderGeometry(radius, radius, heightM * 0.08, 40), new MeshStandardMaterial({ color: 0xf5f5f5 }));
    top.position.y = (bodyZIsTop ? 1 : -1) * (heightM / 2);
    // 回転が見えるよう面内に目印の帯を付ける
    const marker = new Mesh(
      new CylinderGeometry(radius * 0.12, radius * 0.12, heightM * 1.02, 8),
      new MeshStandardMaterial({ color: 0x222222 }),
    );
    marker.position.x = radius * 0.75;
    this.body.add(side, top, marker);
    this.group.add(this.body);

    const origin = new Vector3();
    this.normalArrow = new ArrowHelper(new Vector3(0, 1, 0), origin, AXIS_LENGTH_M, AXIS_COLORS.normal);
    this.spinArrow = new ArrowHelper(new Vector3(0, 1, 0), origin, AXIS_LENGTH_M, AXIS_COLORS.spinAxis);
    this.group.add(this.normalArrow, this.spinArrow);
  }

  setDisplayScale(scale: number): void {
    this.body.scale.setScalar(scale);
  }

  setAxesVisible(visible: boolean): void {
    this.normalArrow.visible = visible;
    this.spinArrow.visible = visible;
  }

  update(frame: InterpolatedFrame): void {
    physicsToThreeVector(frame.positionM, this.group.position);
    physicsToThreeQuaternion(frame.quaternionWxyz, this.body.quaternion);
    if (frame.faceNormalWorld) this.normalArrow.setDirection(physicsToThreeVector(frame.faceNormalWorld).normalize());
    const w = physicsToThreeVector(frame.angularVelocityWorldRadS);
    this.spinArrow.visible = this.normalArrow.visible && w.lengthSq() > 1e-12;
    if (w.lengthSq() > 1e-12) this.spinArrow.setDirection(w.normalize());
  }
}
