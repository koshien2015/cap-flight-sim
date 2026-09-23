/** 投球空間（地面・投手位置・捕手面・座標軸）とカメラ。 */
import {
  AmbientLight,
  CanvasTexture,
  Color,
  DirectionalLight,
  DoubleSide,
  GridHelper,
  Group,
  Mesh,
  MeshBasicMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  Sprite,
  SpriteMaterial,
  Vector3,
  WebGLRenderer,
  ArrowHelper,
} from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { physicsToThreeVector } from '../domain/coordinate-transform';
import type { Vec3 } from '../domain/flight-schema';

export type CameraView = 'top' | 'side' | 'catcher' | 'free';

interface ViewPose {
  position: Vec3; // 物理座標
  target: Vec3;
}

export class FlightScene {
  readonly scene = new Scene();
  readonly camera = new PerspectiveCamera(45, 1, 0.01, 200);
  readonly renderer: WebGLRenderer;
  readonly controls: OrbitControls;
  readonly runsGroup = new Group();
  private pitchDistanceM = 9.22;

  constructor(private readonly container: HTMLElement) {
    this.renderer = new WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);
    this.scene.background = new Color(0x15181d);
    this.scene.add(new AmbientLight(0xffffff, 0.7));
    const sun = new DirectionalLight(0xffffff, 1.4);
    sun.position.set(3, 8, 4);
    this.scene.add(sun, this.runsGroup);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.buildField(this.pitchDistanceM, 0);
    this.setView('free');
    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  private field = new Group();

  buildField(pitchDistanceM: number, groundZM: number): void {
    this.pitchDistanceM = pitchDistanceM;
    this.scene.remove(this.field);
    this.field = new Group();
    const ground = new Mesh(
      new PlaneGeometry(pitchDistanceM + 4, 5),
      new MeshStandardMaterial({ color: 0x3b3326, side: DoubleSide }),
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.copy(physicsToThreeVector([pitchDistanceM / 2, 0, groundZM]));
    const grid = new GridHelper(Math.ceil(pitchDistanceM + 4), Math.ceil(pitchDistanceM + 4), 0x6b5f4a, 0x4a4234);
    grid.position.copy(physicsToThreeVector([pitchDistanceM / 2, 0, groundZM + 0.001]));

    // 長辺(0.6m)を左右(物理 Y)方向に向ける（回転後 PlaneGeometry の幅=物理 X、高さ=物理 Y）
    const pitcher = new Mesh(new PlaneGeometry(0.3, 0.6), new MeshBasicMaterial({ color: 0xffffff, side: DoubleSide }));
    pitcher.rotation.x = -Math.PI / 2;
    pitcher.position.copy(physicsToThreeVector([0, 0, groundZM + 0.002]));

    const catcherPlane = new Mesh(
      new PlaneGeometry(2.0, 2.0),
      new MeshBasicMaterial({ color: 0x55aaff, transparent: true, opacity: 0.12, side: DoubleSide }),
    );
    catcherPlane.rotation.y = Math.PI / 2; // 法線を three +X（= 物理 +X）へ
    catcherPlane.position.copy(physicsToThreeVector([pitchDistanceM, 0, groundZM + 1.0]));

    this.field.add(ground, grid, pitcher, catcherPlane, this.makeAxes(groundZM));
    this.field.add(label('投手', physicsToThreeVector([0, 0, groundZM + 0.25]), 0.2));
    this.field.add(label(`捕手面 X=${pitchDistanceM}m`, physicsToThreeVector([pitchDistanceM, 0, groundZM + 2.15]), 0.2));
    this.scene.add(this.field);
  }

  private makeAxes(groundZM: number): Group {
    const g = new Group();
    const origin = physicsToThreeVector([-0.5, 1.5, groundZM + 0.01]);
    const axes: [Vec3, number, string][] = [
      [[1, 0, 0], 0xff5555, '+X 捕手方向'],
      [[0, 1, 0], 0x55ff55, '+Y 投手の左'],
      [[0, 0, 1], 0x5599ff, '+Z 上'],
    ];
    axes.forEach(([dir, color, text]) => {
      const d = physicsToThreeVector(dir);
      g.add(new ArrowHelper(d, origin, 0.8, color, 0.1, 0.05));
      g.add(label(text, origin.clone().addScaledVector(d, 1.05), 0.14));
    });
    return g;
  }

  setView(view: CameraView): void {
    const mid = this.pitchDistanceM / 2;
    const poses: Record<CameraView, ViewPose> = {
      top: { position: [mid, -0.01, 13], target: [mid, 0, 0] },
      side: { position: [mid, -11, 1.0], target: [mid, 0, 1.0] },
      catcher: { position: [this.pitchDistanceM + 5, 0, 1.2], target: [0, 0, 0.8] },
      free: { position: [-4, -4.5, 3.5], target: [mid, 0, 0.6] },
    };
    const pose = poses[view];
    this.camera.position.copy(physicsToThreeVector(pose.position));
    this.controls.target.copy(physicsToThreeVector(pose.target));
    this.camera.up.set(0, 1, 0);
    this.controls.update();
  }

  resize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  render(): void {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

function label(text: string, position: Vector3, heightM: number): Sprite {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  const fontPx = 48;
  ctx.font = `${fontPx}px sans-serif`;
  canvas.width = Math.ceil(ctx.measureText(text).width) + 16;
  canvas.height = fontPx + 16;
  ctx.font = `${fontPx}px sans-serif`;
  ctx.fillStyle = '#e8e8e8';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 8, canvas.height / 2);
  const sprite = new Sprite(new SpriteMaterial({ map: new CanvasTexture(canvas), depthTest: false }));
  sprite.scale.set((heightM * canvas.width) / canvas.height, heightM, 1);
  sprite.position.copy(position);
  return sprite;
}
