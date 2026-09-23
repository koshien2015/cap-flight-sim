/** パネル内の動的 DOM（ラン一覧・フィルタ・凡例・HUD）。 */
import { VECTOR_COLORS, type VectorKind } from '../scene/force-arrows';
import { AXIS_COLORS } from '../scene/cap-object';
import type { InterpolatedFrame } from '../playback/interpolation';
import { hexColor, type Run } from './run-registry';

export const LABELS = ['good', 'bad', 'virtual'] as const;
export const LABEL_TEXT: Record<string, string> = { good: '良い球', bad: '抜け球', virtual: '仮想条件' };

export const VECTOR_TEXT: Record<VectorKind, string> = {
  velocity: '速度',
  drag: '抗力',
  lift: '揚力',
  magnus: 'マグヌス力',
};

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, props: Partial<HTMLElementTagNameMap[K]> = {}, children: (Node | string)[] = []): HTMLElementTagNameMap[K] {
  const node = Object.assign(document.createElement(tag), props);
  children.forEach((c) => node.append(c));
  return node;
}

export function checkbox(text: string, checked: boolean, onChange: (checked: boolean) => void, color?: number): HTMLLabelElement {
  const input = el('input', { type: 'checkbox', checked });
  input.addEventListener('change', () => onChange(input.checked));
  const children: (Node | string)[] = [input];
  if (color !== undefined) {
    const sw = el('span', { className: 'swatch' });
    sw.style.background = hexColor(color);
    children.push(sw);
  }
  children.push(text);
  return el('label', { className: 'run' }, children);
}

export function renderRunList(container: HTMLElement, runs: Run[], primaryId: number | null, handlers: {
  onToggle: (run: Run, visible: boolean) => void;
  onPrimary: (run: Run) => void;
  onRemove: (run: Run) => void;
}): void {
  container.replaceChildren(
    ...runs.map((run) => {
      const radio = el('input', { type: 'radio', name: 'primary', checked: run.id === primaryId, title: 'HUD に表示する投球' });
      radio.addEventListener('change', () => handlers.onPrimary(run));
      const vis = el('input', { type: 'checkbox', checked: run.visible, title: '表示' });
      vis.addEventListener('change', () => handlers.onToggle(run, vis.checked));
      const remove = el('button', { textContent: '×', title: '削除' });
      remove.addEventListener('click', () => handlers.onRemove(run));
      const sw = el('span', { className: 'swatch' });
      sw.style.background = hexColor(run.color);
      const reason = el('span', { className: 'reason', textContent: `${LABEL_TEXT[run.doc.label] ?? run.doc.label} / ${run.doc.terminationReason ?? ''} / ${run.doc.computedBy ? 'ブラウザ計算' : 'Python'}` });
      return el('div', { className: 'run' }, [radio, vis, sw, el('span', { textContent: run.doc.name }), reason, remove]);
    }),
  );
}

export function renderLegend(container: HTMLElement): void {
  const items: [string, number][] = [
    ...(Object.keys(VECTOR_COLORS) as VectorKind[]).map((k) => [VECTOR_TEXT[k], VECTOR_COLORS[k]] as [string, number]),
    ['回転軸', AXIS_COLORS.spinAxis],
    ['面法線', AXIS_COLORS.normal],
  ];
  container.replaceChildren(
    ...items.map(([text, color]) => {
      const sw = el('span', { className: 'swatch' });
      sw.style.background = hexColor(color);
      return el('div', {}, [sw, text]);
    }),
    el('div', { textContent: '座標: 物理 +X 捕手方向 / +Y 投手の左 / +Z 上 [m]' }),
  );
}

const fmt = (v: number | null | undefined, digits = 1) => (v == null || !Number.isFinite(v) ? '—' : v.toFixed(digits));

export function formatHud(run: Run | null, frame: InterpolatedFrame | null): string {
  if (!run || !frame) return 'データ未読込';
  const [vx, vy, vz] = frame.velocityMS;
  const speed = Math.hypot(vx, vy, vz);
  const [x, y, z] = frame.positionM;
  return [
    `${run.doc.name}`,
    `t      ${fmt(frame.timeSec, 3)} s`,
    `位置   X ${fmt(x, 2)}  Y ${fmt(y, 3)}  Z ${fmt(z, 3)} m`,
    `速度   ${fmt(speed, 2)} m/s (${fmt(speed * 3.6, 1)} km/h)`,
    `回転   ${fmt(frame.rpm, 0)} rpm`,
    `迎角   ${fmt(frame.angleOfAttackDeg)}°`,
    `横滑り ${fmt(frame.sideslipDeg)}°`,
  ].join('\n');
}
