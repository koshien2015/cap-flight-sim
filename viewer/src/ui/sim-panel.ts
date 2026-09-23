/** 「条件を調整して計算」パネル。値の変更ごとに Worker で再計算し、ライブ表示を差し替える。 */
import presetsJson from '../presets/presets.generated.json';
import type { SimResponse } from '../physics/sim-client';
import { SimClient } from '../physics/sim-client';
import {
  buildRawConfig,
  cliCommand,
  DEFAULT_LAUNCH,
  faceNormalFor,
  launchParamsFromPreset,
  type LaunchParams,
} from './launch-params';
import { el } from './panel';

const PRESETS = presetsJson as Record<string, Record<string, unknown>>;
const DEBOUNCE_MS = 40;
export const LIVE_NAME = 'ライブ計算';

type NumKey = { [K in keyof LaunchParams]: LaunchParams[K] extends number ? K : never }[keyof LaunchParams];

interface SliderSpec {
  key: NumKey;
  label: string;
  min: number;
  max: number;
  step: number;
  unit: string;
}

const LAUNCH_SLIDERS: SliderSpec[] = [
  { key: 'speedKmh', label: '初速', min: 10, max: 130, step: 1, unit: 'km/h' },
  { key: 'elevationDeg', label: '仰角', min: -10, max: 25, step: 0.5, unit: '°' },
  { key: 'azimuthLeftDeg', label: '方位（+左）', min: -15, max: 15, step: 0.5, unit: '°' },
  { key: 'releaseHeightM', label: 'リリース高さ', min: 0.2, max: 2.5, step: 0.05, unit: 'm' },
];
const ATTITUDE_SLIDERS: SliderSpec[] = [
  { key: 'bankRightDeg', label: '右傾き', min: -60, max: 60, step: 1, unit: '°' },
  { key: 'noseUpDeg', label: '前縁上げ', min: -45, max: 45, step: 1, unit: '°' },
  { key: 'yawLeftDeg', label: '向き（+左）', min: -60, max: 60, step: 1, unit: '°' },
];
const SPIN_SLIDERS: SliderSpec[] = [{ key: 'rpm', label: '回転数', min: 0, max: 5000, step: 50, unit: 'rpm' }];
const COEFF_SLIDERS: SliderSpec[] = [
  { key: 'cdProjected', label: 'Cd（投影面積基準）', min: 0, max: 2, step: 0.05, unit: '' },
  { key: 'clAlphaPerRad', label: '揚力傾斜 Clα', min: 0, max: 4, step: 0.1, unit: '/rad' },
  { key: 'cl0', label: '非対称揚力 cl0', min: -0.6, max: 0.6, step: 0.02, unit: '' },
  { key: 'magnusCoefficient', label: 'マグヌス係数', min: 0, max: 4, step: 0.1, unit: '' },
  { key: 'cmPitch0', label: 'ピッチモーメント cm0', min: -0.1, max: 0.1, step: 0.005, unit: '' },
  { key: 'spinDamping', label: '自転減衰', min: 0, max: 0.2, step: 0.005, unit: '' },
  { key: 'tumbleDamping', label: 'タンブル減衰', min: 0, max: 0.2, step: 0.005, unit: '' },
];

export interface SimPanelHandlers {
  onLiveResult: (response: SimResponse) => void;
  onPin: (response: SimResponse) => void;
}

const fmt = (v: number, d = 3) => (Number.isFinite(v) ? v.toFixed(d) : '—');
const vecText = (v: readonly number[]) => `[${v.map((x) => fmt(x, 3)).join(', ')}]`;

export function mountSimPanel(container: HTMLElement, handlers: SimPanelHandlers): void {
  const client = new SimClient();
  const initialPreset = PRESETS[DEFAULT_LAUNCH.presetName] ?? Object.values(PRESETS)[0];
  let params: LaunchParams = launchParamsFromPreset(DEFAULT_LAUNCH.presetName, initialPreset, DEFAULT_LAUNCH);
  let last: SimResponse | null = null;
  let pinCount = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const sliderInputs = new Map<NumKey, [HTMLInputElement, HTMLSpanElement]>();

  const status = el('div', { className: 'mono sim-status' });
  const vectors = el('div', { className: 'mono sim-vectors' });

  const currentRaw = (name: string) => buildRawConfig(PRESETS[params.presetName], params, name);

  const schedule = () => {
    clearTimeout(timer);
    timer = setTimeout(recompute, DEBOUNCE_MS);
  };

  async function recompute() {
    const n = faceNormalFor(params);
    vectors.textContent = `面法線 n = ${vecText(n)}\n回転軸 = ${params.spinAxis === 'cap' ? `${params.spinSign > 0 ? '+' : '−'}n（キャップ自身の軸）` : `${params.spinSign > 0 ? '+' : '−'}${params.spinAxis.replace('world_', '')}（world）`}`;
    try {
      const res = await client.run(currentRaw(LIVE_NAME), LIVE_NAME);
      if (!res) return;
      last = res;
      status.textContent = summaryText(res);
      status.classList.remove('error');
      handlers.onLiveResult(res);
    } catch (err) {
      status.textContent = `エラー: ${(err as Error).message}`;
      status.classList.add('error');
    }
  }

  const set = (patch: Partial<LaunchParams>) => {
    params = { ...params, ...patch };
    schedule();
  };

  const slider = (spec: SliderSpec) => {
    const input = el('input', { type: 'range', min: String(spec.min), max: String(spec.max), step: String(spec.step), value: String(params[spec.key]) });
    const value = el('span', { className: 'mono', textContent: `${params[spec.key]} ${spec.unit}` });
    input.addEventListener('input', () => {
      value.textContent = `${input.value} ${spec.unit}`;
      set({ [spec.key]: Number(input.value) } as Partial<LaunchParams>);
    });
    sliderInputs.set(spec.key, [input, value]);
    return el('label', { className: 'slider' }, [el('span', { textContent: spec.label }), value, input]);
  };

  const select = <T extends string>(label: string, options: [T, string][], get: () => T, onChange: (v: T) => void) => {
    const s = el('select');
    options.forEach(([v, text]) => s.append(new Option(text, v, v === get(), v === get())));
    s.addEventListener('change', () => onChange(s.value as T));
    return [el('label', {}, [`${label} `, s]), s] as const;
  };

  const check = (label: string, get: () => boolean, onChange: (v: boolean) => void) => {
    const input = el('input', { type: 'checkbox', checked: get() });
    input.addEventListener('change', () => onChange(input.checked));
    return [el('label', { className: 'inline' }, [input, label]), input] as const;
  };

  const [presetRow, presetSelect] = select(
    '土台プリセット',
    Object.keys(PRESETS).map((k) => [k, k] as [string, string]),
    () => params.presetName,
    (name) => {
      params = launchParamsFromPreset(name, PRESETS[name], params);
      syncControls();
      schedule();
    },
  );
  const [attitudeRow, attitudeSelect] = select(
    '基準姿勢',
    [['flat', '面を水平（弾き投げ型）'], ['edge_on', 'キャップを立てる（エッジオン）']],
    () => params.baseAttitude,
    (v) => set({ baseAttitude: v }),
  );
  const [axisRow, axisSelect] = select(
    '回転軸',
    [['cap', 'キャップ自身の軸（面法線）'], ['world_z', '上下軸（world Z）'], ['world_y', '左右軸（world Y）'], ['world_x', '進行方向軸（world X）']],
    () => params.spinAxis,
    (v) => set({ spinAxis: v }),
  );
  const [reverseRow, reverseInput] = check('逆回転', () => params.spinSign < 0, (c) => set({ spinSign: c ? -1 : 1 }));
  const toggles = (['drag', 'lift', 'magnus', 'moments'] as const).map((k) =>
    check({ drag: '抗力', lift: '揚力', magnus: 'マグヌス', moments: 'モーメント' }[k], () => params.enable[k], (c) => set({ enable: { ...params.enable, [k]: c } })),
  );

  function syncControls() {
    presetSelect.value = params.presetName;
    attitudeSelect.value = params.baseAttitude;
    axisSelect.value = params.spinAxis;
    reverseInput.checked = params.spinSign < 0;
    toggles.forEach(([, input], i) => (input.checked = params.enable[(['drag', 'lift', 'magnus', 'moments'] as const)[i]]));
    const specs = [...LAUNCH_SLIDERS, ...ATTITUDE_SLIDERS, ...SPIN_SLIDERS, ...COEFF_SLIDERS];
    sliderInputs.forEach(([input, value], key) => {
      input.value = String(params[key]);
      value.textContent = `${params[key]} ${specs.find((s) => s.key === key)?.unit ?? ''}`;
    });
  }

  const pinButton = el('button', { textContent: '📌 比較用に固定' });
  pinButton.addEventListener('click', () => {
    if (!last) return;
    pinCount += 1;
    const name = `条件${pinCount}`;
    handlers.onPin({ ...last, doc: { ...last.doc, name } });
  });
  const saveButton = el('button', { textContent: '条件をJSONで保存' });
  saveButton.addEventListener('click', () => {
    const fileName = `browser_${params.presetName}_${Date.now()}.json`;
    const blob = new Blob([JSON.stringify(currentRaw(fileName.replace(/\.json$/, '')), null, 2)], { type: 'application/json' });
    const a = el('a', { href: URL.createObjectURL(blob), download: fileName });
    a.click();
    URL.revokeObjectURL(a.href);
    status.textContent = `${fileName} を保存。Python で再現: ${cliCommand(fileName)}`;
  });
  const copyButton = el('button', { textContent: 'CLIコマンドをコピー' });
  copyButton.addEventListener('click', async () => {
    const cmd = cliCommand('<保存したJSON>');
    await navigator.clipboard?.writeText(cmd).catch(() => undefined);
    status.textContent = `コピー: ${cmd}`;
  });

  container.replaceChildren(
    el('p', { className: 'note', textContent: 'ブラウザ内の TS 移植版で計算（Python と同条件で照合テスト済み）。結果は仮定係数に基づく計算値。' }),
    presetRow,
    el('h3', { textContent: '投射' }), ...LAUNCH_SLIDERS.map(slider),
    el('h3', { textContent: '姿勢（リリース時）' }), attitudeRow, ...ATTITUDE_SLIDERS.map(slider),
    el('h3', { textContent: 'スピン' }), ...SPIN_SLIDERS.map(slider), axisRow, reverseRow,
    vectors,
    el('details', {}, [
      el('summary', { textContent: '空気力・係数（すべて仮定値）' }),
      el('div', { className: 'row' }, toggles.map(([row]) => row)),
      ...COEFF_SLIDERS.map(slider),
    ]),
    el('div', { className: 'row' }, [pinButton, saveButton, copyButton]),
    status,
  );
  schedule();
}

function summaryText(res: SimResponse): string {
  const s = res.summary;
  const reach = s.catcher
    ? `捕手面到達: Y ${fmt(s.catcher.yM)} m（+左） Z ${fmt(s.catcher.zM)} m  ${fmt(s.catcher.speedKmh, 1)} km/h`
    : `未到達（${s.terminationReason}）: 最終 X ${fmt(s.final[0], 2)} Y ${fmt(s.final[1])} Z ${fmt(s.final[2])}`;
  return [
    reach,
    `飛行時間 ${fmt(s.flightTimeSec)} s`,
    `横変化（初速方向の直線から） ${fmt(s.lateralBreakM)} m`,
    `落下量（同） ${fmt(s.dropM)} m`,
    `面法線の変化 ${fmt(s.faceNormalChangeDeg, 1)}°`,
    `計算 ${fmt(res.elapsedMs, 1)} ms（${res.doc.computedBy}）`,
  ].join('\n');
}
