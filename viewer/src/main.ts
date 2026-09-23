/** ビューア本体: 読込・再生ループ・UI 結線。 */
import { parseFlightDocument, type FlightDocument } from './domain/flight-schema';
import { interpolateFrame, type InterpolatedFrame } from './playback/interpolation';
import {
  SYNC_MODE_LABELS,
  advance,
  endTime,
  parameterDomain,
  parameterUnit,
  releaseTime,
  timeForParameter,
  type PlaybackState,
  type SyncMode,
} from './playback/timeline';
import { FlightScene, type CameraView } from './scene/flight-scene';
import type { VectorKind } from './scene/force-arrows';
import { checkbox, formatHud, LABEL_TEXT, LABELS, renderLegend, renderRunList, VECTOR_TEXT } from './ui/panel';
import { createRun, LIVE_COLOR, type Run } from './ui/run-registry';
import { mountSimPanel } from './ui/sim-panel';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const scene = new FlightScene($('viewport'));
let runs: Run[] = [];
let primaryId: number | null = null;
let liveRunId: number | null = null;
let colorCounter = 0;
let syncMode: SyncMode = 'release';
let playback: PlaybackState = { parameter: 0, playing: false, speed: 0.25 };
const labelVisible: Record<string, boolean> = { good: true, bad: true, virtual: true };
const vectorVisible: Record<VectorKind, boolean> = { velocity: true, drag: true, lift: true, magnus: true };
let axesVisible = true;
let forceScale = 10; // m/N
let velocityScale = 0.05; // m/(m/s)
let capScale = 3;

// ------------------------------------------------------------------ データ

function addDocument(doc: FlightDocument, live = false): Run {
  const run = createRun(doc, live ? 0 : colorCounter++, live ? LIVE_COLOR : undefined);
  run.cap.setDisplayScale(capScale);
  scene.runsGroup.add(run.line, run.cap.group, run.arrows.group);
  if (run.observedLine) scene.runsGroup.add(run.observedLine);
  runs = [...runs, run];
  if (primaryId === null) {
    primaryId = run.id;
    scene.buildField(doc.field.pitchDistanceM, doc.field.groundZM);
  }
  refreshRuns();
  return run;
}

/** ライブ計算の結果で前回のライブ表示を置き換える（HUD の選択は維持）。 */
function replaceLive(doc: FlightDocument): void {
  const wasPrimary = primaryId === null || primaryId === liveRunId;
  const previous = runs.find((r) => r.id === liveRunId);
  if (previous) removeRun(previous);
  const run = addDocument(doc, true);
  liveRunId = run.id;
  if (wasPrimary) primaryId = run.id;
  refreshRuns();
}

function removeRun(run: Run): void {
  scene.runsGroup.remove(run.line, run.cap.group, run.arrows.group);
  if (run.observedLine) scene.runsGroup.remove(run.observedLine);
  runs = runs.filter((r) => r.id !== run.id);
  if (primaryId === run.id) primaryId = runs[0]?.id ?? null;
  refreshRuns();
}

async function loadUrl(url: string): Promise<void> {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    addDocument(parseFlightDocument(await res.json()));
  } catch (err) {
    showError(`${url}: ${(err as Error).message}`);
  }
}

async function loadFiles(files: FileList): Promise<void> {
  for (const file of Array.from(files)) {
    try {
      addDocument(parseFlightDocument(JSON.parse(await file.text())));
    } catch (err) {
      showError(`${file.name}: ${(err as Error).message}`);
    }
  }
}

function showError(message: string): void {
  $('warnings').textContent = `読込エラー: ${message}`;
}

async function loadSampleManifest(): Promise<void> {
  try {
    const res = await fetch('samples/manifest.json');
    if (!res.ok) return;
    const manifest: { name: string; file: string }[] = await res.json();
    const box = $('samples');
    box.replaceChildren(
      ...manifest.map((m) => {
        const b = document.createElement('button');
        b.textContent = `+ ${m.name}`;
        b.addEventListener('click', () => loadUrl(`samples/${m.file}`));
        return b;
      }),
    );
  } catch {
    // サンプルが無い構成でも動く
  }
}

// ------------------------------------------------------------------ 再生

const activeRuns = () => runs.filter((r) => r.visible && labelVisible[r.doc.label] !== false);
const domain = () => parameterDomain(activeRuns().map((r) => r.doc), syncMode);
const referenceDuration = () => Math.max(0.01, ...activeRuns().map((r) => endTime(r.doc) - releaseTime(r.doc)));

function frameFor(run: Run): InterpolatedFrame {
  return interpolateFrame(run.doc.frames, timeForParameter(run.doc, syncMode, playback.parameter));
}

function stepFrame(direction: 1 | -1): void {
  const primary = runs.find((r) => r.id === primaryId);
  const [lo, hi] = domain();
  if (!primary) return;
  const frames = primary.doc.frames;
  const dt = frames.length > 1 ? frames[1].timeSec - frames[0].timeSec : 1 / 240;
  const step = syncMode === 'absolute' || syncMode === 'release' ? dt : ((hi - lo) * dt) / referenceDuration();
  playback = { ...playback, playing: false, parameter: Math.min(hi, Math.max(lo, playback.parameter + direction * step)) };
}

function updateScene(): void {
  const visibleIds = new Set(activeRuns().map((r) => r.id));
  let primaryFrame: InterpolatedFrame | null = null;
  for (const run of runs) {
    const show = visibleIds.has(run.id);
    run.line.visible = show;
    run.cap.group.visible = show;
    run.arrows.group.visible = show;
    if (!show) continue;
    const frame = frameFor(run);
    run.cap.update(frame);
    run.cap.setAxesVisible(axesVisible);
    run.arrows.update(frame, { forceMPerN: forceScale, velocityMPerMS: velocityScale }, vectorVisible);
    if (run.id === primaryId) primaryFrame = frame;
  }
  const primary = runs.find((r) => r.id === primaryId) ?? null;
  $('hud').textContent = formatHud(primary, primaryFrame);
  const [lo, hi] = domain();
  const seek = $<HTMLInputElement>('seek');
  if (document.activeElement !== seek) seek.value = String(hi > lo ? ((playback.parameter - lo) / (hi - lo)) * 1000 : 0);
  const unit = parameterUnit(syncMode);
  const shown = unit === '%' ? playback.parameter * 100 : playback.parameter;
  $('seek-label').textContent = `${shown.toFixed(3)} ${unit}  (${SYNC_MODE_LABELS[syncMode]})`;
  $('play').textContent = playback.playing ? '⏸ 停止' : '▶ 再生';
  $('warnings').textContent = primary?.doc.warnings.join('\n') ?? '';
  $('exaggeration').textContent =
    `表示: キャップ ×${capScale} 拡大 / 力 ${forceScale.toFixed(2)} m/N / 速度 ${velocityScale.toFixed(3)} m/(m/s)（物理値は不変）` +
    (primary?.doc.visualExaggeration ? ` / ${primary.doc.visualExaggeration}` : '');
}

let lastTs = performance.now();
function loop(ts: number): void {
  const dt = Math.min((ts - lastTs) / 1000, 0.1);
  lastTs = ts;
  playback = advance(playback, dt, domain(), referenceDuration());
  updateScene();
  scene.render();
  requestAnimationFrame(loop);
}

// ------------------------------------------------------------------ UI 結線

function refreshRuns(): void {
  renderRunList($('runs'), runs, primaryId, {
    onToggle: (run, visible) => {
      runs = runs.map((r) => (r.id === run.id ? { ...r, visible } : r));
      refreshRuns();
    },
    onPrimary: (run) => {
      primaryId = run.id;
    },
    onRemove: removeRun,
  });
}

function setupControls(): void {
  $<HTMLInputElement>('file-input').addEventListener('change', (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files) loadFiles(files);
  });
  $('play').addEventListener('click', () => {
    const [lo, hi] = domain();
    const restart = playback.parameter >= hi;
    playback = { ...playback, playing: !playback.playing, parameter: restart ? lo : playback.parameter };
  });
  $('step-back').addEventListener('click', () => stepFrame(-1));
  $('step-forward').addEventListener('click', () => stepFrame(1));
  $<HTMLSelectElement>('speed').addEventListener('change', (e) => {
    playback = { ...playback, speed: Number((e.target as HTMLSelectElement).value) };
  });
  $<HTMLInputElement>('seek').addEventListener('input', (e) => {
    const [lo, hi] = domain();
    playback = { ...playback, playing: false, parameter: lo + (Number((e.target as HTMLInputElement).value) / 1000) * (hi - lo) };
  });

  const sync = $<HTMLSelectElement>('sync-mode');
  (Object.keys(SYNC_MODE_LABELS) as SyncMode[]).forEach((mode) => {
    sync.append(new Option(SYNC_MODE_LABELS[mode], mode, mode === syncMode, mode === syncMode));
  });
  sync.addEventListener('change', () => {
    syncMode = sync.value as SyncMode;
    playback = { ...playback, parameter: domain()[0] };
  });

  document.querySelectorAll<HTMLButtonElement>('[data-view]').forEach((b) =>
    b.addEventListener('click', () => scene.setView(b.dataset.view as CameraView)),
  );

  $('label-filters').replaceChildren(
    ...LABELS.map((l) => checkbox(LABEL_TEXT[l], true, (c) => (labelVisible[l] = c))),
  );
  $('vector-toggles').replaceChildren(
    ...(Object.keys(VECTOR_TEXT) as VectorKind[]).map((k) => checkbox(VECTOR_TEXT[k], true, (c) => (vectorVisible[k] = c))),
    checkbox('回転軸・面法線', true, (c) => (axesVisible = c)),
  );

  const bindLogSlider = (id: string, labelId: string, set: (v: number) => void, unit: string) => {
    const input = $<HTMLInputElement>(id);
    const apply = () => {
      const v = 10 ** Number(input.value);
      set(v);
      $(labelId).textContent = `${v.toPrecision(3)} ${unit}`;
    };
    input.addEventListener('input', apply);
    apply();
  };
  bindLogSlider('force-scale', 'force-scale-label', (v) => (forceScale = v), 'm/N');
  bindLogSlider('velocity-scale', 'velocity-scale-label', (v) => (velocityScale = v), 'm/(m/s)');
  const cap = $<HTMLInputElement>('cap-scale');
  const applyCap = () => {
    capScale = Number(cap.value);
    runs.forEach((r) => r.cap.setDisplayScale(capScale));
    $('cap-scale-label').textContent = `×${capScale}`;
  };
  cap.addEventListener('input', applyCap);
  applyCap();

  renderLegend($('legend'));
}

setupControls();
mountSimPanel($('sim-panel'), {
  onLiveResult: (res) => replaceLive(res.doc),
  onPin: (res) => {
    const run = addDocument({ ...res.doc, label: 'virtual' });
    primaryId = run.id;
    refreshRuns();
  },
});
loadSampleManifest();
requestAnimationFrame(loop);
