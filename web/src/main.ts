// SuaraRupiah: layar utama. Kamera -> deteksi (Web Worker) -> logika keputusan -> suara, getar, teks aria-live.
import './styles.css';
import { BASE, MODEL, MODEL_CACHE, MODEL_URL, ORT_CACHE, WASM_BYTES, WASM_URL, CACHE_PREFIX, AUDIO_MANIFEST } from './app/assets';
import { CameraError, openCamera, setTorch, stopCamera, type CameraHandle } from './app/camera';
import { Detector } from './app/detector';
import { cleanupCaches } from './app/loader';
import { loadSettings } from './app/settings';
import { Speaker } from './app/speaker';
import { DecisionEngine, GuidanceEngine, DEFAULT_DECISION, type Announcement } from './lib/decision';
import { formatRupiah } from './lib/numwords';
import { announcementSegments, announcementText, vibrationFor, type Segment } from './lib/phrases';
import type { Detection } from './lib/postprocess';

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const video = $<HTMLVideoElement>('#video');
const overlay = $<HTMLCanvasElement>('#overlay');
const statusEl = $<HTMLParagraphElement>('#status');
const resultEl = $<HTMLSpanElement>('#result');
const live = $<HTMLDivElement>('#live');
const progress = $<HTMLDivElement>('#progress');
const progressBar = progress.querySelector('.bar') as HTMLDivElement;
const tap = $<HTMLButtonElement>('#tap');

const MIN_FRAME_MS = 100; // maks. ~10 inferensi/detik: hemat baterai, cukup untuk voting 3 dari 5 frame
const settings = loadSettings();
const speaker = new Speaker(BASE);
speaker.enabled = settings.voice;
speaker.rate = settings.rate;
const decision = new DecisionEngine({ ...DEFAULT_DECISION, threshold: settings.threshold });
let guidance: GuidanceEngine;
let detector: Detector | null = null;
let cam: CameraHandle | null = null;
let torchOn = false;
let running = false;
let lastSaid: { segments: Segment[]; text: string; vibrate?: number[] } | null = null;

// status untuk pengujian E2E & halaman benchmark (tidak berisi gambar kamera)
const debug = { phase: 'boot', announcements: [] as string[], guidance: [] as string[], frames: 0, inferMs: 0, fps: 0 };
(window as unknown as { __suararupiah: typeof debug }).__suararupiah = debug;

const phraseText = (key: string) => (AUDIO_MANIFEST.phrases as Record<string, { text: string }>)[key]?.text ?? key;

function announceLive(text: string) {
  // kosongkan dulu agar pembaca layar membacakan ulang teks yang sama
  live.textContent = '';
  requestAnimationFrame(() => (live.textContent = text));
}

function say(segments: Segment[], text: string, vibrate?: number[], remember = true) {
  announceLive(text);
  void speaker.say(segments);
  if (vibrate && settings.vibrate && 'vibrate' in navigator) navigator.vibrate(vibrate);
  if (remember) lastSaid = { segments, text, vibrate };
}

function sayKey(key: string, remember = false) {
  const text = phraseText(key);
  statusEl.textContent = text;
  say([{ key, text }], text, undefined, remember);
}

function setProgress(frac: number, label: string) {
  const pct = Math.round(frac * 100);
  progress.hidden = false;
  progress.setAttribute('aria-valuenow', String(pct));
  progress.setAttribute('aria-valuetext', `${label} ${pct}%`);
  progressBar.style.width = `${pct}%`;
  statusEl.textContent = `${phraseText('status_preparing')} ${label} ${pct}%`;
}

function announce(a: Announcement, now: number) {
  const text = announcementText(a);
  resultEl.textContent = a.notes.length === 1 ? formatRupiah(a.total) : `${formatRupiah(a.total)} (${a.notes.length} lembar)`;
  say(announcementSegments(a), text, vibrationFor(a));
  guidance.noteAnnounced(now);
  debug.announcements.push(text);
}

async function giveGuidance(g: 'guide_not_found' | 'guide_dark') {
  debug.guidance.push(g);
  if (g === 'guide_dark' && cam?.hasTorch && !torchOn) {
    torchOn = await setTorch(cam, true);
    if (torchOn) {
      const text = `${phraseText('guide_dark')} ${phraseText('guide_torch_on')}`;
      say([{ key: 'guide_dark', text: phraseText('guide_dark') }, { key: 'guide_torch_on', text: phraseText('guide_torch_on') }], text, undefined, false);
      return;
    }
  }
  say([{ key: g, text: phraseText(g) }], phraseText(g), undefined, false);
}

function drawOverlay(dets: Detection[], srcW: number, srcH: number) {
  const cw = overlay.clientWidth;
  const ch = overlay.clientHeight;
  if (overlay.width !== cw || overlay.height !== ch) {
    overlay.width = cw;
    overlay.height = ch;
  }
  const g = overlay.getContext('2d')!;
  g.clearRect(0, 0, cw, ch);
  // video memakai object-fit: cover
  const s = Math.max(cw / srcW, ch / srcH);
  const ox = (cw - srcW * s) / 2;
  const oy = (ch - srcH * s) / 2;
  g.lineJoin = 'round';
  g.font = 'bold 28px system-ui, sans-serif';
  for (const d of dets) {
    if (d.score < settings.threshold) continue;
    const [x0, y0, x1, y1] = d.box.map((v, i) => (i % 2 ? v * s + oy : v * s + ox));
    g.strokeStyle = '#000';
    g.lineWidth = 12;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    g.strokeStyle = '#ffd400';
    g.lineWidth = 6;
    g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const label = formatRupiah([1000, 2000, 5000, 10000, 20000, 50000, 100000][d.cls]);
    const tw = g.measureText(label).width + 16;
    g.fillStyle = '#ffd400';
    g.fillRect(x0, Math.max(0, y0 - 40), tw, 40);
    g.fillStyle = '#000';
    g.fillText(label, x0 + 8, Math.max(30, y0 - 10));
  }
}

async function tick() {
  if (!running || !detector) return;
  if (document.hidden || !video.videoWidth) {
    setTimeout(tick, 400);
    return;
  }
  const t0 = performance.now();
  try {
    const r = await detector.detect(video, video.videoWidth, video.videoHeight, 0.25, 0.45);
    debug.frames++;
    debug.inferMs = debug.inferMs ? debug.inferMs * 0.9 + r.inferMs * 0.1 : r.inferMs;
    drawOverlay(r.dets, r.srcW, r.srcH);
    const now = performance.now();
    const a = decision.update(r.dets, now);
    if (a) announce(a, now);
    else {
      const g = guidance.update(r.dets.some((d) => d.score >= settings.threshold), r.luma, now);
      if (g) void giveGuidance(g);
    }
  } catch (e) {
    console.warn('deteksi gagal', e);
  }
  const dt = performance.now() - t0;
  debug.fps = 1000 / Math.max(dt, MIN_FRAME_MS);
  setTimeout(tick, Math.max(0, MIN_FRAME_MS - dt));
}

async function boot() {
  // status awal hanya lewat layar + aria-live (pembaca layar); audio belum dimuat agar boot ringan
  statusEl.textContent = phraseText('status_preparing');
  announceLive(phraseText('status_preparing'));
  void cleanupCaches(CACHE_PREFIX, [MODEL_CACHE, ORT_CACHE]);
  const camReady = openCamera(video).then(
    (c) => (cam = c),
    (e: unknown) => {
      debug.phase = 'camera-error';
      sayKey(e instanceof CameraError && e.kind === 'denied' ? 'error_camera_denied' : 'error_camera', true);
      throw e;
    },
  );
  try {
    detector = new Detector(MODEL.imgsz, MODEL.classes.length);
    await detector.init(
      { url: new URL(MODEL_URL, location.href).href, cache: MODEL_CACHE, bytes: MODEL.sizeBytes || 6_000_000 },
      { url: new URL(WASM_URL, location.href).href, cache: ORT_CACHE, bytes: WASM_BYTES },
      (r, t) => setProgress(r / t, 'Mengunduh model'),
    );
  } catch (e) {
    console.error(e);
    debug.phase = 'model-error';
    sayKey('error_model', true);
    return;
  }
  progress.hidden = true;
  await speaker.load().catch(() => undefined);
  try {
    await camReady;
  } catch {
    return;
  }
  debug.phase = 'ready';
  sayKey('status_ready', true);
  // decode klip yang paling sering dipakai saat browser senggang
  speaker.warmup([...[1000, 2000, 5000, 10000, 20000, 50000, 100000].flatMap((n) => [`d${n}`, `n${n}`]),
    'guide_not_found', 'guide_dark', 'guide_torch_on', 'c2', 'c3']);
  resultEl.textContent = 'Arahkan kamera ke uang';
  guidance = new GuidanceEngine(performance.now());
  running = true;
  void tick();
}

tap.addEventListener('click', () => {
  void speaker.unlock();
  if (lastSaid) say(lastSaid.segments, lastSaid.text, lastSaid.vibrate, false);
});
document.addEventListener('visibilitychange', () => {
  if (document.hidden && torchOn && cam) {
    void setTorch(cam, false);
    torchOn = false;
  }
});
window.addEventListener('pagehide', () => stopCamera(cam));
// sentuhan/ketukan pertama di mana pun membuka kunci audio browser
window.addEventListener('pointerdown', () => void speaker.unlock(), { once: true });

void boot();
