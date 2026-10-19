// Halaman /benchmark: ukur muat model & inferensi di perangkat nyata. Juga dipakai parity test (?parity=1).
import './styles.css';
import { BASE, MODEL, ORT_CACHE, ORT_VERSION, WASM_BYTES, WASM_URL } from './app/assets';
import { Detector } from './app/detector';

interface VariantInfo {
  file: string;
  variant: string;
  size_mb?: number;
  map50?: number;
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const state = $<HTMLParagraphElement>('state');
const out = $<HTMLPreElement>('out');
const table = $<HTMLTableElement>('table').tBodies[0];
const select = $<HTMLSelectElement>('variant');
const q = new URLSearchParams(location.search);

const stats = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const mean = s.reduce((a, b) => a + b, 0) / s.length;
  return { mean, median: s[Math.floor(s.length / 2)], p95: s[Math.min(s.length - 1, Math.floor(s.length * 0.95))], min: s[0] };
};
const r1 = (v: number) => Math.round(v * 10) / 10;

async function loadImages(urls: string[]): Promise<HTMLImageElement[]> {
  return Promise.all(
    urls.map(
      (u) =>
        new Promise<HTMLImageElement>((res, rej) => {
          const im = new Image();
          im.onload = () => res(im);
          im.onerror = rej;
          im.src = u;
        }),
    ),
  );
}

async function variants(): Promise<VariantInfo[]> {
  try {
    const r = await fetch(`${BASE}models/variants.json`);
    if (r.ok) return (await r.json()) as VariantInfo[];
  } catch {
    /* tidak ada daftar varian */
  }
  return [{ file: MODEL.file, variant: MODEL.variant }];
}

async function makeDetector(file: string): Promise<{ det: Detector; loadMs: number; downloadMs: number; initMs: number; warmupMs: number }> {
  const t0 = performance.now();
  const det = new Detector(MODEL.imgsz, MODEL.classes.length);
  // cache khusus benchmark dengan stempel waktu -> model selalu diunduh ulang (mengukur waktu unduh nyata)
  const ready = await det.init(
    { url: new URL(`${BASE}models/${file}`, location.href).href, cache: `bench-${Date.now()}`, bytes: 6_000_000 },
    { url: new URL(WASM_URL, location.href).href, cache: ORT_CACHE, bytes: WASM_BYTES },
  );
  void caches.keys().then((ks) => ks.filter((k) => k.startsWith('bench-')).forEach((k) => caches.delete(k)));
  return { det, loadMs: performance.now() - t0, downloadMs: ready.downloadMs, initMs: ready.initMs, warmupMs: ready.warmupMs };
}

async function runBenchmark() {
  const file = select.value;
  const iters = Math.max(5, Math.min(500, Number(($('iters') as HTMLInputElement).value) || 50));
  state.textContent = 'Memuat model...';
  const { det, loadMs, downloadMs, initMs, warmupMs } = await makeDetector(file);
  const imgs = await loadImages(['bench/1.jpg', 'bench/2.jpg', 'bench/3.jpg'].map((f) => `${BASE}${f}`));
  const pre: number[] = [];
  const inf: number[] = [];
  const post: number[] = [];
  const total: number[] = [];
  for (let i = 0; i < iters; i++) {
    const im = imgs[i % imgs.length];
    const t = performance.now();
    const r = await det.detect(im, im.naturalWidth, im.naturalHeight);
    total.push(performance.now() - t);
    pre.push(r.preMs);
    inf.push(r.inferMs);
    post.push(r.postMs);
    if (i % 10 === 0) state.textContent = `Iterasi ${i + 1}/${iters}...`;
  }
  const res = {
    model: file,
    imgsz: MODEL.imgsz,
    ort: ORT_VERSION,
    backend: 'wasm',
    threads: det.ready?.threads,
    crossOriginIsolated: self.crossOriginIsolated,
    offscreenWorker: det.ready?.offscreen,
    hardwareConcurrency: navigator.hardwareConcurrency,
    userAgent: navigator.userAgent,
    iterations: iters,
    model_download_ms: r1(downloadMs),
    model_load_total_ms: r1(loadMs),
    session_init_ms: r1(initMs),
    warmup_ms: r1(warmupMs),
    preprocess_ms: r1(stats(pre).mean),
    inference_ms: { mean: r1(stats(inf).mean), median: r1(stats(inf).median), p95: r1(stats(inf).p95), min: r1(stats(inf).min) },
    postprocess_ms: r1(stats(post).mean),
    end_to_end_ms: { mean: r1(stats(total).mean), p95: r1(stats(total).p95) },
    fps: r1(1000 / stats(total).mean),
  };
  out.textContent = JSON.stringify(res, null, 2);
  table.innerHTML = '';
  const rows: [string, string][] = [
    ['Model', `${file} (${MODEL.imgsz}px)`],
    ['Unduh model', `${res.model_download_ms} ms`],
    ['Inisialisasi sesi', `${res.session_init_ms} ms`],
    ['Inferensi (rata-rata / median / p95)', `${res.inference_ms.mean} / ${res.inference_ms.median} / ${res.inference_ms.p95} ms`],
    ['End-to-end per frame', `${res.end_to_end_ms.mean} ms`],
    ['Perkiraan FPS', String(res.fps)],
    ['Thread WASM', String(res.threads)],
  ];
  for (const [k, v] of rows) table.insertRow().innerHTML = `<th scope="row">${k}</th><td>${v}</td>`;
  state.textContent = `Selesai: ${res.fps} FPS.`;
  (window as unknown as { __bench: unknown }).__bench = res;
}

/** Parity test: deteksi pada citra uji yang sama dengan referensi Python (models/parity/). */
async function runParity() {
  state.textContent = 'Parity test berjalan...';
  const ref = await (await fetch(`${BASE}parity/parity_ref.json`)).json();
  const file = q.get('model') ?? MODEL.file;
  const { det } = await makeDetector(file);
  const names = Object.keys(ref.images);
  const imgs = await loadImages(names.map((n) => `${BASE}parity/${n}`));
  const result: Record<string, unknown> = {};
  for (let i = 0; i < names.length; i++) {
    const r = await det.detect(imgs[i], imgs[i].naturalWidth, imgs[i].naturalHeight, ref.conf, ref.iou);
    result[names[i]] = r.dets;
  }
  out.textContent = JSON.stringify(result, null, 1);
  state.textContent = 'Parity selesai.';
  (window as unknown as { __parity: unknown }).__parity = { model: file, result };
}

(async () => {
  for (const v of await variants()) {
    const o = document.createElement('option');
    o.value = v.file;
    o.textContent = `${v.variant}${v.size_mb ? ` - ${v.size_mb} MB` : ''}${v.map50 ? ` - mAP50 ${v.map50}` : ''}`;
    o.selected = v.file === MODEL.file;
    select.append(o);
  }
  $('run').addEventListener('click', () => void runBenchmark().catch((e) => (state.textContent = `Gagal: ${e}`)));
  $('copy').addEventListener('click', () => void navigator.clipboard?.writeText(out.textContent ?? ''));
  if (q.has('parity')) void runParity().catch((e) => (state.textContent = `Gagal: ${e}`));
  if (q.has('auto')) void runBenchmark().catch((e) => (state.textContent = `Gagal: ${e}`));
})();
