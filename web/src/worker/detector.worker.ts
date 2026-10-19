// Web Worker inferensi: letterbox -> normalisasi -> onnxruntime-web (WASM) -> decode + NMS.
// Berjalan terpisah dari thread UI agar tampilan dan suara tidak tersendat.
import * as ort from 'onnxruntime-web/wasm';
import { PAD_VALUE, letterbox, meanLuma, rgbaToTensor, type Letterbox } from '../lib/letterbox';
import { decode } from '../lib/postprocess';
import type { FromWorker, ToWorker } from './protocol';

let session: ort.InferenceSession | null = null;
let inputName = '';
let outputName = '';
let size = 320;
let numClasses = 7;
let ctx: OffscreenCanvasRenderingContext2D | null = null;
let tensorBuf: Float32Array | undefined;

const post = (m: FromWorker) => (self as unknown as Worker).postMessage(m);

async function init(m: Extract<ToWorker, { type: 'init' }>) {
  ort.env.wasm.wasmBinary = m.wasm;
  ort.env.wasm.numThreads = m.threads;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'warning';
  const t0 = performance.now();
  session = await ort.InferenceSession.create(new Uint8Array(m.model), {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  inputName = session.inputNames[0];
  outputName = session.outputNames[0];
  size = m.imgsz;
  numClasses = m.numClasses;
  const initMs = performance.now() - t0;
  const t1 = performance.now();
  const out = await session.run({ [inputName]: new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]) });
  const warmupMs = performance.now() - t1;
  if (typeof OffscreenCanvas !== 'undefined') {
    ctx = new OffscreenCanvas(size, size).getContext('2d', { willReadFrequently: true });
  }
  post({
    type: 'ready',
    initMs,
    warmupMs,
    offscreen: !!ctx,
    numAnchors: out[outputName].dims[2] as number,
    threads: ort.env.wasm.numThreads ?? 1,
  });
}

async function run(id: number, rgba: Uint8ClampedArray, lb: Letterbox, conf: number, iouThr: number) {
  if (!session) throw new Error('model belum siap');
  const t0 = performance.now();
  tensorBuf = rgbaToTensor(rgba, size, tensorBuf);
  const luma = meanLuma(rgba, lb);
  const t1 = performance.now();
  const out = await session.run({ [inputName]: new ort.Tensor('float32', tensorBuf, [1, 3, size, size]) });
  const t2 = performance.now();
  const o = out[outputName];
  const dets = decode(o.data as Float32Array, numClasses, o.dims[2] as number, lb, conf, iouThr);
  const t3 = performance.now();
  post({ type: 'result', id, dets, luma, preMs: t1 - t0, inferMs: t2 - t1, postMs: t3 - t2, srcW: lb.srcW, srcH: lb.srcH });
}

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      await init(m);
    } else if (m.type === 'frame') {
      if (!ctx) throw new Error('OffscreenCanvas tidak tersedia');
      const lb = letterbox(m.bitmap.width, m.bitmap.height, size);
      ctx.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
      ctx.fillRect(0, 0, size, size);
      ctx.drawImage(m.bitmap, lb.padX, lb.padY, lb.newW, lb.newH);
      m.bitmap.close();
      await run(m.id, ctx.getImageData(0, 0, size, size).data, lb, m.conf, m.iou);
    } else if (m.type === 'pixels') {
      await run(m.id, m.rgba, m.lb, m.conf, m.iou);
    }
  } catch (err) {
    post({ type: 'error', id: 'id' in m ? m.id : undefined, message: err instanceof Error ? err.message : String(err) });
  }
};
