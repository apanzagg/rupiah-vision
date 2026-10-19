// Pembungkus Worker di thread utama: inisialisasi, kirim frame, terima hasil (satu frame diproses sekaligus).
import { PAD_VALUE, letterbox } from '../lib/letterbox';
import type { FromWorker, ReadyMsg, ResultMsg } from '../worker/protocol';

export class Detector {
  private worker: Worker;
  private pending = new Map<number, { resolve: (r: ResultMsg) => void; reject: (e: Error) => void }>();
  private nextId = 1;
  private fallbackCanvas: HTMLCanvasElement | null = null;
  ready: ReadyMsg | null = null;

  constructor(private imgsz: number, private numClasses: number) {
    this.worker = new Worker(new URL('../worker/detector.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === 'result') {
        this.pending.get(m.id)?.resolve(m);
        this.pending.delete(m.id);
      } else if (m.type === 'error' && m.id !== undefined) {
        this.pending.get(m.id)?.reject(new Error(m.message));
        this.pending.delete(m.id);
      }
    };
  }

  init(model: ArrayBuffer, wasm: ArrayBuffer): Promise<ReadyMsg> {
    const threads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    return new Promise((resolve, reject) => {
      const onMsg = (e: MessageEvent<FromWorker>) => {
        if (e.data.type === 'ready') {
          this.worker.removeEventListener('message', onMsg);
          this.ready = e.data;
          resolve(e.data);
        } else if (e.data.type === 'error' && e.data.id === undefined) {
          this.worker.removeEventListener('message', onMsg);
          reject(new Error(e.data.message));
        }
      };
      this.worker.addEventListener('message', onMsg);
      this.worker.postMessage(
        { type: 'init', model, wasm, imgsz: this.imgsz, numClasses: this.numClasses, threads },
        [model, wasm],
      );
    });
  }

  /** Deteksi pada sumber gambar (video/gambar/canvas). conf rendah agar logika keputusan yang menyaring. */
  async detect(source: CanvasImageSource & { width?: number; height?: number }, srcW: number, srcH: number, conf = 0.25, iou = 0.45): Promise<ResultMsg> {
    const id = this.nextId++;
    const p = new Promise<ResultMsg>((resolve, reject) => this.pending.set(id, { resolve, reject }));
    if (this.ready?.offscreen && typeof createImageBitmap === 'function') {
      const bitmap = await createImageBitmap(source as ImageBitmapSource);
      this.worker.postMessage({ type: 'frame', id, bitmap, conf, iou }, [bitmap]);
    } else {
      const size = this.imgsz;
      const lb = letterbox(srcW, srcH, size);
      this.fallbackCanvas ??= Object.assign(document.createElement('canvas'), { width: size, height: size });
      const c = this.fallbackCanvas.getContext('2d', { willReadFrequently: true })!;
      c.fillStyle = `rgb(${PAD_VALUE},${PAD_VALUE},${PAD_VALUE})`;
      c.fillRect(0, 0, size, size);
      c.drawImage(source, lb.padX, lb.padY, lb.newW, lb.newH);
      const rgba = c.getImageData(0, 0, size, size).data;
      this.worker.postMessage({ type: 'pixels', id, rgba, lb, conf, iou }, [rgba.buffer]);
    }
    return p;
  }
}
