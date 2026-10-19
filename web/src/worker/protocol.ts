// Pesan antara thread utama dan Web Worker inferensi.
import type { Letterbox } from '../lib/letterbox';
import type { Detection } from '../lib/postprocess';

/** Worker sendiri yang mengunduh & meng-cache model + WASM (agar thread UI tidak menyalin ~20 MB data). */
export interface AssetSpec {
  url: string;
  cache: string;
  bytes: number;
}
export interface InitMsg {
  type: 'init';
  model: AssetSpec;
  wasm: AssetSpec;
  imgsz: number;
  numClasses: number;
  threads: number;
}
export interface FrameMsg {
  type: 'frame';
  id: number;
  bitmap: ImageBitmap;
  conf: number;
  iou: number;
}
/** Jalur cadangan bila worker tidak punya OffscreenCanvas: thread utama mengirim piksel RGBA size x size. */
export interface PixelsMsg {
  type: 'pixels';
  id: number;
  rgba: Uint8ClampedArray;
  lb: Letterbox;
  conf: number;
  iou: number;
}
export type ToWorker = InitMsg | FrameMsg | PixelsMsg;

export interface ProgressMsg {
  type: 'progress';
  received: number;
  total: number;
}
export interface ReadyMsg {
  type: 'ready';
  downloadMs: number;
  initMs: number;
  warmupMs: number;
  offscreen: boolean;
  numAnchors: number;
  threads: number;
}
export interface ResultMsg {
  type: 'result';
  id: number;
  dets: Detection[];
  luma: number;
  preMs: number;
  inferMs: number;
  postMs: number;
  srcW: number;
  srcH: number;
}
export interface ErrorMsg {
  type: 'error';
  id?: number;
  message: string;
}
export type FromWorker = ProgressMsg | ReadyMsg | ResultMsg | ErrorMsg;
