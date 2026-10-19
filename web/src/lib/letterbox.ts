// Letterbox: skala citra agar muat di kotak size x size tanpa mengubah rasio, sisa diberi padding abu (114).
// Rumus HARUS sama dengan training/scripts/infer_ref.py (dipakai untuk parity test).

export const PAD_VALUE = 114;

export interface Letterbox {
  size: number;
  scale: number;
  newW: number;
  newH: number;
  padX: number;
  padY: number;
  srcW: number;
  srcH: number;
}

export function letterbox(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  return {
    size,
    scale,
    newW,
    newH,
    padX: Math.floor((size - newW) / 2),
    padY: Math.floor((size - newH) / 2),
    srcW,
    srcH,
  };
}

/** RGBA (hasil getImageData size x size) -> tensor float32 CHW RGB bernilai 0..1. */
export function rgbaToTensor(rgba: Uint8ClampedArray, size: number, out?: Float32Array): Float32Array {
  const area = size * size;
  const t = out ?? new Float32Array(3 * area);
  for (let i = 0, p = 0; i < area; i++, p += 4) {
    t[i] = rgba[p] / 255;
    t[area + i] = rgba[p + 1] / 255;
    t[2 * area + i] = rgba[p + 2] / 255;
  }
  return t;
}

/** Rata-rata luminans (0..255) area citra asli di dalam letterbox (padding diabaikan), sampel tiap `step` piksel. */
export function meanLuma(rgba: Uint8ClampedArray, lb: Letterbox, step = 4): number {
  let sum = 0;
  let n = 0;
  for (let y = lb.padY; y < lb.padY + lb.newH; y += step) {
    for (let x = lb.padX; x < lb.padX + lb.newW; x += step) {
      const p = (y * lb.size + x) * 4;
      sum += 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
      n++;
    }
  }
  return n ? sum / n : 0;
}
