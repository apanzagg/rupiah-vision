// Decode output YOLOv8 [1, 4 + nc, N] -> deteksi + NMS class-agnostic.
// Baris 0..3: cx, cy, w, h (piksel ruang input letterbox); baris 4..: skor tiap kelas (sudah sigmoid).
import type { Letterbox } from './letterbox';

export interface Detection {
  cls: number;
  score: number;
  /** [x0, y0, x1, y1] dalam piksel citra asli */
  box: [number, number, number, number];
}

export function iou(a: number[], b: number[]): number {
  const x0 = Math.max(a[0], b[0]);
  const y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]);
  const y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return union > 0 ? inter / union : 0;
}

export function nms(boxes: number[][], scores: number[], iouThr: number, maxDet: number): number[] {
  const order = scores.map((_, i) => i).sort((i, j) => scores[j] - scores[i]);
  const keep: number[] = [];
  const removed = new Uint8Array(order.length);
  for (let a = 0; a < order.length && keep.length < maxDet; a++) {
    if (removed[a]) continue;
    const i = order[a];
    keep.push(i);
    for (let b = a + 1; b < order.length; b++) {
      if (!removed[b] && iou(boxes[i], boxes[order[b]]) > iouThr) removed[b] = 1;
    }
  }
  return keep;
}

export function decode(
  data: Float32Array,
  numClasses: number,
  numAnchors: number,
  lb: Letterbox,
  conf = 0.25,
  iouThr = 0.45,
  maxDet = 20,
): Detection[] {
  const boxes: number[][] = [];
  const scores: number[] = [];
  const classes: number[] = [];
  const N = numAnchors;
  for (let i = 0; i < N; i++) {
    let best = -1;
    let bestScore = conf;
    for (let c = 0; c < numClasses; c++) {
      const s = data[(4 + c) * N + i];
      if (s >= bestScore) {
        bestScore = s;
        best = c;
      }
    }
    if (best < 0) continue;
    const cx = data[i];
    const cy = data[N + i];
    const w = data[2 * N + i];
    const h = data[3 * N + i];
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.push(bestScore);
    classes.push(best);
  }
  const keep = nms(boxes, scores, iouThr, maxDet);
  const clamp = (v: number, hi: number) => Math.min(hi, Math.max(0, v));
  return keep.map((k) => {
    const [x0, y0, x1, y1] = boxes[k];
    return {
      cls: classes[k],
      score: scores[k],
      box: [
        clamp((x0 - lb.padX) / lb.scale, lb.srcW),
        clamp((y0 - lb.padY) / lb.scale, lb.srcH),
        clamp((x1 - lb.padX) / lb.scale, lb.srcW),
        clamp((y1 - lb.padY) / lb.scale, lb.srcH),
      ],
    };
  });
}
