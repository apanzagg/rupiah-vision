import { describe, expect, it } from 'vitest';
import { letterbox } from '../../src/lib/letterbox';
import { decode, iou, nms } from '../../src/lib/postprocess';

const NC = 7;

/** Bangun tensor [1, 4+nc, N] dari daftar anchor {cx,cy,w,h,cls,score}. */
function makeOutput(anchors: { cx: number; cy: number; w: number; h: number; cls: number; score: number }[], N = 10) {
  const data = new Float32Array((4 + NC) * N);
  anchors.forEach((a, i) => {
    data[i] = a.cx;
    data[N + i] = a.cy;
    data[2 * N + i] = a.w;
    data[3 * N + i] = a.h;
    data[(4 + a.cls) * N + i] = a.score;
  });
  return { data, N };
}

describe('iou & nms', () => {
  it('iou kotak identik = 1, terpisah = 0', () => {
    expect(iou([0, 0, 10, 10], [0, 0, 10, 10])).toBe(1);
    expect(iou([0, 0, 10, 10], [20, 20, 30, 30])).toBe(0);
    expect(iou([0, 0, 10, 10], [5, 0, 15, 10])).toBeCloseTo(50 / 150);
  });

  it('nms membuang kotak tumpang tindih dengan skor lebih rendah', () => {
    const keep = nms([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], [0.6, 0.9, 0.8], 0.45, 20);
    expect(keep).toEqual([1, 2]);
  });

  it('nms menghormati maxDet', () => {
    const boxes = Array.from({ length: 5 }, (_, i) => [i * 20, 0, i * 20 + 10, 10]);
    expect(nms(boxes, [0.9, 0.8, 0.7, 0.6, 0.5], 0.45, 2)).toEqual([0, 1]);
  });
});

describe('decode', () => {
  const lb = letterbox(640, 480, 320); // scale 0.5, padY 40

  it('memilih kelas skor tertinggi dan mengembalikan koordinat ke citra asli', () => {
    const { data, N } = makeOutput([{ cx: 160, cy: 160, w: 100, h: 60, cls: 5, score: 0.92 }]);
    const d = decode(data, NC, N, lb, 0.25);
    expect(d).toHaveLength(1);
    expect(d[0].cls).toBe(5);
    expect(d[0].score).toBeCloseTo(0.92);
    // x0 = (160-50-0)/0.5 = 220 ; y0 = (160-30-40)/0.5 = 180
    expect(d[0].box.map((v) => Math.round(v))).toEqual([220, 180, 420, 300]);
  });

  it('membuang deteksi di bawah confidence dan menekan duplikat (class-agnostic)', () => {
    const { data, N } = makeOutput([
      { cx: 160, cy: 160, w: 100, h: 60, cls: 5, score: 0.9 },
      { cx: 162, cy: 161, w: 100, h: 60, cls: 4, score: 0.5 }, // kotak sama, kelas lain -> ditekan
      { cx: 60, cy: 100, w: 40, h: 40, cls: 1, score: 0.1 }, // di bawah conf
      { cx: 280, cy: 200, w: 50, h: 50, cls: 0, score: 0.4 },
    ]);
    const d = decode(data, NC, N, lb, 0.25, 0.45);
    expect(d.map((x) => x.cls)).toEqual([5, 0]);
  });

  it('memotong kotak ke batas citra', () => {
    const { data, N } = makeOutput([{ cx: 5, cy: 45, w: 40, h: 40, cls: 2, score: 0.8 }]);
    const [x0, y0] = decode(data, NC, N, lb)[0].box;
    expect(x0).toBe(0);
    expect(y0).toBe(0);
  });
});
