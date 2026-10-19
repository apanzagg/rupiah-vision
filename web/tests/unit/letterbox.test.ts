import { describe, expect, it } from 'vitest';
import { letterbox, meanLuma, rgbaToTensor } from '../../src/lib/letterbox';

describe('letterbox', () => {
  it('citra lanskap: lebar penuh, padding atas-bawah simetris', () => {
    const lb = letterbox(640, 480, 320);
    expect(lb.scale).toBeCloseTo(0.5);
    expect([lb.newW, lb.newH]).toEqual([320, 240]);
    expect([lb.padX, lb.padY]).toEqual([0, 40]);
  });

  it('citra potret: tinggi penuh, padding kiri-kanan', () => {
    const lb = letterbox(720, 1280, 416);
    expect(lb.newH).toBe(416);
    expect(lb.newW).toBe(Math.round(720 * (416 / 1280)));
    expect(lb.padX).toBe(Math.floor((416 - lb.newW) / 2));
    expect(lb.padY).toBe(0);
  });

  it('pembulatan .5 ke atas (sama dengan infer_ref.py)', () => {
    // 3 * (320/2) ... buat kasus newW = x.5: srcW=5, srcH=640, size=320 -> scale 0.5, newW=2.5 -> 3
    const lb = letterbox(5, 640, 320);
    expect(lb.newW).toBe(3);
    expect(lb.padX).toBe(158);
  });

  it('citra persegi tidak diberi padding', () => {
    const lb = letterbox(1000, 1000, 320);
    expect([lb.newW, lb.newH, lb.padX, lb.padY]).toEqual([320, 320, 0, 0]);
  });
});

describe('rgbaToTensor', () => {
  it('menghasilkan CHW RGB 0..1', () => {
    const size = 2;
    const rgba = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 51, 102, 153, 255]);
    const t = rgbaToTensor(rgba, size);
    expect(t.length).toBe(12);
    [1, 0, 0, 0.2].forEach((v, i) => expect(t[i]).toBeCloseTo(v)); // kanal R
    expect(t[4 + 1]).toBe(1); // G piksel 1
    expect(t[8 + 2]).toBe(1); // B piksel 2
    expect(t[8 + 3]).toBeCloseTo(0.6);
  });
});

describe('meanLuma', () => {
  it('mengabaikan area padding', () => {
    const size = 4;
    const lb = letterbox(4, 2, size); // padY = 1, newH = 2
    const rgba = new Uint8ClampedArray(size * size * 4).fill(0);
    for (let y = 1; y < 3; y++) for (let x = 0; x < 4; x++) rgba.set([200, 200, 200, 255], (y * size + x) * 4);
    expect(meanLuma(rgba, lb, 1)).toBeCloseTo(200);
  });
});
