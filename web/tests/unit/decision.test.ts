import { describe, expect, it } from 'vitest';
import { DecisionEngine, GuidanceEngine, frameSignature } from '../../src/lib/decision';
import type { Detection } from '../../src/lib/postprocess';

const det = (cls: number, score = 0.9): Detection => ({ cls, score, box: [0, 0, 10, 10] });
const C = { r1000: 0, r2000: 1, r5000: 2, r10000: 3, r20000: 4, r50000: 5, r100000: 6 };

/** Jalankan urutan frame (selang 200 ms) dan kumpulkan pengumuman beserta waktunya. */
function run(engine: DecisionEngine, frames: Detection[][], t0 = 0, dt = 200) {
  const out: { t: number; notes: number[]; total: number }[] = [];
  frames.forEach((f, i) => {
    const a = engine.update(f, t0 + i * dt);
    if (a) out.push({ t: t0 + i * dt, notes: a.notes, total: a.total });
  });
  return out;
}

describe('frameSignature', () => {
  it('hanya menghitung deteksi >= threshold, urut nominal menurun', () => {
    expect(frameSignature([det(C.r20000), det(C.r50000), det(C.r1000, 0.5)], 0.7)).toBe('50000,20000');
    expect(frameSignature([], 0.7)).toBe('');
  });
});

describe('DecisionEngine voting', () => {
  it('butuh minimal 3 dari 5 frame konsisten', () => {
    const e = new DecisionEngine();
    expect(run(e, [[det(C.r50000)], [det(C.r50000)]])).toEqual([]);
    const e2 = new DecisionEngine();
    const res = run(e2, [[det(C.r50000)], [], [det(C.r50000)], [], [det(C.r50000)]]);
    expect(res).toHaveLength(1);
    expect(res[0].notes).toEqual([50000]);
  });

  it('deteksi berkonfidensi rendah tidak diumumkan', () => {
    const e = new DecisionEngine();
    expect(run(e, Array(5).fill([det(C.r50000, 0.6)]))).toEqual([]);
  });

  it('threshold bisa dikonfigurasi', () => {
    const e = new DecisionEngine({ threshold: 0.5, window: 5, minVotes: 3, cooldownMs: 3000 });
    expect(run(e, Array(3).fill([det(C.r50000, 0.6)]))).toHaveLength(1);
  });

  it('kelas yang berganti-ganti tidak diumumkan', () => {
    const e = new DecisionEngine();
    const frames = [[det(C.r50000)], [det(C.r20000)], [det(C.r50000)], [det(C.r20000)], [det(C.r10000)]];
    expect(run(e, frames)).toEqual([]);
  });
});

describe('DecisionEngine cooldown', () => {
  it('nominal sama tidak diulang dalam 3 detik, diulang setelahnya', () => {
    const e = new DecisionEngine();
    const res = run(e, Array(25).fill([det(C.r100000)])); // 0..4800 ms
    expect(res.map((r) => r.t)).toEqual([400, 3400]);
  });

  it('diumumkan ulang segera jika uang sempat hilang dari kamera', () => {
    const e = new DecisionEngine();
    const frames = [
      ...Array(3).fill([det(C.r5000)]), // diumumkan t=400
      ...Array(4).fill([]), // hilang (kosong stabil)
      ...Array(3).fill([det(C.r5000)]), // muncul lagi < 3 dtk setelahnya
    ];
    const res = run(e, frames);
    expect(res).toHaveLength(2);
    expect(res[1].t - res[0].t).toBeLessThan(3000);
  });

  it('nominal berbeda langsung diumumkan', () => {
    const e = new DecisionEngine();
    const res = run(e, [...Array(3).fill([det(C.r5000)]), ...Array(5).fill([det(C.r10000)])]);
    expect(res.map((r) => r.notes[0])).toEqual([5000, 10000]);
  });
});

describe('DecisionEngine beberapa lembar', () => {
  it('mengumumkan semua lembar beserta total', () => {
    const e = new DecisionEngine();
    const res = run(e, Array(3).fill([det(C.r20000), det(C.r50000)]));
    expect(res[0].notes).toEqual([50000, 20000]);
    expect(res[0].total).toBe(70000);
  });

  it('dua lembar bernominal sama dihitung dua', () => {
    const e = new DecisionEngine();
    const res = run(e, Array(3).fill([det(C.r2000), det(C.r2000), det(C.r1000)]));
    expect(res[0].notes).toEqual([2000, 2000, 1000]);
    expect(res[0].total).toBe(5000);
  });
});

describe('GuidanceEngine', () => {
  it('panduan "belum terdeteksi" setelah 4 detik tanpa uang, lalu tiap 10 detik', () => {
    const g = new GuidanceEngine(0);
    const said: number[] = [];
    for (let t = 0; t <= 16000; t += 250) if (g.update(false, 120, t) === 'guide_not_found') said.push(t);
    expect(said).toEqual([4000, 14000]);
  });

  it('melihat uang menunda panduan', () => {
    const g = new GuidanceEngine(0);
    expect(g.update(false, 120, 3000)).toBeNull();
    expect(g.update(true, 120, 3500)).toBeNull();
    expect(g.update(false, 120, 6000)).toBeNull();
    expect(g.update(false, 120, 7500)).toBe('guide_not_found');
  });

  it('gelap: panduan "terlalu gelap", bukan "belum terdeteksi"', () => {
    const g = new GuidanceEngine(0);
    const said: string[] = [];
    for (let t = 0; t <= 6000; t += 250) {
      const r = g.update(false, 20, t);
      if (r) said.push(r);
    }
    expect(said).toEqual(['guide_dark']);
  });
});
