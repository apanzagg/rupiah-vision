// Logika keputusan: kapan sebuah nominal diumumkan, dan kapan panduan suara diberikan.
import type { Detection } from './postprocess';

export const DENOMS = [1000, 2000, 5000, 10000, 20000, 50000, 100000];

export interface DecisionConfig {
  /** confidence minimum agar deteksi dihitung (default 0,7) */
  threshold: number;
  /** jumlah frame terakhir yang dipertimbangkan */
  window: number;
  /** minimal frame (dari `window`) dengan hasil yang sama */
  minVotes: number;
  /** nominal/kombinasi yang sama tidak diulang dalam rentang ini, kecuali uang sempat hilang */
  cooldownMs: number;
}

export const DEFAULT_DECISION: DecisionConfig = { threshold: 0.7, window: 5, minVotes: 3, cooldownMs: 3000 };

export interface Announcement {
  /** nominal tiap lembar, urut dari terbesar */
  notes: number[];
  total: number;
  signature: string;
}

/** Tanda tangan satu frame: nominal (urut menurun) dari deteksi dengan skor >= threshold. "" = tidak ada uang. */
export function frameSignature(dets: Detection[], threshold: number): string {
  return dets
    .filter((d) => d.score >= threshold)
    .map((d) => DENOMS[d.cls])
    .sort((a, b) => b - a)
    .join(',');
}

export class DecisionEngine {
  private history: string[] = [];
  private lastSaid = new Map<string, number>();

  constructor(public cfg: DecisionConfig = DEFAULT_DECISION) {}

  reset(): void {
    this.history = [];
    this.lastSaid.clear();
  }

  /** Hasil stabil: tanda tangan yang muncul >= minVotes kali di jendela; null jika belum stabil. */
  stable(): string | null {
    const counts = new Map<string, number>();
    for (const s of this.history) counts.set(s, (counts.get(s) ?? 0) + 1);
    let best: string | null = null;
    let bestN = 0;
    for (const [s, n] of counts) {
      if (n > bestN) {
        best = s;
        bestN = n;
      }
    }
    return bestN >= this.cfg.minVotes ? best : null;
  }

  update(dets: Detection[], now: number): Announcement | null {
    this.history.push(frameSignature(dets, this.cfg.threshold));
    if (this.history.length > this.cfg.window) this.history.shift();
    const s = this.stable();
    if (s === null) return null;
    if (s === '') {
      // uang hilang dari kamera: boleh langsung diumumkan lagi saat muncul kembali
      this.lastSaid.clear();
      return null;
    }
    const last = this.lastSaid.get(s);
    if (last !== undefined && now - last < this.cfg.cooldownMs) return null;
    this.lastSaid.set(s, now);
    const notes = s.split(',').map(Number);
    return { notes, total: notes.reduce((a, b) => a + b, 0), signature: s };
  }
}

export interface GuidanceConfig {
  /** panduan "uang belum terdeteksi" setelah sekian ms tanpa deteksi */
  notFoundAfterMs: number;
  /** jeda minimal antar panduan yang sama */
  repeatMs: number;
  /** rata-rata luminans (0..255) di bawah nilai ini dianggap gelap */
  darkLuma: number;
  /** lama gelap berturut-turut sebelum panduan "terlalu gelap" */
  darkAfterMs: number;
}

export const DEFAULT_GUIDANCE: GuidanceConfig = { notFoundAfterMs: 4000, repeatMs: 10000, darkLuma: 45, darkAfterMs: 1500 };

export type Guidance = 'guide_not_found' | 'guide_dark';

export class GuidanceEngine {
  private lastSeen: number;
  private darkSince: number | null = null;
  private lastGiven = new Map<Guidance, number>();

  constructor(start: number, public cfg: GuidanceConfig = DEFAULT_GUIDANCE) {
    this.lastSeen = start;
  }

  /** Panggil tiap frame. `anyMoney` = ada deteksi di atas threshold pada frame ini. */
  update(anyMoney: boolean, luma: number, now: number): Guidance | null {
    if (anyMoney) {
      this.lastSeen = now;
      this.darkSince = null;
      return null;
    }
    if (luma < this.cfg.darkLuma) {
      this.darkSince ??= now;
      if (now - this.darkSince >= this.cfg.darkAfterMs && this.can('guide_dark', now)) return this.give('guide_dark', now);
      return null; // saat gelap, panduan gelap lebih informatif daripada "belum terdeteksi"
    }
    this.darkSince = null;
    if (now - this.lastSeen >= this.cfg.notFoundAfterMs && this.can('guide_not_found', now)) {
      return this.give('guide_not_found', now);
    }
    return null;
  }

  /** Panggil saat pengumuman nominal dibuat agar panduan tidak menyela. */
  noteAnnounced(now: number): void {
    this.lastSeen = now;
  }

  private can(g: Guidance, now: number): boolean {
    const t = this.lastGiven.get(g);
    return t === undefined || now - t >= this.cfg.repeatMs;
  }

  private give(g: Guidance, now: number): Guidance {
    this.lastGiven.set(g, now);
    return g;
  }
}
