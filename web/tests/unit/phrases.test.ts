import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { capitalize, formatRupiah, numToWords } from '../../src/lib/numwords';
import { announcementSegments, announcementText, vibrationFor } from '../../src/lib/phrases';

const vectors: Record<string, string> = JSON.parse(readFileSync(new URL('../numwords_vectors.json', import.meta.url), 'utf8'));
const manifest = JSON.parse(readFileSync(new URL('../../public/audio/manifest.json', import.meta.url), 'utf8'));

describe('numToWords (sama dengan versi Python)', () => {
  for (const [n, words] of Object.entries(vectors)) {
    it(`${n} -> ${words}`, () => expect(numToWords(Number(n))).toBe(words));
  }
  it('menolak angka negatif/pecahan', () => {
    expect(() => numToWords(-1)).toThrow();
    expect(() => numToWords(1.5)).toThrow();
  });
});

describe('perhitungan total & teks pengumuman', () => {
  it('satu lembar', () => {
    const a = { notes: [50000], total: 50000, signature: '50000' };
    expect(announcementSegments(a)).toEqual([{ key: 'd50000', text: 'Lima puluh ribu rupiah.' }]);
    expect(announcementText(a)).toContain('Rp50.000');
  });

  it('dua lembar: jumlah, masing-masing, total', () => {
    const a = { notes: [50000, 20000], total: 70000, signature: '50000,20000' };
    const seg = announcementSegments(a);
    expect(seg.map((s) => s.key)).toEqual(['c2', 'n50000', 'n20000', 't70000']);
    expect(seg.map((s) => s.text).join(' ')).toBe('Dua lembar. lima puluh ribu. dua puluh ribu. Total tujuh puluh ribu rupiah.');
    expect(announcementText(a)).toBe('Dua lembar: Rp50.000 + Rp20.000. Total Rp70.000 (tujuh puluh ribu rupiah)');
  });

  it('setiap kunci untuk 1..3 lembar tersedia di manifest audio', () => {
    const D = [1000, 2000, 5000, 10000, 20000, 50000, 100000];
    for (const a of D) for (const b of D) for (const c of D) {
      for (const notes of [[a], [a, b], [a, b, c]]) {
        const sorted = [...notes].sort((x, y) => y - x);
        const total = sorted.reduce((s, v) => s + v, 0);
        for (const s of announcementSegments({ notes: sorted, total, signature: '' })) {
          expect(manifest.phrases[s.key], s.key).toBeDefined();
          expect(manifest.phrases[s.key].text).toBe(s.text);
        }
      }
    }
  });

  it('format & kapitalisasi', () => {
    expect(formatRupiah(100000)).toBe('Rp100.000');
    expect(capitalize('seribu')).toBe('Seribu');
  });

  it('pola getar: satu lembar vs beberapa lembar', () => {
    expect(vibrationFor({ notes: [1000], total: 1000, signature: '' })).toEqual([120]);
    expect(vibrationFor({ notes: [50000, 1000], total: 51000, signature: '' }).slice(0, 4)).toEqual([80, 80, 80, 300]);
  });
});
