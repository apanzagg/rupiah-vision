// Menyusun pengumuman menjadi segmen suara. Tiap segmen punya kunci klip audio (public/audio/manifest.json)
// dan teks cadangan untuk Web Speech API bila klip tidak tersedia atau gagal diputar.
import { capitalize, formatRupiah, numToWords } from './numwords';
import type { Announcement } from './decision';

export interface Segment {
  key: string;
  text: string;
}

export function denomSegment(n: number): Segment {
  return { key: `d${n}`, text: `${capitalize(numToWords(n))} rupiah.` };
}

export function announcementSegments(a: Announcement): Segment[] {
  if (a.notes.length === 1) return [denomSegment(a.notes[0])];
  const k = a.notes.length;
  return [
    { key: `c${k}`, text: `${capitalize(numToWords(k))} lembar.` },
    ...a.notes.map((n) => ({ key: `n${n}`, text: `${numToWords(n)}.` })),
    { key: `t${a.total}`, text: `Total ${numToWords(a.total)} rupiah.` },
  ];
}

/** Teks untuk region aria-live dan layar. Contoh: "Dua lembar: Rp50.000 + Rp20.000. Total Rp70.000". */
export function announcementText(a: Announcement): string {
  if (a.notes.length === 1) return `${formatRupiah(a.notes[0])} (${numToWords(a.notes[0])} rupiah)`;
  return (
    `${capitalize(numToWords(a.notes.length))} lembar: ${a.notes.map(formatRupiah).join(' + ')}. ` +
    `Total ${formatRupiah(a.total)} (${numToWords(a.total)} rupiah)`
  );
}

/** Pola getar per nominal (ms: getar, jeda, getar, ...). Pendek = 120 ms, panjang = 400 ms. */
export const VIBRATION: Record<number, number[]> = {
  1000: [120],
  2000: [120, 120, 120],
  5000: [400],
  10000: [400, 150, 120],
  20000: [400, 150, 120, 120, 120],
  50000: [400, 150, 400],
  100000: [400, 150, 400, 150, 400],
};
/** Lebih dari satu lembar: getaran ganda cepat lalu pola nominal terbesar. */
export function vibrationFor(a: Announcement): number[] {
  const base = VIBRATION[a.notes[0]] ?? [200];
  return a.notes.length > 1 ? [80, 80, 80, 300, ...base] : base;
}
