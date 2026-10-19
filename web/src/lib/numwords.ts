// Angka -> kata Bahasa Indonesia. Versi TypeScript dari training/scripts/tts_phrases.py:num_to_words
// (diuji dengan vektor yang dibangkitkan dari versi Python, tests/numwords_vectors.json).

const UNITS = ['nol', 'satu', 'dua', 'tiga', 'empat', 'lima', 'enam', 'tujuh', 'delapan', 'sembilan'];

export function numToWords(n: number): string {
  if (!Number.isInteger(n) || n < 0 || n >= 1e9) throw new RangeError(String(n));
  if (n < 10) return UNITS[n];
  if (n === 10) return 'sepuluh';
  if (n === 11) return 'sebelas';
  if (n < 20) return `${UNITS[n - 10]} belas`;
  if (n < 100) {
    const t = Math.floor(n / 10);
    const r = n % 10;
    return `${UNITS[t]} puluh` + (r ? ` ${UNITS[r]}` : '');
  }
  if (n < 1000) {
    const h = Math.floor(n / 100);
    const r = n % 100;
    return (h === 1 ? 'seratus' : `${UNITS[h]} ratus`) + (r ? ` ${numToWords(r)}` : '');
  }
  if (n < 1_000_000) {
    const th = Math.floor(n / 1000);
    const r = n % 1000;
    return (th === 1 ? 'seribu' : `${numToWords(th)} ribu`) + (r ? ` ${numToWords(r)}` : '');
  }
  const m = Math.floor(n / 1_000_000);
  const r = n % 1_000_000;
  return `${numToWords(m)} juta` + (r ? ` ${numToWords(r)}` : '');
}

export const capitalize = (s: string): string => s.charAt(0).toUpperCase() + s.slice(1);

/** "Rp50.000" untuk tampilan teks. */
export const formatRupiah = (n: number): string => 'Rp' + n.toLocaleString('id-ID');
