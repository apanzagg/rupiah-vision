// Pengaturan pengguna (disimpan di localStorage perangkat; bisa juga lewat parameter URL, mis. ?thr=0.6).

export interface Settings {
  /** confidence minimum pengumuman (0,5..0,95) */
  threshold: number;
  /** suara aplikasi (klip audio / Web Speech). Pengguna pembaca layar bisa mematikannya. */
  voice: boolean;
  vibrate: boolean;
  /** kecepatan bicara 0,8..1,5 */
  rate: number;
}

export const DEFAULT_SETTINGS: Settings = { threshold: 0.7, voice: true, vibrate: true, rate: 1 };
const KEY = 'suararupiah.settings.v1';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export function loadSettings(): Settings {
  let s: Settings = { ...DEFAULT_SETTINGS };
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) s = { ...s, ...JSON.parse(raw) };
  } catch {
    /* localStorage tidak tersedia */
  }
  const q = new URLSearchParams(location.search);
  if (q.has('thr')) s.threshold = Number(q.get('thr'));
  s.threshold = clamp(Number.isFinite(s.threshold) ? s.threshold : DEFAULT_SETTINGS.threshold, 0.5, 0.95);
  s.rate = clamp(Number.isFinite(s.rate) ? s.rate : 1, 0.8, 1.5);
  return s;
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* abaikan */
  }
}
