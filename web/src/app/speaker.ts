// Keluaran suara: klip audio TTS (public/audio/*.mp3, mudah diganti rekaman manusia) diputar lewat Web Audio API.
// Jika klip tidak ada / gagal / audio belum diizinkan browser, jatuh ke Web Speech API (lang id-ID).
// Klip diunduh di awal (kecil) tetapi baru di-decode saat dibutuhkan atau saat browser senggang,
// agar thread utama tidak tersendat ketika aplikasi baru dibuka.
import type { Segment } from '../lib/phrases';

interface Manifest {
  phrases: Record<string, { text: string; file: string }>;
}

const hasActivation = () => (navigator as Navigator & { userActivation?: { hasBeenActive: boolean } }).userActivation?.hasBeenActive ?? true;

export class Speaker {
  private ctx: AudioContext | null = null;
  private raw = new Map<string, ArrayBuffer>();
  private buffers = new Map<string, AudioBuffer>();
  private current: AudioBufferSourceNode[] = [];
  private seq = 0;
  manifest: Manifest | null = null;
  enabled = true;
  rate = 1;

  constructor(private base: string) {}

  /** AudioContext dibuat saat pertama dibutuhkan (membuatnya saat halaman dibuka memperlambat boot). */
  private audio(): AudioContext | null {
    if (!this.ctx) {
      try {
        this.ctx = new AudioContext();
      } catch {
        this.ctx = null;
      }
    }
    return this.ctx;
  }

  /** true bila browser sudah mengizinkan audio diputar (autoplay / setelah interaksi). */
  get unlocked(): boolean {
    return this.audio()?.state === 'running';
  }

  /** Panggil dari handler interaksi pengguna (ketukan) untuk membuka kunci audio. */
  async unlock(): Promise<void> {
    try {
      await this.audio()?.resume();
    } catch {
      /* abaikan */
    }
  }

  /** Muat manifest + unduh semua klip (belum di-decode). */
  async load(): Promise<void> {
    const res = await fetch(`${this.base}audio/manifest.json`);
    this.manifest = (await res.json()) as Manifest;
    await Promise.all(
      Object.entries(this.manifest.phrases).map(async ([key, p]) => {
        try {
          const r = await fetch(`${this.base}audio/${p.file}`);
          if (r.ok) this.raw.set(key, await r.arrayBuffer());
        } catch {
          /* klip hilang -> Web Speech */
        }
      }),
    );
  }

  /** Decode klip saat browser senggang (dipanggil setelah aplikasi siap). */
  warmup(keys: string[]): void {
    const idle = (cb: () => void) => ('requestIdleCallback' in self ? requestIdleCallback(cb) : setTimeout(cb, 50));
    const next = () => {
      const k = keys.shift();
      if (!k) return;
      void this.decode(k).finally(() => idle(next));
    };
    idle(next);
  }

  private async decode(key: string): Promise<AudioBuffer | null> {
    const hit = this.buffers.get(key);
    if (hit) return hit;
    const raw = this.raw.get(key);
    if (!raw || !this.audio()) return null;
    try {
      const buf = await this.ctx!.decodeAudioData(raw.slice(0));
      this.buffers.set(key, buf);
      return buf;
    } catch {
      this.raw.delete(key);
      return null;
    }
  }

  stop(): void {
    this.seq++;
    for (const s of this.current) {
      try {
        s.stop();
      } catch {
        /* sudah selesai */
      }
    }
    this.current = [];
    if ('speechSynthesis' in self && hasActivation()) speechSynthesis.cancel();
  }

  /** Ucapkan rangkaian segmen. Pengumuman baru memotong yang sedang diputar. */
  async say(segments: Segment[]): Promise<void> {
    if (!this.enabled || !segments.length) return;
    this.stop();
    const my = this.seq;
    if (this.unlocked && this.ctx) {
      const bufs = await Promise.all(segments.map((s) => this.decode(s.key)));
      if (my !== this.seq) return; // sudah didahului pengumuman lain
      if (bufs.every((b) => b)) {
        let t = this.ctx.currentTime + 0.02;
        for (const b of bufs as AudioBuffer[]) {
          const src = this.ctx.createBufferSource();
          src.buffer = b;
          src.playbackRate.value = this.rate;
          src.connect(this.ctx.destination);
          src.start(t);
          t += b.duration / this.rate + 0.06;
          this.current.push(src);
        }
        return;
      }
    }
    this.speakText(segments.map((s) => s.text).join(' '));
  }

  /** Web Speech API. Tanpa interaksi pengguna browser menolaknya, jadi dilewati (teks tetap ke aria-live). */
  speakText(text: string): void {
    if (!this.enabled || !('speechSynthesis' in self) || !hasActivation()) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = 'id-ID';
    u.rate = this.rate;
    const voice = speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith('id'));
    if (voice) u.voice = voice;
    speechSynthesis.speak(u);
  }

  /** Teks untuk kunci frasa (dipakai juga untuk region aria-live). */
  text(key: string): string {
    return this.manifest?.phrases[key]?.text ?? '';
  }
}
