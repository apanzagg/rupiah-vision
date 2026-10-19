// Keluaran suara: klip audio TTS (public/audio/*.mp3, mudah diganti rekaman manusia) diputar lewat Web Audio API.
// Jika klip tidak ada / gagal / audio belum diizinkan browser, jatuh ke Web Speech API (lang id-ID).
import type { Segment } from '../lib/phrases';

interface Manifest {
  phrases: Record<string, { text: string; file: string }>;
}

export class Speaker {
  private ctx: AudioContext | null = null;
  private buffers = new Map<string, AudioBuffer>();
  private current: AudioBufferSourceNode[] = [];
  private seq = 0;
  manifest: Manifest | null = null;
  enabled = true;
  rate = 1;

  constructor(private base: string) {
    try {
      this.ctx = new AudioContext();
    } catch {
      this.ctx = null;
    }
  }

  /** true bila browser sudah mengizinkan audio diputar (autoplay / setelah interaksi). */
  get unlocked(): boolean {
    return this.ctx?.state === 'running';
  }

  /** Panggil dari handler interaksi pengguna (ketukan) untuk membuka kunci audio. */
  async unlock(): Promise<void> {
    try {
      await this.ctx?.resume();
    } catch {
      /* abaikan */
    }
    if ('speechSynthesis' in self) {
      // "pemanasan" Web Speech pada interaksi pertama (diperlukan sebagian browser)
      const u = new SpeechSynthesisUtterance('');
      u.volume = 0;
      speechSynthesis.speak(u);
    }
  }

  /** Muat manifest + decode semua klip (kecil, ~100 berkas). */
  async load(onProgress?: (done: number, total: number) => void): Promise<void> {
    const res = await fetch(`${this.base}audio/manifest.json`);
    this.manifest = (await res.json()) as Manifest;
    const entries = Object.entries(this.manifest.phrases);
    let done = 0;
    const work = entries.map(async ([key, p]) => {
      try {
        const r = await fetch(`${this.base}audio/${p.file}`);
        if (!r.ok || !this.ctx) return;
        const buf = await this.ctx.decodeAudioData(await r.arrayBuffer());
        this.buffers.set(key, buf);
      } catch {
        /* klip hilang -> Web Speech */
      } finally {
        onProgress?.(++done, entries.length);
      }
    });
    await Promise.all(work);
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
    if ('speechSynthesis' in self) speechSynthesis.cancel();
  }

  /** Ucapkan rangkaian segmen. Pengumuman baru memotong yang sedang diputar. */
  async say(segments: Segment[]): Promise<void> {
    if (!this.enabled || !segments.length) return;
    this.stop();
    const my = this.seq;
    const allClips = segments.every((s) => this.buffers.has(s.key));
    if (this.ctx && this.unlocked && allClips) {
      let t = this.ctx.currentTime + 0.02;
      for (const s of segments) {
        const src = this.ctx.createBufferSource();
        src.buffer = this.buffers.get(s.key)!;
        src.playbackRate.value = this.rate;
        src.connect(this.ctx.destination);
        src.start(t);
        t += src.buffer.duration / this.rate + 0.06;
        this.current.push(src);
      }
      await new Promise((r) => setTimeout(r, (t - this.ctx!.currentTime) * 1000));
      return;
    }
    if (my !== this.seq) return;
    this.speakText(segments.map((s) => s.text).join(' '));
  }

  speakText(text: string): void {
    if (!this.enabled || !('speechSynthesis' in self)) return;
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
