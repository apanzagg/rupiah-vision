import './styles.css';
import { BASE, MODEL, SOURCE_URL } from './app/assets';
import { loadSettings, saveSettings } from './app/settings';
import { Speaker } from './app/speaker';

const s = loadSettings();
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const fmt = (v: number, d: number) => v.toFixed(d).replace('.', ',');

const thr = $<HTMLInputElement>('thr');
const rate = $<HTMLInputElement>('rate');
const voice = $<HTMLInputElement>('voice');
const vibrate = $<HTMLInputElement>('vibrate');
const saved = $<HTMLParagraphElement>('saved');
thr.value = String(s.threshold);
rate.value = String(s.rate);
voice.checked = s.voice;
vibrate.checked = s.vibrate;
$('thr-out').textContent = fmt(s.threshold, 2);
$('rate-out').textContent = fmt(s.rate, 1);

$('settings').addEventListener('input', () => {
  s.threshold = Number(thr.value);
  s.rate = Number(rate.value);
  s.voice = voice.checked;
  s.vibrate = vibrate.checked;
  $('thr-out').textContent = fmt(s.threshold, 2);
  $('rate-out').textContent = fmt(s.rate, 1);
  saveSettings(s);
  saved.textContent = 'Pengaturan disimpan.';
});

$<HTMLAnchorElement>('src-link').href = SOURCE_URL;
$<HTMLAnchorElement>('src-datasets').href = `${SOURCE_URL}/blob/main/docs/SUMBER_DATASET.md`;
$('model-info').textContent = `Versi model: ${MODEL.version} (${MODEL.variant}, input ${MODEL.imgsz}x${MODEL.imgsz} piksel)`;

// speaker dibuat saat tombol diketuk: tidak ada AudioContext/unduhan audio sebelum dibutuhkan
let speaker: Speaker | null = null;
$('play-help').addEventListener('click', async () => {
  if (!speaker) {
    speaker = new Speaker(BASE);
    speaker.rate = s.rate;
    await speaker.unlock();
    await speaker.load().catch(() => undefined);
  }
  await speaker.unlock();
  speaker.enabled = true;
  void speaker.say([{ key: 'help', text: speaker.text('help') }]);
});
