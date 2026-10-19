// Ukur total unduhan pertama dari hasil build (dist/): ukuran mentah & gzip per kelompok aset.
// Yang diunduh saat kunjungan pertama: HTML/JS/CSS halaman utama + worker, model ONNX aktif, runtime WASM,
// manifest & klip audio, ikon, service worker (+ precache, yang sebagian besar sama dengan berkas di atas).
import { readFileSync, readdirSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { join, relative, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(web, 'dist');
const cfg = JSON.parse(readFileSync(join(web, 'src', 'model-config.json'), 'utf8'));
const walk = (d) => readdirSync(d).flatMap((f) => (statSync(join(d, f)).isDirectory() ? walk(join(d, f)) : [join(d, f)]));
const groups = {};
for (const f of walk(dist)) {
  const rel = relative(dist, f).replaceAll('\\', '/');
  let g;
  if (rel.startsWith('models/')) {
    if (!rel.endsWith(cfg.file)) continue; // hanya model aktif yang diunduh aplikasi
    g = 'Model ONNX';
  } else if (rel.endsWith('.wasm') || rel.endsWith('.wasm.gz')) g = 'Runtime ONNX (WASM)';
  else if (rel.startsWith('audio/')) g = 'Audio (klip + manifest)';
  else if (rel.startsWith('icons/')) g = 'Ikon';
  else if (rel.startsWith('bench/') || rel.startsWith('parity/') || rel.startsWith('benchmark/') || rel.startsWith('about/')) continue;
  else g = 'HTML/JS/CSS/service worker';
  const raw = readFileSync(f);
  const gz = rel.endsWith('.gz') ? raw.length : gzipSync(raw, { level: 6 }).length;
  groups[g] ??= { raw: 0, gzip: 0 };
  groups[g].raw += rel.endsWith('.gz') ? 0 : raw.length;
  groups[g].gzip += gz;
}
const mb = (b) => `${(b / 1e6).toFixed(2)} MB`;
const out = {};
let raw = 0;
let gz = 0;
for (const [k, v] of Object.entries(groups)) {
  out[k] = `${mb(v.raw)} mentah / ${mb(v.gzip)} gzip`;
  raw += v.raw;
  gz += v.gzip;
}
out['TOTAL unduhan pertama'] = `${mb(raw)} mentah / ${mb(gz)} gzip`;
mkdirSync(join(web, 'test-results'), { recursive: true });
writeFileSync(join(web, 'test-results', 'download_sizes.json'), JSON.stringify(out, null, 1));
console.log(out);
