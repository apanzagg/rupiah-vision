// Salin aset besar dari folder repo ke web/public sebelum dev/build (agar tidak tersimpan dua kali di git):
//   ../models/<model aktif & varian>  -> public/models/   (+ variants.json untuk halaman benchmark)
//   ../models/parity/*                -> public/parity/   (citra uji + referensi Python, untuk parity test)
//   ort-wasm-simd-threaded.wasm(.gz)  -> public/ort/      (hanya bila VITE_WASM_GZ=1)
import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync } from 'node:zlib';

const web = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoModels = resolve(web, '..', 'models');
const pub = join(web, 'public');
const cfg = JSON.parse(readFileSync(join(web, 'src', 'model-config.json'), 'utf8'));

for (const d of ['models', 'parity', 'ort']) rmSync(join(pub, d), { recursive: true, force: true });
mkdirSync(join(pub, 'models'), { recursive: true });

const variants = [];
if (existsSync(repoModels)) {
  const vjson = join(repoModels, 'variants.json');
  const meta = existsSync(vjson) ? JSON.parse(readFileSync(vjson, 'utf8')).variants ?? {} : {};
  for (const f of readdirSync(repoModels).filter((f) => f.endsWith('.onnx'))) {
    copyFileSync(join(repoModels, f), join(pub, 'models', f));
    const key = Object.keys(meta).find((k) => meta[k].file === f);
    variants.push({ file: f, variant: key ?? f, size_mb: +(statSync(join(repoModels, f)).size / 1e6).toFixed(2), map50: key ? meta[key].map50 : undefined });
  }
  if (existsSync(join(repoModels, 'parity'))) {
    mkdirSync(join(pub, 'parity'), { recursive: true });
    for (const f of readdirSync(join(repoModels, 'parity'))) copyFileSync(join(repoModels, 'parity', f), join(pub, 'parity', f));
  }
}
writeFileSync(join(pub, 'models', 'variants.json'), JSON.stringify(variants, null, 1));
if (!variants.some((v) => v.file === cfg.file)) console.warn(`[prepare-assets] PERINGATAN: model aktif ${cfg.file} belum ada di ../models`);

if (process.env.VITE_WASM_GZ) {
  mkdirSync(join(pub, 'ort'), { recursive: true });
  const wasm = readFileSync(join(web, 'node_modules', 'onnxruntime-web', 'dist', 'ort-wasm-simd-threaded.wasm'));
  writeFileSync(join(pub, 'ort', 'ort-wasm-simd-threaded.wasm.gz'), gzipSync(wasm, { level: 9 }));
}
console.log(`[prepare-assets] ${variants.length} model disalin ke public/models`);
