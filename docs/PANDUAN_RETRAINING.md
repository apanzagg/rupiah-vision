# Panduan Retraining: Menambah Data, Melatih Ulang, Mengganti Model

Seluruh langkah berbasis skrip dan bisa diulang. Training berjalan di Kaggle (GPU gratis 30 jam/minggu);
komputer lokal hanya untuk mengirim job, memeriksa hasil, dan memasang model.

## 0. Persiapan (sekali)

```bash
# Python 3.12
python3.12 -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/Mac: .venv/bin/pip
cp .env.example .env                               # isi ROBOFLOW_API_KEY dan KAGGLE_API_TOKEN
```

- Token Kaggle: Kaggle → Settings → API → buat token; simpan di `~/.kaggle/access_token` atau `.env`.
  Akun Kaggle **wajib terverifikasi nomor HP** agar GPU & internet di kernel aktif.
- Ganti `achmadapanza` di `training/kaggle/*/kernel-metadata.json` dengan username Kaggle Anda.

## 1. Menambah data

### a. Dataset Roboflow Universe baru

1. Tambahkan slug ke `training/configs/roboflow_candidates.txt`, lalu survei metadata & lisensinya:
   ```bash
   cd training/scripts && ../../.venv/Scripts/python.exe rf_survey.py
   ```
2. Jika lisensinya jelas (CC BY 4.0 / Public Domain / MIT / CC0) dan ada versi tanpa augmentasi flip,
   tambahkan entri ke `training/configs/sources.yaml` (`slug`, `version`, `role`, `license`, `aug`).
3. Periksa pemetaan nama kelasnya: `python class_map.py` (semua nama harus terpetakan ke 7 kelas,
   `DROP_BOX` untuk koin/Rp75.000, atau `DROP_IMAGE` untuk kelas tanpa nominal). Tambahkan override di
   `class_map.py: OVERRIDES` bila ada nama yang ambigu.

### b. Foto sendiri

Ikuti `docs/PANDUAN_FOTO.md`, salin ke `data/raw/foto_saya/`, lalu:
```bash
python ingest_user_photos.py --model ../../models/model-v1-<varian>.onnx
```
Foto masuk ke set uji `data/test_conditions/foto_saya_<kondisi>/` (dievaluasi terpisah). Agar foto ikut
training, unggah sebagai dataset Kaggle privat dan tambahkan ke `sources.yaml` bagian `kaggle`.

## 2. Membangun dataset

```bash
cd training/scripts
../../.venv/Scripts/python.exe rf_links.py                         # link ekspor Roboflow (API key tetap lokal)
cd ../.. && .venv/Scripts/python.exe training/kaggle/bundle.py data --links --push --accelerator NvidiaTeslaT4
# setelah selesai (+/- 30 menit):
.venv/Scripts/python.exe training/kaggle/bundle.py build --push     # dedup + split + laporan (CPU)
.venv/Scripts/kaggle kernels output <user>/suararupiah-build -p training/kaggle_out/build --file-pattern "reports\.zip$"
```

Periksa `reports.zip`:
- `reports/sheets/crops_<kelas>.jpg`: tiap sumber harus menampilkan uang dengan nominal yang benar.
- `reports/sheets/sources_*.jpg` + `source_audit.json`: sumber dengan `nest_rate` tinggi = anotasi per-bagian uang → keluarkan.
- `reports/sheets/conflicts_*.jpg`: pasangan duplikat berlabel berbeda.
- `reports/sheets/autolabel_*.jpg`: hasil auto-label.

Keputusan pembersihan dicatat di `training/configs/exclude.yaml` (sumber) atau `training/configs/reject_ids.txt`
(id citra, satu per baris, boleh diberi komentar `# alasan`). Jalankan ulang kernel `build` setelah mengubahnya.

## 3. Training

1. Atur percobaan di `training/configs/train_job.yaml` (nama run, `imgsz`, `epochs`, `patience`, `batch`, override `set`).
2. Kirim:
   ```bash
   .venv/Scripts/python.exe training/kaggle/bundle.py train --push --accelerator NvidiaTeslaT4
   ```
3. Ambil hasil (tanpa test.tar agar cepat):
   ```bash
   .venv/Scripts/kaggle kernels output <user>/suararupiah-train -p training/kaggle_out/train --file-pattern "^(?!test\.tar).*"
   ```
4. Catat setiap run di `docs/EKSPERIMEN.md` (perubahan, mAP50, mAP50-95, catatan). Angka diambil dari
   `runs/<run>/metrics.json` dan `export/<run>/variants.json`, jangan diketik dari ingatan.

## 4. Mengganti model di web

```bash
cd training/scripts
../../.venv/Scripts/python.exe install_model.py --run-dir ../kaggle_out/train/runs/<run> --export-dir ../kaggle_out/train/export/<run> --version v2
```

Skrip ini:
- menyalin varian terpilih ke `models/model-<versi>-<varian>.onnx` (hapus model lama),
- menulis `models/MODEL_CARD.md`, `models/variants.json`, `models/parity/` (citra + output referensi Python),
- memperbarui `web/src/model-config.json` (nama berkas, versi, ukuran, SHA-256).

**Naikkan versi** (`--version v2`, `v3`, …) setiap mengganti model: nama cache di browser memakai versi,
sehingga pengguna otomatis mengunduh model baru dan cache lama dihapus.

Lalu uji dan deploy:
```bash
cd web
npm test                      # unit test
npm run build                 # salin model ke public/, build PWA
npx playwright test           # parity + E2E (perlu: python tests/e2e/make_fake_videos.py --model ...)
git add -A && git commit -m "Model v2" && git push   # GitHub Actions men-deploy ke GitHub Pages
```

## 5. Mengganti suara dengan rekaman manusia

Klip ada di `web/public/audio/<kunci>.mp3`; teks tiap kunci ada di `web/public/audio/manifest.json`.
Rekam ulang dengan **nama berkas sama** (MP3 mono, disarankan 22,05 kHz, awal/akhir tanpa jeda panjang).
Kunci yang tidak ada berkasnya otomatis diucapkan dengan Web Speech API.
Untuk membuat ulang klip TTS: `bundle.py tts --extra "web/public/audio/manifest.json=manifest.json" --push --accelerator NvidiaTeslaT4`.
