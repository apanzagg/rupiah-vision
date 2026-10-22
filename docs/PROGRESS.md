# Progres Proyek SuaraRupiah

Catatan kerja per fase. Jika sesi terputus, lanjutkan dari bagian **Langkah berikutnya** paling bawah.

## Lingkungan (dicek 2026-10-19)

| Komponen | Nilai |
|---|---|
| OS | Windows 10 Pro 10.0.19045, 64-bit |
| CPU / RAM | Intel i7-4770S (4 core, 8 thread) / 16 GB |
| GPU lokal | NVIDIA GeForce GT 710, 2 GB, driver 475.14 (CUDA 11.4), Kepler (compute 3.5): tidak didukung PyTorch modern |
| Python | 3.12.15 di `.venv` (bawaan sistem 3.14.4 terlalu baru untuk PyTorch/Ultralytics) |
| Node.js / npm | v24.15.0 / 11.12.1; Google Chrome terpasang (dipakai Playwright, `channel: 'chrome'`) |
| git / gh | 2.53.0 / GitHub CLI, akun `apanzagg`; repo https://github.com/apanzagg/rupiah-vision |
| Kaggle | CLI 2.2.4, akun `achmadapanza` (HP terverifikasi), GPU T4 x2, kuota 30 jam/minggu; **hanya 1 sesi kernel berjalan bersamaan** |
| Koneksi internet lokal | sekitar 0,1 sampai 0,5 MB/s |

## Fase 0 sampai 1: Setup (selesai)

- Struktur folder, `.gitignore`, `.env` (kredensial, di-gitignore), venv + `requirements.txt` terkunci, LICENSE AGPL-3.0, repo publik.

| Keputusan | Alasan |
|---|---|
| Training & pemrosesan data di Kaggle | GPU lokal tidak didukung; koneksi lokal terlalu lambat untuk ~3 GB data. Kernel Kaggle mengunduh langsung dari Roboflow memakai link ekspor sementara (API key tetap lokal) |
| Kaggle, bukan Colab | Bisa dijalankan penuh lewat API, berjalan di latar belakang, kuota jelas |
| `ultralytics==8.3.253` | Rilis 8.3.x terakhir |
| `--accelerator NvidiaTeslaT4` | Kaggle CLI 2.x mengabaikan `enable_gpu` tanpa opsi ini |

## Fase 2: Dataset (selesai, ronde audit label berjalan)

- Survei 71 kandidat Roboflow + 57 Kaggle; 44 sumber Roboflow + 1 Kaggle terpilih (`training/configs/sources.yaml`, `docs/SUMBER_DATASET.md`).
- Pipeline: `ingest.py` → `autolabel.py` (YOLO-World) → `dedup_split.py` → `analyze.py`/`audit_sources.py`/`diag_dedup.py`.
- Iterasi data D1 sampai D3 dan alasannya: `docs/EKSPERIMEN.md`. Dataset D3: 38.430 citra (train 27.157 / val 5.812 / test 5.461).
- Auto-label YOLO-World untuk dataset Kaggle klasifikasi (emisi 2022) lemah: 307 dari 1.400 diterima (579 tanpa box, 514 confidence rendah).
- Masalah ditemukan & ditangani: sumber berlabel per-bagian uang, foto lampu UV, salah label "jumlah nol", koin Rp1.000 berlabel uang kertas.

## Fase 3 sampai 4: Training & ekspor (berjalan)

- `train.py` (augmentasi kustom, tanpa flip, early stopping, batas waktu), `export_onnx.py` (FP32/FP16/INT8/INT8-mixed + evaluasi), `infer_ref.py`.
- Temuan uji lokal: INT8 penuh merusak skor kelas (output gabungan box+skor); FP16 butuh `Resize` tetap FP32; INT8 per-channel butuh opset 13.

## Fase 5: Aplikasi web (sebagian besar selesai)

- PWA Vite + TypeScript + onnxruntime-web 1.30.0 (WASM, Web Worker), kamera, suara (109 klip Piper TTS, cadangan Web Speech), getar, panduan, senter, halaman Bantuan & Benchmark.
- Smoke test Chrome dengan kamera palsu: siap, inferensi berjalan, **offline setelah reload berhasil**.
- Suara: Piper `id_ID-news_tts-medium` dipilih (CER Whisper 0,9% vs MMS-TTS 4,3%).

## Fase 6 sampai 8 (disiapkan)

- 64 unit test lulus; parity & E2E Playwright siap; skrip uji kondisi, pengukur ukuran, generator laporan.
- GitHub Actions deploy ke Pages (menunggu model); sumber Pages sudah diubah ke "GitHub Actions".

## Fase 3 sampai 4: hasil (selesai)

- Training final e1_r320 & e1_r416 (data D4). Model web v1 = e1_r320 FP16 (6,1 MB): mAP50 pipeline web 0,9154 (target 0,90 tercapai).
  Terpasang di `models/` (`install_model.py`), belum di-commit saat catatan ini dibuat.
- Set uji per kondisi dibangun (`data/test_conditions/`, tidak di-commit) dan dievaluasi v1 (`data/interim/conditions_eval/`):
  bukan_rupiah hanya 0,889 (uang asing disebut rupiah) -> ronde v2 hard negative (lihat `docs/EKSPERIMEN.md`).

## Langkah berikutnya (diperbarui 2026-10-22 WIB)

Status: **BELUM SELESAI**. Sesi otomatis yang melanjutkan: kerjakan berurutan, centang di sini setelah selesai.

- [x] A. Kernel Kaggle `achmadapanza/suararupiah-build` versi 6 (v2, `dedup_split.py --extra-neg`). Setelah COMPLETE: unduh output ke
      `training/kaggle_out/build_v2/` (`kaggle kernels output achmadapanza/suararupiah-build -p training/kaggle_out/build_v2`, cd dulu ke folder),
      pastikan id val/test di `manifest_final.csv` identik dengan `training/kaggle_out/build/reports/manifest_final.csv`.
      Selesai: D5 train 28.231 / val 4.858 / test 4.543, id val/test identik dengan D4.
- [x] B. Push kernel train v2: `PYTHONUTF8=1 .venv/Scripts/python.exe training/kaggle/bundle.py train --push --accelerator NvidiaTeslaT4`
      (job `training/configs/train_job.yaml` = e2_r320_hn, ~1,5 jam). Unduh output ke `training/kaggle_out/train_v2/`.
      Selesai: kernel train versi 3, 40 epoch, 1,34 jam.
- [x] C. Evaluasi v2: `eval_onnx_map.py` (FP32 & FP16 v2, root `data/final_test/final`) dan `eval_conditions.py --model <fp16 v2> --out data/interim/conditions_eval_v2`.
      Terapkan **aturan adopsi** di `docs/EKSPERIMEN.md`. Jika v2 lolos: `install_model.py --run-dir ... --export-dir ... --version v2 --web-map ...`
      lalu `eval_conditions.py` ulang ke `data/interim/conditions_eval`. Catat hasil di EKSPERIMEN.md (lolos atau tidak).
      Selesai: v2 **tidak lolos** syarat (3) (terlipat −5,0, redup −5,8 poin). v1 dipertahankan; hasil di EKSPERIMEN.md.
- [x] D. Web: `cd web && npm test && npm run build`; `python tests/e2e/make_fake_videos.py --model ../models/<model>.onnx`; `npx playwright test`
      (parity, announce, pwa). Benchmark desktop -> `web/test-results/bench_desktop.json`; `node scripts/measure-sizes.mjs`.
      Selesai: 64 unit test, build, 13/13 Playwright lulus; ukuran unduhan pertama 10,81 MB gzip. Benchmark desktop 20 Okt tetap berlaku (model sama).
- [ ] E. Commit `models/` + skrip + docs (JANGAN commit data/, kredensial, bobot .pt) lalu push ke `main` (memicu deploy GitHub Pages).
      Verifikasi https://apanzagg.github.io/rupiah-vision/ (halaman, model termuat, offline). Lighthouse dari URL publik -> `web/test-results/lighthouse.json`.
- [ ] F. Laporan: isi `training/report/*.md`, `make_sources_doc.py`, `make_report.py --run <run>` -> `docs/LAPORAN_EVALUASI.md`;
      perbarui EKSPERIMEN.md, README, MODEL_CARD, ARSITEKTUR/PENJELASAN bila angka berubah. Commit & push.
- [ ] G. Ringkasan akhir untuk pengguna (Bahasa Indonesia): URL, target tercapai/tidak (angka nyata), yang harus dilakukan pengguna
      (uji di Android/iPhone sendiri, foto sendiri sesuai PANDUAN_FOTO, blob data lama di riwayat git commit d69ca74).
      Setelah selesai ubah status di atas menjadi **SELESAI** dan hapus tugas terjadwal `lanjutkan-suararupiah`.
