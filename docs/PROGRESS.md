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

## Langkah berikutnya

1. Ambil hasil training ronde 1 (`e0_r320_audit`) → tinjau `audit.zip` → isi `training/configs/label_fixes.csv` dan/atau
   `force_train_sources` di `exclude.yaml` → push kernel `build`.
2. Training final: `train_job.yaml` dengan `e1_r320` & `e1_r416` (80 epoch, paralel, export + save_test).
3. `install_model.py` → `make_test_conditions.py` → `eval_conditions.py` → `make_fake_videos.py` → Playwright → deploy → `make_report.py`.
