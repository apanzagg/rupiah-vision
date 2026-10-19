# Progres Proyek SuaraRupiah

Catatan kerja per fase. Jika sesi terputus, lanjutkan dari bagian **Langkah berikutnya** paling bawah.

## Lingkungan (dicek 2026-10-19)

| Komponen | Nilai |
|---|---|
| OS | Windows 10 Pro 10.0.19045, 64-bit |
| CPU / RAM | Intel i7-4770S (4 core, 8 thread) / 16 GB |
| GPU lokal | NVIDIA GeForce GT 710, 2 GB, driver 475.14 (CUDA 11.4), arsitektur Kepler (compute capability 3.5) |
| Python | 3.12.15 di `.venv` (bawaan sistem 3.14.4 terlalu baru untuk PyTorch/Ultralytics) |
| Node.js / npm | v24.15.0 / 11.12.1 |
| git / gh | 2.53.0 / GitHub CLI 2.x, login sebagai `apanzagg` |
| Kaggle | CLI 2.2.4, user `achmadapanza`, kuota GPU 30 jam/minggu |
| Koneksi internet | sekitar 0,1 sampai 0,5 MB/s, kadang naik sesaat sampai 3 MB/s |

## Fase 0 sampai 1: Setup (selesai)

**Selesai**

- Struktur folder, `git init`, `.gitignore` (data, bobot, kredensial), `.env` (di-gitignore) untuk `ROBOFLOW_API_KEY` dan `KAGGLE_API_TOKEN`.
- Venv Python 3.12 dengan dependensi terkunci di `requirements.txt` (hasil `pip freeze`). Daftar tingkat atas ada di `training/requirements.in`.
- Kredensial Roboflow, Kaggle, dan GitHub sudah diverifikasi lewat API.
- `LICENSE` AGPL-3.0, `README.md`, `docs/PANDUAN_FOTO.md`.

**Keputusan dan alasannya**

| Keputusan | Alasan |
|---|---|
| Training di Kaggle GPU, bukan GPU lokal | GT 710 (Kepler, sm_35) tidak didukung PyTorch versi sekarang |
| Kaggle, bukan Colab | Kaggle bisa dijalankan penuh lewat API (`kernels push/status/output`), berjalan di latar belakang, dan kuotanya jelas (30 jam/minggu). Colab tidak punya API resmi untuk menjalankan notebook |
| Python 3.12 | Versi stabil yang didukung torch 2.7 dan ultralytics 8.3.x |
| `ultralytics==8.3.253` | Rilis 8.3.x terakhir, sesuai syarat prompt |
| Pemrosesan data berat di Kaggle | Koneksi lokal sangat lambat. Unduh sekitar 3 GB akan memakan 6 sampai 7 jam dan masih harus diunggah ulang ke Kaggle. Server Kaggle mengunduh langsung dari Roboflow memakai link ekspor sementara (API key tidak dikirim) |
| Kernel Kaggle memakai `--accelerator NvidiaTeslaT4` | Kaggle CLI 2.x mengabaikan `enable_gpu` di metadata tanpa opsi ini |

**Masalah yang ditemui**

- Kernel uji Kaggle berjalan tanpa GPU dan tanpa internet (DNS gagal). Penyebab paling mungkin: nomor HP akun Kaggle belum diverifikasi. Sudah diminta ke pemilik akun.

## Fase 2: Dataset (berjalan)

**Selesai**

- Survei Roboflow Universe dengan kata kunci "rupiah", "uang rupiah", "indonesian banknote", "indonesian currency": sekitar 270 proyek object detection ditemukan, 71 kandidat disurvei lewat API (`training/scripts/rf_survey.py`). Hasilnya di `data/interim/rf_survey.csv`: lisensi, jumlah citra, kelas, versi, dan augmentasi.
- Survei Kaggle lewat API: 57 dataset relevan, dipilih yang lisensinya jelas.
- Sumber terpilih dicatat di `training/configs/sources.yaml`, alasan pemilihan dan pengecualian di `docs/SUMBER_DATASET.md`.

## Langkah berikutnya

1. Tunggu verifikasi HP Kaggle, lalu ulangi `training/kaggle/probe` untuk memastikan GPU dan internet aktif.
2. Tulis pipeline data (normalisasi kelas, dedup pHash, split berbasis grup, citra negatif) dan uji pada sampel lokal.
3. Jalankan pipeline penuh di Kaggle, unduh contact sheet untuk pemeriksaan visual.
