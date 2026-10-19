# SuaraRupiah

Aplikasi web progresif (PWA) pengenal nominal uang kertas Rupiah dengan keluaran suara, untuk penyandang tunanetra dan
low vision. Deteksi memakai **YOLOv8n** yang berjalan **sepenuhnya di browser** (ONNX Runtime Web, WASM). Tidak ada
gambar kamera yang dikirim ke server, dan aplikasi tetap berfungsi **offline** setelah kunjungan pertama.

> Proyek capstone: *Rancang Bangun Aplikasi Web Progresif Pengenal Nominal Uang Kertas Rupiah dengan Keluaran Suara
> Menggunakan YOLOv8 untuk Penyandang Tunanetra*.

**Demo:** https://apanzagg.github.io/rupiah-vision/ · Benchmark perangkat: https://apanzagg.github.io/rupiah-vision/benchmark/

## Fitur

- Kamera belakang aktif otomatis; tanpa tombol yang wajib ditekan.
- Mengumumkan nominal dengan suara Bahasa Indonesia (klip TTS, cadangan Web Speech API), getaran berpola per nominal,
  dan teks `aria-live` untuk TalkBack/VoiceOver. Ketuk layar di mana saja untuk mengulang.
- Beberapa lembar sekaligus: jumlah lembar, nominal masing-masing, dan total.
- Anti salah sebut: keyakinan ≥ 0,7 dan konsisten ≥ 3 dari 5 frame; nominal sama tidak diulang dalam 3 detik.
- Panduan suara saat uang tidak terdeteksi atau ruangan terlalu gelap (senter dinyalakan bila didukung).
- Kontras tinggi, teks besar, halaman Bantuan (juga versi audio), pengaturan ambang/suara/getar.

## Ruang lingkup & batasan

- 7 nominal uang kertas: Rp1.000, Rp2.000, Rp5.000, Rp10.000, Rp20.000, Rp50.000, Rp100.000 (TE 2016 & 2022, depan/belakang).
- **Hanya mengenali nominal, tidak memeriksa keaslian.** Uang logam dan Rp75.000 tidak dikenali.
- Hasil evaluasi lengkap (termasuk per kondisi: lusuh, terlipat/tertutup, redup, latar ramai, bukan rupiah):
  [docs/LAPORAN_EVALUASI.md](docs/LAPORAN_EVALUASI.md).

## Menjalankan secara lokal

Prasyarat: Node.js 24+, Python 3.12 (hanya untuk pipeline data/training), Google Chrome (untuk uji E2E).

```bash
cd web
npm ci
npm run dev          # http://localhost:5173 (kamera butuh localhost atau HTTPS)
npm test             # unit test (Vitest)
npm run build        # build produksi ke web/dist (menyalin model dari ../models)
npx playwright test  # parity + E2E (buat video kamera palsu dulu: python tests/e2e/make_fake_videos.py --model ...)
```

Pipeline data & training (berjalan di Kaggle GPU): lihat [docs/PANDUAN_RETRAINING.md](docs/PANDUAN_RETRAINING.md).

## Struktur folder

```
rupiah-vision/
├── data/            # (tidak di-commit) raw/, interim/, final/, test_conditions/
├── training/
│   ├── scripts/     # pipeline data, training, ekspor ONNX, evaluasi, laporan
│   ├── configs/     # sumber dataset, pembersihan data, job training
│   └── kaggle/      # kernel Kaggle (data, build, train, audit, tts) + bundler
├── models/          # model final untuk web + MODEL_CARD.md + data parity test
├── web/             # PWA (Vite + TypeScript + onnxruntime-web)
├── docs/            # laporan, arsitektur, panduan, sumber dataset, progres
└── LICENSE          # GNU AGPL-3.0
```

## Dokumentasi

| Dokumen | Isi |
|---|---|
| [LAPORAN_EVALUASI.md](docs/LAPORAN_EVALUASI.md) | Dataset, training, metrik, uji per kondisi, FP32/FP16/INT8, performa web (Bab 4) |
| [ARSITEKTUR.md](docs/ARSITEKTUR.md) | Diagram arsitektur & alur data |
| [PENJELASAN_UNTUK_SIDANG.md](docs/PENJELASAN_UNTUK_SIDANG.md) | Penjelasan sederhana + daftar pertanyaan penguji |
| [PANDUAN_PENGGUNA.md](docs/PANDUAN_PENGGUNA.md) | Cara memasang & memakai aplikasi |
| [PANDUAN_RETRAINING.md](docs/PANDUAN_RETRAINING.md) | Menambah data, melatih ulang, mengganti model |
| [SUMBER_DATASET.md](docs/SUMBER_DATASET.md) | Sumber, lisensi & atribusi dataset |
| [EKSPERIMEN.md](docs/EKSPERIMEN.md) | Riwayat iterasi data & training |
| [PROGRESS.md](docs/PROGRESS.md) | Catatan progres & keputusan teknis |
| [models/MODEL_CARD.md](models/MODEL_CARD.md) | Spesifikasi model web |

## Lisensi

Kode berlisensi [GNU AGPL-3.0](LICENSE), karena memakai Ultralytics YOLOv8 (AGPL-3.0). Dataset memiliki lisensi
masing-masing (CC BY 4.0, Public Domain, MIT, CC0) yang tercatat di [docs/SUMBER_DATASET.md](docs/SUMBER_DATASET.md).
Klip suara dibuat dengan Piper TTS; lihat [web/public/audio/KREDIT.md](web/public/audio/KREDIT.md).
