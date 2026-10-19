# SuaraRupiah

Aplikasi web progresif (PWA) pengenal nominal uang kertas Rupiah dengan keluaran suara, untuk penyandang tunanetra dan low vision. Deteksi memakai model **YOLOv8n** yang berjalan **sepenuhnya di browser** (on-device). Tidak ada gambar kamera yang dikirim ke server.

> Proyek capstone: *Rancang Bangun Aplikasi Web Progresif Pengenal Nominal Uang Kertas Rupiah dengan Keluaran Suara Menggunakan YOLOv8 untuk Penyandang Tunanetra*.

**Status:** dalam pengembangan. Lihat [docs/PROGRESS.md](docs/PROGRESS.md).

## Ruang lingkup

- 7 nominal uang kertas: Rp1.000, Rp2.000, Rp5.000, Rp10.000, Rp20.000, Rp50.000, Rp100.000 (Tahun Emisi 2016 dan 2022, sisi depan dan belakang).
- Hanya mengenali nominal. Sistem ini **tidak** memeriksa keaslian uang.
- Uang logam dan uang peringatan Rp75.000 tidak termasuk.

## Struktur folder

```
rupiah-vision/
├── data/        # (tidak di-commit) raw/, interim/, final/, test_conditions/
├── training/    # scripts/, notebooks/, configs/, kaggle/
├── models/      # model final untuk web + MODEL_CARD.md
├── web/         # aplikasi PWA (Vite + TypeScript)
├── docs/        # laporan, panduan, sumber dataset, progres
└── LICENSE      # AGPL-3.0
```

## Lisensi

Kode berlisensi [GNU AGPL-3.0](LICENSE), karena memakai Ultralytics YOLOv8 (AGPL-3.0). Dataset memiliki lisensi masing-masing, tercatat di [docs/SUMBER_DATASET.md](docs/SUMBER_DATASET.md).
