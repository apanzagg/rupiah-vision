# Panduan Memotret Uang Asli (Uji Lapangan)

Foto uang asli milik Anda sangat berharga karena memotret kondisi nyata yang tidak ada di dataset publik. Foto ini **utamanya dipakai untuk uji lapangan** (set uji `data/test_conditions/`), sehingga angka akurasi di laporan mencerminkan pemakaian sungguhan. Sebagian kecil dapat dipakai untuk training jika jumlahnya cukup (keputusan dicatat di `docs/PROGRESS.md`).

## Berapa banyak

| Jenis | Minimal | Ideal |
|---|---|---|
| Per nominal (1000, 2000, 5000, 10000, 20000, 50000, 100000) | 20 foto | 40 foto |
| Campuran (2 sampai 3 lembar berbeda dalam satu foto) | 10 foto | 20 foto |
| Bukan rupiah (struk, kertas, kartu, dompet kosong, tangan kosong, uang asing) | 15 foto | 30 foto |

Jika punya, campurkan uang emisi **2016** dan **2022**. Foto sisi **depan dan belakang** kurang lebih sama banyak.

## Variasi kondisi (sebarkan merata)

Untuk tiap nominal, usahakan setiap kondisi berikut terwakili minimal 3 foto:

| Kode kondisi | Cara memotret |
|---|---|
| `baru` | Uang rapi, cahaya terang, latar polos |
| `lusuh` | Uang lecek/kusut/kotor |
| `terlipat` | Uang dilipat dua atau dilipat sebagian |
| `tertutup` | Sebagian uang tertutup jari, tetapi minimal sekitar separuh permukaan masih terlihat |
| `redup` | Lampu ruangan redup/malam hari, tanpa lampu kilat |
| `ramai` | Di atas meja berantakan, kain bermotif, koran, atau di tangan dengan latar ramai |

Tips umum:

- Pegang HP seperti tunanetra akan memakainya: jarak **20 sampai 50 cm**, sudut dan rotasi bebas (miring, terbalik, menyamping). Tidak perlu sempurna.
- Pakai kamera belakang biasa. Matikan filter, mode kecantikan, dan lampu kilat.
- Tidak perlu crop atau edit. Format JPG dari kamera HP sudah cukup.
- Jangan sertakan wajah, KTP, kartu ATM, atau dokumen pribadi di dalam foto.

## Aturan penamaan file

```
<nominal>_<kondisi>_<sisi>_<nomor>.jpg
```

- `nominal`: `1000`, `2000`, `5000`, `10000`, `20000`, `50000`, `100000`
- `kondisi`: `baru`, `lusuh`, `terlipat`, `tertutup`, `redup`, `ramai`
- `sisi`: `depan` atau `belakang`
- `nomor`: dua digit, `01`, `02`, dan seterusnya

Contoh: `50000_lusuh_depan_01.jpg`, `2000_redup_belakang_03.jpg`

Foto **campuran** (beberapa lembar): tulis semua nominalnya dipisah tanda minus.

```
campuran_<nominal1>-<nominal2>[-<nominal3>]_<kondisi>_<nomor>.jpg
```

Contoh: `campuran_20000-50000_baru_01.jpg`

Foto **bukan rupiah**:

```
bukan_<jenis>_<nomor>.jpg
```

Contoh: `bukan_struk_01.jpg`, `bukan_dollar_02.jpg`, `bukan_tangan_01.jpg`

## Menyerahkan foto

1. Salin semua foto ke folder `data/raw/foto_saya/` di dalam proyek (folder ini tidak di-commit ke git, jadi foto Anda tetap pribadi).
2. Beri tahu saya. Saya akan menjalankan skrip `training/scripts/ingest_user_photos.py` yang:
   - memeriksa nama file dan melaporkan nama yang tidak sesuai aturan,
   - mengoreksi orientasi foto (EXIF) dan mengecilkan resolusinya,
   - membuat bounding box otomatis lalu contact sheet untuk diperiksa,
   - memasukkan foto ke set uji per kondisi.
