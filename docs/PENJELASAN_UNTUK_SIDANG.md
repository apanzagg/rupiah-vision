# Penjelasan Sistem untuk Sidang

Bahasa sederhana untuk menjelaskan setiap bagian SuaraRupiah, ditambah daftar pertanyaan penguji yang mungkin muncul.
Angka hasil uji ada di `docs/LAPORAN_EVALUASI.md`; dokumen ini merujuk ke sana agar tidak ada angka ganda yang bisa berbeda.

## 1. Masalah dan solusi dalam satu paragraf

Penyandang tunanetra sulit membedakan nominal uang kertas karena ciri yang bisa diraba (blind code) cepat aus pada uang
lusuh. SuaraRupiah adalah aplikasi web yang dibuka di browser HP: kamera melihat uang, model kecerdasan buatan
(YOLOv8n) menebak nominalnya, lalu aplikasi mengucapkannya. Semua berjalan **di HP**, tanpa mengirim gambar ke server,
dan tetap berfungsi tanpa internet setelah dibuka pertama kali.

## 2. Bagian-bagian sistem

### a. Dataset
- **Apa**: kumpulan foto uang beserta kotak (bounding box) yang menandai letak dan nominal uang.
- **Dari mana**: 34 dataset publik Roboflow Universe dan 1 dataset Kaggle (lisensi CC BY 4.0, Public Domain, MIT, CC0),
  ditambah citra "bukan rupiah" (uang asing, struk, dompet, tangan, koin) agar model tidak asal menyebut nominal.
- **Pembersihan**: nama kelas yang berbeda-beda ("seratus ribu", "100k", "Rp100.000") diseragamkan jadi 7 kelas;
  foto kembar antar-dataset dihapus dengan *perceptual hash* (sidik jari visual); dataset yang memberi kotak hanya pada
  bagian uang (bukan uang utuh) dan foto lampu UV dikeluarkan setelah diperiksa visual.
- **Pembagian**: 70% latih, 15% validasi, 15% uji. Foto yang mirip selalu masuk bagian yang sama supaya model
  tidak "menyontek" (kebocoran data).

### b. Model YOLOv8n
- **YOLO** (*You Only Look Once*) melihat seluruh gambar sekali jalan dan langsung mengeluarkan kotak + kelas + skor
  keyakinan. Varian **n (nano)** dipilih karena paling kecil (sekitar 3 juta parameter) sehingga cukup cepat di HP.
- **Transfer learning**: model dimulai dari bobot yang sudah belajar mengenali objek umum (dataset COCO), lalu dilatih
  ulang khusus untuk uang rupiah.
- **Augmentasi**: saat latihan, gambar diubah-ubah secara acak (diputar bebas, dibuat buram, gelap, diberi tekstur
  kusut, ditutupi "jari") supaya model tahan kondisi nyata. **Tidak ada pencerminan (flip)** karena uang tercermin
  tidak ada di dunia nyata, dan **perubahan warna dibuat sangat kecil** karena warna adalah ciri penting nominal.

### c. Ekspor & kompresi model
- Model PyTorch diubah ke format **ONNX** agar bisa dijalankan di browser oleh **ONNX Runtime Web**.
- Ukuran diperkecil dengan **kuantisasi**: bobot 32-bit (FP32) diubah menjadi 16-bit (FP16) atau 8-bit (INT8).
  Dipilih varian terkecil yang akurasinya turun paling banyak 1 poin mAP50.
- Temuan: INT8 penuh merusak hasil karena keluaran model menggabungkan angka koordinat (0 sampai ratusan) dan skor
  (0 sampai 1) dalam satu tensor 8-bit; solusinya bagian kepala (head) tetap FP32 (**INT8 mixed**).

### d. Aplikasi web progresif (PWA)
- Dibuka seperti situs biasa, bisa dipasang ke layar utama, dan berjalan offline (service worker + Cache Storage).
- **Alur per frame**: ambil gambar kamera → perkecil ke 320×320 dengan padding (letterbox) → model → kotak-kotak
  hasil → buang kotak tumpang tindih (NMS) → logika keputusan → suara.
- **Web Worker**: perhitungan model dipisah dari tampilan agar suara dan layar tidak macet.

### e. Logika keputusan (agar tidak salah sebut)
- Nominal hanya diucapkan jika keyakinan ≥ 0,7 **dan** hasilnya sama di minimal 3 dari 5 frame terakhir.
- Nominal yang sama tidak diulang dalam 3 detik, kecuali uang sempat hilang dari kamera.
- Beberapa lembar sekaligus: diucapkan jumlah lembar, masing-masing nominal, dan totalnya.
- Panduan suara: "Uang belum terdeteksi…" setelah 4 detik tanpa uang; "Ruangan terlalu gelap" bila kecerahan rendah
  (dan senter dinyalakan bila HP mendukung).

### f. Keluaran untuk pengguna
- **Suara**: klip audio Bahasa Indonesia hasil TTS (bisa diganti rekaman manusia dengan nama berkas sama);
  cadangan Web Speech API.
- **Getar**: pola berbeda per nominal (Android).
- **Pembaca layar**: hasil ditulis ke region `aria-live` sehingga dibacakan TalkBack/VoiceOver.
- **Low vision**: kontras tinggi (hitam-kuning), teks besar, kotak tebal di sekitar uang.

## 3. Daftar pertanyaan penguji & jawaban

**T: Kenapa YOLOv8n, bukan model yang lebih besar atau model klasifikasi biasa?**
J: Harus berjalan real-time di browser HP tanpa server. YOLOv8n paling ringan di keluarga YOLOv8. Deteksi (bukan
klasifikasi) dipilih karena perlu mengenali **beberapa lembar sekaligus** dan menolak gambar tanpa uang; klasifikasi
selalu memaksa satu jawaban walau tidak ada uang.

**T: Bagaimana memastikan tidak ada kebocoran data antara data latih dan data uji?**
J: Duplikat dan foto mirip dikelompokkan dengan perceptual hash (diperiksa juga pada rotasi 90/180/270 derajat), dan
satu kelompok selalu masuk split yang sama. Salinan hasil augmentasi Roboflow dipaksa ke data latih. Split dilakukan
**sebelum** training dan tidak diubah sesudahnya.

**T: Kenapa tidak memakai flip saat augmentasi?**
J: Uang yang tercermin tidak pernah ada di dunia nyata. Flip bisa mengajarkan ciri palsu (misalnya angka terbalik).
Sebagai gantinya dipakai rotasi bebas 0 sampai 360 derajat yang memang terjadi saat uang dipegang.

**T: Apa arti mAP50 dan mAP50-95?**
J: mAP (mean Average Precision) mengukur ketepatan sekaligus kelengkapan deteksi, dirata-rata per kelas. "50" berarti
kotak dianggap benar jika tumpang tindihnya (IoU) dengan kotak label ≥ 50%. "50-95" merata-rata dari IoU 50% sampai 95%,
jadi lebih ketat terhadap ketelitian letak kotak. Untuk aplikasi ini mAP50 paling relevan karena yang penting nominalnya
benar, bukan kotaknya presisi.

**T: Apakah aplikasi bisa membedakan uang asli dan palsu?**
J: Tidak, dan ini batasan yang disengaja dan dinyatakan di aplikasi. Keaslian butuh fitur fisik (benang pengaman,
tanda air, UV) yang tidak bisa diandalkan dari kamera HP biasa.

**T: Bagaimana jika uangnya uang asing atau kertas lain?**
J: Model dilatih dengan citra negatif (uang asing, struk, dompet, tangan, koin) sekitar 10% dari data, sehingga belajar
untuk tidak mengeluarkan kotak. Ditambah syarat keyakinan ≥ 0,7 dan konsistensi 3 dari 5 frame. Hasil uji kelompok
"bukan_rupiah" ada di laporan evaluasi.

**T: Kenapa inferensi di browser, bukan di server?**
J: Privasi (gambar tidak meninggalkan HP), bisa offline, tanpa biaya server, dan tanpa jeda jaringan.

**T: Seberapa cepat di HP?**
J: Diukur di halaman `/benchmark`. Angka desktop ada di laporan; HP nyata perlu diukur langsung dengan membuka halaman
tersebut. Laju inferensi menyesuaikan kemampuan perangkat karena frame berikutnya baru diproses setelah frame sebelumnya selesai.

**T: Kenapa suara memakai klip audio, bukan langsung Web Speech API?**
J: Kualitas Web Speech berbeda-beda antar-HP dan kadang tidak ada suara Bahasa Indonesia. Klip audio konsisten, bisa
offline, dan mudah diganti rekaman manusia. Web Speech tetap dipakai sebagai cadangan.

**T: Mengapa tidak langsung bersuara saat aplikasi dibuka di iPhone?**
J: Safari (dan Chrome sebelum aplikasi dipasang ke layar utama) melarang audio sebelum ada sentuhan pengguna. Solusinya:
status juga dikirim ke pembaca layar (aria-live) yang tidak terkena larangan itu, dan sentuhan pertama di mana saja
membuka kunci audio.

**T: Bagaimana menangani lisensi dataset dan YOLOv8?**
J: Hanya dataset dengan lisensi jelas yang dipakai; yang lisensinya "Unknown" dikeluarkan. Atribusi tercatat di
`docs/SUMBER_DATASET.md`. Ultralytics YOLOv8 berlisensi AGPL-3.0, sehingga seluruh kode proyek juga AGPL-3.0 dan
kode sumbernya terbuka di GitHub (ditautkan dari halaman Bantuan).

**T: Apa kelemahan sistem ini dan rencana pengembangannya?**
J: (1) Data publik didominasi foto uang yang cukup rapi; uji lapangan dengan foto uang asli pengguna masih perlu
diperbanyak. (2) Di GitHub Pages, WASM hanya satu thread. (3) Belum diuji langsung oleh pengguna tunanetra. Rencana:
uji pengguna, menambah foto kondisi nyata, rekaman suara manusia, dan hosting dengan header COOP/COEP untuk multi-thread.
