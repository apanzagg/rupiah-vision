- **Hanya mengenali nominal, tidak memeriksa keaslian.** Uang palsu yang mirip tetap akan disebut nominalnya.
- **Ruang lingkup:** 7 nominal uang kertas emisi 2016 dan 2022. Uang logam, Rp75.000, dan emisi lama tidak dikenali
  atau belum diuji.
- **Uang asing masih bisa disebut sebagai rupiah**: 11% citra uang asing pada set uji (22 dari 199) salah diumumkan.
  Ronde v2 memperbaikinya, tetapi mengorbankan deteksi uang terlipat/redup, sehingga belum dipakai.
- **Set uji kondisi sebagian sintetis.** `terlipat_tertutup` seluruhnya dibuat dengan oklusi jari dan lipatan buatan
  pada citra test. Angka kondisi ini bukan pengganti uji lapangan.
- **Belum ada uji lapangan dengan pengguna tunanetra** dan belum ada pengukuran di ponsel Android/iPhone nyata. Semua
  angka kecepatan berasal dari desktop dan emulasi Lighthouse.
- **Kualitas label dataset publik.** Sumber Roboflow memuat salah label (pola "salah jumlah nol"). Sebagian sudah
  dikoreksi, tetapi 63 kasus di val/test dibiarkan, dan set uji per kondisi juga memuat sedikitnya satu label keliru.
- **Kinerja di browser terbatas 1 thread WASM** karena GitHub Pages tidak mendukung header COOP/COEP. Hosting dengan
  header tersebut akan mengaktifkan multi-thread.
- **Selisih protokol evaluasi:** validator Ultralytics bawaan (`rect=True`) memberi mAP50 sekitar 4 poin lebih tinggi
  daripada pipeline aplikasi untuk model yang sama. Laporan ini memakai angka pipeline aplikasi sebagai angka utama.
