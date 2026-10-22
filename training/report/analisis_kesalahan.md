### Analisis kesalahan

Kesalahan model web v1 pada set uji per kondisi (1.079 citra, ambang 0,7) terbagi menjadi dua jenis yang dampaknya
berbeda bagi pengguna tunanetra:

**Salah sebut (29 citra): ada nominal yang diumumkan tetapi tidak ada di citra.** Ini kesalahan paling berbahaya.
- 22 citra **uang asing** diumumkan sebagai rupiah dengan skor 0,70 sampai 0,92: real Brasil biru menjadi
  Rp50.000/Rp2.000, baht Thailand menjadi Rp2.000/Rp100.000, kip Laos menjadi Rp20.000. Penyebabnya, data train hanya
  memuat 258 citra uang asing sebagai negatif. Warna dan tata letak uang asing itu mirip nominal rupiah tertentu,
  sedangkan menaikkan ambang tidak efektif karena skornya tinggi.
- 3 citra **lusuh** tertukar Rp10.000 ↔ Rp1.000, pola "beda satu nol" yang juga ditemukan sebagai salah label di
  beberapa sumber data. Kemungkinan sisa label keliru di data latih ikut terpelajari.
- 3 citra **terlipat/tertutup** (Rp50.000 → Rp20.000, Rp1.000 → Rp2.000, Rp10.000 → Rp2.000): bagian pembeda (angka,
  warna utama) tertutup jari atau lipatan.
- 1 citra latar ramai ternyata **salah label di set uji**: dilabeli 2 lembar Rp10.000, padahal fotonya selembar
  Rp50.000. Model mengumumkan Rp50.000 dengan benar. Akurasi `latar_ramai` sebenarnya sedikit lebih tinggi daripada
  yang tercatat.

**Diam (37 citra): uang ada, tetapi tidak ada deteksi dengan skor ≥ 0,7.** Kesalahan ini lebih aman karena aplikasi
memberi panduan suara "uang belum terdeteksi" dan pengguna bisa menggeser kamera. Kasus terbanyak ada di latar ramai
(17 citra, umumnya uang kecil di lantai bermotif dengan kelas benar tetapi skor 0,5 sampai 0,7) dan di terlipat/tertutup
(10 citra).

**Percobaan perbaikan (ronde v2).** Fine-tune dengan 1.770 negatif tambahan, uang asing di-oversample 3x, mengurangi
salah sebut dari 29 menjadi 6, dan akurasi uang asing naik dari 0,889 ke 0,995. mAP50 pipeline web juga naik ke 0,9304.
Namun kasus diam naik dari 37 menjadi 52, terutama pada citra redup (−5,8 poin) dan terlipat (−5,0 poin). Karena
melanggar aturan adopsi yang ditetapkan sebelum hasil keluar ("kondisi lain turun ≤ 3 poin"), v2 **tidak dipakai**.
Kompromi ini menunjukkan arah perbaikan berikutnya: kalibrasi ambang per model pada split val, atau penambahan data
positif terlipat/redup bersamaan dengan negatif uang asing.
