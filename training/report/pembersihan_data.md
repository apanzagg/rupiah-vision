### Pembersihan data

Pool awal 69.624 citra dari 44 sumber Roboflow dan 1 dataset Kaggle menyusut menjadi 35.862 citra (data D4) melalui
empat iterasi. Alasan setiap iterasi dicatat di `docs/EKSPERIMEN.md`. Ringkasnya:

1. **Normalisasi kelas dan validasi box.** Ratusan variasi nama kelas (`10rb`, `Rp. 10.000`, `SEPULUH RIBU ASLI`, dan
   seterusnya) dipetakan ke 7 nominal. Box koin dan Rp75.000 dijadikan latar.
2. **Auto-label.** Dataset Kaggle berbentuk klasifikasi (tanpa box) diberi box oleh YOLO-World. Hanya 307 dari 1.400
   citra yang diterima: 579 tanpa box dan 514 dengan confidence rendah dibuang.
3. **Penghapusan duplikat** dengan pHash pada 4 rotasi, ditambah ciri warna dan gradien, secara serakah non-transitif.
   Foto identik dengan label berbeda dibuang sebagai konflik label (258 citra). Pemeriksaan visual menunjukkan
   sebagian besar konflik adalah salah label sungguhan berpola "salah jumlah nol", misalnya Rp1.000 ditulis 10000.
4. **Sumber dikeluarkan** karena gaya anotasinya tidak cocok dengan tugas deteksi per lembar: anotasi per-bagian uang,
   foto lampu UV, dan koin berlabel uang kertas. Box bagian yang bersarang di dalam box sekelas yang lebih besar juga
   dibuang (11.359 box).
5. **Audit berbasis model** (model e0): 65 label berpola salah-nol dikoreksi, dan 42 citra train dengan kelas meragukan
   dibuang. Sebanyak 63 kasus meragukan yang tidak sistematis di val/test sengaja **dibiarkan** agar evaluasi tidak bias
   ke model; akibatnya angka test sedikit pesimistis.
6. **Split 70/15/15 berbasis kelompok kemiripan** sehingga foto yang mirip tidak muncul sekaligus di train dan test.
   Negatif (uang asing, dompet, struk, tangan, koin) dibatasi kuota 10%.
