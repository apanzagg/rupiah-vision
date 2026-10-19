# Catatan Eksperimen

Setiap angka di bawah berasal dari eksekusi nyata: `runs/<run>/metrics.json` (PyTorch, Ultralytics val) dan
`export/<run>/variants.json` (ONNX) yang diunduh dari kernel Kaggle `suararupiah-train` ke `training/kaggle_out/train/`.
mAP dihitung pada **split test** (tidak dipakai untuk training maupun pemilihan model), kecuali disebut lain.

## Iterasi data (sebelum training)

| Iterasi | Perubahan | Hasil |
|---|---|---|
| D1 | Pool awal dari 44 sumber; dedup union-find transitif (pHash + warna + gradien) | 69.624 citra di pool, 38.250 setelah dedup; 5.299 citra terbuang sebagai "konflik label" (824 klaster) |
| D2 | Analisis konflik: mayoritas hanya beda **jumlah** box untuk foto sama, diperparah rantai union-find (klaster hingga 153 citra). Dedup diganti serakah **non-transitif**; beda jumlah box bukan konflik (anotasi per lembar dipertahankan) | lihat bawah |
| D3 | Pemeriksaan visual crop per sumber: `malik-project/deteksi-uang-rupiah-50fx4` memberi box pada **bagian** uang (angka/potret/sudut), dikeluarkan; foto lampu UV (`bima-dcfqn/*`) disaring | lihat bawah |

## Training

| Run | Perubahan | mAP50 (test) | mAP50-95 (test) | Catatan |
|---|---|---|---|---|
