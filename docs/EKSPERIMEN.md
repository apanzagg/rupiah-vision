# Catatan Eksperimen

Setiap angka di bawah berasal dari eksekusi nyata: `dedup_report.json` (kernel `suararupiah-data`/`suararupiah-build`),
`runs/<run>/metrics.json` (PyTorch, Ultralytics val) dan `export/<run>/variants.json` (ONNX) yang diunduh dari kernel
Kaggle ke `training/kaggle_out/`. mAP dihitung pada **split test** (tidak dipakai untuk training maupun pemilihan model),
kecuali disebut lain.

## Iterasi data (sebelum training)

Pool awal: **69.624 citra** dari 44 sumber Roboflow + 1 dataset Kaggle (setelah normalisasi kelas & validasi box).

| Iterasi | Perubahan | Citra final | Konflik label (citra dibuang) | Catatan |
|---|---|---|---|---|
| D1 | Dedup union-find transitif (pHash 4 rotasi + warna + gradien) | 38.250 | 5.299 (824 klaster) | Klaster terbesar 153 citra: satu beda label membuang seluruh rantai |
| D2 | Dedup serakah **non-transitif**; beda *jumlah* box bukan konflik (anotasi per lembar dipertahankan); `malik-project` dikeluarkan (anotasi per-bagian); filter foto UV untuk `bima-dcfqn/*` | 39.681 | 250 (125 pasangan) | 2.663 duplikat beda-jumlah-box, 1.632 foto UV terbuang |
| D3 | Audit gaya anotasi (`audit_sources.py`): `hmmakmur/nominal-rupiah`, `putra-p21x4/project-ka` (dominan per-bagian) dan `bima-dcfqn/*` (UV lolos filter) dikeluarkan; **box bagian bersarang** (≥80% di dalam box sekelas yang lebih besar) dibuang untuk semua sumber | **38.430** | 246 (123 pasangan) | 11.359 box bagian dibuang; nest_rate maks. turun 0,73 → 0,05 |

| D4 | Hasil audit model ronde 1 (`audit_labels.py`, model e0): koreksi 65 label pola "salah ketik nol" (47 train, 6 val, 12 test; verifikasi visual sampel `uang-wfprv`: 37/37 memang salah label), 42 citra train dengan kelas meragukan dibuang, 63 kasus non-sistematis di val/test **dibiarkan** (tidak bias ke model). Sumber dikeluarkan: `tugas-akhir-xq8lu/train-6bmh8` (salah label parah, 45% box tidak cocok), `sitis-workspace-k9sh9/deteksi-uang-rupiah-09ea1` (koin berlabel uang kertas) | **35.862** | 258 (129 pasangan) | train 26.461 / val 4.858 / test 4.543 (74 / 13,5 / 12,7%; salinan augmentasi dipaksa ke train) |

Pemeriksaan visual konflik D2/D3 (`reports/sheets/conflicts_*.jpg`): mayoritas adalah **salah label sungguhan** pada foto
identik, pola "salah jumlah nol" (Rp1.000 ditulis 10000, Rp2.000 ditulis 20000, Rp5.000 ditulis 50000), terutama di
`duit/deteksi-uang-4lt3e`, `teamkito-labs/uang-wfprv`, `test-eyksp/tugas-pemodelan-data-rupiah`,
`remedial-data-mining/deteksi-uang-zyvb3`. Konflik yang tertangkap dedup hanya yang kebetulan punya duplikat; sisanya
dicari dengan **audit berbasis model** (ronde 1).

Split D3 (objek per kelas; D4 di `docs/LAPORAN_EVALUASI.md`):

| Split | Citra | Negatif | Rp1.000 | Rp2.000 | Rp5.000 | Rp10.000 | Rp20.000 | Rp50.000 | Rp100.000 |
|---|---|---|---|---|---|---|---|---|---|
| train | 27.157 | 2.692 | 3.424 | 3.185 | 4.221 | 4.074 | 3.745 | 4.397 | 4.155 |
| val | 5.812 | 576 | 765 | 680 | 785 | 771 | 800 | 812 | 781 |
| test | 5.461 | 576 | 779 | 682 | 796 | 782 | 803 | 821 | 800 |

Rasio kelas terbanyak/tersedikit di train = 1,38 sehingga **oversampling tidak dipakai** (daftar `train_balanced.txt`
tetap dibuat untuk eksperimen bila diperlukan).

## Training

| Run | Data | Perubahan | mAP50 (test) | mAP50-95 (test) | Catatan |
|---|---|---|---|---|---|
| e0_r320_audit | D3 | YOLOv8n, imgsz 320, 30 epoch (patience 10), batch 64, cache RAM, augmentasi kustom; tujuan: model audit label | 0,8393 | 0,6647 | 57 menit (T4). val mAP50 0,8440. AP50 per kelas: 1000 0,903 · 2000 0,833 · 5000 0,767 · 10000 0,776 · 20000 0,808 · 50000 0,954 · 100000 0,835. Confusion matrix: salah kelas antar-nominal ≤3%; kesalahan dominan = uang tidak terdeteksi/terlokalisasi (5 sampai 18% per kelas) dan prediksi tanpa label (10 sampai 18%) |
