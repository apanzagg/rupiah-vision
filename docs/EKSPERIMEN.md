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
| D5 | D4 + negatif yang terbuang kuota dimasukkan ke train saja (`dedup_split.py --extra-neg`, ronde v2): koin Rp75.000 hasil potong citra positif 971, uang asing 368, struk 154, dompet 135, koin 80, tangan 62 | **37.632** | sama dengan D4 | train 28.231 / val 4.858 / test 4.543; id val/test **identik** dengan D4 (dicek dari `manifest_final.csv`) |

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
| e1_r320 | D4 | imgsz 320, 80 epoch (patience 20), batch 64, cache RAM, augmentasi kustom | 0,9566* | 0,8056* | 2,49 jam (T4); epoch terbaik 80 (masih naik). val mAP50 0,9733*. AP50 test per kelas*: 1000 0,956 · 2000 0,950 · 5000 0,943 · 10000 0,964 · 20000 0,964 · 50000 0,955 · 100000 0,964 |
| e1_r416 | D4 | sama, imgsz 416 | 0,9614* | 0,8107* | 3,77 jam (T4); epoch terbaik 80. val mAP50 0,9785* |
| e2_r320_hn | D5 | fine-tune dari e1_r320, 40 epoch (patience 15), SGD lr0 0,005, warmup 1, `train_hardneg.txt` (uang asing 3x) | 0,9548* | 0,8057* | 1,34 jam (T4); epoch terbaik 40. val mAP50 0,9717*. AP50 test per kelas*: 1000 0,939 · 2000 0,954 · 5000 0,940 · 10000 0,963 · 20000 0,964 · 50000 0,959 · 100000 0,964. **Tidak diadopsi** (lihat ronde v2) |

\* Angka validator Ultralytics bawaan (`rect=True`). **Lihat catatan protokol di bawah**: angka yang mewakili pipeline
aplikasi lebih rendah.

### Catatan protokol evaluasi (penting)

Model e1_r320 yang SAMA pada split test yang SAMA:

| Protokol | mAP50 | mAP50-95 |
|---|---|---|
| Ultralytics `val`, `rect=True` (bawaan, batch 32 maupun 1) | 0,9566 | 0,8054 |
| Ultralytics `val`, `rect=False` (letterbox persegi, seperti model ONNX statis) | 0,9285 | 0,6801 |
| Metrik sendiri (`ap_per_class` Ultralytics) + prediksi PyTorch padding minimal | 0,9199 | 0,6793 |
| Metrik sendiri + **pipeline web** ONNX FP32 (`eval_onnx_map.py`, letterbox tengah, NMS class-agnostic) | **0,9154** | **0,6771** |

Prediksi PyTorch vs ONNX identik pada citra yang sama (kelas, skor, box), dan perbedaan padding minimal vs persegi
hanya mengubah prediksi sedikit (150 citra non-persegi: selisih skor median 0,0075, IoU box median 0,984). Karena
tiga pengukuran independen sepakat di ~0,92 / ~0,68, angka mode `rect=True` dianggap **terlalu optimistis** untuk
data ini dan **angka pipeline web dipakai sebagai angka utama** di laporan. Penyebab pasti selisih di validator
Ultralytics tidak diselidiki lebih jauh. Angka ronde e0 di atas juga memakai protokol `rect=True`.

## Uji per kondisi v1 dan ronde v2 (hard negative)

Model web v1 (`model-v1-fp16.onnx`, dari e1_r320) pada set uji per kondisi (`make_test_conditions.py`, diperiksa visual;
ambang 0,7, akurasi = nominal yang diumumkan per citra persis sama dengan label):

| Kondisi | n | Akurasi | Catatan |
|---|---|---|---|
| baru | 120 | 0,992 | |
| lusuh_kusut | 201 | 0,960 | pindaian Kaggle CC0, label manual |
| terlipat_tertutup | 120 | 0,892 | sintetis |
| cahaya_redup | 120 | 0,967 | |
| latar_ramai | 120 | 0,842 | mayoritas gagal: uang kecil di lantai marmer merah, kelas benar tetapi skor 0,5 sampai 0,7 |
| bukan_rupiah | 199 | **0,889** | 22 citra uang asing diumumkan sebagai rupiah (skor 0,70 sampai 0,92): real Brasil biru → Rp50.000/Rp2.000, baht Thailand → Rp2.000/Rp100.000, kip Laos → Rp20.000 |
| baru_pindaian | 99 | 1,000 | |

Diagnosis: train D4 hanya memuat **258 citra uang asing** sebagai negatif (val 262, test 294). Kuota negatif
(`NEG_FRAC` 0,10) membuang 1.899 negatif secara proporsional, dan split bergrup menaruh sebagian besar uang asing di
luar train. Menaikkan ambang tidak efektif karena skor salah sebut tinggi (sampai 0,92).

**Ronde v2 (direncanakan sebelum hasil keluar):**

- Data D5 = D4 + negatif yang terbuang kuota dimasukkan ke **train** saja (`dedup_split.py --extra-neg`), kecuali yang
  satu grup kemiripan dengan citra val/test. Split val/test **identik** dengan D4 (diverifikasi dari manifest).
- `train_hardneg.txt`: negatif uang asing di-oversample 3x.
- Run `e2_r320_hn`: fine-tune dari bobot e1_r320, 40 epoch (patience 15), SGD lr0 0,005, warmup 1 epoch, hyperparameter
  augmentasi sama.
- **Aturan adopsi** (ditetapkan sebelum melihat hasil): v2 menggantikan v1 bila (1) mAP50 pipeline web pada test
  ≥ 0,90 dan turun ≤ 1 poin dari v1, (2) akurasi `bukan_rupiah` naik ≥ 3 poin, dan (3) akurasi kondisi lain tidak turun
  lebih dari 3 poin. Jika tidak, v1 dipertahankan dan v2 dilaporkan sebagai percobaan.

**Hasil ronde v2** (`training/kaggle_out/train_v2/`, `data/interim/conditions_eval_v2/`):

mAP pipeline web pada test (4.543 citra, `eval_onnx_map.py`): v2 FP32 0,9304 / 0,6890, FP16 **0,9304** / 0,6888
(v1 FP16 0,9154 / 0,6773). AP50 per kelas v2 FP16: 1000 0,887 · 2000 0,924 · 5000 0,928 · 10000 0,942 · 20000 0,931 ·
50000 0,950 · 100000 0,951.

| Kondisi | n | v1 | v2 | Selisih (poin) | Salah sebut v1 → v2 | Diam v1 → v2 |
|---|---|---|---|---|---|---|
| baru | 120 | 0,992 | 0,992 | 0,0 | 0 → 0 | 1 → 1 |
| lusuh_kusut | 201 | 0,960 | 0,950 | −1,0 | 3 → 3 | 5 → 7 |
| terlipat_tertutup | 120 | 0,892 | 0,842 | **−5,0** | 3 → 1 | 10 → 18 |
| cahaya_redup | 120 | 0,967 | 0,908 | **−5,8** | 0 → 0 | 4 → 11 |
| latar_ramai | 120 | 0,842 | 0,883 | +4,2 | 1 → 1 | 17 → 12 |
| bukan_rupiah | 199 | 0,889 | **0,995** | +10,6 | 22 → 1 | 0 → 0 |
| baru_pindaian | 99 | 1,000 | 0,970 | −3,0 | 0 → 0 | 0 → 3 |
| **Jumlah** | 1.079 | | | | **29 → 6** | **37 → 52** |

"Salah sebut" = ada nominal diumumkan yang tidak ada di citra; "diam" = tidak ada deteksi ≥ 0,7 (latar_ramai juga
memuat 1 kasus kurang lembar di kedua model).

Penerapan aturan adopsi: (1) lolos (+1,5 poin), (2) lolos (+10,6 poin), (3) **tidak lolos** (terlipat_tertutup −5,0 dan
cahaya_redup −5,8). **Keputusan: v1 dipertahankan sebagai model web**, v2 dilaporkan sebagai percobaan.

Analisis: semua citra yang benar di v1 tetapi gagal di v2 (25 citra) gagal karena **diam**: skor v1 untuk citra itu
0,70 sampai 0,89, sedangkan v2 tidak memberi deteksi ≥ 0,7. Tidak ada kasus baru salah sebut. Tambahan negatif membuat
model lebih "hati-hati": salah sebut turun dari 29 ke 6, tetapi kasus diam naik dari 37 ke 52. Bagi pengguna tunanetra,
salah sebut lebih berbahaya daripada diam (aplikasi memberi panduan suara bila uang tidak terdeteksi), sehingga
kompromi ini layak dikaji pada ronde berikutnya (mis. kalibrasi ambang pada split val, atau menambah data positif
terlipat/redup) dengan aturan adopsi yang ditetapkan ulang sebelum hasil keluar.
