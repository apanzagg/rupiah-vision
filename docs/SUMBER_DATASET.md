# Sumber Dataset

Semua dataset di bawah dipakai sesuai lisensinya. Dataset ber-lisensi **CC BY 4.0** mewajibkan atribusi:
tabel ini berfungsi sebagai atribusi (nama pemilik workspace/akun, judul, URL, lisensi).
Berkas ini dibangkitkan oleh `training/scripts/make_sources_doc.py`; jangan diedit manual.

Kolom:
- **citra sumber**: jumlah citra asli di proyek (sebelum augmentasi Roboflow), dari API Roboflow/Kaggle.
- **versi & augmentasi**: versi yang diunduh dan augmentasi bawaan versi itu (`none` = tanpa augmentasi).
- **dipakai**: jumlah citra yang lolos normalisasi kelas dan penghapusan duplikat (diisi setelah pipeline data berjalan).

## Roboflow Universe (object detection, format YOLOv8)

| # | Dataset (pemilik/proyek) | Lisensi | Peran | Citra sumber | Versi & augmentasi | Kelas asli | Dipakai |
|---|---|---|---|---|---|---|---|
| 1 | [Uang Rupiah Kertas dan Koin by Adelia Clarissa](https://universe.roboflow.com/coin-new/uang-rupiah-kertas-dan-koin-by-adelia-clarissa-zzt2j/dataset/11) (`coin-new/uang-rupiah-kertas-dan-koin-by-adelia-clarissa-zzt2j`) | CC BY 4.0 | positive | 9000 | v11, none | 500, 1000, 10rb, 50rb, 100rb, 5rb, 20rb, 1rb, 2rb |  |
| 2 | [rupiah-currentcy-public-V2](https://universe.roboflow.com/craft-model/rupiah-currentcy-public-v2/dataset/8) (`craft-model/rupiah-currentcy-public-v2`) | Public Domain | positive | 3405 | v8, none | 1000, 2000, 5000, 10000, 20000, 50000, 75000, 100000 |  |
| 3 | [tugas-pemodelan-data-rupiah](https://universe.roboflow.com/test-eyksp/tugas-pemodelan-data-rupiah/dataset/1) (`test-eyksp/tugas-pemodelan-data-rupiah`) | CC BY 4.0 | positive | 3402 | v1, none | 1000, 2000, 5000, 10000, 20000, 50000, 75000, 100000 |  |
| 4 | [deteksiuang[]](https://universe.roboflow.com/currency-yimpm/deteksiuang-xllap/dataset/1) (`currency-yimpm/deteksiuang-xllap`) | CC BY 4.0 | positive | 3213 | v1, none | 10ribu, 100ribu, 1ribu, 20ribu, 2ribu, 50ribu, 75ribu, 5ribu |  |
| 5 | [IDR_Money_detection](https://universe.roboflow.com/andri/idr_money_detection/dataset/3) (`andri/idr_money_detection`) | Public Domain | positive | 2700 | v3, none | Rp20000, Rp2000, Rp1000, Rp10000, Rp50000, Rp5000, Rp100000 |  |
| 6 | [Dataset Uang Rupiah Kertas](https://universe.roboflow.com/2210511040-mahasiswa-upnvj-ac-id/dataset-uang-rupiah-kertas/dataset/5) (`2210511040-mahasiswa-upnvj-ac-id/dataset-uang-rupiah-kertas`) | CC BY 4.0 | positive | 2672 | v5, none | 1000, 2000, 5000, 10000, 20000, 50000, 100000 |  |
| 7 | [deteksi uang](https://universe.roboflow.com/duit/deteksi-uang-4lt3e/dataset/1) (`duit/deteksi-uang-4lt3e`) | Public Domain | positive | 2549 | v1, none | 1000, 2000, 5000, 10000, 20000, 50000, 100000 |  |
| 8 | [Train](https://universe.roboflow.com/tugas-akhir-xq8lu/train-6bmh8/dataset/1) (`tugas-akhir-xq8lu/train-6bmh8`) | CC BY 4.0 | positive | 2394 | v1, none | Lima Puluh Ribu, Dua Ribu, Dua Puluh Ribu, Dua Ratus Perak, Sepuluh Ribu, Seratus Ribu, Lima Ribu, Seribu, 20k, Lima Ratus Perak, Seratus Perak |  |
| 9 | [INI UANG](https://universe.roboflow.com/bima-dcfqn/ini-uang/dataset/1) (`bima-dcfqn/ini-uang`) | CC BY 4.0 | positive | 2206 | v1, none | DUA PULUH RIBU ASLII, LIMA PULUH RIBU ASLI, SERATUS RIBU ASLI, SEPULUH RIBU ASLII, DUA RIBU ASLI, SERIBU ASLI, LIMA RIBU ASLII, SEPULUH RIBU ASLI, UANGKU, LIMA RIBU ASLI, DUAPULUH RIBU ASLI, DUA PULUH RIBU ASLI |  |
| 10 | [UANG 1](https://universe.roboflow.com/bima-dcfqn/uang-1/dataset/2) (`bima-dcfqn/uang-1`) | CC BY 4.0 | positive | 745 | v2, none | LIMA PULUH RIBU ASLI, UANG, SERATUSRIBU ASLI, DUA RIBU ASLI, SEPULUH RIBU ASLI, SERIBU ASLI, LIMA RIBU ASLI, DUA PULUH RIBU ASLI |  |
| 11 | [rupiah detector](https://universe.roboflow.com/nanabanana/rupiah-detector-zrmyf/dataset/1) (`nanabanana/rupiah-detector-zrmyf`) | CC BY 4.0 | positive | 1582 | v1, none | Dua puluh ribu rupiah, lima puluh ribu rupiah, seribu rupiah, lima ribu rupiah, dua ribu rupiah, sepuluh ribu rupiah, seratus ribu rupiah |  |
| 12 | [Uang Rupiah](https://universe.roboflow.com/ramzi-guyzk/uang-rupiah-li6sa/dataset/2) (`ramzi-guyzk/uang-rupiah-li6sa`) | CC BY 4.0 | positive | 1552 | v2, none | 2k, 5k, 10k, 100k, 1k, 50k, 20k |  |
| 13 | [moneydetection](https://universe.roboflow.com/4ia17ottos-workspace/moneydetection-uetq8/dataset/14) (`4ia17ottos-workspace/moneydetection-uetq8`) | CC BY 4.0 | positive | 1418 | v14, none | 100k, 10k, 50k, 1k, 20k, 5k, 2k |  |
| 14 | [Deteksi Uang Kertas Rupiah](https://universe.roboflow.com/dakhmadlearner/deteksi-uang-kertas-rupiah/dataset/3) (`dakhmadlearner/deteksi-uang-kertas-rupiah`) | CC BY 4.0 | positive | 1400 | v3, none | Rp. 2.000, Rp. 5.000, Rp. 20.000, Rp. 10.000, Rp. 50.000, Rp. 100.000, Rp. 1.000, Hiu png |  |
| 15 | [uang-yolo](https://universe.roboflow.com/ramziahmar-sdckt/uang-yolo/dataset/1) (`ramziahmar-sdckt/uang-yolo`) | CC BY 4.0 | positive | 1287 | v1, none | objects, 100k, 20k, 2k, 50k, 5k, 1k, 10k |  |
| 16 | [EAS Vision](https://universe.roboflow.com/shuttlecock-6uoyv/eas-vision/dataset/1) (`shuttlecock-6uoyv/eas-vision`) | CC BY 4.0 | positive | 1080 | v1, none | 50K, 1K, 20K, 5K, 10K, 100K, 2K |  |
| 17 | [Rupiah Detector](https://universe.roboflow.com/rupiah-detector/rupiah-detector-qzmb7/dataset/1) (`rupiah-detector/rupiah-detector-qzmb7`) | CC BY 4.0 | positive | 1142 | v1, none | lima puluh ribu rupiah, dua ribu rupiah, dua puluh ribu rupiah, lima ribu rupiah, seribu rupiah, seratusribu rupiah, sepuluhribu rupiah |  |
| 18 | [rupiah](https://universe.roboflow.com/tugas-akhir-afuhz/rupiah-m55hl/dataset/1) (`tugas-akhir-afuhz/rupiah-m55hl`) | CC BY 4.0 | positive | 746 | v1, none | 1000, 2000, 5000, 10000, 20000, 50000, 100000 |  |
| 19 | [data mining](https://universe.roboflow.com/ilham-project/data-mining-n52lu/dataset/1) (`ilham-project/data-mining-n52lu`) | Public Domain | positive | 692 | v1, none | 1000, 2000, 5000, 100 ribu, 10ribu, 50 ribu, 20 ribu |  |
| 20 | [Nominal Rupiah](https://universe.roboflow.com/hmmakmur/nominal-rupiah/dataset/17) (`hmmakmur/nominal-rupiah`) | CC BY 4.0 | positive | 654 | v17, none | Seratus-Ribu, Lima-Puluh-RIbu, Sepuluh-Ribu, Dua-Puluh-Ribu, Lima Ribu, Dua-Ribu, Seribu |  |
| 21 | [FIX Indonesia rupiah banknote 7 cls](https://universe.roboflow.com/agil-skripsi-3/fix-indonesia-rupiah-banknote-7-cls/dataset/13) (`agil-skripsi-3/fix-indonesia-rupiah-banknote-7-cls`) | CC BY 4.0 | positive | 644 | v13, none | lima ribu, dua ribu, lima puluh ribu, seratus ribu, sepuluh ribu, dua puluh ribu, seribu |  |
| 22 | [1-indonesian-rupiah-banknotes-bg-black](https://universe.roboflow.com/agil-skripsi-fix/1-indonesian-rupiah-banknotes-bg-black/dataset/16) (`agil-skripsi-fix/1-indonesian-rupiah-banknotes-bg-black`) | CC BY 4.0 | positive | 420 | v16, none | sepuluhribu 2022, dua ribu 2016, lima ribu 2022, lima puluh ribu 2016, seratus ribu 2016, dua puluh ribu 2022, seribu 2016, dua ribu 2022, lima ribu 2016, seribu 2022, dua puluhribu 2016, lima puluh ribu 2022, seratus ribu 2022, sepuluhribu 2016 | 408 |
| 23 | [Deteksi Uang 2](https://universe.roboflow.com/universitas-amikom-yogyakarta-uxsjx/deteksi-uang-2/dataset/2) (`universitas-amikom-yogyakarta-uxsjx/deteksi-uang-2`) | CC BY 4.0 | positive | 500 | v2, none | 50k, 5k, 2k, 1k, 10k, 20k, 100k |  |
| 24 | [Uang rupiah](https://universe.roboflow.com/uang-rupiah/uang-rupiah-oyrxk/dataset/2) (`uang-rupiah/uang-rupiah-oyrxk`) | CC BY 4.0 | positive | 422 | v2, none | 1000, 2000, 5000, 10000, 20000, 50000, 100000 |  |
| 25 | [rupiah detection](https://universe.roboflow.com/image-processing-re77r/rupiah-detection-7nuxw/dataset/1) (`image-processing-re77r/rupiah-detection-7nuxw`) | CC BY 4.0 | positive | 392 | v1, none | 50k, 5k, 2k, 1k, 10k, 20k, 100k |  |
| 26 | [Deteksi Uang Rupiah](https://universe.roboflow.com/malik-project/deteksi-uang-rupiah-50fx4/dataset/2) (`malik-project/deteksi-uang-rupiah-50fx4`) | CC BY 4.0 | positive | 358 | v2, none | 1000, 2000, 5000, 10000, 20000, 50000, 100000 | 354 |
| 27 | [Deteksi Uang Rupiah](https://universe.roboflow.com/sitis-workspace-k9sh9/deteksi-uang-rupiah-09ea1/dataset/1) (`sitis-workspace-k9sh9/deteksi-uang-rupiah-09ea1`) | CC BY 4.0 | positive | 343 | v1, none | Rp. 5.000, objects, Rp. 1.000, Rp. 2.000, Rp. 50.000, Rp. 100.000, Rp. 20.000, Rp. 10.000 |  |
| 28 | [Uang Rupiah](https://universe.roboflow.com/yudhi/uang-rupiah-sjt3b/dataset/1) (`yudhi/uang-rupiah-sjt3b`) | CC BY 4.0 | positive | 333 | v1, none | 20000 rupiah, 2000 rupiah, 50000 rupiah, 5000 rupiah, 10000 rupiah, 100000 rupiah, 1000 rupiah | 333 |
| 29 | [project ka](https://universe.roboflow.com/putra-p21x4/project-ka/dataset/5) (`putra-p21x4/project-ka`) | CC BY 4.0 | positive | 299 | v5, none | dua puluh ribu rupiah, lima ribu rupiah, sepuluh ribu rupiah, 100-perak, seribu rupiah, dua ribu rupiah, 500-perak, lima puluh ribu rupiah, 1000-perak, 200 perak, seratus ribu rupiah |  |
| 30 | [Uang](https://universe.roboflow.com/teamkito-labs/uang-wfprv/dataset/3) (`teamkito-labs/uang-wfprv`) | CC BY 4.0 | positive | 5435 | v3, x2 rotate ninety | 1000, 2000, 5000, 10000, 20000, 50000, 75000, 100000 |  |
| 31 | [Rupiah Detection](https://universe.roboflow.com/rizkynindra-wsxpl/rupiah-detection/dataset/11) (`rizkynindra-wsxpl/rupiah-detection`) | CC BY 4.0 | positive | 972 | v11, x3 rotate | 1000, 2000, 5000, 10000, 20000, 50000, 100000 |  |
| 32 | [Deteksi Uang](https://universe.roboflow.com/remedial-data-mining/deteksi-uang-zyvb3/dataset/1) (`remedial-data-mining/deteksi-uang-zyvb3`) | CC BY 4.0 | positive | 560 | v1, x3 rotate ninety | 1000, 2000, 5000, 10000, 20000, 50000, 100000 |  |
| 33 | [uang](https://universe.roboflow.com/blind-aid/uang-5z9jp/dataset/1) (`blind-aid/uang-5z9jp`) | CC BY 4.0 | positive | 391 | v1, x3 ninety blur brightness rotate | koin_1k, 10k, 20k, koin_100, 5k, 75k, koin_500, 1k, 50k, 2k, 100k |  |
| 34 | [NetraSense](https://universe.roboflow.com/netrasense/netrasense/dataset/1) (`netrasense/netrasense`) | CC BY 4.0 | positive | 1514 | v1, x3 brightness rotate blur shear noise exposure ninety | 100K, 10K, 50K, 5K, 20K |  |
| 35 | [currency detection](https://universe.roboflow.com/mata-uang/currency-detection-y4tb2/dataset/9) (`mata-uang/currency-detection-y4tb2`) | CC BY 4.0 | negative | 2593 | v9, none | Vietnam Dong, Brunei Dollar, Singapore Dollar, Thailand Baht, Philippines Peso, Myanmar Kyat, Laos Kip, Indonesian Rupiah, Malaysian Ringgit, Cambodian Riel |  |
| 36 | [Banknote](https://universe.roboflow.com/kalumkoi/banknote-68gsn/dataset/1) (`kalumkoi/banknote-68gsn`) | CC BY 4.0 | negative | 800 | v1, none | THAI50, THAI20, THAI1000, THAI500, THAI100 | 122 |
| 37 | [banknote-recognition](https://universe.roboflow.com/dell-lead-7wvtm/banknote-recognition/dataset/9) (`dell-lead-7wvtm/banknote-recognition`) | CC BY 4.0 | negative | 3255 | v9, none | 05reais-cara, 20reais-cara, 05reais-coroa, 02reais-cara, 20reais-coroa, 10reais-cara, 02reais-coroa, 50reais-cara, 10reais-coroa, 50reais-coroa, 200reais-cara, 200reais-coroa, 100reais-cara, 100reais-coroa |  |
| 38 | [cj-gl8m1/receipt-ocr-cvfxx](https://universe.roboflow.com/cj-gl8m1/receipt-ocr-cvfxx/dataset/4) (`cj-gl8m1/receipt-ocr-cvfxx`) | CC BY 4.0 | negative |  | v4, none |  |  |
| 39 | [receipt-lbosb/receipt-msfg0](https://universe.roboflow.com/receipt-lbosb/receipt-msfg0/dataset/1) (`receipt-lbosb/receipt-msfg0`) | CC BY 4.0 | negative |  | v1, x3 shear |  |  |
| 40 | [mohamed-hussam-cq81o/wallet-sn9n2](https://universe.roboflow.com/mohamed-hussam-cq81o/wallet-sn9n2/dataset/1) (`mohamed-hussam-cq81o/wallet-sn9n2`) | CC BY 4.0 | negative |  | v1, none |  |  |
| 41 | [project-qysdi/wallet-iiyy3](https://universe.roboflow.com/project-qysdi/wallet-iiyy3/dataset/1) (`project-qysdi/wallet-iiyy3`) | CC BY 4.0 | negative |  | v1, none |  |  |
| 42 | [fastnucesuakqb/hand-object-detection-wyhyi](https://universe.roboflow.com/fastnucesuakqb/hand-object-detection-wyhyi/dataset/5) (`fastnucesuakqb/hand-object-detection-wyhyi`) | CC BY 4.0 | negative |  | v5, none |  |  |
| 43 | [holeno/holding-hand-only](https://universe.roboflow.com/holeno/holding-hand-only/dataset/3) (`holeno/holding-hand-only`) | CC BY 4.0 | negative |  | v3, none |  |  |
| 44 | [tugas-akhir-xq8lu/coin-rupiah](https://universe.roboflow.com/tugas-akhir-xq8lu/coin-rupiah/dataset/1) (`tugas-akhir-xq8lu/coin-rupiah`) | CC BY 4.0 | negative |  | v1, none |  |  |

## Kaggle

| # | Dataset | Lisensi | Peran | Format | Dipakai |
|---|---|---|---|---|---|
| 1 | [daniakhmadmaulana/citra-uang-rupiah-emisi-2022](https://www.kaggle.com/datasets/daniakhmadmaulana/citra-uang-rupiah-emisi-2022) | MIT | positive_cls | klasifikasi per folder (1k..100k), emisi 2022 |  |
| 2 | [indahtricahyani/kondisi-uang-kertas-rupiah](https://www.kaggle.com/datasets/indahtricahyani/kondisi-uang-kertas-rupiah) | CC0-1.0 | condition_unlabeled | folder per kondisi, tanpa nominal |  |

## Pembersihan setelah pemeriksaan visual (`training/configs/exclude.yaml`)

| Sumber | Tindakan | Alasan |
|---|---|---|
| `malik-project/deteksi-uang-rupiah-50fx4` | seluruh citra dikeluarkan | anotasi per-bagian uang (angka/potret/sudut), bukan uang utuh (nest_rate 0,68) |
| `bima-dcfqn/ini-uang` | foto lampu UV (dominan ungu) dibuang | foto untuk cek keaslian, tidak mewakili pemakaian |
| `bima-dcfqn/uang-1` | foto lampu UV (dominan ungu) dibuang | foto untuk cek keaslian, tidak mewakili pemakaian |

## Dataset yang dikecualikan sejak seleksi awal

| Dataset | Alasan |
|---|---|
| `craft-model/rupiah-currentcy-detection`, `craft-model/rupiah-currentcy-detection-v2`, `machine-learning-pmhkb/deteksi-rupiah-jkkmu` (Roboflow) | Lisensi "Private" |
| Proyek Roboflow yang hanya punya versi augmentasi dengan **flip** (cermin), mis. `4ia17ottos-workspace/uang_baru`, `detection-twl6q/rupiah_emisi-baru`, `test-sirmi/deteksi-uang-rupiah-8zlc2` | Uang yang tercermin tidak ada di dunia nyata dan bisa mengajari model ciri yang salah |
| `orbitaibanknotes/indonesian-banknotes` | Hanya satu kelas "rupiah" tanpa nominal |
| `muhammad-ihsan-sinaga/dataset-gabungan-rupiah`, `first-oi8bn/money-detection-q0jko` | Nama kelas tidak konsisten/tidak bisa dipetakan dengan pasti; sebagian besar isinya gabungan dataset lain |
| Kaggle: `anidwiastuti/rupiah-banknotes-dataset`, `fannyzahrahramadhan/*`, `nurulalfiyyah/rupiah-banknotes`, `brotoa/*`, `dimaswara/*`, `kevinwaska/uang-baru`, `muhrisal100786/*`, `yunhann/*` | Lisensi "Unknown" (tidak dinyatakan) |
| Kaggle: `fzmhww8782/rupiah-22`, `hashimatulzaria/data-uang-rupiah` | Lisensi "Other (specified in description)" tetapi deskripsinya kosong |
| Kaggle: `elvinjaya6/deteksi-uang-kertas`, `erichaw/uang-rupiah-wendy` | Hanya citra tanpa label; tampak turunan dataset berlisensi "Unknown" |

Catatan: banyak proyek Roboflow merupakan salinan (fork) satu sama lain. Duplikat lintas sumber dihapus dengan
perceptual hash sebelum split (lihat `docs/LAPORAN_EVALUASI.md`).
