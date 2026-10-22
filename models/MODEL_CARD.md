# Model Card: SuaraRupiah v1 (fp16)

Detektor nominal uang kertas Rupiah, **YOLOv8n** (Ultralytics 8.3.253), diekspor ke ONNX
untuk dijalankan di browser dengan ONNX Runtime Web (backend WASM). Dibangkitkan oleh `training/scripts/install_model.py`.

## Berkas

| Item | Nilai |
|---|---|
| Berkas | `model-v1-fp16.onnx` |
| Ukuran | 6.11 MB |
| SHA-256 | `05f7d311be4223be14b1e6c73b4fece9fa177ec32206e0466311c9759666c2e4` |
| Varian | `fp16` (dipilih: terkecil dengan drop mAP50 <= 1 poin) |
| Opset ONNX | 12 |
| Run training | `e1_r320` |

## Masukan & keluaran

| | Nama | Bentuk | Keterangan |
|---|---|---|---|
| Input | `images` | [1, 3, 320, 320] | float32, RGB, nilai 0..1, letterbox ke 320x320 dengan padding abu (114), posisi tengah |
| Output | `output0` | [1, 11, 2100] | baris 0..3 = cx, cy, w, h (piksel ruang input); baris 4..10 = skor 7 kelas (sudah sigmoid). NMS dilakukan di aplikasi |

## Kelas (urutan indeks)

| Indeks | Nama | Nominal | AP50 test (validator Ultralytics) | AP50-95 test (validator Ultralytics) |
|---|---|---|---|---|
| 0 | `1000` | Rp1.000 | 0.9560 | 0.8103 |
| 1 | `2000` | Rp2.000 | 0.9498 | 0.8174 |
| 2 | `5000` | Rp5.000 | 0.9431 | 0.7954 |
| 3 | `10000` | Rp10.000 | 0.9639 | 0.8137 |
| 4 | `20000` | Rp20.000 | 0.9642 | 0.7859 |
| 5 | `50000` | Rp50.000 | 0.9548 | 0.8013 |
| 6 | `100000` | Rp100.000 | 0.9642 | 0.8148 |

## Ambang yang disarankan

- Confidence filter awal: 0,25; NMS class-agnostic IoU 0,45 (maks. 20 deteksi).
- Pengumuman suara: confidence >= **0,7** dan konsisten di >= 3 dari 5 frame terakhir (lihat `web/src/lib/decision.ts`).

## Metrik split test

PyTorch (`best.pt`, validator Ultralytics bawaan `rect=True`, cenderung optimistis): mAP50 **0.9566**, mAP50-95 **0.8056**, precision 0.9535, recall 0.8936.

Perbandingan varian ONNX (validator Ultralytics, backend onnxruntime CPU, batch 1, letterbox persegi; waktu CPU = Kaggle 4 vCPU):

| Varian | Ukuran (MB) | mAP50 | mAP50-95 | Precision | Recall | ms/citra (CPU) |
|---|---|---|---|---|---|---|
| fp32 | 12.14 | 0.9285 | 0.6801 | 0.9281 | 0.8860 | 15.12 |
| fp16 | 6.11 | 0.9286 | 0.6800 | 0.9283 | 0.8865 | 15.93 |
| int8 | 3.38 | 0.8760 | 0.6481 | 0.9440 | 0.8197 | 15.04 |
| int8_mixed | 5.58 | 0.9147 | 0.6633 | 0.9127 | 0.8710 | 23.87 |


## Metrik pipeline aplikasi web (angka utama)

Protokol: pipeline referensi web (infer_ref.py), letterbox tengah, NMS class-agnostic, conf 0.001, IoU NMS 0.7; split test 4543 citra. Validator Ultralytics bawaan (`rect=True`) memberi angka lebih tinggi untuk model yang sama; lihat `docs/EKSPERIMEN.md` (catatan protokol).

| Varian | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| model-v1-fp32.onnx | 0.9154 | 0.6771 | 0.9277 | 0.8866 |
| model-v1-fp16.onnx | 0.9154 | 0.6773 | 0.9279 | 0.8870 |
| model-v1-int8.onnx | 0.8770 | 0.6539 | 0.9447 | 0.8202 |
| model-v1-int8_mixed.onnx | 0.8997 | 0.6590 | 0.9146 | 0.8708 |

## Data & batasan

Lihat `docs/SUMBER_DATASET.md` (sumber & lisensi) dan `docs/LAPORAN_EVALUASI.md` (evaluasi per kondisi, analisis kesalahan).
Model **hanya mengenali nominal**, bukan keaslian. Uang logam dan Rp75.000 tidak dikenali (dilatih sebagai latar).
