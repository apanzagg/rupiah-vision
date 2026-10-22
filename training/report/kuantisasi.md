### Pembahasan kuantisasi

- **FP16** (`onnxruntime.transformers.float16`, input/output tetap float32) memangkas ukuran dari 12,14 MB menjadi
  6,11 MB tanpa penurunan akurasi: mAP50 pipeline web sama, 0,9154. Operator `Resize` harus dibiarkan FP32 karena
  input `scales`-nya wajib float32. Tanpa pengecualian ini, model hasil konversi tidak valid.
- **INT8 penuh** (kuantisasi statis QDQ per-channel, kalibrasi citra train) adalah yang terkecil (3,38 MB), tetapi
  mAP50 web turun ke 0,8770 dan recall ke 0,82. Penyebabnya, keluaran YOLOv8 menggabungkan koordinat box (0 sampai
  320 piksel) dan skor kelas (0 sampai 1) dalam satu tensor, sehingga satu skala kuantisasi merusak resolusi skor
  kelas. Pada model imgsz 416 efeknya jauh lebih parah: recall INT8 hanya 0,014. Kuantisasi per-channel juga
  mengharuskan graf dinaikkan dari opset 12 ke 13.
- **INT8 campuran** (head deteksi `/model.22/*` tetap FP32) memulihkan sebagian besar akurasi (mAP50 web 0,8997),
  tetapi masih turun 1,6 poin dibanding FP32. Ukurannya juga hanya 0,53 MB lebih kecil dari FP16, dan di CPU Kaggle
  justru lebih lambat (23,9 vs 15,9 ms/citra) karena banyak operasi kuantisasi/dekuantisasi.
- Sesuai aturan "varian terkecil dengan penurunan mAP50 ≤ 1 poin", **FP16 dipilih** sebagai model web.
