# Arsitektur SuaraRupiah

Dokumen ini menjelaskan alur data dari dataset sampai suara keluar di ponsel pengguna.

## 1. Gambaran besar

```mermaid
flowchart LR
  subgraph OFF["Luring (sekali, di Kaggle)"]
    A[Dataset publik<br/>Roboflow Universe + Kaggle] --> B[Pipeline data<br/>normalisasi kelas, dedup, split]
    B --> C[Training YOLOv8n<br/>Kaggle GPU T4]
    C --> D[Ekspor ONNX<br/>FP32 / FP16 / INT8]
  end
  D -->|model-v1-*.onnx| E[GitHub Pages<br/>berkas statis HTTPS]
  subgraph HP["Ponsel pengguna (browser, offline setelah kunjungan pertama)"]
    E --> F[Service worker<br/>+ Cache Storage]
    F --> G[Aplikasi PWA]
    K[Kamera belakang] --> G
    G --> H[Web Worker<br/>ONNX Runtime Web WASM]
    H --> G
    G --> I[Suara: klip audio / Web Speech]
    G --> J[Getar + teks aria-live<br/>TalkBack / VoiceOver]
  end
```

Tidak ada server inferensi. GitHub Pages hanya menyajikan berkas statis (HTML, JS, WASM, model, audio).
Gambar kamera tidak pernah meninggalkan ponsel.

## 2. Pipeline data & training (folder `training/`)

```mermaid
flowchart TD
  S1[rf_survey.py<br/>survei 71 kandidat Roboflow] --> S2[sources.yaml<br/>44 sumber terpilih + lisensi]
  S2 --> L[rf_links.py lokal<br/>link ekspor sementara, API key tetap lokal]
  L --> K1
  subgraph K1["Kernel suararupiah-data (GPU)"]
    D1[fetch_rf_links.py<br/>unduh & ekstrak YOLOv8] --> D2[ingest.py<br/>peta 100+ nama kelas ke 7 kelas<br/>validasi box, resize 640, pHash 4 rotasi]
    D2 --> D3[autolabel.py<br/>YOLO-World untuk dataset klasifikasi]
  end
  K1 -->|pool.tar| K2
  subgraph K2["Kernel suararupiah-build (CPU)"]
    B1[dedup_split.py<br/>dedup pHash + warna + gradien<br/>split 70/15/15 berbasis grup] --> B2[analyze.py, audit_sources.py,<br/>diag_dedup.py: grafik & contact sheet]
  end
  B2 -->|pemeriksaan visual| X[exclude.yaml / reject_ids.txt]
  X --> B1
  K2 -->|final.tar| K3
  subgraph K3["Kernel suararupiah-train (GPU T4 x2)"]
    T1[train.py YOLOv8n<br/>augmentasi kustom, tanpa flip] --> T2[export_onnx.py<br/>FP32 / FP16 / INT8 + evaluasi mAP]
  end
  K3 --> M[install_model.py<br/>models/ + MODEL_CARD.md + web/src/model-config.json]
```

## 3. Alur per frame di aplikasi (folder `web/src/`)

```mermaid
sequenceDiagram
  participant Cam as Kamera (video)
  participant UI as main.ts (thread UI)
  participant W as detector.worker.ts
  participant D as decision.ts
  participant S as speaker.ts
  loop maks. ~10 kali/detik, satu frame diproses sekaligus
    UI->>Cam: createImageBitmap(video)
    UI->>W: postMessage(bitmap)
    W->>W: letterbox ke 320x320 (padding 114) di OffscreenCanvas
    W->>W: RGB/255 -> tensor [1,3,320,320]
    W->>W: ONNX Runtime Web (WASM) -> [1,11,2100]
    W->>W: decode + NMS class-agnostic, kecerahan rata-rata
    W-->>UI: deteksi + luma + waktu (ms)
    UI->>UI: gambar kotak (low vision)
    UI->>D: update(deteksi)
    alt stabil >= 3 dari 5 frame, skor >= 0,7, lewat cooldown 3 dtk
      D-->>UI: pengumuman (nominal, jumlah lembar, total)
      UI->>S: klip "c2", "n50000", "n20000", "t70000"
      UI->>UI: aria-live + getar (pola per nominal)
    else tidak ada uang 4 dtk / terlalu gelap
      UI->>S: panduan suara (+ nyalakan senter)
    end
  end
```

## 4. Komponen web

| Berkas | Tugas |
|---|---|
| `src/main.ts` | Boot (kamera, unduh model + WASM dengan progres, siap), loop frame, pengumuman, ketuk untuk ulang |
| `src/worker/detector.worker.ts` | Inferensi di Web Worker: letterbox, normalisasi, ONNX Runtime Web, decode + NMS |
| `src/lib/letterbox.ts`, `postprocess.ts` | Fungsi murni pra/pasca-proses (diuji unit, sama dengan `infer_ref.py`) |
| `src/lib/decision.ts` | Voting 3 dari 5 frame, threshold, cooldown, reset saat uang hilang; panduan "belum terdeteksi" & "gelap" |
| `src/lib/phrases.ts`, `numwords.ts` | Susunan kalimat (jumlah lembar, masing-masing, total), angka ke kata, pola getar |
| `src/app/speaker.ts` | Putar klip MP3 (Web Audio), cadangan Web Speech API `id-ID` |
| `src/app/loader.ts` | Unduh model & WASM dengan progres, simpan di Cache Storage (offline), hapus cache versi lama |
| `src/app/camera.ts` | Kamera belakang (`facingMode: environment`), senter (`torch`) bila didukung |
| `vite.config.ts` | Build multi-halaman (`/`, `/about/`, `/benchmark/`) + PWA (manifest, service worker Workbox) |

## 5. Strategi offline & pembaruan model

- **Service worker (Workbox)** mem-precache HTML, JS, CSS, ikon, manifest, dan semua klip audio.
- **Model ONNX & runtime WASM** disimpan sendiri oleh aplikasi di Cache Storage (`suararupiah-asset-model-<versi>-<varian>`,
  `suararupiah-asset-ort-<versi ORT>`), agar progres unduhan bisa ditampilkan dan berkas tidak terunduh dua kali.
- Nama model berversi (`model-v1-<varian>.onnx`). Model baru = versi baru: cache baru dibuat, cache lama dihapus saat aplikasi dibuka.
- Setelah kunjungan pertama, aplikasi berjalan tanpa internet (diuji E2E dengan mode offline).

## 6. Keputusan teknis penting

| Keputusan | Alasan |
|---|---|
| Inferensi di browser (WASM) | Privasi (tanpa unggah gambar), offline, tanpa biaya server |
| Web Worker | Inferensi 50 sampai 300 ms per frame tidak membekukan UI/suara |
| Satu frame diproses sekaligus | Laju inferensi otomatis menyesuaikan kemampuan perangkat |
| NMS class-agnostic | Satu lembar uang tidak boleh punya dua nominal |
| Voting 3/5 + threshold 0,7 | Menekan salah sebut akibat frame buram/transisi |
| Klip audio + cadangan Web Speech | Suara konsisten & offline; klip mudah diganti rekaman manusia (nama berkas sama) |
| GitHub Pages | HTTPS gratis; keterbatasan: tanpa header COOP/COEP sehingga WASM hanya 1 thread (diukur di laporan) |
