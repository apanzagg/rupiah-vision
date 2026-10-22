### Pembahasan performa

- **Unduhan pertama** 10,81 MB terkompresi gzip (22,03 MB mentah). Komponen terbesar adalah model (5,61 MB gzip) dan
  runtime ONNX WASM (3,70 MB gzip). Setelah kunjungan pertama, semua aset disajikan dari cache service worker, jadi
  aplikasi berjalan **offline**.
- **Kecepatan inferensi** di desktop Intel i7-4770S (2013), Chrome, backend WASM: median 130 ms per frame (p95 161 ms),
  end-to-end sekitar 139 ms, atau sekitar **7 FPS**. WASM berjalan **1 thread** karena GitHub Pages tidak mengirim
  header COOP/COEP, sehingga halaman tidak `crossOriginIsolated` dan multi-thread WASM tidak tersedia. Dengan aturan
  konsistensi 3 dari 5 frame, nominal diumumkan kira-kira 0,4 sampai 0,7 detik setelah uang stabil di depan kamera.
- **Waktu muat** (build produksi lokal sampai model siap): kunjungan pertama 2,0 detik, dari cache 1,0 detik, offline
  0,95 detik. Aplikasi lolos kriteria instalasi PWA Chrome tanpa error manifest.
- **Lighthouse** dari URL publik, dengan emulasi ponsel Moto G Power dan throttling jaringan/CPU: Performance 99,
  Accessibility 100, Best Practices 100, SEO 100.
- **Verifikasi deploy publik** (22 Oktober 2026): uji E2E dengan kamera palsu terhadap
  https://apanzagg.github.io/rupiah-vision/ lulus untuk pengumuman multi-lembar, panduan "uang belum terdeteksi", dan
  pengumuman setelah reload **tanpa jaringan**.
- **Belum diukur:** kecepatan di ponsel nyata. CPU ponsel kelas menengah umumnya lebih lambat daripada desktop ini,
  sehingga FPS di ponsel perlu diukur sendiri lewat halaman `/benchmark/`.
