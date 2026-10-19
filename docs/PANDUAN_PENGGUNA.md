# Panduan Pengguna SuaraRupiah

SuaraRupiah membantu penyandang tunanetra dan low vision mengenali nominal uang kertas Rupiah. Arahkan kamera ponsel
ke uang, dan aplikasi menyebutkan nominalnya dengan suara.

> Untuk pendamping: bagian "Pemasangan" sebaiknya dibantu sekali di awal. Setelah itu pengguna cukup membuka aplikasi.

## Kebutuhan

- Ponsel Android dengan Chrome versi terbaru, atau iPhone dengan Safari (iOS 16.4 atau lebih baru).
- Internet hanya untuk kunjungan pertama (sekitar 10 MB). Setelah itu aplikasi berfungsi tanpa internet.

## Pemasangan (sekali saja)

1. Buka alamat aplikasi di browser: **https://apanzagg.github.io/rupiah-vision/**
2. Izinkan akses kamera bila diminta.
3. Tunggu sampai terdengar "Aplikasi siap, arahkan kamera ke uang". Di layar ada tulisan progres unduhan.
4. Pasang ke layar utama agar mudah dibuka dan bisa dipakai offline:
   - **Android (Chrome):** ketuk menu titik tiga, lalu **Tambahkan ke layar utama** atau **Instal aplikasi**.
   - **iPhone (Safari):** ketuk tombol **Bagikan**, lalu **Tambah ke Layar Utama**.

## Memakai

1. Buka SuaraRupiah dari layar utama. Kamera belakang menyala otomatis, tanpa tombol yang harus ditekan.
2. Pegang uang di depan kamera, jarak sekitar **20 sampai 50 cm** (kira-kira sejengkal sampai dua jengkal).
3. Aplikasi menyebutkan nominal, misalnya **"Lima puluh ribu rupiah"**. Di Android, ponsel juga bergetar dengan pola khas per nominal.
4. Jika ada beberapa lembar sekaligus, aplikasi menyebutkan jumlah lembar, nominal masing-masing, dan totalnya. Contoh: **"Dua lembar. Lima puluh ribu. Dua puluh ribu. Total tujuh puluh ribu rupiah."**
5. **Ketuk layar di mana saja** untuk mendengar ulang hasil terakhir.

### Tips agar cepat dikenali

- Jika terdengar **"Uang belum terdeteksi, coba balik uang atau jauhkan sedikit"**: balik uang, jauhkan sedikit, atau geser ke tengah kamera.
- Jika terdengar **"Ruangan terlalu gelap"**: pindah ke tempat lebih terang. Aplikasi mencoba menyalakan senter ponsel (jika ponsel mendukung).
- Usahakan minimal separuh permukaan uang terlihat (tidak tertutup jari).
- Uang yang sama tidak diulang dalam 3 detik. Jauhkan lalu dekatkan lagi untuk mengulang lebih cepat, atau ketuk layar.

### Pengguna TalkBack / VoiceOver

- Hasil juga dikirim ke pembaca layar, sehingga terdengar dari TalkBack/VoiceOver.
- Jika suara terdengar dua kali (suara aplikasi + pembaca layar), buka **Bantuan**, lalu matikan **Suara aplikasi**.
- Seluruh layar adalah satu tombol "Ulangi hasil terakhir". Ketuk dua kali (gaya pembaca layar) untuk mengulang.
- Tombol **Bantuan** ada di kanan atas.

### Jika browser menahan suara

Beberapa browser (terutama Safari iPhone, atau Chrome sebelum aplikasi dipasang ke layar utama) tidak mengizinkan suara
sebelum layar disentuh. **Ketuk layar sekali** setelah aplikasi siap; setelah itu suara berjalan normal.

## Pola getaran (Android)

| Nominal | Pola |
|---|---|
| Rp1.000 | 1 pendek |
| Rp2.000 | 2 pendek |
| Rp5.000 | 1 panjang |
| Rp10.000 | 1 panjang, 1 pendek |
| Rp20.000 | 1 panjang, 2 pendek |
| Rp50.000 | 2 panjang |
| Rp100.000 | 3 panjang |
| Beberapa lembar | 2 getar cepat, lalu pola nominal terbesar |

## Pengaturan (halaman Bantuan)

| Pengaturan | Bawaan | Keterangan |
|---|---|---|
| Ambang keyakinan | 0,70 | Lebih tinggi = lebih jarang salah sebut, tetapi lebih lama mengenali |
| Kecepatan bicara | 1,0 | 0,8 sampai 1,5 |
| Suara aplikasi | aktif | Matikan bila memakai pembaca layar dan suara terdengar dobel |
| Getaran | aktif | Tidak tersedia di iPhone |

## Batasan penting

- **Hanya mengenali nominal, tidak memeriksa keaslian uang.**
- Uang logam dan uang peringatan Rp75.000 tidak dikenali.
- Pada uang yang sangat rusak, sangat gelap, atau tertutup lebih dari separuh, hasil bisa salah atau tidak muncul.
  Untuk transaksi penting, ulangi dari sisi lain uang atau minta konfirmasi.

## Privasi

Semua pengenalan berjalan di ponsel. Tidak ada gambar kamera yang dikirim ke internet. Aplikasi tidak memakai analitik atau pelacak.

## Masalah umum

| Masalah | Solusi |
|---|---|
| "Izin kamera ditolak" | Buka pengaturan situs di browser, izinkan Kamera, lalu muat ulang |
| Progres unduhan berhenti | Periksa internet, lalu muat ulang. Unduhan pertama sekitar 10 MB |
| Tidak ada suara | Ketuk layar sekali; pastikan volume media menyala; cek "Suara aplikasi" di Bantuan |
| Aplikasi lambat | Tutup aplikasi lain; ponsel lama mengenali sekitar 2 sampai 5 kali per detik dan tetap bisa dipakai |
