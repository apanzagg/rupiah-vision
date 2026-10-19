# Kredit & lisensi audio

Klip `*.mp3` di folder ini dibuat dengan **Piper TTS** (https://github.com/rhasspy/piper, kode berlisensi MIT)
memakai suara **`id_ID-news_tts-medium`** dari repositori `rhasspy/piper-voices` (Hugging Face, berlisensi MIT).
Kartu model suara tersebut tidak menyatakan lisensi dataset rekaman aslinya dengan jelas ("See URL"), sehingga
klip ini diperlakukan sebagai bahan **prototipe akademik non-komersial**. Untuk distribusi luas/komersial,
ganti klip dengan rekaman suara manusia.

Cara mengganti: rekam ulang setiap kalimat di `manifest.json` (kolom `text`) dan simpan dengan **nama berkas yang sama**
(MP3 mono). Klip yang tidak ada otomatis diucapkan oleh Web Speech API browser (`id-ID`).

Pemilihan mesin TTS: dibandingkan dengan MMS-TTS `facebook/mms-tts-ind` (CC-BY-NC 4.0) memakai uji kejelasan otomatis
(transkripsi Whisper large-v3-turbo, character error rate). Piper: CER rata-rata 0,9% (106/109 klip persis benar);
MMS-TTS: CER 4,3% (90/109), termasuk salah ucap pada nominal. Detail: `docs/tts_asr_report.json`.
Kode pembuatan: `training/kaggle/tts/main.py`.
