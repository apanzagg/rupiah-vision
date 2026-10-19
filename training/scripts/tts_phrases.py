"""Daftar frasa suara SuaraRupiah (kunci -> teks) + angka ke kata Bahasa Indonesia.

Satu sumber kebenaran untuk: pembuatan audio TTS (kernel Kaggle tts), web/public/audio/manifest.json,
dan vektor uji fungsi angka->kata versi TypeScript (web/src/lib/numwords.ts).
Pemakaian: python tts_phrases.py <keluaran manifest.json> [<keluaran vektor uji json>]
"""
from __future__ import annotations

import json
import sys
from itertools import combinations_with_replacement

DENOMS = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
MAX_NOTES_AUDIO = 3  # total untuk kombinasi 2..3 lembar dibuat audionya; lebih dari itu memakai Web Speech
UNITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]


def num_to_words(n: int) -> str:
    if n < 0:
        raise ValueError(n)
    if n < 10:
        return UNITS[n]
    if n == 10:
        return "sepuluh"
    if n == 11:
        return "sebelas"
    if n < 20:
        return f"{UNITS[n - 10]} belas"
    if n < 100:
        t, r = divmod(n, 10)
        return f"{UNITS[t]} puluh" + (f" {UNITS[r]}" if r else "")
    if n < 1000:
        h, r = divmod(n, 100)
        head = "seratus" if h == 1 else f"{UNITS[h]} ratus"
        return head + (f" {num_to_words(r)}" if r else "")
    if n < 1_000_000:
        th, r = divmod(n, 1000)
        head = "seribu" if th == 1 else f"{num_to_words(th)} ribu"
        return head + (f" {num_to_words(r)}" if r else "")
    if n < 1_000_000_000:
        m, r = divmod(n, 1_000_000)
        return f"{num_to_words(m)} juta" + (f" {num_to_words(r)}" if r else "")
    raise ValueError(n)


def cap(s: str) -> str:
    return s[:1].upper() + s[1:]


HELP_TEXT = (
    "Cara memakai SuaraRupiah. Arahkan kamera belakang ke uang kertas, dengan jarak sekitar dua puluh "
    "sampai lima puluh sentimeter. Aplikasi akan menyebutkan nominal uang secara otomatis. "
    "Jika ada lebih dari satu lembar, aplikasi menyebutkan jumlah lembar, nominal masing-masing, dan totalnya. "
    "Ketuk layar di mana saja untuk mengulang hasil terakhir. "
    "Jika uang belum terdeteksi, coba balik uang, jauhkan sedikit, atau pindah ke tempat yang lebih terang. "
    "Aplikasi ini hanya mengenali nominal, tidak memeriksa keaslian uang. "
    "Uang logam dan uang peringatan tujuh puluh lima ribu rupiah tidak dikenali. "
    "Semua proses berjalan di ponsel Anda. Tidak ada gambar kamera yang dikirim ke internet."
)


def phrases() -> dict[str, str]:
    p = {
        "status_preparing": "Sedang menyiapkan aplikasi, mohon tunggu.",
        "status_ready": "Aplikasi siap, arahkan kamera ke uang.",
        "guide_not_found": "Uang belum terdeteksi, coba balik uang atau jauhkan sedikit.",
        "guide_dark": "Ruangan terlalu gelap.",
        "guide_torch_on": "Senter dinyalakan.",
        "error_camera_denied": "Izin kamera ditolak. Mohon izinkan akses kamera di pengaturan browser, lalu muat ulang aplikasi.",
        "error_camera": "Kamera tidak dapat dibuka.",
        "error_model": "Gagal memuat model. Periksa koneksi internet, lalu muat ulang aplikasi.",
        "help": HELP_TEXT,
    }
    for d in DENOMS:
        p[f"d{d}"] = cap(num_to_words(d)) + " rupiah."
        p[f"n{d}"] = num_to_words(d) + "."
    for k in range(2, 6):
        p[f"c{k}"] = f"{cap(num_to_words(k))} lembar."
    totals = sorted({sum(c) for n in range(2, MAX_NOTES_AUDIO + 1) for c in combinations_with_replacement(DENOMS, n)})
    for t in totals:
        p[f"t{t}"] = f"Total {num_to_words(t)} rupiah."
    return p


def main() -> None:
    p = phrases()
    manifest = {"lang": "id-ID", "format": "mp3", "max_notes_audio": MAX_NOTES_AUDIO,
                "phrases": {k: {"text": v, "file": f"{k}.mp3"} for k, v in p.items()}}
    with open(sys.argv[1], "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    if len(sys.argv) > 2:
        vec = {str(n): num_to_words(n) for n in
               [0, 1, 9, 10, 11, 12, 19, 20, 21, 99, 100, 101, 111, 150, 999, 1000, 1001, 1500, 2000, 10000, 11000,
                21000, 70000, 100000, 101000, 120000, 152000, 200000, 215000, 300000, 999999, 1000000, 1250000]}
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            json.dump(vec, f, ensure_ascii=False, indent=1)
    print(f"{len(p)} frasa")


if __name__ == "__main__":
    main()
