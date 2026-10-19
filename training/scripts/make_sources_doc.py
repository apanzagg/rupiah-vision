"""Bangun docs/SUMBER_DATASET.md dari training/configs/sources.yaml + data/interim/rf_survey.json.

Jika data/interim/source_stats.json ada (hasil build_dataset.py), kolom "dipakai" diisi
dengan jumlah citra yang lolos normalisasi & dedup.
"""
from __future__ import annotations

import json

import yaml

from common import DATA, ROOT

HEADER = """# Sumber Dataset

Semua dataset di bawah dipakai sesuai lisensinya. Dataset ber-lisensi **CC BY 4.0** mewajibkan atribusi:
tabel ini berfungsi sebagai atribusi (nama pemilik workspace/akun, judul, URL, lisensi).
Berkas ini dibangkitkan oleh `training/scripts/make_sources_doc.py`; jangan diedit manual.

Kolom:
- **citra sumber**: jumlah citra asli di proyek (sebelum augmentasi Roboflow), dari API Roboflow/Kaggle.
- **versi & augmentasi**: versi yang diunduh dan augmentasi bawaan versi itu (`none` = tanpa augmentasi).
- **dipakai**: jumlah citra yang lolos normalisasi kelas dan penghapusan duplikat (diisi setelah pipeline data berjalan).
"""


def main() -> None:
    cfg = yaml.safe_load((ROOT / "training/configs/sources.yaml").read_text(encoding="utf-8"))
    survey = json.loads((DATA / "interim/rf_survey.json").read_text(encoding="utf-8"))
    stats_path = DATA / "interim/source_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

    lines = [HEADER, "## Roboflow Universe (object detection, format YOLOv8)\n"]
    lines.append("| # | Dataset (pemilik/proyek) | Lisensi | Peran | Citra sumber | Versi & augmentasi | Kelas asli | Dipakai |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, s in enumerate(cfg["roboflow"], 1):
        p = survey.get(s["slug"], {}).get("project", {})
        classes = ", ".join(k for k in (p.get("classes") or {}).keys())
        url = f"https://universe.roboflow.com/{s['slug']}/dataset/{s['version']}"
        used = stats.get(s["slug"], {}).get("kept", "")
        lines.append(
            f"| {i} | [{p.get('name', s['slug'])}]({url}) (`{s['slug']}`) | {s['license']} | {s['role']} | "
            f"{p.get('images', '')} | v{s['version']}, {s['aug']} | {classes} | {used} |"
        )
    lines.append("\n## Kaggle\n")
    lines.append("| # | Dataset | Lisensi | Peran | Format | Dipakai |")
    lines.append("|---|---|---|---|---|---|")
    for i, s in enumerate(cfg.get("kaggle", []), 1):
        url = f"https://www.kaggle.com/datasets/{s['ref']}"
        used = stats.get(s["ref"], {}).get("kept", "")
        lines.append(f"| {i} | [{s['ref']}]({url}) | {s['license']} | {s['role']} | {s.get('format', 'klasifikasi per folder')} | {used} |")
    lines.append(
        """
## Dataset yang dikecualikan

| Dataset | Alasan |
|---|---|
| `craft-model/rupiah-currentcy-detection`, `craft-model/rupiah-currentcy-detection-v2`, `machine-learning-pmhkb/deteksi-rupiah-jkkmu` (Roboflow) | Lisensi "Private" |
| Proyek Roboflow yang hanya punya versi augmentasi dengan **flip** (cermin), mis. `4ia17ottos-workspace/uang_baru`, `detection-twl6q/rupiah_emisi-baru`, `test-sirmi/deteksi-uang-rupiah-8zlc2` | Uang yang tercermin tidak ada di dunia nyata dan bisa mengajari model ciri yang salah |
| `orbitaibanknotes/indonesian-banknotes` | Hanya satu kelas "rupiah" tanpa nominal |
| `muhammad-ihsan-sinaga/dataset-gabungan-rupiah`, `first-oi8bn/money-detection-q0jko` | Nama kelas tidak konsisten/tidak bisa dipetakan dengan pasti; sebagian besar isinya gabungan dataset lain |
| Kaggle: `anidwiastuti/rupiah-banknotes-dataset`, `fannyzahrahramadhan/*`, `nurulalfiyyah/rupiah-banknotes`, `brotoa/*`, `dimaswara/*`, `kevinwaska/uang-baru`, `muhrisal100786/*`, `yunhann/*` | Lisensi "Unknown" (tidak dinyatakan) |
| Kaggle: `fzmhww8782/rupiah-22`, `hashimatulzaria/data-uang-rupiah` | Lisensi "Other (specified in description)" tetapi deskripsinya kosong |

Catatan: banyak proyek Roboflow merupakan salinan (fork) satu sama lain. Duplikat lintas sumber dihapus dengan
perceptual hash sebelum split (lihat `docs/LAPORAN_EVALUASI.md`).
"""
    )
    (ROOT / "docs/SUMBER_DATASET.md").write_text("\n".join(lines), encoding="utf-8")
    print("docs/SUMBER_DATASET.md diperbarui")


if __name__ == "__main__":
    main()
