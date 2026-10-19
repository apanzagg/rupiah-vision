"""Masukkan foto uang asli milik pengguna (data/raw/foto_saya/) ke set uji per kondisi.

Aturan nama berkas (lihat docs/PANDUAN_FOTO.md):
  <nominal>_<kondisi>_<sisi>_<nomor>.jpg                 contoh 50000_lusuh_depan_01.jpg
  campuran_<n1>-<n2>[-<n3>]_<kondisi>_<nomor>.jpg         contoh campuran_20000-50000_baru_01.jpg
  bukan_<jenis>_<nomor>.jpg                               contoh bukan_struk_01.jpg

Langkah:
1. validasi nama berkas (yang tidak sesuai dilaporkan & dilewati);
2. koreksi orientasi EXIF, perkecil sisi terpanjang ke 1280 px;
3. bounding box diusulkan model ONNX; nominal dari NAMA BERKAS adalah label kebenaran:
   - jumlah & nominal deteksi cocok dengan nama berkas -> box model dipakai (label = nama berkas)
   - tidak cocok -> citra masuk daftar tinjau (contact sheet) dan TIDAK dipakai sampai diperiksa,
     agar evaluasi tidak bias ke kemampuan model sendiri: tambahkan box manual di
     training/configs/user_photo_boxes.csv (file,kelas,x0,y0,x1,y1 relatif) lalu jalankan ulang;
4. disalin ke data/test_conditions/foto_saya_<kondisi>/ (kondisi: baru, lusuh_kusut, terlipat_tertutup,
   cahaya_redup, latar_ramai, bukan_rupiah) agar dilaporkan terpisah dari set uji publik.

Pemakaian: python ingest_user_photos.py --model ../../models/model-v1-int8_mixed.onnx
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from common import CLASSES, CONFIGS, DATA
from infer_ref import OnnxDetector

SRC = DATA / "raw" / "foto_saya"
OUT = DATA / "test_conditions"
KONDISI = {"baru": "baru", "lusuh": "lusuh_kusut", "terlipat": "terlipat_tertutup", "tertutup": "terlipat_tertutup",
           "redup": "cahaya_redup", "ramai": "latar_ramai"}
RE_SINGLE = re.compile(r"^(1000|2000|5000|10000|20000|50000|100000)_(baru|lusuh|terlipat|tertutup|redup|ramai)_(depan|belakang)_\d+$")
RE_MULTI = re.compile(r"^campuran_((?:1000|2000|5000|10000|20000|50000|100000)(?:-(?:1000|2000|5000|10000|20000|50000|100000)){1,4})_(baru|lusuh|terlipat|tertutup|redup|ramai)_\d+$")
RE_NEG = re.compile(r"^bukan_[a-z0-9]+_\d+$")


def parse(stem: str) -> tuple[list[str], str] | None:
    s = stem.lower()
    if m := RE_SINGLE.match(s):
        return [m.group(1)], KONDISI[m.group(2)]
    if m := RE_MULTI.match(s):
        return m.group(1).split("-"), KONDISI[m.group(2)]
    if RE_NEG.match(s):
        return [], "bukan_rupiah"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    files = sorted(p for p in SRC.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".heic", ".webp"}) if SRC.exists() else []
    if not files:
        print(f"Tidak ada foto di {SRC}. Lihat docs/PANDUAN_FOTO.md.")
        return
    manual = {}
    mf = CONFIGS / "user_photo_boxes.csv"
    if mf.exists():
        for r in csv.DictReader(open(mf, encoding="utf-8")):
            manual.setdefault(r["file"], []).append(r)
    det = OnnxDetector(args.model)
    for d in OUT.glob("foto_saya_*"):
        shutil.rmtree(d)
    report = {"total": len(files), "bad_name": [], "review": [], "used": 0}
    review_tiles = []
    for p in files:
        parsed = parse(p.stem)
        if parsed is None:
            report["bad_name"].append(p.name)
            continue
        gt, cond = parsed
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((1280, 1280))
        rgb = np.asarray(im)
        h, w = rgb.shape[:2]
        if p.name in manual:
            lines = []
            for r in manual[p.name]:
                x0, y0, x1, y1 = (float(r[c]) for c in ("x0", "y0", "x1", "y1"))
                lines.append(f"{CLASSES.index(r['kelas'])} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}")
        else:
            preds = [q for q in det(rgb, conf=0.4) if q["score"] >= 0.4]
            if sorted(CLASSES[q["cls"]] for q in preds) != sorted(gt):
                report["review"].append({"file": p.name, "label": gt, "pred": [CLASSES[q["cls"]] for q in preds]})
                review_tiles.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                continue
            # label kebenaran dari nama berkas; box dari model (urut sama karena multiset identik)
            lines = []
            for q in preds:
                x0, y0, x1, y1 = q["box"]
                lines.append(f"{q['cls']} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
        dst = OUT / f"foto_saya_{cond}"
        (dst / "images").mkdir(parents=True, exist_ok=True)
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        im.save(dst / "images" / f"{p.stem}.jpg", quality=92)
        (dst / "labels" / f"{p.stem}.txt").write_text("".join(l + "\n" for l in lines))
        report["used"] += 1
    for d in OUT.glob("foto_saya_*"):
        (d / "meta.json").write_text(json.dumps({"real": len(list((d / "images").glob("*.jpg"))), "synthetic": 0,
                                                 "source": "foto uang asli milik pengguna"}, indent=1), encoding="utf-8")
    if review_tiles:
        th = 200
        sheet = np.full(((len(review_tiles) + 7) // 8 * th, 8 * th, 3), 255, np.uint8)
        for k, t in enumerate(review_tiles):
            s = th / max(t.shape[:2])
            t = cv2.resize(t, (int(t.shape[1] * s), int(t.shape[0] * s)))
            sheet[(k // 8) * th : (k // 8) * th + t.shape[0], (k % 8) * th : (k % 8) * th + t.shape[1]] = t
        cv2.imwrite(str(DATA / "interim" / "foto_saya_tinjau.jpg"), sheet)
    (DATA / "interim" / "foto_saya_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
