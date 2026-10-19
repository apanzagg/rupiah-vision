"""Audit label berbasis model: cari label yang kemungkinan salah (mis. 2000 tertulis 20000) per sumber.

Model (ronde audit) memprediksi semua citra train/val/test. Untuk setiap box label dicari prediksi dengan IoU >= 0,5:
  wrong_class : ada prediksi yakin (>= CONF_WRONG) di lokasi yang sama tetapi kelasnya beda dengan label
  missed      : tidak ada prediksi sama sekali di lokasi label
  extra       : prediksi yakin tanpa label di lokasinya (kemungkinan label hilang)
Hasil per sumber dipakai untuk keputusan pembersihan (dicatat di docs/EKSPERIMEN.md); citra yang dicurigai
diperiksa visual lewat contact sheet sebelum diputuskan.

Keluaran (--out): audit.csv, audit_summary.json, sheets/wrong_<sumber>.jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from common import CLASSES, SEED, WORK

CONF_WRONG = 0.6


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data-root", default="/tmp/final")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from ultralytics import YOLO

    out = Path(args.out)
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    root = Path(args.data_root)
    man = {r["id"]: r for r in csv.DictReader(open(WORK / "manifest_final.csv", encoding="utf-8"))}
    model = YOLO(args.weights)
    rows = []
    for split in ("train", "val", "test"):
        imgs = sorted((root / "images" / split).glob("*.jpg"))
        for res in model.predict([str(p) for p in imgs], imgsz=args.imgsz, conf=0.25, iou=0.45, agnostic_nms=True,
                                 stream=True, verbose=False, batch=64):
            p = Path(res.path)
            h, w = res.orig_shape
            gt = []
            for line in (root / "labels" / split / f"{p.stem}.txt").read_text().splitlines():
                v = line.split()
                if len(v) == 5:
                    c, xc, yc, bw, bh = int(v[0]), *map(float, v[1:])
                    gt.append((c, [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
            pb = res.boxes.xyxy.cpu().numpy().tolist()
            pc = res.boxes.cls.cpu().numpy().astype(int).tolist()
            ps = res.boxes.conf.cpu().numpy().tolist()
            used, wrong, missed = set(), [], 0
            for c, b in gt:
                cand = sorted(((iou(b, q), j) for j, q in enumerate(pb) if j not in used), reverse=True)
                if cand and cand[0][0] >= 0.5:
                    j = cand[0][1]
                    used.add(j)
                    if pc[j] != c and ps[j] >= CONF_WRONG:
                        wrong.append((CLASSES[c], CLASSES[pc[j]], round(ps[j], 3)))
                else:
                    missed += 1
            extra = sum(1 for j in range(len(pb)) if j not in used and ps[j] >= CONF_WRONG)
            m = man.get(p.stem, {})
            rows.append({"id": p.stem, "split": split, "slug": m.get("slug", "?"), "role": m.get("role", "?"),
                         "gt": " ".join(CLASSES[c] for c, _ in gt), "n_wrong": len(wrong),
                         "wrong": ";".join(f"{a}>{b}@{s}" for a, b, s in wrong), "missed": missed, "extra": extra,
                         "max_wrong_conf": max((s for _, _, s in wrong), default=0)})
    with open(out / "audit.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["slug"]].append(r)
    summary = {}
    for slug, rs in by_src.items():
        pos = [r for r in rs if r["role"] == "pos"]
        neg = [r for r in rs if r["role"] == "neg"]
        pairs = Counter(x.split("@")[0] for r in pos for x in r["wrong"].split(";") if x)
        summary[slug] = {"images": len(rs), "pos": len(pos),
                         "wrong_img_rate": round(sum(r["n_wrong"] > 0 for r in pos) / max(1, len(pos)), 4),
                         "missed_img_rate": round(sum(r["missed"] > 0 for r in pos) / max(1, len(pos)), 4),
                         "neg_fp_rate": round(sum(r["extra"] > 0 for r in neg) / max(1, len(neg)), 4) if neg else None,
                         "top_pairs": pairs.most_common(6)}
    summary = dict(sorted(summary.items(), key=lambda kv: -kv[1]["wrong_img_rate"]))
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for slug, s in summary.items():
        print(f"{s['wrong_img_rate']:.3f} salah-kelas | {s['missed_img_rate']:.3f} terlewat | {s['pos']:5d} | {slug} {s['top_pairs'][:3]}")
    # contact sheet citra salah-kelas untuk 10 sumber teratas
    rng = random.Random(SEED)
    th, cols = 160, 8
    for slug in [k for k, v in summary.items() if v["wrong_img_rate"] > 0][:10]:
        wr_rows = [r for r in by_src[slug] if r["n_wrong"] > 0]
        rng.shuffle(wr_rows)
        wr_rows = wr_rows[:48]
        sheet = np.full((((len(wr_rows) + cols - 1) // cols) * (th + 16), cols * th, 3), 255, np.uint8)
        for k, r in enumerate(wr_rows):
            im = cv2.imread(str(root / "images" / r["split"] / f"{r['id']}.jpg"))
            s = th / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1] * s), int(im.shape[0] * s)))
            y, x = (k // cols) * (th + 16), (k % cols) * th
            sheet[y : y + im.shape[0], x : x + im.shape[1]] = im
            cv2.putText(sheet, r["wrong"].split("@")[0].replace(">", " vs "), (x + 2, y + th + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        cv2.imwrite(str(out / "sheets" / f"wrong_{slug.replace('/', '__')}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
