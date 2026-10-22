"""Grid contoh deteksi benar & salah dari hasil eval_conditions.py (untuk Bab 4.8 laporan).

Memilih citra dari conditions_metrics.json (deterministik: urutan nama berkas), menjalankan ulang model untuk
menggambar box. Hijau = diumumkan (skor >= ambang), abu = di bawah ambang (tidak diumumkan).
Keluaran: docs/figures/examples_correct.jpg dan docs/figures/examples_wrong.jpg

Pemakaian: python make_examples.py --model ../../models/model-v1-fp16.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import CLASSES, DATA, ROOT
from infer_ref import OnnxDetector

TW, TH, CAP = 400, 300, 44
LABEL = {c: f"Rp{int(c):,}".replace(",", ".") for c in CLASSES}


def kind(r: dict) -> str:
    if r["ok"]:
        return "benar"
    if not r["pred"]:
        return "diam"
    return "salah_sebut" if set(r["pred"]) - set(r["gt"]) else "kurang_lembar"


def tile(det: OnnxDetector, img_path: Path, cond: str, r: dict, thr: float) -> np.ndarray:
    img = cv2.imread(str(img_path))
    dets = det(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), conf=0.25, iou=0.45)
    s = min(TW / img.shape[1], (TH - CAP) / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    canvas = np.full((TH, TW, 3), 255, np.uint8)
    ox, oy = (TW - im.shape[1]) // 2, CAP + (TH - CAP - im.shape[0]) // 2
    canvas[oy:oy + im.shape[0], ox:ox + im.shape[1]] = im
    for d in sorted(dets, key=lambda d: d["score"]):
        on = d["score"] >= thr
        x0, y0, x1, y1 = (int(v * s) for v in d["box"])
        col = (40, 170, 40) if on else (150, 150, 150)
        cv2.rectangle(canvas, (ox + x0, oy + y0), (ox + x1, oy + y1), col, 2 if on else 1)
        cv2.putText(canvas, f"{LABEL[CLASSES[d['cls']]]} {d['score']:.2f}", (ox + x0 + 3, oy + max(y0, 0) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    gt = ", ".join(LABEL[c] for c in r["gt"]) or "bukan rupiah"
    said = ", ".join(LABEL[c] for c in r["pred"]) or "(diam)"
    cv2.putText(canvas, f"{cond}  [{kind(r)}]", (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"label: {gt} | diumumkan: {said}"[:60], (6, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return canvas


def grid(tiles: list[np.ndarray], cols: int = 4) -> np.ndarray:
    while len(tiles) % cols:
        tiles.append(np.full((TH, TW, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    return np.vstack(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--metrics", default=str(DATA / "interim" / "conditions_eval" / "conditions_metrics.json"))
    ap.add_argument("--root", default=str(DATA / "test_conditions"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()
    res = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    det, root, out = OnnxDetector(args.model), Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    good, bad = [], []
    for cond, c in res["conditions"].items():
        rows = sorted(c["details"], key=lambda r: r["image"])
        ok = [r for r in rows if r["ok"]]
        # satu contoh benar per kondisi (utamakan multi-lembar); salah: maks. 2 salah sebut + 1 diam per kondisi
        ok.sort(key=lambda r: -len(r["gt"]))
        good += [(cond, r) for r in ok[:1]]
        bad += [(cond, r) for r in rows if kind(r) == "salah_sebut"][:2]
        bad += [(cond, r) for r in rows if kind(r) == "diam"][:1]
    bad.sort(key=lambda x: kind(x[1]) != "salah_sebut")
    for name, items in (("examples_correct.jpg", good[:8]), ("examples_wrong.jpg", bad[:8])):
        tiles = [tile(det, root / cond / "images" / r["image"], cond, r, res["threshold"]) for cond, r in items]
        cv2.imwrite(str(out / name), grid(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(name, [f"{cond}/{r['image']}" for cond, r in items])


if __name__ == "__main__":
    main()
