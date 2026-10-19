"""Evaluasi model ONNX per kondisi pada data/test_conditions/<kondisi>/{images,labels}.

Memakai pipeline referensi yang sama dengan web (infer_ref.py: letterbox, decode, NMS class-agnostic).
Metrik per kondisi:
- akurasi citra : multiset nominal yang diumumkan (skor >= threshold) sama persis dengan label
                  (untuk 'bukan_rupiah': benar bila tidak ada nominal yang diumumkan)
- precision / recall / F1 per kelas (pencocokan box IoU >= 0.5, class-agnostic lalu dibandingkan kelasnya)
- confusion matrix 8x8 (7 kelas + 'latar'): baris = label, kolom = prediksi
Keluaran: <out>/conditions_metrics.json dan <out>/confusion_<kondisi>.png

Pemakaian: python eval_conditions.py --model ../../models/model-v1-int8_mixed.onnx [--thr 0.7] [--out ../../docs/figures]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import CLASSES, DATA
from infer_ref import OnnxDetector

CONDITIONS = ["baru", "lusuh_kusut", "terlipat_tertutup", "cahaya_redup", "latar_ramai", "bukan_rupiah"]
NAMES = [f"Rp{int(c):,}".replace(",", ".") for c in CLASSES] + ["latar"]


def iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def read_gt(lbl: Path, w: int, h: int) -> list[tuple[int, list[float]]]:
    out = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                c, xc, yc, bw, bh = int(v[0]), *map(float, v[1:])
                out.append((c, [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]))
    return out


def evaluate_dir(det: OnnxDetector, d: Path, thr: float) -> dict:
    nc = len(CLASSES)
    cm = np.zeros((nc + 1, nc + 1), int)
    correct, n_img, details = 0, 0, []
    for img_path in sorted((d / "images").glob("*.jpg")):
        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt = read_gt(d / "labels" / f"{img_path.stem}.txt", w, h)
        preds = [p for p in det(img, conf=0.25, iou=0.45) if p["score"] >= thr]
        used = set()
        for gc, gb in gt:
            best, bj = 0.5, -1
            for j, p in enumerate(preds):
                if j not in used and iou(gb, p["box"]) >= best:
                    best, bj = iou(gb, p["box"]), j
            if bj >= 0:
                used.add(bj)
                cm[gc, preds[bj]["cls"]] += 1
            else:
                cm[gc, nc] += 1
        for j, p in enumerate(preds):
            if j not in used:
                cm[nc, p["cls"]] += 1
        ok = sorted(c for c, _ in gt) == sorted(p["cls"] for p in preds)
        correct += ok
        n_img += 1
        details.append({"image": img_path.name, "gt": [CLASSES[c] for c, _ in gt], "pred": [CLASSES[p["cls"]] for p in preds],
                        "scores": [round(p["score"], 3) for p in preds], "ok": ok})
    per_class = {}
    for c in range(nc):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        per_class[CLASSES[c]] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                                 "support": int(cm[c, :].sum())}
    tp = np.trace(cm[:nc, :nc])
    fp = cm[:, :nc].sum() - tp
    fn = cm[:nc, :].sum() - tp
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"images": n_img, "accuracy": round(correct / max(1, n_img), 4), "precision": round(P, 4), "recall": round(R, 4),
            "f1": round(2 * P * R / (P + R), 4) if P + R else 0.0, "per_class": per_class, "confusion": cm.tolist(), "details": details}


def plot_cm(cm: np.ndarray, title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=150)
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(NAMES)), NAMES, fontsize=8)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label")
    ax.set_title(title, loc="left", fontsize=10)
    vmax = cm.max() or 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7, color="white" if cm[i, j] > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--root", default=str(DATA / "test_conditions"))
    ap.add_argument("--out", default=str(DATA / "interim" / "conditions_eval"))
    args = ap.parse_args()
    det = OnnxDetector(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    res = {"model": Path(args.model).name, "threshold": args.thr, "conditions": {}}
    for cond in CONDITIONS + sorted(p.name for p in root.iterdir() if p.is_dir() and p.name not in CONDITIONS):
        d = root / cond
        if not (d / "images").exists():
            continue
        r = evaluate_dir(det, d, args.thr)
        meta = d / "meta.json"
        r["meta"] = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        res["conditions"][cond] = r
        plot_cm(np.array(r["confusion"]), f"Confusion matrix: {cond} (n={r['images']})", out / f"confusion_{cond}.png")
        print(f"{cond:20s} n={r['images']:4d}  akurasi={r['accuracy']:.3f}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}")
    (out / "conditions_metrics.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
