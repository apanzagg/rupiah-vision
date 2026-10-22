"""mAP varian ONNX memakai pipeline REFERENSI web (infer_ref.py), bukan validator Ultralytics.

Alasan: validator Ultralytics untuk model ONNX statis memakai letterbox persegi (rect=False, batch=1). Pada
Ultralytics 8.3.253 mode itu menghasilkan mAP lebih rendah daripada rect=True untuk MODEL YANG SAMA
(PyTorch e1_r320 pada test: rect=True 0,9566/0,8054; rect=False 0,9285/0,6801), padahal prediksi PyTorch dan
ONNX identik. Skrip ini menghitung mAP dari prediksi pipeline aplikasi (letterbox tengah, decode, NMS),
memakai fungsi metrik Ultralytics (ap_per_class) agar definisinya sama.

Pemakaian: python eval_onnx_map.py --models a.onnx b.onnx --root ../../data/final_test/final --out hasil.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.utils.metrics import ap_per_class, box_iou

from infer_ref import OnnxDetector

IOUV = torch.linspace(0.5, 0.95, 10)


def match(pred_cls, gt_cls, iou):
    """Matriks benar [n_pred, 10] seperti DetectionValidator.match_predictions (tanpa scipy)."""
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    if len(gt_cls) == 0 or len(pred_cls) == 0:
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    iou = iou.numpy()
    for i, thr in enumerate(IOUV.tolist()):
        m = np.array(np.nonzero(iou >= thr)).T
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return correct


def evaluate(model: str, root: Path, conf: float, iou_nms: float) -> dict:
    det = OnnxDetector(model)
    stats = {"tp": [], "conf": [], "pred_cls": [], "target_cls": []}
    t0 = time.time()
    imgs = sorted((root / "images" / "test").glob("*.jpg"))
    for p in imgs:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt = [l.split() for l in (root / "labels" / "test" / f"{p.stem}.txt").read_text().splitlines() if len(l.split()) == 5]
        gcls = np.array([int(g[0]) for g in gt])
        gbox = torch.tensor([[(float(g[1]) - float(g[3]) / 2) * w, (float(g[2]) - float(g[4]) / 2) * h,
                              (float(g[1]) + float(g[3]) / 2) * w, (float(g[2]) + float(g[4]) / 2) * h] for g in gt]).reshape(-1, 4)
        preds = det(img, conf=conf, iou=iou_nms, max_det=300)
        pcls = np.array([d["cls"] for d in preds])
        pconf = np.array([d["score"] for d in preds])
        pbox = torch.tensor([d["box"] for d in preds]).reshape(-1, 4)
        iou = box_iou(gbox, pbox) if len(gt) and len(preds) else torch.zeros((len(gt), len(preds)))
        stats["tp"].append(match(pcls, gcls, iou))
        stats["conf"].append(pconf)
        stats["pred_cls"].append(pcls)
        stats["target_cls"].append(gcls)
    s = {k: np.concatenate(v, 0) for k, v in stats.items()}
    res = ap_per_class(s["tp"], s["conf"], s["pred_cls"], s["target_cls"])
    p, r, ap = res[2], res[3], res[5]  # per kelas: precision, recall, AP[10 IoU]
    return {"model": Path(model).name, "images": len(imgs), "map50": round(float(ap[:, 0].mean()), 4),
            "map50_95": round(float(ap.mean()), 4), "precision": round(float(p.mean()), 4), "recall": round(float(r.mean()), 4),
            "ap50_per_class": [round(float(x), 4) for x in ap[:, 0]], "seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    args = ap.parse_args()
    out = []
    for m in args.models:
        r = evaluate(m, Path(args.root), args.conf, args.iou)
        print(json.dumps(r))
        out.append(r)
    Path(args.out).write_text(json.dumps({"protocol": "pipeline referensi web (infer_ref.py), letterbox tengah, NMS class-agnostic, "
                                                      f"conf {args.conf}, IoU NMS {args.iou}", "results": out}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
