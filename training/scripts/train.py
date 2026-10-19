"""Training YOLOv8n + evaluasi val/test.

Augmentasi: rotasi bebas (degrees=180), perspektif & shear kecil, skala, mosaic, hue shift KECIL
(warna adalah ciri pembeda nominal), TANPA flip, ditambah augmentasi kustom (augment_custom.py).
Early stopping lewat `patience`. Hasil: <project>/<name>/ (best.pt, results.csv, kurva, confusion matrix)
dan <project>/<name>/metrics.json.

Pemakaian: python train.py --data /tmp/final/data.yaml --imgsz 320 --name r320 [--device 0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import CLASSES

HYP = dict(
    degrees=180.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
    fliplr=0.0, flipud=0.0, hsv_h=0.005, hsv_s=0.5, hsv_v=0.4,
    mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0,
)


def metrics_dict(r) -> dict:
    b = r.box
    per_class = {CLASSES[int(c)]: {"ap50": round(float(b.ap50[i]), 4), "ap50_95": round(float(b.ap[i]), 4)}
                 for i, c in enumerate(b.ap_class_index)}
    return {"map50": round(float(b.map50), 4), "map50_95": round(float(b.map), 4),
            "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4), "per_class": per_class}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default="/kaggle/working/runs")
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-custom-aug", action="store_true")
    ap.add_argument("--cache", default="false", help="false | ram | disk (ram mempercepat bila RAM cukup)")
    ap.add_argument("--train-list", default="", help="ganti entri train di data.yaml, mis. train_balanced.txt")
    ap.add_argument("--max-hours", type=float, default=0, help="hentikan training dengan rapi setelah N jam (batas sesi Kaggle)")
    ap.add_argument("--set", nargs="*", default=[], help="override hyperparameter: key=value")
    args = ap.parse_args()

    from ultralytics import YOLO

    from augment_custom import build_augmentations

    hyp = dict(HYP)
    for kv in args.set:
        k, v = kv.split("=", 1)
        hyp[k] = type(hyp[k])(v) if k in hyp else (float(v) if v.replace(".", "", 1).isdigit() else v)
    if args.train_list:  # data.yaml varian dengan daftar train lain (mis. oversampling)
        import yaml

        d = yaml.safe_load(open(args.data, encoding="utf-8"))
        d["train"] = args.train_list
        alt = Path(args.data).with_name(f"data_{Path(args.train_list).stem}.yaml")
        alt.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
        args.data = str(alt)
    t0 = time.time()
    model = YOLO(args.model)
    if args.max_hours:
        def _time_guard(trainer):
            if time.time() - t0 > args.max_hours * 3600 and not trainer.stop:
                print(f"Batas waktu {args.max_hours} jam tercapai: training dihentikan setelah epoch {trainer.epoch + 1}")
                trainer.stop = True

        model.add_callback("on_fit_epoch_end", _time_guard)
    train_kw = dict(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        device=args.device, workers=args.workers, project=args.project, name=args.name, exist_ok=True,
        seed=args.seed, deterministic=False, plots=True, cache={"false": False, "ram": "ram", "disk": "disk"}[args.cache], **hyp,
    )
    if not args.no_custom_aug:
        train_kw["augmentations"] = build_augmentations()
    model.train(**train_kw)
    train_s = time.time() - t0

    run = Path(args.project) / args.name
    best = YOLO(run / "weights" / "best.pt")
    out = {"name": args.name, "imgsz": args.imgsz, "epochs_requested": args.epochs, "patience": args.patience,
           "batch": args.batch, "cache": args.cache, "train_list": args.train_list or "images/train", "max_hours": args.max_hours, "hyp": hyp, "custom_aug": not args.no_custom_aug, "train_seconds": round(train_s)}
    for split in ("val", "test"):
        r = best.val(data=args.data, split=split, imgsz=args.imgsz, batch=32, device=args.device, plots=True,
                     project=str(run), name=f"eval_{split}", exist_ok=True)
        out[split] = metrics_dict(r)
    import csv

    rows = list(csv.DictReader(open(run / "results.csv", encoding="utf-8")))
    out["epochs_trained"] = len(rows)
    best_row = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"].strip() or 0))
    out["best_epoch"] = int(float(best_row["epoch"]))
    (run / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
