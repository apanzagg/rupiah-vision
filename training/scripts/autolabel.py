"""Tahap 2 pipeline data: bounding box otomatis untuk dataset klasifikasi (tanpa box).

Detektor zero-shot YOLO-World (Ultralytics) dengan prompt teks "banknote"/"paper money" mencari letak uang;
label kelas diambil dari nama folder (kolom al_class di manifest).

Aturan penerimaan:
- tolak jika tidak ada box, atau confidence terbaik < MIN_CONF;
- box disimpan jika conf >= max(KEEP_ABS, KEEP_REL * conf_terbaik) dan luasnya >= MIN_AREA;
- box yang hampir seluruhnya berada di dalam box lain yang lebih yakin dibuang (bagian dari uang yang sama);
- citra dengan > 1 box ditandai al_multi=1 dan ditinjau di contact sheet.
Hasil ditinjau visual lewat contact sheet (analyze.py); id yang salah dimasukkan ke configs/reject_ids.txt.

Keluaran: SR_WORK/pool/labels/<id>.txt (ditimpa), SR_WORK/pool/manifest_autolabel.csv
"""
from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from common import WORK

PROMPTS = ["banknote", "paper money"]
MIN_CONF, KEEP_ABS, KEEP_REL, MIN_AREA = 0.20, 0.25, 0.5, 0.02


def contained(a: np.ndarray, b: np.ndarray) -> float:
    """Proporsi luas box a (xyxy) yang berada di dalam box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8x-worldv2.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    from ultralytics import YOLOWorld

    t0 = time.time()
    pool = WORK / "pool"
    rows = [r for r in csv.DictReader(open(pool / "manifest.csv", encoding="utf-8")) if r["status"] == "needs_autolabel"]
    print(f"{len(rows)} citra perlu auto-label")
    if not rows:
        return
    model = YOLOWorld(args.weights)
    model.set_classes(PROMPTS)
    out = []
    for s in range(0, len(rows), args.batch):
        chunk = rows[s : s + args.batch]
        paths = [str(pool / "images" / f"{r['id']}.jpg") for r in chunk]
        results = model.predict(paths, imgsz=args.imgsz, conf=0.05, iou=0.5, verbose=False)
        for r, res in zip(chunk, results):
            b = res.boxes
            rec = {"id": r["id"], "status": "ok", "classes": "", "n_boxes": 0, "al_conf": 0.0, "al_multi": 0}
            if b is None or len(b) == 0:
                rec["status"] = "al_reject_nobox"
                out.append(rec)
                continue
            conf = b.conf.cpu().numpy()
            xyxyn = b.xyxyn.cpu().numpy()
            order = np.argsort(-conf)
            best = float(conf[order[0]])
            rec["al_conf"] = round(best, 3)
            if best < MIN_CONF:
                rec["status"] = "al_reject_lowconf"
                out.append(rec)
                continue
            keep: list[np.ndarray] = []
            for k in order:
                if conf[k] < max(KEEP_ABS, KEEP_REL * best) and keep:
                    break
                box = xyxyn[k]
                if (box[2] - box[0]) * (box[3] - box[1]) < MIN_AREA:
                    continue
                if any(contained(box, kb) > 0.8 or contained(kb, box) > 0.8 for kb in keep):
                    continue
                keep.append(box)
            if not keep:
                rec["status"] = "al_reject_small"
                out.append(rec)
                continue
            c = int(r["al_class"])
            lines = []
            for x0, y0, x1, y1 in keep:
                lines.append(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}\n")
            (pool / "labels" / f"{r['id']}.txt").write_text("".join(lines))
            rec.update({"classes": " ".join([str(c)] * len(keep)), "n_boxes": len(keep), "al_multi": int(len(keep) > 1)})
            out.append(rec)
        print(f"  {min(s + args.batch, len(rows))}/{len(rows)} ({time.time() - t0:.0f} dtk)")
    with open(pool / "manifest_autolabel.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    st = {}
    for o in out:
        st[o["status"]] = st.get(o["status"], 0) + 1
    print("Selesai auto-label:", st, f"multi-box: {sum(o['al_multi'] for o in out)}")


if __name__ == "__main__":
    main()
