"""Ekspor best.pt -> ONNX (FP32, FP16, INT8) + evaluasi mAP tiap varian + referensi parity test.

Varian:
- fp32       : ekspor Ultralytics, opset 12, simplify
- fp16       : onnxruntime.transformers.float16, keep_io_types (input/output tetap float32), Resize tetap FP32
- int8       : kuantisasi statis QDQ per-channel (onnxruntime, graf dinaikkan ke opset 13), kalibrasi citra train
- int8_mixed : seperti int8 tetapi head deteksi (/model.22/*) tetap FP32 (sering lebih akurat)
Setiap varian dievaluasi pada split test dengan Ultralytics val (backend onnxruntime) sehingga
semua varian (termasuk baseline FP32) memakai pipeline evaluasi yang sama.
Pemilihan: varian TERKECIL dengan penurunan mAP50 <= 1 poin dibanding FP32 ONNX.

Keluaran di --out: model-<ver>-<varian>.onnx, variants.json, parity/ (citra uji + parity_ref.json)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np

from common import CLASSES, SEED
from infer_ref import OnnxDetector, preprocess


class CalibReader:
    def __init__(self, paths: list[Path], size: int, input_name: str):
        import cv2

        self.items = []
        for p in paths:
            img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
            self.items.append({input_name: preprocess(img, size)[0]})
        self.it = iter(self.items)

    def get_next(self):
        return next(self.it, None)

    def rewind(self):
        self.it = iter(self.items)


def quantize_int8(fp32: Path, out: Path, calib: list[Path], size: int, exclude_head: bool) -> None:
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # kuantisasi per-channel (DequantizeLinear dengan atribut axis) butuh opset >= 13
    from onnx import version_converter

    up = out.with_suffix(".op13.onnx")
    onnx.save(version_converter.convert_version(onnx.load(str(fp32)), 13), str(up))
    pre = out.with_suffix(".pre.onnx")
    quant_pre_process(str(up), str(pre), skip_symbolic_shape=True)
    up.unlink(missing_ok=True)
    m = onnx.load(str(pre))
    exclude = [n.name for n in m.graph.node if n.name.startswith("/model.22/")] if exclude_head else []
    reader = CalibReader(calib, size, m.graph.input[0].name)
    quantize_static(
        str(pre), str(out), reader, quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=exclude,
    )
    pre.unlink(missing_ok=True)


def evaluate(path: Path, data: str, size: int) -> dict:
    from ultralytics import YOLO

    r = YOLO(str(path), task="detect").val(data=data, split="test", imgsz=size, batch=1, device="cpu",
                                           plots=False, verbose=False)
    return {"map50": round(float(r.box.map50), 4), "map50_95": round(float(r.box.map), 4),
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4)}


def bench(path: Path, size: int, n: int = 50) -> float:
    det = OnnxDetector(str(path))
    img = (np.random.default_rng(0).random((size, size, 3)) * 255).astype(np.uint8)
    for _ in range(5):
        det(img)
    t = time.perf_counter()
    for _ in range(n):
        det(img)
    return round((time.perf_counter() - t) / n * 1000, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--max-drop", type=float, default=0.01)
    args = ap.parse_args()
    import cv2
    import yaml
    from onnxruntime.transformers import float16
    import onnx
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data).parent
    rng = random.Random(SEED)

    exported = Path(YOLO(args.weights).export(format="onnx", opset=12, simplify=True, imgsz=args.imgsz, dynamic=False))
    paths = {"fp32": out / f"model-{args.version}-fp32.onnx"}
    shutil.copy2(exported, paths["fp32"])
    # Resize dibiarkan FP32 (input 'scales'-nya harus float32); konverter onnxruntime.transformers
    # menyisipkan Cast dengan anotasi tipe yang benar (konverter onnxconverter-common gagal di sini).
    m16 = float16.convert_float_to_float16(onnx.load(str(paths["fp32"])), keep_io_types=True,
                                           op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + ["Resize"])
    paths["fp16"] = out / f"model-{args.version}-fp16.onnx"
    onnx.save(m16, str(paths["fp16"]))
    train_imgs = sorted((data_root / "images" / "train").glob("*.jpg"))
    calib = rng.sample(train_imgs, min(args.n_calib, len(train_imgs)))
    paths["int8"] = out / f"model-{args.version}-int8.onnx"
    quantize_int8(paths["fp32"], paths["int8"], calib, args.imgsz, exclude_head=False)
    paths["int8_mixed"] = out / f"model-{args.version}-int8_mixed.onnx"
    quantize_int8(paths["fp32"], paths["int8_mixed"], calib, args.imgsz, exclude_head=True)

    res = {}
    for k, p in paths.items():
        t = time.time()
        res[k] = {"file": p.name, "size_mb": round(p.stat().st_size / 1e6, 2), **evaluate(p, args.data, args.imgsz),
                  "cpu_ms": bench(p, args.imgsz)}
        print(k, res[k], f"({time.time() - t:.0f} dtk)", flush=True)
    base = res["fp32"]["map50"]
    ok = [k for k in res if base - res[k]["map50"] <= args.max_drop]
    chosen = min(ok, key=lambda k: res[k]["size_mb"])
    summary = {"imgsz": args.imgsz, "classes": CLASSES, "variants": res, "rule": f"terkecil dengan drop mAP50 <= {args.max_drop * 100:.0f} poin",
               "chosen": chosen, "input": "float32 [1,3,%d,%d] RGB 0..1 letterbox pad 114" % (args.imgsz, args.imgsz),
               "output": f"float32 [1,{4 + len(CLASSES)},N] (cx,cy,w,h dalam piksel input, lalu skor 7 kelas)"}

    # referensi parity: beberapa citra test (satu per kelas + satu negatif)
    pdir = out / "parity"
    pdir.mkdir(exist_ok=True)
    test_lbl = data_root / "labels" / "test"
    by_cls: dict[int, list[Path]] = {}
    negs = []
    for lp in sorted(test_lbl.glob("*.txt")):
        lines = lp.read_text().split()
        if not lines:
            negs.append(lp)
        else:
            by_cls.setdefault(int(lines[0]), []).append(lp)
    picks = [rng.choice(v) for _, v in sorted(by_cls.items())] + ([rng.choice(negs)] if negs else [])
    refs = {}
    dets = {k: OnnxDetector(str(paths[k])) for k in ("fp32", chosen)}
    for lp in picks:
        src = data_root / "images" / "test" / f"{lp.stem}.jpg"
        shutil.copy2(src, pdir / src.name)
        img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
        refs[src.name] = {k: d(img, conf=0.25, iou=0.45) for k, d in dets.items()}
    (pdir / "parity_ref.json").write_text(json.dumps({"conf": 0.25, "iou": 0.45, "images": refs}, indent=1), encoding="utf-8")
    (out / "variants.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
