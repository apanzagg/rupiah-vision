"""Pasang model hasil training ke repo: models/ (model web + MODEL_CARD.md + parity) dan web/src/model-config.json.

Pemakaian: python install_model.py --run-dir <kaggle_out/train/runs/NAMA> --export-dir <kaggle_out/train/export/NAMA> [--variant int8_mixed]
Tanpa --variant, dipakai varian 'chosen' dari variants.json (terkecil dengan penurunan mAP50 <= 1 poin).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import onnx

from common import CLASSES, ROOT

MODELS = ROOT / "models"
LABEL = {c: f"Rp{int(c):,}".replace(",", ".") for c in CLASSES}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    run, exp = Path(args.run_dir), Path(args.export_dir)
    variants = json.loads((exp / "variants.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    chosen = args.variant or variants["chosen"]
    v = variants["variants"][chosen]
    src = exp / v["file"]
    MODELS.mkdir(exist_ok=True)
    for old in MODELS.glob("*.onnx"):
        old.unlink()
    dst = MODELS / v["file"]
    shutil.copy2(src, dst)
    if (MODELS / "parity").exists():
        shutil.rmtree(MODELS / "parity")
    shutil.copytree(exp / "parity", MODELS / "parity")
    shutil.copy2(exp / "variants.json", MODELS / "variants.json")
    m = onnx.load(str(dst))
    inp = m.graph.input[0]
    out = m.graph.output[0]
    dims = lambda t: [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]  # noqa: E731
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    cfg = {"version": args.version, "variant": chosen, "file": v["file"], "imgsz": variants["imgsz"], "classes": CLASSES,
           "sizeBytes": dst.stat().st_size, "sha256": sha}
    (ROOT / "web" / "src" / "model-config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    t = metrics["test"]
    rows = "\n".join(
        f"| {k} | {vv['size_mb']} | {vv['map50']:.4f} | {vv['map50_95']:.4f} | {vv['precision']:.4f} | {vv['recall']:.4f} | {vv['cpu_ms']} |"
        for k, vv in variants["variants"].items())
    pcs = "\n".join(f"| {i} | `{c}` | {LABEL[c]} | {t['per_class'].get(c, {}).get('ap50', 0):.4f} | {t['per_class'].get(c, {}).get('ap50_95', 0):.4f} |"
                    for i, c in enumerate(CLASSES))
    card = f"""# Model Card: SuaraRupiah {args.version} ({chosen})

Detektor nominal uang kertas Rupiah, **YOLOv8n** (Ultralytics {metrics.get('ultralytics', '8.3.253')}), diekspor ke ONNX
untuk dijalankan di browser dengan ONNX Runtime Web (backend WASM). Dibangkitkan oleh `training/scripts/install_model.py`.

## Berkas

| Item | Nilai |
|---|---|
| Berkas | `{dst.name}` |
| Ukuran | {dst.stat().st_size / 1e6:.2f} MB |
| SHA-256 | `{sha}` |
| Varian | `{chosen}` (dipilih: {variants['rule']}) |
| Opset ONNX | {m.opset_import[0].version} |
| Run training | `{metrics['name']}` |

## Masukan & keluaran

| | Nama | Bentuk | Keterangan |
|---|---|---|---|
| Input | `{inp.name}` | {dims(inp)} | float32, RGB, nilai 0..1, letterbox ke {variants['imgsz']}x{variants['imgsz']} dengan padding abu (114), posisi tengah |
| Output | `{out.name}` | {dims(out)} | baris 0..3 = cx, cy, w, h (piksel ruang input); baris 4..10 = skor 7 kelas (sudah sigmoid). NMS dilakukan di aplikasi |

## Kelas (urutan indeks)

| Indeks | Nama | Nominal | AP50 (test) | AP50-95 (test) |
|---|---|---|---|---|
{pcs}

## Ambang yang disarankan

- Confidence filter awal: 0,25; NMS class-agnostic IoU 0,45 (maks. 20 deteksi).
- Pengumuman suara: confidence >= **0,7** dan konsisten di >= 3 dari 5 frame terakhir (lihat `web/src/lib/decision.ts`).

## Metrik (split test, {t.get('images', '')})

PyTorch (`best.pt`, Ultralytics val): mAP50 **{t['map50']:.4f}**, mAP50-95 **{t['map50_95']:.4f}**, precision {t['precision']:.4f}, recall {t['recall']:.4f}.

Perbandingan varian ONNX (Ultralytics val, backend onnxruntime CPU, batch 1, letterbox persegi; waktu CPU = Kaggle 4 vCPU):

| Varian | Ukuran (MB) | mAP50 | mAP50-95 | Precision | Recall | ms/citra (CPU) |
|---|---|---|---|---|---|---|
{rows}

## Data & batasan

Lihat `docs/SUMBER_DATASET.md` (sumber & lisensi) dan `docs/LAPORAN_EVALUASI.md` (evaluasi per kondisi, analisis kesalahan).
Model **hanya mengenali nominal**, bukan keaslian. Uang logam dan Rp75.000 tidak dikenali (dilatih sebagai latar).
"""
    (MODELS / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    print(f"Terpasang: {dst} ({dst.stat().st_size / 1e6:.2f} MB), varian {chosen}")


if __name__ == "__main__":
    main()
