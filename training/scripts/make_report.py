"""Bangun docs/LAPORAN_EVALUASI.md dari berkas hasil eksekusi (tidak ada angka yang diketik manual).

Masukan (yang tersedia saja yang dipakai):
  training/kaggle_out/build/reports/   dedup_report.json, ingest_stats.json, source_stats.json, source_audit.json, figures/
  training/kaggle_out/train/           summary.json, runs/<run>/metrics.json & grafik, export/<run>/variants.json
  data/interim/conditions_eval/        conditions_metrics.json + confusion_<kondisi>.png
  web/test-results/                    bench_desktop.json, e2e-results.json, lighthouse.json
  training/report/*.md                 bagian naratif (analisis kesalahan, keterbatasan) yang ditulis setelah melihat hasil
Keluaran: docs/LAPORAN_EVALUASI.md + salinan grafik di docs/figures/
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import CLASSES, DATA, ROOT

LABEL = {c: f"Rp{int(c):,}".replace(",", ".") for c in CLASSES}
FIG = ROOT / "docs" / "figures"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def fmt(v, d=4):
    return f"{v:.{d}f}".replace(".", ",") if isinstance(v, (int, float)) else str(v)


def copy_fig(src: Path, name: str) -> str | None:
    if src.exists():
        FIG.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, FIG / name)
        return f"figures/{name}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="nama run final, mis. e1_r320")
    ap.add_argument("--build", default=str(ROOT / "training/kaggle_out/build/reports"))
    ap.add_argument("--train", default=str(ROOT / "training/kaggle_out/train"))
    args = ap.parse_args()
    build, train = Path(args.build), Path(args.train)
    rep = load(build / "dedup_report.json") or {}
    src_stats = load(build / "source_stats.json") or {}
    summary = load(train / "summary.json") or {}
    m = load(train / "runs" / args.run / "metrics.json") or {}
    var = load(train / "export" / args.run / "variants.json") or {}
    cond = load(DATA / "interim" / "conditions_eval" / "conditions_metrics.json")
    bench = load(ROOT / "web" / "test-results" / "bench_desktop.json")
    lh = load(ROOT / "web" / "test-results" / "lighthouse.json")
    sizes = load(ROOT / "web" / "test-results" / "download_sizes.json")
    parts = {p.stem: p.read_text(encoding="utf-8") for p in (ROOT / "training" / "report").glob("*.md")}
    L: list[str] = []
    w = L.append

    w("# Laporan Evaluasi SuaraRupiah\n")
    w("Dokumen ini dibangkitkan oleh `training/scripts/make_report.py` dari berkas hasil eksekusi (log/JSON di repo dan "
      "output kernel Kaggle). Strukturnya mengikuti Bab 4 laporan capstone.\n")

    # ---------------- 4.1 dataset
    w("## 4.1 Dataset\n")
    n_src = len([k for k, v in src_stats.items() if v.get("kept")])
    w(f"Dataset dibangun dari **{n_src} sumber** yang lolos seleksi lisensi (daftar & atribusi: `docs/SUMBER_DATASET.md`).\n")
    if rep:
        w("| Tahap | Jumlah citra |\n|---|---|")
        w(f"| Pool setelah normalisasi kelas | {rep['input']:,} |".replace(",", "."))
        for k, v in sorted(rep["drop"].items(), key=lambda kv: -kv[1]):
            w(f"| Dibuang: `{k}` | {v:,} |".replace(",", "."))
        w(f"| **Dataset final** | **{rep['after_dedup']:,}** |".replace(",", "."))
        w(f"| Kelompok kemiripan (unit split) | {rep['groups']:,} |\n".replace(",", "."))
        w("Pembagian 70/15/15 berbasis kelompok dan stratified per kelas:\n")
        w("| Split | Citra | Negatif | " + " | ".join(LABEL[c] for c in CLASSES) + " |")
        w("|---|---|---|" + "---|" * len(CLASSES))
        for s in ("train", "val", "test"):
            st = rep["splits"][s]
            w(f"| {s} | {st['images']} | {st['negatives']} | " + " | ".join(str(st["instances"][c]) for c in CLASSES) + " |")
        w("\n(angka per kelas = jumlah objek/bounding box)\n")
    for f, cap in (("class_distribution.png", "Distribusi kelas per split"), ("bbox_size_hist.png", "Distribusi ukuran bounding box"),
                   ("boxes_per_image.png", "Jumlah uang per citra"), ("source_contribution.png", "Kontribusi sumber"),
                   ("samples_grid.png", "Contoh citra latih beserta label")):
        p = copy_fig(build / "figures" / f, f)
        if p:
            w(f"![{cap}]({p})\n")
    if "pembersihan_data" in parts:
        w(parts["pembersihan_data"] + "\n")

    # ---------------- 4.2 konfigurasi training
    w("## 4.2 Konfigurasi training\n")
    if m:
        w("| Parameter | Nilai |\n|---|---|")
        w(f"| Model | YOLOv8n (bobot awal COCO `yolov8n.pt`), Ultralytics 8.3.253 |")
        w(f"| Ukuran input | {m['imgsz']} x {m['imgsz']} |")
        w(f"| Epoch | {m['epochs_trained']} dijalankan (maks. {m['epochs_requested']}, early stopping patience {m['patience']}); epoch terbaik {m['best_epoch']} |")
        w(f"| Batch | {m['batch']} |")
        w(f"| Waktu training | {m['train_seconds'] / 3600:.1f} jam (GPU NVIDIA Tesla T4, Kaggle) |".replace(".", ","))
        hyp = m["hyp"]
        w(f"| Augmentasi geometri | rotasi bebas ±{hyp['degrees']:.0f}°, translate {hyp['translate']}, skala {hyp['scale']}, shear {hyp['shear']}, perspektif {hyp['perspective']}, mosaic {hyp['mosaic']} (dimatikan {hyp['close_mosaic']} epoch terakhir) |")
        w(f"| Augmentasi warna | hue ±{hyp['hsv_h']} (kecil, warna = ciri nominal), saturasi {hyp['hsv_s']}, value {hyp['hsv_v']} |")
        w(f"| Flip | horizontal {hyp['fliplr']}, vertikal {hyp['flipud']} (tidak dipakai) |")
        w("| Augmentasi kustom | tekstur kusut + lipatan (p=0,25), oklusi jari (p=0,2), cahaya redup (p=0,12), motion blur, noise, brightness/contrast, gamma, kompresi JPEG, downscale |")
        w("")
        for f, cap in (("results.png", "Kurva training"), ("eval_test/confusion_matrix_normalized.png", "Confusion matrix (test, ternormalisasi)"),
                       ("eval_test/BoxPR_curve.png", "Kurva precision-recall (test)")):
            p = copy_fig(train / "runs" / args.run / f, f"{args.run}_{Path(f).name}")
            if p:
                w(f"![{cap}]({p})\n")

    # ---------------- 4.3 eksperimen
    w("## 4.3 Tabel eksperimen\n")
    runs = summary.get("runs", {})
    if runs:
        w("| Run | imgsz | Epoch | mAP50 val | mAP50 test | mAP50-95 test | Precision test | Recall test |\n|---|---|---|---|---|---|---|---|")
        for name, r in runs.items():
            if "test" in r:
                w(f"| {name} | {r['imgsz']} | {r['epochs_trained']} | {fmt(r['val']['map50'])} | {fmt(r['test']['map50'])} | "
                  f"{fmt(r['test']['map50_95'])} | {fmt(r['test']['precision'])} | {fmt(r['test']['recall'])} |")
        w("\nRiwayat lengkap & alasan setiap perubahan: `docs/EKSPERIMEN.md`.\n")

    # ---------------- 4.4 metrik akhir
    w("## 4.4 Metrik akhir model terpilih\n")
    if m:
        t = m["test"]
        tgt = "tercapai" if t["map50"] >= 0.90 else "BELUM tercapai"
        w(f"Run **{args.run}** pada split test: mAP50 **{fmt(t['map50'])}**, mAP50-95 **{fmt(t['map50_95'])}**, "
          f"precision **{fmt(t['precision'])}**, recall **{fmt(t['recall'])}**. Target mAP50 ≥ 0,90: **{tgt}**.\n")
        w("| Kelas | AP50 | AP50-95 |\n|---|---|---|")
        for c in CLASSES:
            pc = t["per_class"].get(c)
            if pc:
                w(f"| {LABEL[c]} | {fmt(pc['ap50'])} | {fmt(pc['ap50_95'])} |")
        w("")

    # ---------------- 4.5 kondisi
    w("## 4.5 Hasil uji per kondisi\n")
    if cond:
        w(f"Model web (`{cond['model']}`), ambang pengumuman {fmt(cond['threshold'], 2)}. Akurasi = persentase citra yang "
          "nominal-nominalnya diumumkan persis benar (untuk `bukan_rupiah`: tidak ada nominal yang diumumkan).\n")
        w("| Kondisi | Citra (nyata + sintetis) | Akurasi | Precision | Recall | F1 |\n|---|---|---|---|---|---|")
        for k, r in cond["conditions"].items():
            meta = r.get("meta", {})
            src = f"{meta.get('real', '?')} + {meta.get('synthetic', 0)}"
            w(f"| {k} | {r['images']} ({src}) | {fmt(r['accuracy'], 3)} | {fmt(r['precision'], 3)} | {fmt(r['recall'], 3)} | {fmt(r['f1'], 3)} |")
        w("\nSet `terlipat_tertutup` **sepenuhnya sintetis** dan sebagian `cahaya_redup` (berkas berawalan `sint_`) dibuat "
          "dengan penggelapan buatan; angka keduanya bukan pengganti uji lapangan.\n")
        for k, r in cond["conditions"].items():
            pcs = r["per_class"]
            if pcs:
                w(f"<details><summary>Per kelas: {k}</summary>\n")
                w("| Kelas | Precision | Recall | F1 | Dukungan |\n|---|---|---|---|---|")
                for c, v in pcs.items():
                    w(f"| {LABEL[c]} | {fmt(v['precision'], 3)} | {fmt(v['recall'], 3)} | {fmt(v['f1'], 3)} | {v['support']} |")
                w("\n</details>\n")
            p = copy_fig(DATA / "interim" / "conditions_eval" / f"confusion_{k}.png", f"confusion_{k}.png")
            if p:
                w(f"![Confusion matrix {k}]({p})\n")

    # ---------------- 4.6 varian
    w("## 4.6 Perbandingan FP32, FP16, INT8\n")
    if var:
        w("Evaluasi split test dengan backend onnxruntime (CPU), pipeline evaluasi sama untuk semua varian.\n")
        w("| Varian | Ukuran (MB) | mAP50 | mAP50-95 | Precision | Recall | ms/citra (CPU Kaggle) |\n|---|---|---|---|---|---|---|")
        for k, v in var["variants"].items():
            w(f"| {k} | {fmt(v['size_mb'], 2)} | {fmt(v['map50'])} | {fmt(v['map50_95'])} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['cpu_ms'], 1)} |")
        w(f"\nAturan pemilihan: {var['rule']}. Terpilih: **{var['chosen']}**.\n")
        if "kuantisasi" in parts:
            w(parts["kuantisasi"] + "\n")

    # ---------------- 4.7 performa web
    w("## 4.7 Performa aplikasi web\n")
    if sizes:
        w("| Aset unduhan pertama | Ukuran |\n|---|---|")
        for k, v in sizes.items():
            w(f"| {k} | {v} |")
        w("")
    if bench:
        w("Benchmark desktop (halaman `/benchmark`, Chrome, backend WASM):\n")
        w("```json\n" + json.dumps(bench, indent=1, ensure_ascii=False) + "\n```\n")
    if lh:
        w("| Lighthouse | Skor |\n|---|---|")
        for k, v in lh.items():
            w(f"| {k} | {v} |")
        w("")
    if "performa" in parts:
        w(parts["performa"] + "\n")

    # ---------------- 4.8 contoh & analisis
    w("## 4.8 Contoh deteksi benar dan salah\n")
    for f, cap in (("examples_correct.jpg", "Contoh deteksi benar"), ("examples_wrong.jpg", "Contoh deteksi salah")):
        if (FIG / f).exists():
            w(f"![{cap}](figures/{f})\n")
    if "analisis_kesalahan" in parts:
        w(parts["analisis_kesalahan"] + "\n")

    # ---------------- 4.9 keterbatasan
    w("## 4.9 Keterbatasan sistem\n")
    w(parts.get("keterbatasan", "-") + "\n")
    (ROOT / "docs" / "LAPORAN_EVALUASI.md").write_text("\n".join(L), encoding="utf-8")
    print("docs/LAPORAN_EVALUASI.md ditulis")


if __name__ == "__main__":
    main()
