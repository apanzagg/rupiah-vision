"""Unduh semua sumber di training/configs/sources.yaml ke data/raw/ (atau SR_RAW).

- Roboflow: format YOLOv8 -> data/raw/roboflow/<workspace>__<project>__v<ver>/
- Kaggle:   data/raw/kaggle/<owner>__<slug>/
Aman diulang: sumber yang sudah lengkap (ada penanda .done) dilewati.

Pemakaian: python training/scripts/download_sources.py [--only roboflow|kaggle] [--filter teks]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import yaml

from common import RAW, ROOT, load_env, roboflow_key


def dl_roboflow(src: dict, key: str) -> None:
    from roboflow import Roboflow

    ws, proj = src["slug"].split("/")
    out = RAW / "roboflow" / f"{ws}__{proj}__v{src['version']}"
    if (out / ".done").exists():
        print(f"  lewati (sudah ada): {out.name}")
        return
    if out.exists():
        shutil.rmtree(out)
    rf = Roboflow(api_key=key)
    t0 = time.time()
    for attempt in range(3):
        try:
            rf.workspace(ws).project(proj).version(int(src["version"])).download(
                "yolov8", location=str(out), overwrite=True
            )
            break
        except Exception as e:  # noqa: BLE001 - jaringan / export belum siap
            print(f"  percobaan {attempt + 1} gagal: {e}")
            time.sleep(20 * (attempt + 1))
    else:
        raise RuntimeError(f"gagal mengunduh {src['slug']}")
    n = sum(1 for _ in out.rglob("*.jpg")) + sum(1 for _ in out.rglob("*.png"))
    (out / ".done").write_text(f"{n} images, {time.time() - t0:.0f}s\n", encoding="utf-8")
    print(f"  selesai: {out.name} ({n} citra, {time.time() - t0:.0f} dtk)")


def dl_kaggle(src: dict) -> None:
    owner, slug = src["ref"].split("/")
    out = RAW / "kaggle" / f"{owner}__{slug}"
    if (out / ".done").exists():
        print(f"  lewati (sudah ada): {out.name}")
        return
    out.mkdir(parents=True, exist_ok=True)
    kaggle = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    t0 = time.time()
    subprocess.run([str(kaggle), "datasets", "download", "-d", src["ref"], "-p", str(out)], check=True)
    for z in out.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                target = (out / m.filename).resolve()
                if not str(target).startswith(str(out.resolve())):
                    raise RuntimeError(f"path mencurigakan di zip: {m.filename}")
            zf.extractall(out)
        z.unlink()
    n = sum(1 for p in out.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    (out / ".done").write_text(f"{n} images, {time.time() - t0:.0f}s\n", encoding="utf-8")
    print(f"  selesai: {out.name} ({n} citra, {time.time() - t0:.0f} dtk)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["roboflow", "kaggle"])
    ap.add_argument("--filter", default="")
    args = ap.parse_args()
    load_env()
    cfg = yaml.safe_load((ROOT / "training/configs/sources.yaml").read_text(encoding="utf-8"))
    if args.only != "roboflow":
        for s in cfg.get("kaggle", []):
            if args.filter in s["ref"]:
                print(f"[kaggle] {s['ref']}")
                dl_kaggle(s)
    if args.only != "kaggle":
        key = roboflow_key()
        for s in cfg.get("roboflow", []):
            if args.filter in s["slug"]:
                print(f"[roboflow] {s['slug']} v{s['version']}")
                dl_roboflow(s, key)


if __name__ == "__main__":
    main()
