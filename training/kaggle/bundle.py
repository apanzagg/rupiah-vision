"""Bungkus kode proyek menjadi satu skrip kernel Kaggle, lalu (opsional) push.

Kernel script Kaggle hanya bisa berisi satu berkas, jadi training/scripts dan training/configs dikemas
sebagai zip base64 di dalam run.py, diekstrak ke /tmp/src saat kernel berjalan, lalu isi
training/kaggle/<nama>/main.py dijalankan.

Pemakaian:
  python training/kaggle/bundle.py data  --links --push --accelerator NvidiaTeslaT4
  python training/kaggle/bundle.py train --push --accelerator NvidiaTeslaT4

Hasil di training/kaggle/_build/<nama>/ (di-gitignore karena bisa memuat link unduhan sementara).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = '''# Dibangkitkan oleh training/kaggle/bundle.py dari training/kaggle/{name}/main.py. Jangan diedit manual.
# Kode sumber lengkap: https://github.com/apanzagg/rupiah-vision (AGPL-3.0)
import base64 as _b64, io as _io, sys as _sys, zipfile as _zf
_BUNDLE = """{bundle}"""
_zf.ZipFile(_io.BytesIO(_b64.b64decode(_BUNDLE))).extractall("/tmp/src")
_sys.path.insert(0, "/tmp/src/scripts")
# ---------------------------------------------------------------- main.py
'''


def build(name: str, links: bool, extra: list[str]) -> Path:
    kdir = ROOT / "training" / "kaggle" / name
    out = ROOT / "training" / "kaggle" / "_build" / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "training" / "scripts").glob("*.py")):
            z.write(p, f"scripts/{p.name}")
        for p in sorted((ROOT / "training" / "configs").rglob("*")):
            if p.is_file():
                z.write(p, f"configs/{p.relative_to(ROOT / 'training' / 'configs').as_posix()}")
        if links:
            z.write(ROOT / "data" / "interim" / "rf_links.json", "rf_links.json")
        for e in extra:
            src, _, arc = e.partition("=")
            z.write(ROOT / src, arc or Path(src).name)
    bundle = base64.b64encode(buf.getvalue()).decode()
    main = (kdir / "main.py").read_text(encoding="utf-8")
    (out / "run.py").write_text(HEADER.format(name=name, bundle=bundle) + main, encoding="utf-8")
    meta = json.loads((kdir / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["code_file"] = "run.py"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"bundel {name}: {len(bundle) / 1e3:.0f} kB -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--links", action="store_true", help="sertakan data/interim/rf_links.json")
    ap.add_argument("--extra", nargs="*", default=[], help="berkas tambahan: path[=nama_dalam_bundel]")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--accelerator", default="")
    args = ap.parse_args()
    out = build(args.name, args.links, args.extra)
    if args.push:
        kaggle = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
        cmd = [str(kaggle), "kernels", "push", "-p", str(out)]
        if args.accelerator:
            cmd += ["--accelerator", args.accelerator]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
