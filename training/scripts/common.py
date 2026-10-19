"""Utilitas bersama untuk skrip data/training.

Lokasi data bisa diganti lewat environment variable (dipakai saat berjalan di Kaggle):
  SR_RAW      data mentah (default data/raw)
  SR_KAGGLE   akar dataset Kaggle yang di-mount (default data/raw/kaggle)
  SR_WORK     data antara: pool, hash, manifest (default data/interim)
  SR_FINAL    dataset final format YOLO (default data/final)
  SR_REPORTS  laporan, grafik, contact sheet (default data/interim/reports)
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CLASSES = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
SEED = 20261019


def _p(env: str, default: Path) -> Path:
    return Path(os.environ[env]) if os.environ.get(env) else default


RAW = _p("SR_RAW", DATA / "raw")
KAGGLE_IN = _p("SR_KAGGLE", RAW / "kaggle")
WORK = _p("SR_WORK", DATA / "interim")
FINAL = _p("SR_FINAL", DATA / "final")
REPORTS = _p("SR_REPORTS", WORK / "reports")
CONFIGS = Path(os.environ.get("SR_CONFIGS", ROOT / "training" / "configs"))


def source_id(slug: str) -> str:
    """ID pendek & stabil untuk sebuah sumber (dipakai sebagai prefiks nama berkas)."""
    return hashlib.sha1(slug.encode()).hexdigest()[:6]


def load_env() -> None:
    """Muat variabel dari ROOT/.env ke os.environ (tanpa mencetak nilainya)."""
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$", line)
        if m and m.group(2) and m.group(1) not in os.environ:
            os.environ[m.group(1)] = m.group(2)


def roboflow_key() -> str:
    load_env()
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY belum diisi di .env")
    return key


def load_sources() -> dict:
    import yaml

    return yaml.safe_load((CONFIGS / "sources.yaml").read_text(encoding="utf-8"))


def roboflow_dir(src: dict) -> Path:
    ws, proj = src["slug"].split("/")
    return RAW / "roboflow" / f"{ws}__{proj}__v{src['version']}"


def kaggle_dir(ref: str) -> Path | None:
    """Cari folder dataset Kaggle: lokal (owner__slug) atau mount Kaggle (/kaggle/input/...)."""
    owner, slug = ref.split("/")
    for cand in (
        KAGGLE_IN / f"{owner}__{slug}",
        KAGGLE_IN / slug,
        KAGGLE_IN / "datasets" / owner / slug,
    ):
        if cand.exists():
            return cand
    return None
