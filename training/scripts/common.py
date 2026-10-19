"""Utilitas bersama untuk skrip data/training."""
from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CLASSES = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]


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
