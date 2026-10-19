"""Unduh & ekstrak ekspor Roboflow dari rf_links.json (dijalankan di Kaggle, tanpa API key).

Pemakaian: python fetch_rf_links.py <rf_links.json>
Hasil: SR_RAW/roboflow/<workspace>__<project>__v<ver>/ (format YOLOv8), penanda .done per sumber.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import RAW


def fetch(item: tuple[str, dict]) -> str:
    slug, info = item
    out = RAW / "roboflow" / info["dir"]
    if (out / ".done").exists():
        return f"lewati {slug}"
    out.mkdir(parents=True, exist_ok=True)
    zpath = RAW / "zips" / f"{info['dir']}.zip"
    zpath.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for attempt in range(4):
        try:
            req = urllib.request.Request(info["link"], headers={"User-Agent": "suararupiah-data/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(zpath, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except Exception as e:  # noqa: BLE001
            if attempt == 3:
                raise RuntimeError(f"{slug}: {e}") from e
            time.sleep(10 * (attempt + 1))
    root = out.resolve()
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.infolist():
            if not str((out / m.filename).resolve()).startswith(str(root)):
                raise RuntimeError(f"path mencurigakan di zip {slug}: {m.filename}")
        zf.extractall(out)
    size = zpath.stat().st_size / 1e6
    zpath.unlink()
    n = sum(1 for p in out.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    (out / ".done").write_text(f"{n}\n")
    return f"ok {slug}: {n} citra, {size:.0f} MB, {time.time() - t0:.0f} dtk"


def main() -> None:
    links = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    t0 = time.time()
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(fetch, links.items()):
            print(" ", msg, flush=True)
    print(f"Selesai mengunduh {len(links)} sumber dalam {time.time() - t0:.0f} dtk")


if __name__ == "__main__":
    main()
