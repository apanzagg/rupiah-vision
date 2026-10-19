"""Minta link ekspor YOLOv8 untuk semua sumber Roboflow (dijalankan LOKAL, memakai API key dari .env).

Link ekspor Roboflow adalah URL unduhan sementara yang TIDAK memuat API key, sehingga aman diteruskan
ke kernel Kaggle; API key tetap hanya ada di komputer lokal.
Keluaran: data/interim/rf_links.json (di-gitignore).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from common import WORK, load_sources, roboflow_dir, roboflow_key


def export_link(slug: str, version: int, key: str, timeout_s: int = 900) -> str:
    url = f"https://api.roboflow.com/{slug}/{version}/yolov8?api_key={key}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(15)
                continue
            raise
        link = (d.get("export") or {}).get("link")
        if link:
            assert key not in link
            return link
        time.sleep(10)  # ekspor sedang dibuat Roboflow
    raise TimeoutError(f"ekspor {slug} v{version} belum siap")


def main() -> None:
    key = roboflow_key()
    cfg = load_sources()
    out = {}
    for s in cfg["roboflow"]:
        link = export_link(s["slug"], int(s["version"]), key)
        out[s["slug"]] = {"version": s["version"], "dir": roboflow_dir(s).name, "link": link}
        print(f"  ok: {s['slug']} v{s['version']}")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "rf_links.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} link disimpan ke {WORK / 'rf_links.json'}")


if __name__ == "__main__":
    main()
