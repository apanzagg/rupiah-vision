"""Survei metadata kandidat dataset Roboflow Universe lewat API.

Keluaran: data/interim/rf_survey.json (lengkap) dan data/interim/rf_survey.csv (ringkas).
Pemakaian: python training/scripts/rf_survey.py [training/configs/roboflow_candidates.txt]
"""
from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request

from common import DATA, ROOT, roboflow_key


def fetch(slug: str, key: str) -> dict:
    url = f"https://api.roboflow.com/{slug}?api_key={key}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            return {"error": f"HTTP {e.code}"}
        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(5)
                continue
            return {"error": str(e.reason)}
    return {"error": "gagal"}


def summarize_aug(aug: dict) -> str:
    if not aug:
        return "none"
    parts = []
    for k, v in aug.items():
        if k == "image":
            parts.append(f"x{v.get('versions')}")
        elif k == "flip":
            parts.append("flip:" + "".join(n[0] for n, on in v.items() if on))
        else:
            parts.append(k)
    return ",".join(parts)


def main() -> None:
    cand_file = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "training/configs/roboflow_candidates.txt")
    slugs = [
        s.strip()
        for s in cand_file.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.startswith("#")
    ]
    key = roboflow_key()
    out_dir = DATA / "interim"
    out_dir.mkdir(parents=True, exist_ok=True)
    full, rows = {}, []
    for i, slug in enumerate(slugs, 1):
        d = fetch(slug, key)
        full[slug] = d
        if "error" in d:
            print(f"[{i}/{len(slugs)}] {slug}: {d['error']}")
            rows.append({"slug": slug, "error": d["error"]})
            continue
        p = d.get("project", {})
        vers = d.get("versions", [])
        vinfo = []
        for v in vers:
            vid = v["id"].split("/")[-1]
            resize = (v.get("preprocessing") or {}).get("resize") or {}
            rs = f"|{resize.get('format', '')}{resize.get('width', '')}" if resize else ""
            y8 = "|yolov8" if "yolov8" in (v.get("exports") or []) else ""
            vinfo.append(f"v{vid}:{v.get('images')}img[{summarize_aug(v.get('augmentation') or {})}]{rs}{y8}")
        row = {
            "slug": slug,
            "name": p.get("name"),
            "type": p.get("type"),
            "license": p.get("license"),
            "source_images": p.get("images"),
            "unannotated": p.get("unannotated"),
            "classes": json.dumps(p.get("classes", {}), ensure_ascii=False),
            "n_versions": len(vers),
            "versions": " ; ".join(vinfo),
            "updated": time.strftime("%Y-%m-%d", time.gmtime(p.get("updated", 0) or 0)),
        }
        rows.append(row)
        print(f"[{i}/{len(slugs)}] {slug}: {row['license']} | {row['source_images']} img | {len(vers)} versi")
        time.sleep(0.4)
    (out_dir / "rf_survey.json").write_text(json.dumps(full, indent=1, ensure_ascii=False), encoding="utf-8")
    fields = ["slug", "name", "type", "license", "source_images", "unannotated", "classes",
              "n_versions", "versions", "updated", "error"]
    with open(out_dir / "rf_survey.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("Tersimpan:", out_dir / "rf_survey.csv")


if __name__ == "__main__":
    main()
