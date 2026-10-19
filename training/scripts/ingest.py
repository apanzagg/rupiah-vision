"""Tahap 1 pipeline data: baca semua sumber mentah -> pool seragam.

Untuk setiap citra:
- kelas asli dipetakan ke 7 kelas baku (class_map.py); box di luar lingkup dibuang,
  citra dengan kelas generik tanpa nominal dibuang;
- box divalidasi (dipotong ke [0,1], box terlalu kecil dibuang); poligon diubah ke bbox;
- orientasi EXIF dikoreksi, sisi terpanjang dikecilkan ke maks. 640 px, disimpan JPEG q90;
- dihitung pHash 64-bit untuk 4 rotasi (0/90/180/270) + thumbnail warna 8x8 (untuk dedup).

Keluaran (di SR_WORK):
  pool/images/*.jpg, pool/labels/*.txt, pool/manifest.csv, pool/hashes.npz,
  conditions_raw/<kondisi>/*.jpg (dataset kondisi tanpa label nominal), ingest_stats.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from class_map import DROP_BOX, DROP_IMAGE, map_class
from common import CLASS_ID, SEED, WORK, kaggle_dir, load_sources, roboflow_dir, source_id

MAX_SIDE = 640
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_WH = 0.01  # box lebih sempit dari 1% sisi citra dianggap tidak valid
FIELDS = [
    "id", "src", "slug", "role", "kind", "orig_split", "orig_name", "origin_key", "augmented",
    "w", "h", "n_boxes", "classes", "dropped_boxes", "autolabel", "al_class", "status",
]


def phash_rot(img: Image.Image) -> tuple[list[int], np.ndarray]:
    from scipy.fft import dct

    g = np.asarray(img.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    t = np.asarray(img.convert("RGB").resize((8, 8), Image.Resampling.BOX), dtype=np.uint8)
    hashes, thumbs = [], []
    for k in range(4):
        a = np.rot90(g, k)
        d = dct(dct(a, axis=0), axis=1)[:8, :8]
        bits = (d > np.median(d)).flatten()
        hashes.append(int(np.packbits(bits).view(">u8")[0]))
        thumbs.append(np.rot90(t, k))
    return hashes, np.stack(thumbs)


def parse_label_file(path: Path) -> list[tuple[int, list[float]]]:
    """Baca label YOLO (bbox atau poligon) -> [(cls_asli, [xc, yc, w, h])]."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
        if len(vals) == 4:
            xc, yc, w, h = vals
        else:  # poligon x1 y1 x2 y2 ...
            xs, ys = vals[0::2], vals[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        out.append((c, [xc, yc, w, h]))
    return out


def clean_box(b: list[float]) -> list[float] | None:
    xc, yc, w, h = b
    x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    w, h = x1 - x0, y1 - y0
    if w < MIN_WH or h < MIN_WH:
        return None
    return [(x0 + x1) / 2, (y0 + y1) / 2, w, h]


def process(task: dict) -> dict | None:
    """Dijalankan di worker: simpan citra ter-resize + label, hitung hash."""
    try:
        with Image.open(task["path"]) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            s = MAX_SIDE / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
            out_img = Path(task["out_dir"]) / "images" / f"{task['id']}.jpg"
            im.save(out_img, quality=90)
            hashes, thumbs = phash_rot(im)
            w, h = im.size
    except Exception as e:  # citra rusak
        return {"id": task["id"], "error": str(e)}
    lab = Path(task["out_dir"]) / "labels" / f"{task['id']}.txt"
    lab.write_text("".join(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n" for c, b in task["boxes"]))
    return {"id": task["id"], "w": w, "h": h, "hashes": hashes, "thumbs": thumbs}


def origin_key(name: str) -> str:
    """Roboflow menamai salinan augmentasi '<asal>_jpg.rf.<hash>.jpg' -> kunci = '<asal>'."""
    return name.split(".rf.")[0]


def collect_roboflow(src: dict, rng: random.Random, stats: dict) -> list[dict]:
    d = roboflow_dir(src)
    st = stats[src["slug"]]
    if not (d / "data.yaml").exists():
        print(f"  [lewati] belum diunduh: {d.name}")
        st["missing"] = 1
        return []
    names = yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8"))["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    mapping = [map_class(n, src["slug"]) for n in names]
    drop_with = set(src.get("drop_images_with", []))
    sid = source_id(src["slug"])
    is_aug = str(src.get("aug", "none")) != "none"
    rows = []
    for split_dir in sorted(p for p in d.iterdir() if (p / "images").is_dir()):
        for img in sorted((split_dir / "images").iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            st["read"] += 1
            raw = parse_label_file(split_dir / "labels" / f"{img.stem}.txt")
            raw_names = [names[c] if 0 <= c < len(names) else "?" for c, _ in raw]
            boxes, dropped, generic = [], 0, False
            if src["role"] == "positive":
                for (c, b), rn in zip(raw, raw_names):
                    m = mapping[c] if 0 <= c < len(mapping) else DROP_IMAGE
                    if m == DROP_IMAGE:
                        generic = True
                        break
                    if m == DROP_BOX:
                        dropped += 1
                        continue
                    cb = clean_box(b)
                    if cb is None:
                        st["box_invalid"] += 1
                        continue
                    boxes.append((CLASS_ID[m], cb))
                if generic:
                    st["drop_generic"] += 1
                    continue
                if not boxes:
                    if dropped:  # hanya koin / Rp75.000 -> jadi citra negatif
                        kind, role = "koin_75k_dari_pos", "neg"
                    else:
                        st["drop_unlabeled"] += 1
                        continue
                else:
                    kind, role = "positive", "pos"
            else:
                if drop_with & set(raw_names):
                    st["drop_contains_idr"] += 1
                    continue
                kind, role = src.get("kind", "negative"), "neg"
            rel = f"{split_dir.name}/{img.name}"
            rows.append({
                "id": f"{sid}_{hashlib.sha1(rel.encode()).hexdigest()[:10]}",
                "src": sid, "slug": src["slug"], "role": role, "kind": kind,
                "orig_split": split_dir.name, "orig_name": img.name,
                "origin_key": f"{sid}:{origin_key(img.name)}",
                "augmented": int(is_aug and split_dir.name == "train"),
                "classes": " ".join(sorted(str(c) for c, _ in boxes)),
                "n_boxes": len(boxes), "dropped_boxes": dropped, "autolabel": 0, "status": "ok",
                "_path": str(img), "_boxes": boxes,
            })
    if src["role"] == "negative" and src.get("max_images") and len(rows) > src["max_images"]:
        rng.shuffle(rows)
        st["neg_sampled_out"] = len(rows) - src["max_images"]
        rows = rows[: src["max_images"]]
    return rows


def collect_kaggle_cls(src: dict, stats: dict) -> list[dict]:
    d = kaggle_dir(src["ref"])
    st = stats[src["ref"]]
    if d is None:
        print(f"  [lewati] dataset Kaggle tidak ditemukan: {src['ref']}")
        st["missing"] = 1
        return []
    sid = source_id(src["ref"])
    fmap = {k.lower(): v for k, v in src["folder_map"].items()}
    rows = []
    for img in sorted(d.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        folder = img.parent.name.lower()
        if folder not in fmap:
            continue
        st["read"] += 1
        rel = str(img.relative_to(d)).replace("\\", "/")
        rows.append({
            "id": f"{sid}_{hashlib.sha1(rel.encode()).hexdigest()[:10]}",
            "src": sid, "slug": src["ref"], "role": "pos", "kind": "positive",
            "orig_split": "kaggle", "orig_name": rel, "origin_key": f"{sid}:{rel}", "augmented": 0,
            "classes": "", "al_class": CLASS_ID[fmap[folder]], "n_boxes": 0, "dropped_boxes": 0, "autolabel": 1,
            "status": "needs_autolabel", "_path": str(img), "_boxes": [],
        })
    return rows


def copy_conditions(src: dict, stats: dict) -> None:
    d = kaggle_dir(src["ref"])
    if d is None:
        stats[src["ref"]]["missing"] = 1
        return
    out = WORK / "conditions_raw" / source_id(src["ref"])
    n = 0
    for img in sorted(d.rglob("*")):
        if img.suffix.lower() not in IMG_EXT:
            continue
        dst = out / img.parent.name / f"{img.stem}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(img) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            s = MAX_SIDE / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
            im.save(dst, quality=90)
        n += 1
    stats[src["ref"]]["read"] = n
    print(f"  kondisi: {n} citra -> {out}")


def main() -> None:
    t0 = time.time()
    cfg = load_sources()
    rng = random.Random(SEED)
    pool = WORK / "pool"
    if pool.exists():
        shutil.rmtree(pool)
    (pool / "images").mkdir(parents=True)
    (pool / "labels").mkdir(parents=True)
    stats: dict = defaultdict(Counter)

    rows: list[dict] = []
    for src in cfg["roboflow"]:
        r = collect_roboflow(src, rng, stats)
        print(f"  {src['slug']}: {len(r)} citra")
        rows += r
    for src in cfg.get("kaggle", []):
        if src["role"] == "positive_cls":
            r = collect_kaggle_cls(src, stats)
            print(f"  {src['ref']}: {len(r)} citra (perlu auto-label)")
            rows += r
        elif src["role"] == "condition_unlabeled":
            copy_conditions(src, stats)

    # buang id ganda (seharusnya tidak ada)
    seen, uniq = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            uniq.append(r)
    rows = uniq

    tasks = [{"id": r["id"], "path": r["_path"], "boxes": r["_boxes"], "out_dir": str(pool)} for r in rows]
    by_id = {r["id"]: r for r in rows}
    ids, hashes, thumbs = [], [], []
    nproc = max(1, min(8, (__import__("os").cpu_count() or 2)))
    with Pool(nproc) as p:
        for i, res in enumerate(p.imap_unordered(process, tasks, chunksize=32), 1):
            if "error" in res:
                by_id[res["id"]]["status"] = "broken"
                stats[by_id[res["id"]]["slug"]]["broken"] += 1
                continue
            r = by_id[res["id"]]
            r["w"], r["h"] = res["w"], res["h"]
            ids.append(res["id"])
            hashes.append(res["hashes"])
            thumbs.append(res["thumbs"])
            if i % 5000 == 0:
                print(f"  diproses {i}/{len(tasks)} ({time.time() - t0:.0f} dtk)")
    rows = [r for r in rows if r["status"] != "broken"]
    np.savez_compressed(
        pool / "hashes.npz", ids=np.array(ids), phash=np.array(hashes, dtype=np.uint64), thumbs=np.array(thumbs)
    )
    with open(pool / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    by_src = Counter(r["slug"] for r in rows)
    for k, v in by_src.items():
        stats[k]["pooled"] = v
    (WORK / "ingest_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    kinds = Counter(r["kind"] for r in rows)
    print(f"Selesai: {len(rows)} citra di pool, {dict(kinds)} ({time.time() - t0:.0f} dtk)")


if __name__ == "__main__":
    sys.exit(main())
