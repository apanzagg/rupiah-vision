"""Audit gaya anotasi per sumber: apakah box menandai uang UTUH atau hanya BAGIAN uang (angka, potret, sudut).

Statistik per sumber (dari pool, sebelum dedup):
- boxes_per_img : rata-rata box per citra positif
- nest_rate     : proporsi box yang >= 80% luasnya berada di dalam box lain (ciri anotasi per-bagian)
- same_cls_multi: proporsi citra yang punya > 1 box berkelas sama
- med_size      : median akar(luas box relatif)
Keluaran: SR_REPORTS/source_audit.json dan SR_REPORTS/sheets/sources_<n>.jpg (4 citra utuh + box per sumber).
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from collections import defaultdict

from PIL import Image, ImageDraw

from common import CLASSES, REPORTS, SEED, WORK

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def boxes(img_id: str) -> list[tuple[int, float, float, float, float]]:
    out = []
    p = WORK / "pool" / "labels" / f"{img_id}.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            v = line.split()
            if len(v) == 5:
                out.append((int(v[0]), *map(float, v[1:])))
    return out


def inside(a, b) -> float:
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    return iw * ih / max(1e-9, a[3] * a[4])


def main() -> None:
    rng = random.Random(SEED)
    rows = [r for r in csv.DictReader(open(WORK / "pool" / "manifest.csv", encoding="utf-8")) if r["role"] == "pos" and r["status"] == "ok"]
    by_src = defaultdict(list)
    for r in rows:
        by_src[r["slug"]].append(r)
    audit = {}
    for slug, rs in sorted(by_src.items()):
        sample = rng.sample(rs, min(400, len(rs)))
        nb, nested, total, multi, sizes = [], 0, 0, 0, []
        for r in sample:
            b = boxes(r["id"])
            nb.append(len(b))
            total += len(b)
            sizes += [(x[3] * x[4]) ** 0.5 for x in b]
            for i, x in enumerate(b):
                if any(inside(x, y) > 0.8 and x[3] * x[4] < y[3] * y[4] for j, y in enumerate(b) if j != i):
                    nested += 1
            cls = [x[0] for x in b]
            multi += int(len(cls) != len(set(cls)))
        audit[slug] = {
            "images": len(rs), "boxes_per_img": round(sum(nb) / max(1, len(nb)), 2),
            "nest_rate": round(nested / max(1, total), 3), "same_cls_multi": round(multi / max(1, len(sample)), 3),
            "med_size": round(statistics.median(sizes), 3) if sizes else 0,
        }
    (REPORTS / "source_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")
    for slug, a in sorted(audit.items(), key=lambda kv: -kv[1]["nest_rate"]):
        print(f"{a['nest_rate']:6.3f} nest | {a['boxes_per_img']:5.2f} box/img | {a['same_cls_multi']:5.2f} multi | "
              f"{a['med_size']:5.2f} size | {a['images']:5d} | {slug}")

    # contact sheet: 4 citra utuh (dengan box) per sumber, 3 sumber per baris
    (REPORTS / "sheets").mkdir(parents=True, exist_ok=True)
    th, per_row, per_sheet = 140, 3, 12
    slugs = sorted(by_src)
    for page in range(0, len(slugs), per_sheet):
        chunk = slugs[page : page + per_sheet]
        rows_n = (len(chunk) + per_row - 1) // per_row
        sheet = Image.new("RGB", (per_row * (4 * th + 10), rows_n * (th + 16)), "white")
        d = ImageDraw.Draw(sheet)
        for k, slug in enumerate(chunk):
            x0, y0 = (k % per_row) * (4 * th + 10), (k // per_row) * (th + 16)
            for t, r in enumerate(rng.sample(by_src[slug], min(4, len(by_src[slug])))):
                im = Image.open(WORK / "pool" / "images" / f"{r['id']}.jpg").convert("RGB")
                dr = ImageDraw.Draw(im)
                W, H = im.size
                for c, xc, yc, w, h in boxes(r["id"]):
                    dr.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H], outline=COLORS[c], width=max(2, W // 120))
                    dr.text(((xc - w / 2) * W + 3, (yc - h / 2) * H + 2), CLASSES[c], fill=COLORS[c])
                im.thumbnail((th - 2, th - 2))
                sheet.paste(im, (x0 + t * th, y0))
            d.text((x0, y0 + th), f"{slug.split('/')[1][:40]}  nest={audit[slug]['nest_rate']:.2f} b/img={audit[slug]['boxes_per_img']}", fill="black")
        sheet.save(REPORTS / "sheets" / f"sources_{page // per_sheet + 1}.jpg", quality=80)


if __name__ == "__main__":
    main()
