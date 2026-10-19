"""Diagnosis dedup: gambar pasangan citra yang dinilai duplikat tetapi berlabel kelas berbeda (konflik label).

Keluaran: SR_REPORTS/sheets/conflicts_<n>.jpg dan SR_REPORTS/conflict_stats.json
Dipakai untuk memeriksa secara visual apakah konflik = label salah di sumber, atau citra berbeda yang
keliru dianggap duplikat (jika yang kedua, ambang dedup perlu diperketat).
"""
from __future__ import annotations

import json
import random
from collections import Counter

from PIL import Image, ImageDraw

from common import CLASSES, REPORTS, SEED, WORK


def label(cls_str: str) -> str:
    return "+".join(CLASSES[int(c)] for c in cls_str.split()) or "-"


def main() -> None:
    rep = json.loads((WORK / "dedup_report.json").read_text(encoding="utf-8"))
    conf = rep.get("conflict_clusters", [])
    (REPORTS / "sheets").mkdir(parents=True, exist_ok=True)
    stats = {
        "n": rep.get("n_conflict_clusters", len(conf)),
        "grad_hist": Counter(round(c["grad"], 2) // 0.02 * 0.02 for c in conf),
        "source_pairs": Counter(" | ".join(sorted({c["a_src"].split("/")[1], c["b_src"].split("/")[1]})) for c in conf).most_common(25),
        "label_pairs": Counter(" vs ".join(sorted({label(c["a_cls"]), label(c["b_cls"])})) for c in conf).most_common(25),
    }
    stats["grad_hist"] = {f"{k:.2f}": v for k, v in sorted(stats["grad_hist"].items())}
    (REPORTS / "conflict_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    rng = random.Random(SEED)
    sample = rng.sample(conf, min(72, len(conf)))
    sample.sort(key=lambda c: c["grad"])
    th, per_page, cols = 150, 24, 4  # 4 pasangan per baris
    for page in range(0, len(sample), per_page):
        chunk = sample[page : page + per_page]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (2 * th + 12), rows * (th + 30)), "white")
        d = ImageDraw.Draw(sheet)
        for k, c in enumerate(chunk):
            x0, y0 = (k % cols) * (2 * th + 12), (k // cols) * (th + 30)
            for t, (img_id, cls, src) in enumerate(((c["a"], c["a_cls"], c["a_src"]), (c["b"], c["b_cls"], c["b_src"]))):
                im = Image.open(WORK / "pool" / "images" / f"{img_id}.jpg").convert("RGB")
                im.thumbnail((th - 2, th - 2))
                sheet.paste(im, (x0 + t * th, y0))
                d.text((x0 + t * th, y0 + th + 1), f"{label(cls)}", fill="red" if t else "blue")
                d.text((x0 + t * th, y0 + th + 13), src.split("/")[1][:22], fill="black")
            d.text((x0 + 2, y0 + 2), f"g={c['grad']:.2f}", fill="yellow")
        sheet.save(REPORTS / "sheets" / f"conflicts_{page // per_page + 1}.jpg", quality=80)
    print(json.dumps({k: v for k, v in stats.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
