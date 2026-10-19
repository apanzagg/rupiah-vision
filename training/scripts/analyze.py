"""Tahap 4 pipeline data: analisis dataset final + contact sheet untuk pemeriksaan visual.

Keluaran di SR_REPORTS:
  figures/class_distribution.png, bbox_size_hist.png, boxes_per_image.png, source_contribution.png, samples_grid.png
  sheets/crops_<kelas>.jpg      crop box per sumber (cek pemetaan kelas per sumber)
  sheets/negatives.jpg          contoh citra negatif per jenis
  sheets/autolabel_<kelas>.jpg  semua hasil auto-label (merah = ditolak otomatis)
  summary.md, source_stats.json
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from common import CLASSES, FINAL, REPORTS, SEED, WORK  # noqa: E402

LABEL = {c: f"Rp{int(c):,}".replace(",", ".") for c in CLASSES}
# Palet kategorikal tervalidasi (slot 1..3) + tinta teks; lihat skill dataviz/references/palette.md
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BOX_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    out = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                out.append((int(p[0]), *map(float, p[1:])))
    return out


def draw(img: Image.Image, boxes, width: int = 3) -> Image.Image:
    d = ImageDraw.Draw(img)
    W, H = img.size
    for c, xc, yc, w, h in boxes:
        d.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H],
                    outline=BOX_COLORS[c], width=width)
        d.text(((xc - w / 2) * W + 4, (yc - h / 2) * H + 2), LABEL[CLASSES[c]], fill=BOX_COLORS[c])
    return img


def grid(tiles: list[tuple[Image.Image, str]], cols: int, size: int, title: str = "") -> Image.Image:
    rows = max(1, (len(tiles) + cols - 1) // cols)
    top = 22 if title else 0
    sheet = Image.new("RGB", (cols * size, top + rows * (size + 14)), "white")
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((4, 4), title, fill="black")
    for k, (im, cap) in enumerate(tiles):
        im = im.copy()
        im.thumbnail((size - 4, size - 4))
        x, y = (k % cols) * size, top + (k // cols) * (size + 14)
        sheet.paste(im, (x + 2, y + 2))
        d.text((x + 2, y + size), cap[:26], fill="black")
    return sheet


def main() -> None:
    rng = random.Random(SEED)
    figs, sheets = REPORTS / "figures", REPORTS / "sheets"
    figs.mkdir(parents=True, exist_ok=True)
    sheets.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(WORK / "manifest_final.csv", encoding="utf-8")))
    rep = json.loads((WORK / "dedup_report.json").read_text(encoding="utf-8"))
    ingest = json.loads((WORK / "ingest_stats.json").read_text(encoding="utf-8")) if (WORK / "ingest_stats.json").exists() else {}

    # ---------- statistik per sumber ----------
    src_stats = defaultdict(lambda: Counter())
    for r in rows:
        src_stats[r["slug"]]["kept"] += 1
        src_stats[r["slug"]][r["split"]] += 1
    for slug, st in ingest.items():
        src_stats[slug]["read"] = st.get("read", 0)
        src_stats[slug]["pooled"] = st.get("pooled", 0)
    (REPORTS / "source_stats.json").write_text(json.dumps(src_stats, indent=1), encoding="utf-8")
    (WORK / "source_stats.json").write_text(json.dumps(src_stats, indent=1), encoding="utf-8")

    # ---------- grafik distribusi kelas ----------
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    x = range(len(CLASSES))
    wbar = 0.26
    for k, s in enumerate(("train", "val", "test")):
        vals = [rep["splits"][s]["instances"][c] for c in CLASSES]
        ax.bar([i + (k - 1) * (wbar + 0.02) for i in x], vals, width=wbar, color=SERIES[k], label=s)
    ax.set_xticks(list(x), [LABEL[c] for c in CLASSES])
    ax.set_ylabel("Jumlah objek (bounding box)", color=INK2)
    ax.set_title("Distribusi kelas per split", color=INK, loc="left", fontsize=11)
    ax.legend(frameon=False, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(figs / "class_distribution.png")
    plt.close(fig)

    # ---------- ukuran box & box per citra ----------
    sizes, per_img = [], Counter()
    for r in rows:
        if r["split"] != "train":
            continue
        b = read_boxes(FINAL / "labels" / r["split"] / f"{r['id']}.txt")
        per_img[min(len(b), 5)] += 1
        sizes += [(w * h) ** 0.5 for _, _, _, w, h in b]
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.hist(sizes, bins=40, range=(0, 1), color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax.set_xlabel("Ukuran relatif box = akar(luas box / luas citra)", color=INK2)
    ax.set_ylabel("Jumlah box (train)", color=INK2)
    ax.set_title("Distribusi ukuran bounding box", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "bbox_size_hist.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ks = list(range(6))
    ax.bar(ks, [per_img.get(k, 0) for k in ks], color=SERIES[0], width=0.6)
    ax.set_xticks(ks, ["0 (negatif)", "1", "2", "3", "4", "5+"])
    ax.set_xlabel("Jumlah uang per citra", color=INK2)
    ax.set_ylabel("Jumlah citra (train)", color=INK2)
    ax.set_title("Jumlah objek per citra", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "boxes_per_image.png")
    plt.close(fig)

    # ---------- kontribusi sumber ----------
    top = sorted(((v["kept"], k) for k, v in src_stats.items() if v.get("kept")), reverse=True)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.24 * len(top))), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.barh([k for _, k in top][::-1], [v for v, _ in top][::-1], color=SERIES[0], height=0.6)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlabel("Citra di dataset final", color=INK2)
    ax.set_title("Kontribusi tiap sumber setelah dedup", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / "source_contribution.png")
    plt.close(fig)

    # ---------- contoh citra ----------
    pos = [r for r in rows if r["role"] == "pos" and r["split"] == "train"]
    rng.shuffle(pos)
    tiles = []
    for r in pos[:16]:
        im = Image.open(FINAL / "images" / r["split"] / f"{r['id']}.jpg").convert("RGB")
        tiles.append((draw(im, read_boxes(FINAL / "labels" / r["split"] / f"{r['id']}.txt")), ""))
    if tiles:
        grid(tiles, 4, 300).save(figs / "samples_grid.png")

    # ---------- contact sheet crop per kelas per sumber ----------
    by_cls_src = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["role"] != "pos":
            continue
        for b in read_boxes(FINAL / "labels" / r["split"] / f"{r['id']}.txt"):
            by_cls_src[b[0]][r["slug"]].append((r, b))
    for c in range(len(CLASSES)):
        tiles = []
        for slug in sorted(by_cls_src[c]):
            items = by_cls_src[c][slug]
            rng.shuffle(items)
            for r, (_, xc, yc, w, h) in items[:3]:
                im = Image.open(FINAL / "images" / r["split"] / f"{r['id']}.jpg").convert("RGB")
                W, H = im.size
                crop = im.crop((max(0, (xc - w / 2) * W), max(0, (yc - h / 2) * H),
                                min(W, (xc + w / 2) * W), min(H, (yc + h / 2) * H)))
                tiles.append((crop, slug.split("/")[1][:18]))
        if tiles:
            grid(tiles, 12, 112, f"Kelas {LABEL[CLASSES[c]]}: 3 crop acak per sumber").save(
                sheets / f"crops_{CLASSES[c]}.jpg", quality=80)

    # ---------- contoh negatif ----------
    neg = defaultdict(list)
    for r in rows:
        if r["role"] == "neg":
            neg[r["kind"]].append(r)
    tiles = []
    for kind in sorted(neg):
        rng.shuffle(neg[kind])
        for r in neg[kind][:12]:
            tiles.append((Image.open(FINAL / "images" / r["split"] / f"{r['id']}.jpg").convert("RGB"), kind))
    if tiles:
        grid(tiles, 12, 112, "Citra negatif per jenis").save(sheets / "negatives.jpg", quality=80)

    # ---------- contact sheet auto-label ----------
    al_path = WORK / "pool" / "manifest_autolabel.csv"
    if al_path.exists():
        man = {r["id"]: r for r in csv.DictReader(open(WORK / "pool" / "manifest.csv", encoding="utf-8"))}
        al = list(csv.DictReader(open(al_path, encoding="utf-8")))
        by_c = defaultdict(list)
        for a in al:
            by_c[int(man[a["id"]]["al_class"])].append(a)
        for c, items in sorted(by_c.items()):
            tiles = []
            for a in sorted(items, key=lambda a: a["id"]):
                im = Image.open(WORK / "pool" / "images" / f"{a['id']}.jpg").convert("RGB")
                if a["status"] == "ok":
                    im = draw(im, read_boxes(WORK / "pool" / "labels" / f"{a['id']}.txt"), width=4)
                else:
                    ImageDraw.Draw(im).rectangle([0, 0, im.width - 1, im.height - 1], outline="red", width=12)
                tiles.append((im, a["id"][-10:] + (" M" if a["al_multi"] == "1" else "")))
            for page in range(0, len(tiles), 120):
                grid(tiles[page : page + 120], 12, 112,
                     f"Auto-label {LABEL[CLASSES[c]]} ({page // 120 + 1}); merah = ditolak otomatis; M = multi-box").save(
                    sheets / f"autolabel_{CLASSES[c]}_{page // 120 + 1}.jpg", quality=80)

    # ---------- ringkasan teks ----------
    lines = ["# Ringkasan dataset final\n", "| Split | Citra | Negatif | " + " | ".join(LABEL[c] for c in CLASSES) + " |",
             "|---|---|---|" + "---|" * len(CLASSES)]
    for s in ("train", "val", "test"):
        st = rep["splits"][s]
        lines.append(f"| {s} | {st['images']} | {st['negatives']} | " + " | ".join(str(st["instances"][c]) for c in CLASSES) + " |")
    lines.append(f"\nInput pool: {rep['input']}; setelah dedup/kuota: {rep['after_dedup']}; grup: {rep['groups']}")
    lines.append(f"Dibuang: {json.dumps(rep['drop'])}")
    (REPORTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
