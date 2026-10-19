"""Bangkitkan ikon PWA SuaraRupiah (dijalankan sekali; hasil di-commit di public/icons)."""
from pathlib import Path

import matplotlib
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "public" / "icons"
FONT = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
YELLOW, BLACK = (255, 212, 0), (0, 0, 0)


def icon(size: int, safe: float) -> Image.Image:
    im = Image.new("RGB", (size, size), BLACK)
    d = ImageDraw.Draw(im)
    m = size * (1 - safe) / 2
    d.rounded_rectangle([m, m, size - m, size - m], radius=size * 0.14 * safe, fill=YELLOW)
    inner = size - 2 * m
    font = ImageFont.truetype(str(FONT), int(inner * 0.40))
    d.text((m + inner * 0.08, size / 2), "Rp", font=font, fill=BLACK, anchor="lm")
    cx, cy = m + inner * 0.74, size / 2
    for k in range(3):  # gelombang suara
        r = inner * (0.08 + 0.085 * k)
        d.arc([cx - r, cy - r, cx + r, cy + r], start=-50, end=50, fill=BLACK, width=max(2, int(inner * 0.045)))
    return im


OUT.mkdir(parents=True, exist_ok=True)
icon(192, 0.92).save(OUT / "icon-192.png")
icon(512, 0.92).save(OUT / "icon-512.png")
icon(512, 0.72).save(OUT / "icon-maskable-512.png")
icon(180, 0.92).save(OUT / "apple-touch-icon.png")
icon(64, 0.95).save(OUT / "favicon.png")
print("ikon dibuat di", OUT)
