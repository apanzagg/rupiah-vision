"""Augmentasi tambahan (Albumentations) untuk kondisi nyata pengguna tunanetra.

Dipakai lewat model.train(augmentations=build_augmentations()). Ultralytics menerapkannya setelah
mosaic + random_perspective (rotasi/skala/perspektif) dan sebelum augmentasi HSV.
Semua transformasi di sini hanya mengubah piksel (bukan geometri), sehingga bounding box tidak berubah.
Citra di pipeline Ultralytics berformat BGR uint8.

- CrumpleOverlay: tekstur kusut/lipatan sintetis (medan acak frekuensi rendah + garis lipatan).
- FingerOcclusion: 1-3 "jari" berwarna kulit menutupi sebagian citra (simulasi uang dipegang).
- LowLight: cahaya redup (gamma tinggi + noise sensor).
- Bawaan Albumentations: motion blur, noise, brightness/contrast, gamma, kompresi JPEG, downscale.
Tidak ada flip (uang tercermin tidak ada di dunia nyata); hue shift kecil diatur lewat hsv_h saat training.
"""
from __future__ import annotations

import cv2
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform

# warna kulit (BGR), dari terang ke gelap
SKIN_BGR = np.array([[180, 200, 235], [150, 175, 220], [120, 150, 200], [95, 125, 175], [70, 95, 140], [55, 75, 110]])


class CrumpleOverlay(ImageOnlyTransform):
    def __init__(self, strength=(0.15, 0.4), folds=(0, 3), p: float = 0.25):
        super().__init__(p=p)
        self.strength, self.folds = strength, folds

    def apply(self, img, **params):
        rng = self.random_generator
        h, w = img.shape[:2]
        field = np.zeros((h, w), np.float32)
        for scale in (4, 9, 17):  # beberapa oktaf noise halus
            small = rng.random((scale, scale)).astype(np.float32)
            field += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) / scale**0.5
        field = (field - field.min()) / (np.ptp(field) + 1e-6) - 0.5
        for _ in range(int(rng.integers(self.folds[0], self.folds[1] + 1))):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            ang = rng.uniform(0, np.pi)
            dx, dy = np.cos(ang) * 2 * max(w, h), np.sin(ang) * 2 * max(w, h)
            line = np.zeros((h, w), np.float32)
            cv2.line(line, (int(x0 - dx), int(y0 - dy)), (int(x0 + dx), int(y0 + dy)), 1.0, int(rng.integers(1, 4)))
            line = cv2.GaussianBlur(line, (0, 0), float(rng.uniform(1.5, 4)))
            field += line * float(rng.choice([-1.0, 1.0])) * 1.5
        s = float(rng.uniform(*self.strength))
        out = img.astype(np.float32) * (1.0 + s * field[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("strength", "folds")


class FingerOcclusion(ImageOnlyTransform):
    def __init__(self, max_fingers: int = 3, max_area: float = 0.15, p: float = 0.2):
        super().__init__(p=p)
        self.max_fingers, self.max_area = max_fingers, max_area

    def apply(self, img, **params):
        rng = self.random_generator
        h, w = img.shape[:2]
        out = img.copy()
        base = SKIN_BGR[int(rng.integers(0, len(SKIN_BGR)))].astype(np.float32)
        n = int(rng.integers(1, self.max_fingers + 1))
        # jari-jari berdekatan dan sejajar, seperti tangan memegang uang
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        ang = rng.uniform(0, 180)
        flen = rng.uniform(0.25, 0.5) * min(h, w)
        fwid = flen * rng.uniform(0.22, 0.32)
        if n * flen * fwid > self.max_area * h * w:
            n = max(1, int(self.max_area * h * w / (flen * fwid)))
        mask = np.zeros((h, w), np.float32)
        nx, ny = np.cos(np.deg2rad(ang + 90)), np.sin(np.deg2rad(ang + 90))
        for i in range(n):
            off = (i - (n - 1) / 2) * fwid * 1.05
            center = (float(cx + nx * off), float(cy + ny * off))
            cv2.ellipse(mask, (center, (flen, fwid), ang), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.0, fwid * 0.08))[..., None]
        shade = cv2.GaussianBlur(rng.random((h // 16 + 1, w // 16 + 1)).astype(np.float32), (0, 0), 1)
        shade = cv2.resize(shade, (w, h))[..., None] * 0.35 + 0.8
        skin = np.clip(base[None, None, :] * shade, 0, 255)
        out = out.astype(np.float32) * (1 - mask) + skin * mask
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("max_fingers", "max_area")


class LowLight(ImageOnlyTransform):
    def __init__(self, gamma=(1.6, 3.0), noise=(4.0, 14.0), p: float = 0.12):
        super().__init__(p=p)
        self.gamma, self.noise = gamma, noise

    def apply(self, img, **params):
        rng = self.random_generator
        g = float(rng.uniform(*self.gamma))
        out = 255.0 * (img.astype(np.float32) / 255.0) ** g
        out += rng.normal(0, float(rng.uniform(*self.noise)), img.shape).astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("gamma", "noise")


def build_augmentations() -> list:
    import albumentations as A

    return [
        CrumpleOverlay(p=0.25),
        FingerOcclusion(p=0.2),
        LowLight(p=0.12),
        A.MotionBlur(blur_limit=(3, 9), p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.05),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.3),
        A.RandomGamma(gamma_limit=(70, 150), p=0.15),
        A.ImageCompression(quality_range=(45, 95), p=0.2),
        A.Downscale(scale_range=(0.5, 0.9), p=0.1),
    ]


if __name__ == "__main__":
    # Pratinjau: python augment_custom.py <citra> <keluaran.jpg>
    import sys

    import albumentations as A

    src = cv2.imread(sys.argv[1])
    src = cv2.resize(src, (320, int(320 * src.shape[0] / src.shape[1])))
    tiles = [src]
    for t in [CrumpleOverlay(p=1), FingerOcclusion(p=1), LowLight(p=1), A.MotionBlur(blur_limit=(7, 9), p=1)]:
        for _ in range(2):
            tiles.append(t(image=src)["image"])
    rows = [np.hstack(tiles[i : i + 3]) for i in range(0, 9, 3)]
    cv2.imwrite(sys.argv[2], np.vstack(rows))
