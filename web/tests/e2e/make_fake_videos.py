"""Buat video kamera palsu (.y4m) untuk E2E Playwright dari citra set uji.

Chrome memutar berkas ini sebagai kamera (--use-file-for-fake-video-capture), berulang terus.
Kasus:
  nominal : satu lembar uang (diambil dari data/test_conditions/baru) -> harus diumumkan nominalnya
  multi   : dua lembar berbeda nominal digabung berdampingan            -> "Dua lembar ... Total ..."
  kosong  : meja tanpa uang (citra negatif 'bukan_rupiah')             -> panduan "uang belum terdeteksi"
  gelap   : citra yang sangat gelap                                    -> panduan "ruangan terlalu gelap"
Keluaran: web/tests/e2e/.fake/*.y4m + cases.json (diabaikan git).

Pemakaian: python make_fake_videos.py --model ../../../models/model-v1-int8_mixed.onnx
(model dipakai hanya untuk MEMILIH citra yang labelnya pasti dan terdeteksi jelas, agar E2E menguji
aplikasi, bukan batas akurasi model; akurasi diukur terpisah di eval_conditions.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
from common import CLASSES  # noqa: E402
from infer_ref import OnnxDetector  # noqa: E402

W, H, FPS, SECONDS = 640, 480, 10, 6
WORDS = {"1000": "Rp1.000", "2000": "Rp2.000", "5000": "Rp5.000", "10000": "Rp10.000", "20000": "Rp20.000", "50000": "Rp50.000", "100000": "Rp100.000"}


def fit(img: np.ndarray, w: int = W, h: int = H, bg: int = 40) -> np.ndarray:
    s = min(w / img.shape[1], h / img.shape[0])
    im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
    canvas = np.full((h, w, 3), bg, np.uint8)
    y, x = (h - im.shape[0]) // 2, (w - im.shape[1]) // 2
    canvas[y : y + im.shape[0], x : x + im.shape[1]] = im
    return canvas


def write_y4m(path: Path, frame_bgr: np.ndarray) -> None:
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV_I420)
    with open(path, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
        for i in range(FPS * SECONDS):
            # sedikit getaran kamera agar mirip tangan memegang ponsel
            dx, dy = int(3 * np.sin(i / 3)), int(2 * np.cos(i / 4))
            m = np.float32([[1, 0, dx], [0, 1, dy]])
            fr = cv2.warpAffine(frame_bgr, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
            f.write(b"FRAME\n" + cv2.cvtColor(fr, cv2.COLOR_BGR2YUV_I420).tobytes())
    del yuv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    det = OnnxDetector(args.model)
    out = HERE / ".fake"
    out.mkdir(exist_ok=True)
    base = ROOT / "data" / "test_conditions"
    cases = []
    singles = {}
    for img_path in sorted((base / "baru" / "images").glob("*.jpg")):
        lbl = (base / "baru" / "labels" / f"{img_path.stem}.txt").read_text().split()
        if len(lbl) != 5:
            continue
        c = CLASSES[int(lbl[0])]
        if c in singles:
            continue
        frame = fit(cv2.imread(str(img_path)))
        p = det(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.25)
        if len(p) == 1 and CLASSES[p[0]["cls"]] == c and p[0]["score"] >= 0.85:
            singles[c] = frame
    for c in ("50000", "2000"):  # dua kasus nominal (biru & abu-abu: mudah tertukar dengan 20000)
        if c in singles:
            write_y4m(out / f"rp{c}.y4m", singles[c])
            cases.append({"video": f"rp{c}.y4m", "kind": "nominal", "expect": WORDS[c]})
    pair = [c for c in ("100000", "20000", "10000", "5000") if c in singles][:2]
    if len(pair) == 2:
        a, b = (cv2.resize(singles[c][60:420, 80:560], (320, 360)) for c in pair)
        frame = fit(np.hstack([a, b]))
        p = det(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.25)
        if sorted(CLASSES[q["cls"]] for q in p if q["score"] >= 0.7) == sorted(pair):
            write_y4m(out / "multi.y4m", frame)
            total = sum(int(c) for c in pair)
            cases.append({"video": "multi.y4m", "kind": "nominal", "expect": f"Total Rp{total:,}".replace(",", ".")})
    neg = sorted((base / "bukan_rupiah" / "images").glob("*.jpg"))
    for img_path in neg:
        frame = cv2.convertScaleAbs(fit(cv2.imread(str(img_path))), alpha=1.0, beta=10)
        if cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean() < 80:
            continue  # kasus 'kosong' harus cukup terang agar panduannya "belum terdeteksi", bukan "gelap"
        if not [q for q in det(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), conf=0.25) if q["score"] >= 0.5]:
            write_y4m(out / "kosong.y4m", frame)
            cases.append({"video": "kosong.y4m", "kind": "panduan", "expect": "guide_not_found"})
            # versi sangat gelap dari adegan yang sama (tanpa uang) -> panduan "ruangan terlalu gelap"
            write_y4m(out / "gelap.y4m", (frame * 0.1).astype(np.uint8))
            cases.append({"video": "gelap.y4m", "kind": "panduan", "expect": "guide_dark"})
            break
    (out / "cases.json").write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(json.dumps(cases, indent=1))


if __name__ == "__main__":
    main()
