"""Pipeline inferensi REFERENSI (Python + onnxruntime) yang meniru persis pipeline web.

Dipakai untuk: output referensi parity test web, evaluasi per kondisi, dan benchmark CPU.
Langkah: letterbox ke SxS (padding 114, posisi tengah) -> RGB/255 NCHW -> model -> output [1, 4+nc, N]
-> filter confidence -> NMS class-agnostic -> koordinat dikembalikan ke piksel citra asli.
"""
from __future__ import annotations

import numpy as np

PAD_VALUE = 114


def letterbox_params(w: int, h: int, size: int) -> tuple[float, int, int, int, int]:
    """(skala, lebar_baru, tinggi_baru, pad_kiri, pad_atas) - rumus yang sama dipakai di web (letterbox.ts)."""
    scale = min(size / w, size / h)
    # pembulatan setengah-ke-atas agar sama dengan Math.round di JavaScript (round() Python = banker's rounding)
    nw, nh = int(w * scale + 0.5), int(h * scale + 0.5)
    return scale, nw, nh, (size - nw) // 2, (size - nh) // 2


def preprocess(img_rgb: np.ndarray, size: int) -> tuple[np.ndarray, tuple]:
    import cv2

    h, w = img_rgb.shape[:2]
    scale, nw, nh, px, py = letterbox_params(w, h, size)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[py : py + nh, px : px + nw] = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, (scale, px, py, w, h)


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x0 = np.maximum(a[0], b[:, 0])
    y0 = np.maximum(a[1], b[:, 1])
    x1 = np.minimum(a[2], b[:, 2])
    y1 = np.minimum(a[3], b[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


def decode(out: np.ndarray, meta: tuple, conf: float = 0.25, iou: float = 0.45, max_det: int = 20) -> list[dict]:
    """out: [1, 4+nc, N]. Kembalikan [{cls, score, box:[x0,y0,x1,y1] piksel citra asli}] terurut skor."""
    scale, px, py, w, h = meta
    p = out[0].T  # [N, 4+nc]
    scores = p[:, 4:]
    cls = scores.argmax(1)
    sc = scores[np.arange(len(cls)), cls]
    keep = sc >= conf
    p, cls, sc = p[keep], cls[keep], sc[keep]
    if len(p) == 0:
        return []
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    order = np.argsort(-sc)
    picked = []
    while len(order) and len(picked) < max_det:
        i = order[0]
        picked.append(i)
        if len(order) == 1:
            break
        ious = iou_xyxy(boxes[i], boxes[order[1:]])
        order = order[1:][ious <= iou]
    dets = []
    for i in picked:
        x0, y0, x1, y1 = boxes[i]
        x0, x1 = np.clip([(x0 - px) / scale, (x1 - px) / scale], 0, w)
        y0, y1 = np.clip([(y0 - py) / scale, (y1 - py) / scale], 0, h)
        dets.append({"cls": int(cls[i]), "score": round(float(sc[i]), 5),
                     "box": [round(float(v), 2) for v in (x0, y0, x1, y1)]})
    return dets


class OnnxDetector:
    def __init__(self, path: str, threads: int = 0):
        import onnxruntime as ort

        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.size = int(self.inp.shape[2])

    def __call__(self, img_rgb: np.ndarray, conf: float = 0.25, iou: float = 0.45, max_det: int = 20) -> list[dict]:
        x, meta = preprocess(img_rgb, self.size)
        out = self.sess.run(None, {self.inp.name: x})[0]
        return decode(out, meta, conf, iou, max_det)
