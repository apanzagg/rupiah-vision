"""Bangun set uji per kondisi di data/test_conditions/<kondisi>/{images,labels,meta.json}.

Sumber (TIDAK pernah dipakai training): split test dataset final (data/final_test/final/...) dan
dataset kondisi Kaggle CC0 (folder lusuh/sedikit_lusuh/uang_baru, dilabeli manual via labels_kondisi.csv).

Langkah:
  python make_test_conditions.py candidates   -> pilih kandidat dengan heuristik + contact sheet bernomor
  (periksa contact sheet; tulis id yang SALAH ke training/configs/test_conditions_reject.txt)
  python make_test_conditions.py build        -> salin citra terpilih (+ versi sintetis bertanda 'sint_')

Kondisi:
  baru               citra test 1 lembar, terang, latar polos, tajam (nyata)
  latar_ramai        citra test dengan kepadatan tepi tinggi di luar box (nyata)
  cahaya_redup       citra test gelap (nyata) + penggelapan sintetis bila kurang dari MIN_REAL
  lusuh_kusut        foto uang lusuh/sedikit lusuh (Kaggle CC0, nyata) dengan label manual
  terlipat_tertutup  SINTETIS: oklusi jari besar + garis lipatan pada citra test
  bukan_rupiah       citra negatif test (uang asing, struk, dompet, tangan, koin)
"""
from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from common import CLASSES, CONFIGS, DATA, SEED

TEST = DATA / "final_test" / "final"
OUT = DATA / "test_conditions"
CAND = DATA / "interim" / "conditions_candidates"
KONDISI_RAW = DATA / "final_test" / "conditions_raw"
N_PER = 120
MIN_REAL = 60


def boxes(lbl: Path) -> list[list[float]]:
    return [list(map(float, l.split())) for l in lbl.read_text().splitlines() if len(l.split()) == 5] if lbl.exists() else []


def features(img: np.ndarray, bxs: list[list[float]]) -> dict:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.ones((h, w), bool)
    for _, xc, yc, bw, bh in bxs:
        mask[int(max(0, (yc - bh / 2) * h)) : int((yc + bh / 2) * h), int(max(0, (xc - bw / 2) * w)) : int((xc + bw / 2) * w)] = False
    edges = cv2.Canny(gray, 80, 160) > 0
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()
    return {
        "luma": float(gray.mean()),
        "bg_edges": float(edges[mask].mean()) if mask.any() else 0.0,
        "bg_frac": float(mask.mean()),
        "sharp": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "sat": float(sat),
    }


def sheet(items: list[tuple[str, np.ndarray, str]], path: Path, th: int = 150, cols: int = 10) -> None:
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * (th + 18), cols * th, 3), 255, np.uint8)
    for k, (name, img, cap) in enumerate(items):
        s = th / max(img.shape[:2])
        im = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))
        y, x = (k // cols) * (th + 18), (k % cols) * th
        canvas[y : y + im.shape[0], x : x + im.shape[1]] = im
        cv2.putText(canvas, f"{k}:{cap}", (x + 2, y + th + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])


def draw_boxes(img: np.ndarray, bxs: list[list[float]]) -> np.ndarray:
    im = img.copy()
    h, w = im.shape[:2]
    for c, xc, yc, bw, bh in bxs:
        p0 = (int((xc - bw / 2) * w), int((yc - bh / 2) * h))
        p1 = (int((xc + bw / 2) * w), int((yc + bh / 2) * h))
        cv2.rectangle(im, p0, p1, (0, 212, 255), max(2, w // 150))
        cv2.putText(im, CLASSES[int(c)], (p0[0] + 4, p0[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, max(0.8, w / 600), (0, 212, 255), 2)
    return im


def candidates() -> None:
    rng = random.Random(SEED)
    man = [r for r in csv.DictReader(open(TEST / "manifest_final.csv", encoding="utf-8")) if r["split"] == "test"]
    feats = {}
    for r in man:
        img = cv2.imread(str(TEST / "images" / "test" / f"{r['id']}.jpg"))
        if img is None:
            continue
        feats[r["id"]] = {**features(img, boxes(TEST / "labels" / "test" / f"{r['id']}.txt")), "role": r["role"], "kind": r["kind"],
                          "n": int(r["n_boxes"] or 0), "slug": r["slug"]}
    pos = {k: f for k, f in feats.items() if f["role"] == "pos" and f["sat"] > 25}  # buang citra hitam-putih
    pick = {
        # uang terlihat utuh dengan latar (box 10-70% luas citra), 1 lembar, terang, tajam, latar polos
        "baru": sorted((k for k, f in pos.items() if f["n"] == 1 and 0.3 < f["bg_frac"] < 0.9 and 90 < f["luma"] < 200
                        and f["bg_edges"] < 0.05 and f["sharp"] > 100),
                       key=lambda k: pos[k]["bg_edges"])[: N_PER + 40],
        "latar_ramai": sorted((k for k, f in pos.items() if f["bg_frac"] > 0.35 and f["bg_edges"] > 0.12), key=lambda k: -pos[k]["bg_edges"])[: N_PER + 40],
        "cahaya_redup": sorted((k for k, f in pos.items() if f["luma"] < 70), key=lambda k: pos[k]["luma"])[: N_PER + 30],
    }
    negs = [k for k, f in feats.items() if f["role"] == "neg"]
    rng.shuffle(negs)
    pick["bukan_rupiah"] = negs[:200]
    used = set().union(*pick.values())
    rest = [k for k, f in pos.items() if k not in used and f["n"] == 1 and f["luma"] > 90]
    rng.shuffle(rest)
    pick["terlipat_tertutup"] = rest[: N_PER]
    pick["cahaya_redup_sint"] = rest[N_PER : N_PER + max(0, N_PER - len(pick["cahaya_redup"]))]
    CAND.mkdir(parents=True, exist_ok=True)
    (CAND / "picks.json").write_text(json.dumps(pick, indent=1), encoding="utf-8")
    for cond, ids in pick.items():
        if cond in ("terlipat_tertutup", "cahaya_redup_sint"):
            continue  # sintetis dari citra test yang labelnya sudah terverifikasi
        for page in range(0, len(ids), 80):
            items = []
            for i in ids[page : page + 80]:
                img = cv2.imread(str(TEST / "images" / "test" / f"{i}.jpg"))
                items.append((i, draw_boxes(img, boxes(TEST / "labels" / "test" / f"{i}.txt")), i[-6:]))
            sheet(items, CAND / f"{cond}_{page // 80 + 1}.jpg")
    print({k: len(v) for k, v in pick.items()})


def synth_fold_occlude(img: np.ndarray, bxs: list[list[float]], rng: np.random.Generator) -> np.ndarray:
    """Oklusi jari 20-45% box + garis lipatan (bayangan) pada uang."""
    sys.path.insert(0, str(Path(__file__).parent))
    from augment_custom import CrumpleOverlay, FingerOcclusion

    out = CrumpleOverlay(strength=(0.25, 0.45), folds=(1, 2), p=1.0)(image=img)["image"]
    h, w = out.shape[:2]
    _, xc, yc, bw, bh = bxs[0]
    x0, y0 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
    x1, y1 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
    roi = out[max(0, y0) : y1, max(0, x0) : x1]
    if roi.size:
        occ = FingerOcclusion(max_fingers=3, max_area=0.45, p=1.0)
        occ.set_random_seed(int(rng.integers(1 << 30)))
        out[max(0, y0) : y1, max(0, x0) : x1] = occ(image=roi)["image"]
    return out


def synth_dark(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    g = rng.uniform(2.0, 3.0)
    out = 255.0 * (img.astype(np.float32) / 255.0) ** g + rng.normal(0, rng.uniform(4, 10), img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def build() -> None:
    rng = np.random.default_rng(SEED)
    pick = json.loads((CAND / "picks.json").read_text(encoding="utf-8"))
    rej_file = CONFIGS / "test_conditions_reject.txt"
    reject = {l.split("#")[0].strip() for l in rej_file.read_text(encoding="utf-8").splitlines()} if rej_file.exists() else set()
    if OUT.exists():
        shutil.rmtree(OUT)

    def put(cond: str, name: str, img: np.ndarray, lbl_lines: list[str]) -> None:
        (OUT / cond / "images").mkdir(parents=True, exist_ok=True)
        (OUT / cond / "labels").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / cond / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (OUT / cond / "labels" / f"{name}.txt").write_text("".join(l + "\n" for l in lbl_lines))

    meta: dict[str, dict] = {}
    for cond in ("baru", "latar_ramai", "cahaya_redup", "bukan_rupiah"):
        ids = [i for i in pick[cond] if i not in reject][: (200 if cond == "bukan_rupiah" else N_PER)]
        for i in ids:
            img = cv2.imread(str(TEST / "images" / "test" / f"{i}.jpg"))
            put(cond, i, img, (TEST / "labels" / "test" / f"{i}.txt").read_text().splitlines())
        meta[cond] = {"real": len(ids), "synthetic": 0, "source": "split test dataset final (dipilih heuristik + diperiksa visual)"}
    for i in pick["cahaya_redup_sint"]:
        if i in reject:
            continue
        img = cv2.imread(str(TEST / "images" / "test" / f"{i}.jpg"))
        put("cahaya_redup", f"sint_{i}", synth_dark(img, rng), (TEST / "labels" / "test" / f"{i}.txt").read_text().splitlines())
        meta["cahaya_redup"]["synthetic"] += 1
    n = 0
    for i in pick["terlipat_tertutup"]:
        lbl = TEST / "labels" / "test" / f"{i}.txt"
        img = cv2.imread(str(TEST / "images" / "test" / f"{i}.jpg"))
        put("terlipat_tertutup", f"sint_{i}", synth_fold_occlude(img, boxes(lbl), rng), lbl.read_text().splitlines())
        n += 1
    meta["terlipat_tertutup"] = {"real": 0, "synthetic": n, "source": "SINTETIS: oklusi jari 20-45% box + lipatan pada citra test"}
    # lusuh_kusut & baru_pindaian: label manual (configs/labels_kondisi.csv: file,kondisi,kelas,x0,y0,x1,y1,catatan;
    # kelas kosong = dibuang, mis. emisi lama / Rp75.000)
    lab_csv = CONFIGS / "labels_kondisi.csv"
    if lab_csv.exists():
        cnt: dict[str, int] = {}
        for r in csv.DictReader(open(lab_csv, encoding="utf-8")):
            if not r["kelas"]:
                continue
            img = cv2.imread(str(KONDISI_RAW / r["file"]))
            if img is None:
                continue
            x0, y0, x1, y1 = (float(r[c]) for c in ("x0", "y0", "x1", "y1"))
            line = f"{CLASSES.index(r['kelas'])} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"
            name = Path(r["file"]).parent.name + "_" + Path(r["file"]).stem.replace(" ", "_").replace(".", "_")
            put(r["kondisi"], name, img, [line])
            cnt[r["kondisi"]] = cnt.get(r["kondisi"], 0) + 1
        for k, n in cnt.items():
            meta[k] = {"real": n, "synthetic": 0, "source": "Kaggle indahtricahyani/kondisi-uang-kertas-rupiah (CC0), "
                       "pindaian uang (memenuhi bingkai), label nominal manual"}
    for cond, m in meta.items():
        (OUT / cond).mkdir(parents=True, exist_ok=True)
        (OUT / cond / "meta.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    {"candidates": candidates, "build": build}[sys.argv[1]]()
