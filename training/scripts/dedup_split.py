"""Tahap 3 pipeline data: hapus duplikat, kelompokkan citra mirip, split 70/15/15, tulis dataset final.

Kemiripan dua citra = jarak Hamming pHash minimum atas 4 rotasi (0/90/180/270), DIKONFIRMASI dengan
selisih rata-rata thumbnail warna 8x8 pada rotasi yang sama (pHash hanya melihat luminans, sehingga dua
nominal berbeda dengan tata letak sama bisa ber-pHash mirip; warna membedakannya).

- Duplikat: jarak pHash <= T_DUP & warna <= C_DUP, DAN jarak gradien ternormalisasi (128x128) <= G_DUP.
  Cek gradien diperlukan karena uang berbeda nominal dengan pose/latar sama (mis. dataset latar hitam
  hitam-putih) bisa lolos pHash+warna; gradien menangkap perbedaan detail (angka, gambar) tanpa
  terpengaruh luas latar polos. Kalibrasi: duplikat buatan (resize+JPEG) median 0,13; uang beda nominal
  minimum 0,18. Satu citra per klaster dipertahankan; jika label kelasnya berbeda (konflik label),
  semua anggota dibuang. Citra negatif yang menduplikasi citra positif dibuang.
- Grup (jarak <= T_GROUP & warna <= C_GROUP, ditambah salinan augmentasi dari citra asal yang sama):
  satu grup selalu masuk split yang sama -> tidak ada kebocoran train/val/test.
- Grup berisi salinan augmentasi Roboflow dipaksa ke train, supaya val/test hanya berisi citra asli.
- Split stratified per kelas (berdasarkan jumlah citra yang memuat tiap kelas + negatif).
- Citra negatif disampel agar sekitar NEG_FRAC dari total.

Keluaran: SR_FINAL/{images,labels}/{train,val,test}/, data.yaml, train_balanced.txt (opsional),
          SR_WORK/manifest_final.csv, SR_WORK/dedup_report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import time
from collections import Counter, defaultdict

import numpy as np

from common import CLASSES, CONFIGS, FINAL, SEED, WORK

T_DUP, C_DUP, G_DUP = 4, 14.0, 0.15
T_GROUP, C_GROUP = 10, 24.0
SPLITS = ("train", "val", "test")
FRACS = np.array([0.70, 0.15, 0.15])
NEG_FRAC = 0.10


def popcount64(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(*x.shape, 8), axis=-1).sum(-1)


class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def find_pairs(phash: np.ndarray, thumbs: np.ndarray, block: int = 512) -> list[tuple[int, int, int, float, int]]:
    """Semua pasangan (i, j, jarak, selisih_warna, rotasi_j) dengan jarak pHash <= T_GROUP (rotasi terbaik)."""
    n = len(phash)
    p0 = phash[:, 0]
    best: dict[tuple[int, int], tuple[int, float, int]] = {}
    for k in range(4):
        pk = phash[:, k]
        for s in range(0, n, block):
            d = popcount64(p0[s : s + block, None] ^ pk[None, :])
            ii, jj = np.nonzero(d <= T_GROUP)
            for i, j in zip(ii + s, jj):
                if i == j:
                    continue
                dist = int(d[i - s, j])
                cdiff = float(np.abs(thumbs[i, 0].astype(np.int16) - thumbs[j, k].astype(np.int16)).mean())
                key = (min(i, j), max(i, j))
                # rotasi relatif dinyatakan untuk citra kedua (j dirotasi k kali 90 derajat)
                rot = k if i < j else (4 - k) % 4
                old = best.get(key)
                if old is None or (dist, cdiff) < old[:2]:
                    best[key] = (dist, cdiff, rot)
    return [(i, j, d, c, r) for (i, j), (d, c, r) in best.items()]


_GRAD_CACHE: dict[str, np.ndarray] = {}


def grad_map(img_id: str) -> np.ndarray:
    g = _GRAD_CACHE.get(img_id)
    if g is None:
        from PIL import Image

        a = np.asarray(
            Image.open(WORK / "pool" / "images" / f"{img_id}.jpg").convert("L").resize((128, 128), Image.Resampling.BOX),
            dtype=np.float32,
        )
        g = np.hypot(np.diff(a, axis=1)[:-1, :], np.diff(a, axis=0)[:, :-1])
        if len(_GRAD_CACHE) > 20000:
            _GRAD_CACHE.clear()
        _GRAD_CACHE[img_id] = g
    return g


def grad_dist(id_a: str, id_b: str, rot_b: int) -> float:
    """Jarak gradien ternormalisasi; rotasi diterapkan pada citra 128x128 sebelum gradien."""
    from PIL import Image

    a = grad_map(id_a)
    if rot_b == 0:
        b = grad_map(id_b)
    else:
        im = np.asarray(
            Image.open(WORK / "pool" / "images" / f"{id_b}.jpg").convert("L").resize((128, 128), Image.Resampling.BOX),
            dtype=np.float32,
        )
        im = np.rot90(im, rot_b)
        b = np.hypot(np.diff(im, axis=1)[:-1, :], np.diff(im, axis=0)[:, :-1])
    return float(np.abs(a - b).sum() / (a.sum() + b.sum() + 1e-6))


def stratified_group_split(groups: dict[int, list[int]], rows: list[dict], rng: random.Random) -> dict[int, str]:
    cats = CLASSES + ["neg"]
    cidx = {c: i for i, c in enumerate(cats)}
    vec, forced = {}, {}
    for g, members in groups.items():
        v = np.zeros(len(cats))
        for m in members:
            r = rows[m]
            if r["role"] == "neg":
                v[cidx["neg"]] += 1
            for c in set(r["classes"].split()):
                v[cidx[CLASSES[int(c)]]] += 1
        vec[g] = v
        forced[g] = any(int(rows[m]["augmented"]) for m in members)
    total = sum(vec.values())
    target = FRACS[:, None] * np.maximum(total[None, :], 1)
    cur = np.zeros((3, len(cats)))
    assign = {}
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: (not forced[g], -len(groups[g])))
    for g in order:
        v = vec[g]
        if forced[g]:
            s = 0
        else:
            w = v / max(v.sum(), 1)
            fill = [float((w * (cur[k] + v) / target[k]).sum()) for k in range(3)]
            s = int(np.argmin(fill))
        assign[g] = SPLITS[s]
        cur[s] += v
    return assign


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--balance", action="store_true", help="tulis train_balanced.txt (oversampling kelas minoritas)")
    ap.add_argument("--copy", action="store_true", help="salin berkas (default: hardlink jika bisa)")
    args = ap.parse_args()
    t0 = time.time()
    rng = random.Random(SEED)
    pool = WORK / "pool"
    rows = list(csv.DictReader(open(pool / "manifest.csv", encoding="utf-8")))
    hz = np.load(pool / "hashes.npz")
    idx_of = {k: i for i, k in enumerate(hz["ids"].tolist())}

    # autolabel: status diperbarui oleh autolabel.py di manifest_autolabel.csv
    al_path = pool / "manifest_autolabel.csv"
    if al_path.exists():
        al = {r["id"]: r for r in csv.DictReader(open(al_path, encoding="utf-8"))}
        for r in rows:
            if r["id"] in al:
                r.update({k: al[r["id"]][k] for k in ("status", "classes", "n_boxes")})
    rejects = set()
    rej_file = CONFIGS / "reject_ids.txt"
    if rej_file.exists():
        for line in rej_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#")[0].strip()
            if line:
                rejects.add(line)

    report: dict = {"input": len(rows), "drop": Counter()}
    kept = []
    for r in rows:
        if r["status"] != "ok":
            report["drop"][f"status_{r['status']}"] += 1
        elif r["id"] in rejects:
            report["drop"]["manual_reject"] += 1
        elif r["id"] not in idx_of:
            report["drop"]["no_hash"] += 1
        else:
            kept.append(r)
    rows = kept
    sel = np.array([idx_of[r["id"]] for r in rows])
    phash, thumbs = hz["phash"][sel], hz["thumbs"][sel]
    n = len(rows)
    print(f"{n} citra masuk dedup ({time.time() - t0:.0f} dtk)")

    pairs = find_pairs(phash, thumbs)
    print(f"{len(pairs)} pasangan kandidat ({time.time() - t0:.0f} dtk)")
    dup_uf, grp_uf = UF(n), UF(n)
    n_dup_pairs = 0
    for i, j, d, c, k in pairs:
        if d <= T_DUP and c <= C_DUP and grad_dist(rows[i]["id"], rows[j]["id"], k) <= G_DUP:
            dup_uf.union(i, j)
            n_dup_pairs += 1
        if d <= T_GROUP and c <= C_GROUP:
            grp_uf.union(i, j)
    by_origin = defaultdict(list)
    for i, r in enumerate(rows):
        by_origin[r["origin_key"]].append(i)
    for members in by_origin.values():
        for m in members[1:]:
            grp_uf.union(members[0], m)

    # selesaikan klaster duplikat
    clusters = defaultdict(list)
    for i in range(n):
        clusters[dup_uf.find(i)].append(i)
    removed = set()
    conflicts = []
    for members in clusters.values():
        if len(members) == 1:
            continue
        pos = [m for m in members if rows[m]["role"] == "pos"]
        neg = [m for m in members if rows[m]["role"] == "neg"]
        if pos and neg:
            removed.update(neg)
            report["drop"]["dup_neg_of_pos"] += len(neg)
        if pos:
            label_sets = {rows[m]["classes"] for m in pos}
            if len(label_sets) > 1:
                removed.update(pos)
                report["drop"]["label_conflict"] += len(pos)
                conflicts.append([rows[m]["id"] for m in pos])
                continue
            pos.sort(key=lambda m: (int(rows[m]["augmented"]), int(rows[m]["autolabel"]),
                                    -int(rows[m]["w"]) * int(rows[m]["h"]), rows[m]["id"]))
            removed.update(pos[1:])
            report["drop"]["duplicate"] += len(pos) - 1
        elif len(neg) > 1:
            neg.sort(key=lambda m: rows[m]["id"])
            removed.update(neg[1:])
            report["drop"]["duplicate"] += len(neg) - 1

    alive = [i for i in range(n) if i not in removed]
    # kuota citra negatif ~NEG_FRAC dari total
    n_pos = sum(1 for i in alive if rows[i]["role"] == "pos")
    negs = [i for i in alive if rows[i]["role"] == "neg"]
    target_neg = round(n_pos * NEG_FRAC / (1 - NEG_FRAC))
    if len(negs) > target_neg:
        by_kind = defaultdict(list)
        for i in negs:
            by_kind[rows[i]["kind"]].append(i)
        keep_neg = []
        for kind, lst in sorted(by_kind.items()):
            rng.shuffle(lst)
            keep_neg += lst[: max(1, round(len(lst) * target_neg / len(negs)))]
        drop_neg = set(negs) - set(keep_neg)
        report["drop"]["neg_quota"] += len(drop_neg)
        alive = [i for i in alive if i not in drop_neg]

    groups = defaultdict(list)
    for i in alive:
        groups[grp_uf.find(i)].append(i)
    assign = stratified_group_split(groups, rows, rng)

    # tulis dataset final
    if FINAL.exists():
        shutil.rmtree(FINAL)
    for s in SPLITS:
        (FINAL / "images" / s).mkdir(parents=True)
        (FINAL / "labels" / s).mkdir(parents=True)
    out_rows = []
    for g, members in groups.items():
        s = assign[g]
        for i in members:
            r = dict(rows[i])
            r["split"], r["group"] = s, f"g{g}"
            src_img, src_lab = pool / "images" / f"{r['id']}.jpg", pool / "labels" / f"{r['id']}.txt"
            dst_img, dst_lab = FINAL / "images" / s / f"{r['id']}.jpg", FINAL / "labels" / s / f"{r['id']}.txt"
            if args.copy:
                shutil.copy2(src_img, dst_img)
            else:
                try:
                    dst_img.hardlink_to(src_img)
                except OSError:
                    shutil.copy2(src_img, dst_img)
            shutil.copy2(src_lab, dst_lab)
            out_rows.append(r)
    out_rows.sort(key=lambda r: (r["split"], r["id"]))
    fields = list(out_rows[0].keys())
    with open(WORK / "manifest_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # statistik per split
    split_stats = {}
    for s in SPLITS:
        rs = [r for r in out_rows if r["split"] == s]
        inst = Counter(CLASSES[int(c)] for r in rs for c in r["classes"].split())
        imgs = Counter(CLASSES[int(c)] for r in rs for c in set(r["classes"].split()))
        split_stats[s] = {
            "images": len(rs), "negatives": sum(r["role"] == "neg" for r in rs),
            "instances": {c: inst.get(c, 0) for c in CLASSES}, "images_with_class": {c: imgs.get(c, 0) for c in CLASSES},
            "augmented": sum(int(r["augmented"]) for r in rs), "autolabel": sum(int(r["autolabel"]) for r in rs),
        }
    train_file = "images/train"
    if args.balance:
        tr = [r for r in out_rows if r["split"] == "train"]
        cnt = split_stats["train"]["images_with_class"]
        med = float(np.median(list(cnt.values())))
        lines = []
        for r in tr:
            cl = [CLASSES[int(c)] for c in set(r["classes"].split())]
            rep = 1 if not cl else min(3, max(1, math.ceil(med / max(1, min(cnt[c] for c in cl)))))
            lines += [f"./images/train/{r['id']}.jpg"] * rep
        (FINAL / "train_balanced.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        train_file = "train_balanced.txt"
        report["balanced_train_entries"] = len(lines)
    names = "\n".join(f"  {i}: '{c}'" for i, c in enumerate(CLASSES))
    (FINAL / "data.yaml").write_text(
        f"# Dataset SuaraRupiah (dibangkitkan dedup_split.py)\ntrain: {train_file}\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames:\n{names}\n",
        encoding="utf-8",
    )
    report.update({
        "after_dedup": len(alive), "groups": len(groups), "pairs": len(pairs), "dup_pairs": n_dup_pairs, "conflict_clusters": conflicts[:200],
        "n_conflict_clusters": len(conflicts), "splits": split_stats, "thresholds": {
            "T_DUP": T_DUP, "C_DUP": C_DUP, "G_DUP": G_DUP, "T_GROUP": T_GROUP, "C_GROUP": C_GROUP, "NEG_FRAC": NEG_FRAC},
        "seconds": round(time.time() - t0),
    })
    (WORK / "dedup_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("conflict_clusters",)}, indent=1))


if __name__ == "__main__":
    main()
