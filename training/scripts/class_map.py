"""Pemetaan nama kelas asli (beragam per dataset) ke 7 kelas baku.

Hasil map_class():
- "1000" ... "100000"  -> kelas baku
- DROP_BOX             -> objek di luar lingkup (koin, Rp75.000, kelas 'negative'): box dibuang, citra tetap dipakai
- DROP_IMAGE           -> kelas generik tanpa nominal (mis. "money", "rupiah"): seluruh citra dibuang,
                          karena uang di dalamnya tidak bisa diberi label nominal yang benar
"""
from __future__ import annotations

import re

from common import CLASSES

DROP_BOX = "DROP_BOX"
DROP_IMAGE = "DROP_IMAGE"

# Override per sumber, untuk nama yang ambigu. Kunci: slug sumber -> {nama asli: hasil}
OVERRIDES: dict[str, dict[str, str]] = {
    # Dataset ini punya kelas uang kertas "1rb" dan kelas koin "1000" & "500"
    "coin-new/uang-rupiah-kertas-dan-koin-by-adelia-clarissa-zzt2j": {"1000": DROP_BOX, "500": DROP_BOX},
}

WORDS = [
    # urutan penting: frasa "puluh" dicek sebelum "ribu" tunggal
    (r"seratus\s*ribu", 100000),
    (r"lima\s*puluh\s*ribu", 50000),
    (r"dua\s*puluh\s*ribu", 20000),
    (r"sepuluh\s*ribu", 10000),
    (r"lima\s*ribu", 5000),
    (r"dua\s*ribu", 2000),
    (r"seribu", 1000),
]
GENERIC = {
    "money", "uang", "cash", "objects", "object", "rupiah", "uangrupiah", "notes", "banknote", "currency",
    "uangrupiahemisi2022", "idr", "rp", "0", "hiupng", "uangku", "duit",
}


def _norm(name: str) -> str:
    s = re.sub(r"[-_/]+", " ", name.lower().strip())
    s = re.sub(r"\b(2016|2022)\b", " ", s)  # tahun emisi, bukan nominal
    s = s.replace("indonesianrupiah", " ")
    return s


def map_class(raw: str, source: str = "") -> str:
    ov = OVERRIDES.get(source, {})
    if raw in ov:
        return ov[raw]
    s = _norm(raw)
    if re.search(r"koin|coin|perak|logam", s):
        return DROP_BOX
    if re.search(r"tujuh\s*puluh\s*lima|\b75\s*(k|rb|ribu|\.?000)\b|75000|75ribu|75rb|75k", s):
        return DROP_BOX
    if re.search(r"negative|negatif|background|palsu|fake|mainan", s):
        return DROP_BOX
    for pat, val in WORDS:
        if re.search(pat, s):
            return str(val)
    compact = re.sub(r"[^a-z0-9.]", "", s)
    m = re.search(r"(\d+(?:\.\d{3})*)(k|rb|ribu)?", compact)
    if m:
        num = int(m.group(1).replace(".", ""))
        if m.group(2):
            num *= 1000
        if str(num) in CLASSES:
            return str(num)
        if num in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
            return DROP_BOX  # pecahan koin / sen
        return DROP_IMAGE
    if re.sub(r"[^a-z0-9]", "", s) in GENERIC:
        return DROP_IMAGE
    return DROP_IMAGE


if __name__ == "__main__":
    # Cetak tabel pemetaan semua nama kelas dari sumber terpilih, untuk diperiksa manual.
    import json
    import sys

    import yaml

    from common import DATA, ROOT

    cfg = yaml.safe_load((ROOT / "training/configs/sources.yaml").read_text(encoding="utf-8"))
    survey = {}
    for f in ("rf_survey.json", "rf_survey_neg.json"):
        p = DATA / "interim" / f
        if p.exists():
            survey.update(json.loads(p.read_text(encoding="utf-8")))
    rows = []
    for s in cfg["roboflow"]:
        if s["role"] != "positive":
            continue
        classes = survey.get(s["slug"], {}).get("project", {}).get("classes", {})
        for name, cnt in classes.items():
            rows.append((map_class(name, s["slug"]), name, cnt, s["slug"]))
    rows.sort()
    out = sys.stdout
    for r in rows:
        out.write(f"{r[0]:>10} <- {r[1]!r:40} n={r[2]:<5} {r[3]}\n")
