# Kernel Kaggle "suararupiah-build" (CPU): bangun ulang split final dari pool (output suararupiah-data)
# dengan kode dedup terbaru + daftar reject manual, lalu analisis & diagnosis konflik label.
# Output: reports.zip (grafik, contact sheet, statistik). Tidak memakai kuota GPU.
import glob
import os
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

T0 = time.time()


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}   [{time.time() - T0:.0f} dtk]", flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.environ.update(SR_WORK="/tmp/work", SR_FINAL="/tmp/final", SR_REPORTS="/tmp/work/reports",
                  SR_CONFIGS="/tmp/src/configs", PYTHONUNBUFFERED="1")
Path("/tmp/work").mkdir(parents=True, exist_ok=True)
with tarfile.open(sorted(glob.glob("/kaggle/input/**/pool.tar", recursive=True))[0]) as t:
    t.extractall("/tmp/work", filter="data")
S = "cd /tmp/src/scripts && python"
sh(f"{S} dedup_split.py --balance")
sh(f"{S} diag_dedup.py")
sh(f"{S} analyze.py")
with zipfile.ZipFile("/kaggle/working/reports.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("/tmp/work").rglob("*"):
        if p.is_file() and ("reports" in p.parts or p.suffix in {".json", ".csv"}) and "pool" not in p.parts:
            z.write(p, p.relative_to("/tmp/work").as_posix())
print(f"SELESAI dalam {time.time() - T0:.0f} dtk")
