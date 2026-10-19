# Kernel Kaggle "suararupiah-audit": audit label berbasis model (best.pt ronde audit) pada dataset final
# dari kernel suararupiah-build. Output: audit.zip (audit.csv, audit_summary.json, contact sheet salah-kelas).
import glob
import os
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path

T0 = time.time()


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}   [{time.time() - T0:.0f} dtk]", flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.environ.update(SR_WORK="/tmp/work", SR_FINAL="/tmp/final", SR_CONFIGS="/tmp/src/configs", PYTHONUNBUFFERED="1")
sh("pip install -q ultralytics==8.3.253")
with tarfile.open(sorted(glob.glob("/kaggle/input/**/final.tar", recursive=True))[0]) as t:
    t.extractall("/tmp", filter="data")
Path("/tmp/work").mkdir(parents=True, exist_ok=True)
shutil.copy2("/tmp/final/manifest_final.csv", "/tmp/work/manifest_final.csv")
best = sorted(glob.glob("/kaggle/input/**/e0_r320_audit_best.pt", recursive=True))[0]
print("bobot:", best)
out = Path("/kaggle/working/audit/e0_r320_audit")
sh(f"cd /tmp/src/scripts && python audit_labels.py --weights {best} --imgsz 320 --out {out} --device 0")
with zipfile.ZipFile("/kaggle/working/audit.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("/kaggle/working/audit").rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to("/kaggle/working").as_posix())
print(f"SELESAI dalam {time.time() - T0:.0f} dtk")
