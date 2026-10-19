# Kernel Kaggle "suararupiah-data": bangun dataset dari nol.
# unduh Roboflow (link ekspor) -> ingest -> auto-label (YOLO-World, GPU) -> dedup & split -> analisis.
# Output (/kaggle/working): pool.tar, conditions_raw.tar, reports.zip, final.tar
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


os.environ.update(
    SR_RAW="/tmp/raw", SR_KAGGLE="/kaggle/input", SR_WORK="/tmp/work", SR_FINAL="/tmp/final",
    SR_REPORTS="/tmp/work/reports", SR_CONFIGS="/tmp/src/configs", PYTHONUNBUFFERED="1",
)
sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true")
sh("find /kaggle/input -maxdepth 3 -type d | head -30")
sh("pip install -q ultralytics==8.3.253")
S = "cd /tmp/src/scripts && python"
sh(f"{S} fetch_rf_links.py /tmp/src/rf_links.json")
sh(f"{S} ingest.py")
sh(f"{S} autolabel.py")
sh(f"{S} dedup_split.py --balance")
sh(f"{S} analyze.py")

out = Path("/kaggle/working")
for name, path in [("pool.tar", "/tmp/work/pool"), ("conditions_raw.tar", "/tmp/work/conditions_raw"), ("final.tar", "/tmp/final")]:
    if Path(path).exists():
        with tarfile.open(out / name, "w") as t:
            t.add(path, arcname=Path(path).name)
with zipfile.ZipFile(out / "reports.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in Path("/tmp/work").rglob("*"):
        if p.is_file() and ("reports" in p.parts or p.suffix in {".json", ".csv"}) and "pool" not in p.parts:
            z.write(p, p.relative_to("/tmp/work").as_posix())
    for p in Path("/tmp/work/pool").glob("manifest*.csv"):
        z.write(p, f"pool/{p.name}")
sh("ls -la /kaggle/working && du -sh /tmp/work/pool /tmp/final")
print(f"SELESAI dalam {time.time() - T0:.0f} dtk")
