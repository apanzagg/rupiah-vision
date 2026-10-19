# Kernel Kaggle "suararupiah-train": bangun split final dari pool (output kernel suararupiah-data),
# latih YOLOv8n sesuai configs/train_job.yaml, evaluasi, lalu ekspor ONNX FP32/FP16/INT8.
# Output (/kaggle/working): runs/<run>/ (best.pt, results.csv, grafik, metrics.json),
#                           export/<run>/ (onnx + variants.json + parity/), data_info/, test.tar, summary.json
import glob
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import yaml

T0 = time.time()
W = Path("/kaggle/working")


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}   [{time.time() - T0:.0f} dtk]", flush=True)
    subprocess.run(cmd, shell=True, check=True)


os.environ.update(SR_WORK="/tmp/work", SR_FINAL="/tmp/final", SR_REPORTS="/tmp/work/reports",
                  SR_CONFIGS="/tmp/src/configs", PYTHONUNBUFFERED="1")
job = yaml.safe_load(open("/tmp/src/configs/train_job.yaml", encoding="utf-8"))
print(json.dumps(job, indent=1))
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh("pip install -q ultralytics==8.3.253 albumentations==2.0.8 onnx onnxslim onnxruntime")

S = "/tmp/src/scripts"
Path("/tmp/work").mkdir(parents=True, exist_ok=True)
final_tar = sorted(glob.glob("/kaggle/input/**/final.tar", recursive=True))
if final_tar:  # dataset final dari kernel suararupiah-build (dedup sudah dilakukan di CPU)
    print("final.tar:", final_tar[0])
    with tarfile.open(final_tar[0]) as t:
        t.extractall("/tmp", filter="data")
    for f in ("dedup_report.json", "manifest_final.csv"):
        shutil.copy2(f"/tmp/final/{f}", f"/tmp/work/{f}")
else:  # cadangan: bangun dari pool
    pool_tar = sorted(glob.glob("/kaggle/input/**/pool.tar", recursive=True))
    with tarfile.open(pool_tar[0]) as t:
        t.extractall("/tmp/work", filter="data")
    sh(f"cd {S} && python dedup_split.py --balance")
(W / "data_info").mkdir(parents=True, exist_ok=True)
for f in ("dedup_report.json", "manifest_final.csv"):
    shutil.copy2(f"/tmp/work/{f}", W / "data_info" / f)
shutil.copy2("/tmp/final/data.yaml", W / "data_info" / "data.yaml")
sh(f"cd {S} && python -c \"from ultralytics import YOLO; YOLO('yolov8n.pt')\"")

import torch  # noqa: E402

ngpu = max(1, torch.cuda.device_count())
runs = job["runs"]
procs = []
for i, r in enumerate(runs):
    dev = i % ngpu if job.get("parallel") else 0
    workers = 3 if job.get("parallel") and len(runs) > 1 else 4
    cmd = (f"cd {S} && python train.py --data /tmp/final/data.yaml --imgsz {r['imgsz']} --epochs {r['epochs']} "
           f"--patience {r['patience']} --batch {r['batch']} --name {r['name']} --device {dev} --workers {workers} "
           f"--project {W / 'runs'} --max-hours {job.get('max_hours', 0)} "
           + " ".join(f"--{k} {v}" for k, v in r.get("args", {}).items())
           + (" --set " + " ".join(f"{k}={v}" for k, v in r["set"].items()) if r.get("set") else ""))
    (W / "runs").mkdir(parents=True, exist_ok=True)
    log = open(W / "runs" / f"{r['name']}.log", "w")
    print(f"\n$ {cmd}", flush=True)
    p = subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT)
    if job.get("parallel"):
        procs.append((r["name"], p))
    else:
        print(r["name"], "selesai, kode", p.wait(), flush=True)
for name, p in procs:
    print(name, "selesai, kode", p.wait(), f"[{time.time() - T0:.0f} dtk]", flush=True)

summary = {"job": job, "runs": {}}
for r in runs:
    run_dir = W / "runs" / r["name"]
    m = run_dir / "metrics.json"
    summary["runs"][r["name"]] = json.loads(m.read_text()) if m.exists() else {"error": "metrics.json tidak ada"}
    best = run_dir / "weights" / "best.pt"
    if job.get("export") and best.exists():
        try:
            sh(f"cd {S} && python export_onnx.py --weights {best} --data /tmp/final/data.yaml --imgsz {r['imgsz']} "
               f"--out {W / 'export' / r['name']} --version {job.get('version', 'v1')}")
            summary["runs"][r["name"]]["export"] = json.loads((W / "export" / r["name"] / "variants.json").read_text())
        except subprocess.CalledProcessError as e:
            summary["runs"][r["name"]]["export_error"] = str(e)
    (W / "summary.json").write_text(json.dumps(summary, indent=1))

if job.get("save_test"):
    with tarfile.open(W / "test.tar", "w") as t:
        for sub in ("images/test", "labels/test"):
            t.add(f"/tmp/final/{sub}", arcname=f"final/{sub}")
        t.add("/tmp/final/data.yaml", arcname="final/data.yaml")
        t.add("/tmp/work/manifest_final.csv", arcname="final/manifest_final.csv")
# bobot last.pt tidak diperlukan (best.pt cukup) -> hemat ukuran output
for p in (W / "runs").glob("*/weights/last.pt"):
    p.unlink()
sh("du -sh /kaggle/working/* || true")
print(json.dumps(summary, indent=1)[:4000])
print(f"SELESAI dalam {time.time() - T0:.0f} dtk")
