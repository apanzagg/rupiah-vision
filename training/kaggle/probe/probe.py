# Uji kemampuan akun Kaggle: GPU, internet, dan kecepatan unduh.
import shutil, subprocess, time, urllib.request
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
else:
    print("nvidia-smi TIDAK ADA")
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
t = time.time()
try:
    with urllib.request.urlopen("https://speed.cloudflare.com/__down?bytes=50000000", timeout=60) as r:
        n = len(r.read())
    print(f"internet OK: {n/1e6:.0f} MB dalam {time.time()-t:.1f} dtk")
except Exception as e:
    print("internet GAGAL:", e)
print(subprocess.run(["df", "-h", "/kaggle/working"], capture_output=True, text=True).stdout)
print("nproc", subprocess.run(["nproc"], capture_output=True, text=True).stdout)
print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
