# Uji kemampuan akun Kaggle: GPU, internet, dan kecepatan unduh.
import os, shutil, socket, subprocess, time, urllib.request
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
print("proxy env:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()})
print(open("/etc/resolv.conf").read())
for host in ["pypi.org", "github.com", "www.google.com", "app.roboflow.com", "storage.googleapis.com", "speed.cloudflare.com"]:
    try:
        print(host, "->", socket.gethostbyname(host))
    except Exception as e:
        print(host, "DNS GAGAL:", e)
for url in ["https://pypi.org/simple/ultralytics/", "https://speed.cloudflare.com/__down?bytes=50000000"]:
    t = time.time()
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            n = len(r.read())
        print(f"OK {url[:40]}: {n/1e6:.1f} MB dalam {time.time()-t:.1f} dtk")
    except Exception as e:
        print("GAGAL", url[:40], e)
