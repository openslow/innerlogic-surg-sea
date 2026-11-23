import sys, time, urllib.request
print("python:", sys.version)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch import failed:", e)
try:
    import ultralytics
    print("ultralytics:", ultralytics.__version__)
except Exception as e:
    print("ultralytics not installed:", e)
# GitHub reachability + speed
t0 = time.time()
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/HyunD-init/spineendo-yolov11-seg/main/weights/best_n.pt.enc",
    "/tmp/best_n.pt.enc")
dt = time.time() - t0
import os
print(f"github download best_n (6MB): {dt:.1f}s = {6/dt:.1f} MB/s, size={os.path.getsize('/tmp/best_n.pt.enc')}")
