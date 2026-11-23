import importlib.machinery, importlib.util, urllib.request
# fetch loader.pyc from GitHub (fast on Colab) and try importing it
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/HyunD-init/spineendo-yolov11-seg/main/weights/loader.pyc",
    "/tmp/loader.pyc")
try:
    loader = importlib.machinery.SourcelessFileLoader("weights.loader", "/tmp/loader.pyc")
    spec = importlib.util.spec_from_loader("weights.loader", loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    print("LOADER IMPORTED OK")
except ImportError as e:
    print("IMPORT ERROR (magic/version?):", e)
except ModuleNotFoundError as e:
    print("MAGIC ACCEPTED, missing dep:", e)
except Exception as e:
    print("OTHER:", type(e).__name__, e)
