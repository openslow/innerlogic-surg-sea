"""环境预检。不挂载外部存储，不启动训练。"""
import shutil
import socket
from pathlib import Path


def environment_report(path="."):
    target = Path(path)
    probe = target if target.exists() else (target.parent if target.parent.exists() else Path("/"))
    usage = shutil.disk_usage(probe)
    cuda = False
    bf16 = False
    gpu_name = None
    torch_version = None
    try:
        import torch
        torch_version = torch.__version__
        cuda = bool(torch.cuda.is_available())
        if cuda:
            gpu_name = torch.cuda.get_device_name(0)
            bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    except Exception:
        pass
    return {
        "hostname": socket.gethostname(),
        "path": str(target),
        "path_exists": target.exists(),
        "disk_free_bytes": int(usage.free),
        "cuda": cuda,
        "bf16": bf16,
        "gpu_name": gpu_name,
        "torch": torch_version,
    }
