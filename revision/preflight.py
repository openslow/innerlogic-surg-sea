"""环境预检。不挂载外部存储，不启动训练。"""
import os
import shutil
import socket
from pathlib import Path

from revision.identity import RevisionError


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
        "fp16": cuda,
        "gpu_name": gpu_name,
        "torch": torch_version,
    }


def assert_output_ready(path, min_free_bytes):
    """真实写出之前检查目录可写、剩余空间够。不够就停，不创建半套结果。"""
    target = Path(path)
    parent = target if target.exists() else target.parent
    if not parent.exists() or not os.access(parent, os.W_OK):
        raise RevisionError(
            "not_writable",
            f"输出路径不可写：{target}",
            {"path": str(target)},
        )
    free = shutil.disk_usage(parent).free
    if free < int(min_free_bytes):
        raise RevisionError(
            "disk_insufficient",
            f"剩余 {free} 字节，需要至少 {int(min_free_bytes)} 字节。",
            {"path": str(target), "free": int(free), "required": int(min_free_bytes)},
        )
    return {"path": str(target), "free": int(free)}
