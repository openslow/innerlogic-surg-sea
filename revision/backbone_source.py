"""官方 surgfm_probe 的来源。本仓库不携带这份实现。"""
import importlib.util
from pathlib import Path

from revision.identity import RevisionError

# 2026-11-25 从 GitHub 核对：tag v0.1.0 指向这一次提交。
# 包名是 pyproject 里的 surgfm-probe，导入名是 surgfm_probe。
SURGFM_GIT = "https://github.com/semaphor-surgical/Innerlogic-Surg.git"
SURGFM_TAG = "v0.1.0"
SURGFM_COMMIT = "99a3e12489c11edc681cc8cc2ce4e4e481c21e4d"
SURGFM_VERSION = "0.1.0"
INSTALL_COMMANDS = (
    "git clone --branch v0.1.0 --depth 1 https://github.com/semaphor-surgical/Innerlogic-Surg",
    "python -m pip install -e Innerlogic-Surg",
)
PIP_SPEC = (
    "surgfm-probe @ git+https://github.com/semaphor-surgical/Innerlogic-Surg.git"
    "@99a3e12489c11edc681cc8cc2ce4e4e481c21e4d"
)

# load_backbone(checkpoint, arch=None, device="cpu") 读扁平 DINOv3 state_dict，
# 用 cls_token 和 patch_embed.proj.weight 推断结构。训练 checkpoint 的 head_state
# 包一层，不能直接交给它。上游 README 写 fp16 会在这个骨干里溢出，推荐 bfloat16。


def probe_status():
    """没有安装时返回 BLOCKED。装在本仓库里面则拒绝，避免把自造模块当成官方骨干。"""
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.find_spec("surgfm_probe")
    origin = None if spec is None else spec.origin
    if origin:
        origin_path = Path(origin).resolve()
        if repo == origin_path or repo in origin_path.parents:
            raise RevisionError(
                "fake_backbone",
                "surgfm_probe 出现在本仓库里。公平入口不接受自造的同名实现。",
                {"origin": str(origin_path)},
            )
    if spec is None or not origin:
        return {
            "available": False,
            "blocked": True,
            "code": "BLOCKED",
            "missing": "surgfm_probe（Innerlogic-Surg v0.1.0，不是本仓库的文件）",
            "expected_version": SURGFM_VERSION,
            "expected_commit": SURGFM_COMMIT,
            "install": list(INSTALL_COMMANDS),
            "pip_spec": PIP_SPEC,
            "vendored": False,
        }
    module = importlib.import_module("surgfm_probe")
    return {
        "available": callable(getattr(module, "load_backbone", None)),
        "blocked": not callable(getattr(module, "load_backbone", None)),
        "code": "ready" if callable(getattr(module, "load_backbone", None)) else "BLOCKED",
        "version": getattr(module, "__version__", None),
        "expected_version": SURGFM_VERSION,
        "expected_commit": SURGFM_COMMIT,
        "origin": str(Path(origin).resolve()),
        "vendored": False,
        "install": list(INSTALL_COMMANDS),
        "pip_spec": PIP_SPEC,
    }


def require_official_loader():
    status = probe_status()
    if status["blocked"] or not status["available"]:
        raise RevisionError(
            "backbone_loader_missing",
            "BLOCKED：缺少官方 surgfm_probe。它来自 Innerlogic-Surg "
            f"{SURGFM_TAG}（{SURGFM_COMMIT}），本仓库没有这份源码，也不提供同名替代。",
            status,
        )
    return status
