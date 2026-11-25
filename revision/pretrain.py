"""两条 ViT-B/16 的来源。本模块不下载、不把权重读进模型。"""
import hashlib
from pathlib import Path

from revision.identity import RevisionError

# 与仓库里既有训练脚本的校验和、以及上游文件名 8 位后缀一致。
# 本次修订没有重新下载或重新哈希权重文件。
ILS_SHA256 = "a516a775805b9093f15e5e7f6fa023663a48d81ca211d6552b4fa87be03c8806"

_STEPS = {
    "innerlogic_surg_vitb16": (
        "获取步骤：从 https://github.com/semaphor-surgical/Innerlogic-Surg "
        "Release v0.1.0 取 dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth，"
        "或使用项目里已经核过的同一文件。期望 sha256 "
        f"{ILS_SHA256}。不要换成 ViT-L（22622135，embed 1024），"
        "也不要用通用 DINOv3 LVD 权重顶替。代码和权重受 DINOv3 License 约束。"
        "运行时把本地路径传进来；本修订不下载。"
    ),
    "dinov3_vitb16_lvd1689m": (
        "获取步骤：先接受 DINOv3 License（https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md "
        "与 https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m），"
        "再取 LVD-1689M 的 ViT-B/16 distilled（约 86M，patch 16），不要取 SAT-493M、ViT-L 或 7B。"
        "官方加载方式是本地 torch.hub dinov3_vitb16。本次没有核到可钉死的 sha256，"
        "所以在把校验和写进注册表之前一律拒绝加载。把 InnerLogic 文件"
        f"（sha256 {ILS_SHA256}）指到这个配置上会按偷换骨干拒绝。"
    ),
}


def decide_weight(backbone_id, sha256):
    """只根据骨干 id 和 sha256 决定。不读张量。"""
    sha = str(sha256).lower()
    if backbone_id == "dinov3_vitb16_lvd1689m" and sha == ILS_SHA256:
        raise RevisionError(
            "backbone_substitution",
            "这个文件是 InnerLogic-Surg ViT-B/16，不能当作通用 DINOv3 LVD 骨干。",
            {"weights_loaded": False, "steps_zh": _STEPS[backbone_id], "sha256": sha},
        )
    if backbone_id == "innerlogic_surg_vitb16":
        if sha == ILS_SHA256:
            return {
                "hash_accepted": True,
                "weights_loaded": False,
                "sha256": sha,
                "backbone_id": backbone_id,
            }
        raise RevisionError(
            "hash_mismatch",
            "InnerLogic-Surg ViT-B/16 的 sha256 与已钉死的 a516a775… 不一致。",
            {"weights_loaded": False, "steps_zh": _STEPS[backbone_id], "sha256": sha},
        )
    if backbone_id == "dinov3_vitb16_lvd1689m":
        raise RevisionError(
            "weight_unverified",
            "通用 DINOv3 ViT-B/16 的 sha256 尚未钉死，拒绝加载，也不改用 InnerLogic 权重。",
            {"weights_loaded": False, "steps_zh": _STEPS[backbone_id], "sha256": sha},
        )
    raise RevisionError(
        "unknown_backbone",
        f"未知骨干 {backbone_id}",
        {"weights_loaded": False},
    )


def resolve_backbone(backbone_id, path):
    steps = _STEPS.get(backbone_id, "未知骨干，没有获取步骤。")
    file_path = Path(path) if path else None
    if file_path is None or not file_path.is_file():
        raise RevisionError(
            "weight_missing",
            "本地没有可用的预训练权重，拒绝启动。",
            {"weights_loaded": False, "steps_zh": steps, "path": None if file_path is None else str(file_path)},
        )
    digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    return decide_weight(backbone_id, digest)
