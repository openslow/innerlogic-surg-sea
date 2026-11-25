"""四组公平对比。几何备选不进入训练矩阵。"""
from revision.identity import RevisionError

_BACKBONES = ("dinov3_vitb16_lvd1689m", "innerlogic_surg_vitb16")
_MODES = ("frozen_probe", "last4_finetune")


def _seed_list(phase, seed):
    if phase == "single":
        return [int(seed)]
    if phase == "triple":
        return [int(seed), int(seed) + 1, int(seed) + 2]
    raise RevisionError("seed_phase", "phase 只能是 single 或 triple", {"phase": phase})


def training_matrix(phase="single", seed=7):
    """先单 seed（4 组）。phase='triple' 才展开成 12 组，不另加几何轴。"""
    runs = []
    for backbone_id in _BACKBONES:
        for mode in _MODES:
            for one_seed in _seed_list(phase, seed):
                runs.append({
                    "run_id": f"{backbone_id}__{mode}__s{one_seed}",
                    "backbone_id": backbone_id,
                    "architecture": {
                        "family": "dinov3_vit",
                        "variant": "vit_b16",
                        "embed_dim": 768,
                        "depth": 12,
                        "patch_size": 16,
                        "params_m": 86,
                    },
                    "train_mode": mode,
                    "trainable_blocks": [] if mode == "frozen_probe" else [8, 9, 10, 11],
                    "zero_shot": False,
                    "protocol_name": (
                        "frozen_backbone_supervised_head"
                        if mode == "frozen_probe"
                        else "last4_block_finetune_supervised_head"
                    ),
                    "geometry": "square768",
                    "split": "manifests_paper",
                    "split_seed": 7,
                    "head": "conv1x1",
                    "head_init": "random",
                    "classes": 7,
                    "epochs": 6,
                    "early_stop": False,
                    "effective_batch": 8,
                    "microbatch": 1,
                    "grad_accum": 8,
                    "lr_backbone": None if mode == "frozen_probe" else 1e-5,
                    "lr_head": 1e-4,
                    "precision_policy": "auto",
                    "selection_split": "val",
                    # Stage E val_dice：帧等权 binary foreground。semantic 只记录，不选点。
                    "selection_score": "binary_foreground_dice_frame_equal",
                    "selection_protocol": "stage_e_binary_foreground_frame_equal_val",
                    "reported_but_not_used_for_selection": [
                        "gt_class_dice", "class_aware_macro_dice", "naming_accuracy",
                    ],
                    "resume_from_ep10": False,
                    "seed": one_seed,
                    "seeds": [one_seed],
                    "allow_backbone_substitution": False,
                })
    return runs


def prepared_geometries():
    """stretch / letterbox / 原生裁剪只作记录。训练矩阵只用 square768。"""
    return {
        "square768": {
            "in_training_matrix": True,
            "output_hw": [768, 768],
            "mode": "center_square_crop_then_resize",
        },
        "stretch_768x432": {
            "in_training_matrix": False,
            "output_hw": [432, 768],
            "mode": "direct_resize",
            "historical": "Stage B/C/D",
        },
        "letterbox_768": {
            "in_training_matrix": False,
            "output_hw": [768, 768],
            "mode": "letterbox_pad",
        },
        "center_crop_native": {
            "in_training_matrix": False,
            "output_hw": None,
            "mode": "center_square_crop_no_resize",
        },
    }


def select_epoch(val_scores):
    """只看 val。分数高者优先，同分取更早的 epoch。"""
    if not val_scores:
        raise RevisionError("empty_val", "没有 val 分数，不能选 checkpoint", {})
    return min(val_scores, key=lambda epoch: (-float(val_scores[epoch]), epoch))


def choose_precision(device, cuda, bf16, requested="auto", fp16=None):
    """按设备和请求选精度。不改 epoch、有效 batch 或几何。

    auto：有 bf16 用 bf16；CUDA 但没有 bf16（T4）用 fp16 加 GradScaler；否则 fp32。
    点名 bf16 而卡不支持时直接失败，不偷偷改成别的精度。
    """
    if device not in ("cpu", "cuda"):
        raise RevisionError("bad_device", "device 只能是 cpu 或 cuda", {"device": device})
    if requested not in ("auto", "bf16", "fp16", "fp32"):
        raise RevisionError("bad_precision", "precision 只能是 auto、bf16、fp16 或 fp32", {"precision": requested})
    if fp16 is None:
        fp16 = bool(device == "cuda" and cuda)
    budget = {"epochs": 6, "effective_batch": 8, "geometry": "square768"}
    if device == "cpu":
        if requested not in ("auto", "fp32"):
            raise RevisionError(
                "precision_unavailable",
                "CPU 路径固定 fp32。fp16 或 bf16 只在 CUDA 上选择。",
                {"requested": requested, "budget": budget},
            )
        return {
            "autocast": "fp32",
            "grad_scaler": False,
            "precision_reason": "cpu",
            "budget": budget,
        }
    if requested == "bf16" or (requested == "auto" and bf16):
        if not bf16:
            raise RevisionError(
                "bf16_unavailable",
                "这张卡没有 bf16。T4 请用 precision=auto 或 fp16，不要点名 bf16。",
                {"cuda": cuda, "bf16": bf16, "budget": budget},
            )
        return {
            "autocast": "bf16",
            "grad_scaler": False,
            "precision_reason": "bf16_supported",
            "budget": budget,
        }
    if requested == "fp32" or (requested == "auto" and not fp16):
        return {
            "autocast": "fp32",
            "grad_scaler": False,
            "precision_reason": "explicit_fp32" if requested == "fp32" else "no_reduced_precision",
            "budget": budget,
        }
    if not fp16:
        raise RevisionError(
            "fp16_unavailable",
            "要求 fp16，但这张卡没有 fp16。",
            {"cuda": cuda, "fp16": fp16, "budget": budget},
        )
    return {
        "autocast": "fp16",
        "grad_scaler": True,
        "precision_reason": "no_bf16_use_fp16",
        "upstream_fp16_note": (
            "Innerlogic-Surg v0.1.0 的 README 写这个骨干 fp16 会溢出，并推荐 bfloat16。"
            "T4 没有 bf16，所以这里显式用 fp16 加 GradScaler。epoch、有效 batch 和几何不改。"
        ),
        "budget": budget,
    }


def runtime_plan(real, cuda, bf16, device="cpu", precision="auto", fp16=None):
    """device 默认 cpu。只有调用方写明 cuda 才会用 GPU。精度按卡选择，不强制 bf16。"""
    if device not in ("cpu", "cuda"):
        raise RevisionError("bad_device", "device 只能是 cpu 或 cuda", {"device": device})
    plan = {
        "microbatch": 1,
        "grad_accum": 8,
        "effective_batch": 8,
        "epochs": 6,
        "assumed_gpu": None,
        "device": device,
        "precision_requested": precision,
    }
    if device == "cuda":
        if not real:
            raise RevisionError(
                "gpu_requires_real_switch",
                "GPU 只有在显式真实开关打开后才能选用。",
                {"device": device},
            )
        if not cuda:
            raise RevisionError(
                "gpu_unavailable",
                "要求使用 CUDA，但这台机器没有可用 GPU。",
                {"cuda": cuda, "bf16": bf16},
            )
    chosen = choose_precision(device, cuda, bf16, precision, fp16=fp16)
    plan.update(chosen)
    plan["launch"] = "train" if (device == "cuda" or real) else "cpu_smoke"
    return plan
