"""公开训练脚本里的 checkpoint 形状。不读取私有权重。"""
from revision.identity import RevisionError

# modal_stage_e.py train_group 保存：
# epoch, best_val_dice, bad_epochs, history, head_state, opt_state，
# 微调才另有 backbone_state。head 是 Conv2d(768, 7, 1)。
# modal_stage_e_cont2.py 的 ep10 文件只保存 epoch, best_val_dice, history,
# head_state, backbone_state，没有 opt_state。推理脚本只 load 这两个 state。
# load_backbone 只接受带 cls_token / patch_embed.proj.weight 的扁平 state_dict。


def classify_checkpoint(blob):
    """区分预训练权重、公平训练断点，和只能做推理重放的历史 checkpoint。"""
    if not isinstance(blob, dict):
        raise RevisionError("unrecognized_checkpoint", "checkpoint 不是字典", {})
    if "head_state" not in blob and ("cls_token" in blob or "patch_embed.proj.weight" in blob):
        return {
            "schema": "dinov3_pretrain",
            "use": "new_training_backbone",
            "ckpt_kind": None,
            "new_training_loads_this_file": True,
        }
    if "head_state" not in blob:
        raise RevisionError(
            "unrecognized_checkpoint",
            "既不是 DINOv3 预训练 state_dict，也没有 head_state。",
            {"keys": list(blob)[:8]},
        )
    has_backbone = isinstance(blob.get("backbone_state"), dict)
    fair = all(key in blob for key in ("optimizer_step", "sched_state", "rng_state"))
    epoch = blob.get("epoch")
    continued = bool(blob.get("continued_from_ep10")) or (epoch is not None and int(epoch) >= 10)
    use = "fair_resume" if fair and not continued else "inference_replay"
    return {
        "schema": "stage_e_training_ckpt",
        "ckpt_kind": "full_finetune" if has_backbone else "control_head",
        "use": use,
        "has_opt_state": "opt_state" in blob,
        "has_backbone_state": has_backbone,
        "has_scaler_state": "scaler_state" in blob,
        "inference_loads": ["head_state"] + (["backbone_state"] if has_backbone else []),
        "new_training_loads_this_file": False,
        "note_zh": (
            "ep10 续训的公开保存没有 opt_state。推理重放加载 head_state，"
            "有 backbone_state 时再加载它。新训练调用 load_backbone 读扁平预训练权重，"
            "不从这份训练 checkpoint 往下续。"
        ),
    }
