"""最小训练 / 断点 / 推理。合成骨干走同一条循环，不下载受限权重。"""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from revision.fair_config import runtime_plan, select_epoch, training_matrix
from revision.identity import RevisionError, create_run_dir, verify_identity
from revision.metrics import CLS_NAMES, score_mask
from revision.overlays import write_review
from revision.preflight import assert_output_ready
from revision.pretrain import _STEPS, resolve_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SELECTION_SCORE = "binary_foreground_dice_frame_equal"
CLS2ID = {name: i for i, name in enumerate(CLS_NAMES)}


class TinyDino(torch.nn.Module):
    """与 DINOv3 dense 接口相同的小网络：12 个 block、get_intermediate_layers。

    它不是 ViT-B/16 权重，只用来在 CPU 上跑通冻结、累积梯度和选点。
    """

    def __init__(self, embed=8, depth=12, patch=4):
        super().__init__()
        self.embed_dim = embed
        self.patch_size = patch
        self.depth = depth
        self.patch_embed = torch.nn.Conv2d(3, embed, kernel_size=patch, stride=patch, bias=False)
        torch.nn.init.normal_(self.patch_embed.weight, std=0.02)
        blocks = []
        for _ in range(depth):
            conv = torch.nn.Conv2d(embed, embed, 1, bias=False)
            with torch.no_grad():
                conv.weight.zero_()
                for i in range(embed):
                    conv.weight[i, i, 0, 0] = 1.0
            blocks.append(conv)
        self.blocks = torch.nn.ModuleList(blocks)

    def get_intermediate_layers(self, x, n=1, reshape=True, return_class_token=True):
        hidden = self.patch_embed(x)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        pooled = hidden.mean(dim=(2, 3))
        if return_class_token:
            return ((hidden, pooled),)
        return (hidden,)


def square_crop_box(width, height):
    """Stage E：side = min(W, H)，x0 = (W - side) // 2，y0 = 0。坐标在原图像素上。"""
    side = int(min(width, height))
    x0 = int((width - side) // 2)
    return {
        "x0": x0,
        "y0": 0,
        "side": side,
        "crop_box_original_xyxy": [x0, 0, x0 + side, side],
        "original_hw": [int(height), int(width)],
    }


def apply_train_mode(backbone, mode):
    backbone.requires_grad_(False)
    if mode == "frozen_probe":
        return []
    if mode == "last4_finetune":
        chosen = []
        for block in backbone.blocks[8:]:
            for param in block.parameters():
                param.requires_grad = True
                chosen.append(param)
        return chosen
    raise RevisionError("bad_train_mode", f"未知 train_mode {mode}", {"train_mode": mode})


def make_head(embed_dim, num_classes, seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    head = torch.nn.Conv2d(embed_dim, num_classes, 1)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.02)
        head.bias.copy_(torch.randn(head.bias.shape, generator=generator) * 0.02)
    return head


def forward_dense(backbone, images):
    ((features, _cls),) = backbone.get_intermediate_layers(
        images, n=1, reshape=True, return_class_token=True)
    return features


def _class_mask(mask, cls_name):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    out = np.zeros(mask.shape[:2], dtype=np.int16)
    cls_id = CLS2ID[cls_name]
    out[np.asarray(mask) > 0] = cls_id
    return out


def prepare_sample(row, input_size, patch, augment):
    if isinstance(row.get("image"), np.ndarray):
        image = np.asarray(row["image"])
        mask = _class_mask(row["mask"], row["cls"])
    else:
        root = Path(row["root"])
        image = np.asarray(Image.open(root / row["image"]).convert("RGB"))
        mask_path = root / row["mask"]
        if mask_path.suffix == ".nrrd":
            mask = _read_nrrd_mask(mask_path, row["cls"])
        else:
            mask = _class_mask(np.asarray(Image.open(mask_path)), row["cls"])
    height, width = image.shape[:2]
    box = square_crop_box(width, height)
    x0, y0, side = box["x0"], box["y0"], box["side"]
    image_c = image[y0:y0 + side, x0:x0 + side]
    mask_c = mask[y0:y0 + side, x0:x0 + side]
    if augment and random.random() < 0.5:
        image_c = np.ascontiguousarray(image_c[:, ::-1])
        mask_c = np.ascontiguousarray(mask_c[:, ::-1])
    image_in = np.array(
        Image.fromarray(image_c).resize((input_size, input_size), Image.BICUBIC),
        dtype=np.uint8, copy=True,
    )
    mask_in = np.array(
        Image.fromarray(mask_c.astype(np.uint8)).resize((input_size, input_size), Image.NEAREST),
        dtype=np.uint8, copy=True,
    ).astype(np.int16)
    tensor = torch.from_numpy(image_in).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = (tensor - mean) / std
    labels = _patch_labels(mask_in, patch)
    return {
        "tensor": tensor,
        "labels": torch.from_numpy(labels),
        "crop_mask": mask_c.astype(np.int16),
        "crop_image": image_c,
        "crop": box,
        "patient": row.get("patient"),
        "cls": row.get("cls"),
    }


def _read_nrrd_mask(path, cls_name):
    try:
        import nrrd
    except ImportError as exc:
        raise RevisionError(
            "dependency_missing",
            "读 nrrd 需要 pynrrd。合成测试不需要它。",
            {"package": "pynrrd"},
        ) from exc
    arr, _hdr = nrrd.read(str(path))
    mask = np.asarray(arr)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.T > 0
    return _class_mask(mask.astype(np.uint8) * 255, cls_name)


def _patch_labels(class_mask, patch):
    height, width = class_mask.shape
    if height % patch or width % patch:
        raise RevisionError(
            "patch_alignment",
            f"输入边长 {height}x{width} 不能被 patch {patch} 整除",
            {},
        )
    grid_h, grid_w = height // patch, width // patch
    foreground = (class_mask > 0).astype(np.float32)
    small = foreground.reshape(grid_h, patch, grid_w, patch).mean(axis=(1, 3))
    present = class_mask[class_mask > 0]
    cls_id = int(present[0]) if len(present) else 0
    return np.where(small >= 0.5, cls_id, 0).astype(np.int64)


def _loss(logits, labels):
    ce = F.cross_entropy(logits.float(), labels)
    probs = 1.0 - torch.softmax(logits.float(), dim=1)[:, 0]
    target = (labels != 0).float()
    inter = (probs * target).sum(dim=(1, 2))
    denom = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (1.0 - (2 * inter + 1e-6) / (denom + 1e-6)).mean()
    return ce + dice


def _capture_rng():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optimizer(backbone, head, mode, lr_head, lr_backbone):
    if mode == "frozen_probe":
        groups = [{"params": [p for p in head.parameters() if p.requires_grad], "lr": lr_head}]
    else:
        groups = [
            {"params": [p for p in apply_train_mode(backbone, mode) if p.requires_grad], "lr": lr_backbone},
            {"params": [p for p in head.parameters() if p.requires_grad], "lr": lr_head},
        ]
        apply_train_mode(backbone, mode)
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def _scheduler(opt, steps_per_epoch, epochs):
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=max(steps_per_epoch, 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(steps_per_epoch * max(epochs - 1, 1), 1))
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[max(steps_per_epoch, 1)])


def evaluate_split(backbone, head, rows, *, input_size, patch, device, split):
    if split != "val":
        raise RevisionError("test_leak", "选点只允许读取 val。独立测试不能参加选点。", {"split": split})
    backbone.eval()
    head.eval()
    binary_scores = []
    class_scores = []
    frames = []
    with torch.no_grad():
        for index, row in enumerate(rows):
            prepared = prepare_sample(row, input_size, patch, augment=False)
            images = prepared["tensor"].unsqueeze(0).to(device)
            logits = head(forward_dense(backbone, images))
            side = prepared["crop"]["side"]
            up = F.interpolate(logits.float(), size=(side, side), mode="bilinear", align_corners=False)
            pred = up.argmax(1)[0].cpu().numpy().astype(np.int16)
            scored = score_mask(pred, prepared["crop_mask"])
            binary_scores.append(scored["binary_foreground_dice"])
            if scored["gt_class_dice"] is not None:
                class_scores.append(scored["gt_class_dice"])
            frames.append({
                "index": index,
                "patient": prepared["patient"],
                "binary_foreground_dice": scored["binary_foreground_dice"],
                "gt_class_dice": scored["gt_class_dice"],
                "pred_name": scored["pred_name"],
                "gt_name": scored["gt_name"],
            })
    return {
        "binary_foreground_dice": float(sum(binary_scores) / len(binary_scores)),
        "gt_class_dice": float(sum(class_scores) / len(class_scores)) if class_scores else None,
        "frames": frames,
    }


def _ckpt_kind(mode):
    return "control_head" if mode == "frozen_probe" else "full_finetune"


def _pack_checkpoint(backbone, head, mode, opt, sched, history, best, config, step):
    blob = {
        "epoch": int(history[-1]["epoch"]) if history else -1,
        "optimizer_step": int(step),
        "best_val_dice": best["score"],
        "best_step": best["step"],
        "selection_metric": SELECTION_SCORE,
        "selection_split": "val",
        "history": [{k: v for k, v in item.items() if k != "frames"} for item in history],
        "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict(),
        "rng_state": _capture_rng(),
        "config": config,
        "ckpt_kind": _ckpt_kind(mode),
        "train_mode": mode,
        "role": config.get("protocol_name"),
        "backbone_id": config.get("backbone_id"),
        "geometry": config.get("geometry"),
        "embed_dim": int(backbone.embed_dim),
        "depth": int(getattr(backbone, "depth", len(backbone.blocks))),
        "patch_size": int(backbone.patch_size),
        "num_classes": int(head.weight.shape[0]),
        "continued_from_ep10": False,
    }
    if mode == "last4_finetune":
        blob["backbone_state"] = {k: v.detach().cpu() for k, v in backbone.state_dict().items()}
    return blob


def _save_blob(blob, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, temporary)
    temporary.replace(path)


def _grad_report(backbone, head):
    def active(params):
        return any(p.grad is not None and float(p.grad.detach().abs().sum()) > 0 for p in params)

    early = [p for block in backbone.blocks[:8] for p in block.parameters()]
    late = [p for block in backbone.blocks[8:] for p in block.parameters()]
    return {
        "early_requires_grad": any(p.requires_grad for p in early),
        "last4_requires_grad": any(p.requires_grad for p in late),
        "early_blocks_have_grad": active(early),
        "last4_have_grad": active(late),
        "head_has_grad": active(head.parameters()),
    }


def train(config, train_rows, val_rows, backbone, parent, run_name, *,
          device="cpu", max_optimizer_steps=None, resume=False, evaluate_fn=None,
          val_every="epoch"):
    """训练。选点指标固定为 val 上的帧等权 binary foreground Dice。"""
    if config.get("selection_score") != SELECTION_SCORE:
        raise RevisionError(
            "selection_protocol",
            "选点必须是 Stage E 的 binary foreground 帧等权 Dice，不能改成 semantic。",
            {"selection_score": config.get("selection_score")},
        )
    if config.get("selection_split") != "val":
        raise RevisionError("test_leak", "selection_split 必须是 val", {})
    if any(row.get("split") in {"test", "local_test", "external_test"} for row in list(train_rows) + list(val_rows)):
        raise RevisionError("test_leak", "train/val 列表里混入了独立测试划分", {})
    parent = Path(parent)
    mode = config["train_mode"]
    input_size = int(config.get("input_size", 768))
    patch = int(config.get("patch_size_runtime", config["architecture"]["patch_size"]))
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RevisionError("gpu_unavailable", "指定了 cuda，但当前进程没有可用 GPU。", {})
    apply_train_mode(backbone, mode)
    head = make_head(backbone.embed_dim, int(config.get("classes", 7)), config["seed"])
    opt = _optimizer(
        backbone, head, mode,
        lr_head=float(config.get("lr_head", 1e-4)),
        lr_backbone=float(config.get("lr_backbone") or 1e-5),
    )
    micro = int(config["microbatch"])
    accum = int(config["grad_accum"])
    if micro * accum != int(config["effective_batch"]):
        raise RevisionError(
            "batch_mismatch",
            "microbatch * grad_accum 必须等于 effective_batch",
            {"microbatch": micro, "grad_accum": accum},
        )
    if micro != 1:
        raise RevisionError("batch_mismatch", "这一版的微批固定为 1，再用梯度累积凑有效 batch。", {})
    steps_per_epoch = max(1, math.ceil(len(train_rows) / int(config["effective_batch"])))
    sched = _scheduler(opt, steps_per_epoch, int(config["epochs"]))
    autocast = device.type == "cuda"
    if config.get("fp16_forbidden") and config.get("autocast") == "fp16":
        raise RevisionError("fp16_forbidden", "禁止 fp16。", {})

    if resume:
        run = parent / run_name
        if not run.is_dir() or not (run / "last.pt").is_file():
            raise RevisionError("resume_missing", f"找不到可恢复的运行 {run}", {"path": str(run)})
        blob = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        if blob.get("config", {}).get("run_id") != config.get("run_id") or blob.get("train_mode") != mode:
            raise RevisionError("resume_forbidden", "断点配置和当前运行不一致", {})
        if blob.get("ckpt_kind") != _ckpt_kind(mode):
            raise RevisionError("ckpt_kind_mismatch", "断点种类和 train_mode 不一致", {})
        if blob.get("continued_from_ep10"):
            raise RevisionError("resume_forbidden", "不能从 ep10 历史微调续训", {})
        head.load_state_dict(blob["head_state"])
        if mode == "last4_finetune":
            backbone.load_state_dict(blob["backbone_state"])
        apply_train_mode(backbone, mode)
        opt.load_state_dict(blob["opt_state"])
        sched.load_state_dict(blob["sched_state"])
        _restore_rng(blob["rng_state"])
        history = list(blob["history"])
        step = int(blob["optimizer_step"])
        best = {"step": blob["best_step"], "score": blob["best_val_dice"]}
    else:
        run = create_run_dir(parent, run_name, config=config)
        random.seed(int(config["seed"]))
        np.random.seed(int(config["seed"]))
        torch.manual_seed(int(config["seed"]))
        history = []
        step = 0
        best = {"step": None, "score": -1.0}

    backbone.to(device)
    head.to(device)
    target = step + int(max_optimizer_steps) if max_optimizer_steps is not None else steps_per_epoch * int(config["epochs"])
    backwards = 0
    grad_probe = None
    opt.zero_grad(set_to_none=False)
    pending = 0
    evaluator = evaluate_fn or evaluate_split
    while step < target:
        backbone.train()
        head.train()
        row = train_rows[random.randrange(len(train_rows))]
        prepared = prepare_sample(row, input_size, patch, augment=True)
        images = prepared["tensor"].unsqueeze(0).to(device)
        labels = prepared["labels"].unsqueeze(0).to(device)
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(forward_dense(backbone, images))
                loss = _loss(logits, labels) / accum
        else:
            logits = head(forward_dense(backbone, images))
            loss = _loss(logits, labels) / accum
        loss.backward()
        backwards += 1
        pending += 1
        if grad_probe is None:
            grad_probe = _grad_report(backbone, head)
        if pending < accum:
            continue
        torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]], 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=False)
        pending = 0
        step += 1
        epoch = (step - 1) // steps_per_epoch
        record_val = val_every == "optimizer_step" or (val_every == "epoch" and step % steps_per_epoch == 0)
        if record_val:
            metrics = evaluator(
                backbone, head, val_rows,
                input_size=input_size, patch=patch, device=device, split="val",
            )
            item = {
                "epoch": int(epoch),
                "optimizer_step": int(step),
                "val_binary_foreground_dice": metrics["binary_foreground_dice"],
                "val_gt_class_dice": metrics.get("gt_class_dice"),
                "frames": metrics.get("frames", []),
            }
            history.append(item)
            if metrics["binary_foreground_dice"] > best["score"]:
                best = {"step": int(step), "score": metrics["binary_foreground_dice"]}
                _save_blob(
                    _pack_checkpoint(backbone, head, mode, opt, sched, history, best, config, step),
                    run / "best.pt",
                )
        _save_blob(
            _pack_checkpoint(backbone, head, mode, opt, sched, history, best, config, step),
            run / "last.pt",
        )
    scores = {
        item["optimizer_step"]: item["val_binary_foreground_dice"]
        for item in history if "val_binary_foreground_dice" in item
    }
    chosen = select_epoch(scores) if scores else None
    if chosen is not None and best["step"] != chosen:
        raise RevisionError("selection_protocol", "best.pt 与 binary 选点规则不一致", {"best": best, "chosen": chosen})
    return {
        "run_dir": str(run),
        "optimizer_steps": int(step),
        "backward_passes": int(backwards),
        "best_step": best["step"],
        "best_val_binary_foreground_dice": best["score"],
        "selection_metric": SELECTION_SCORE,
        "selection_split": "val",
        "history": [{k: v for k, v in item.items() if k != "frames"} for item in history],
        "grad_probe": grad_probe,
        "synthetic_backbone": bool(config.get("synthetic_backbone")),
    }


def infer_checkpoint(path, expect, rows, backbone, head, out_dir, *, device="cpu", input_size=768, patch=16):
    """加载 ep10 兼容字典（head_state，微调则另有 backbone_state）并写出 class-id。"""
    path = Path(path)
    out_dir = Path(out_dir)
    if not path.is_file():
        raise RevisionError("weight_missing", f"找不到 checkpoint {path}", {"path": str(path)})
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "head_state" not in blob:
        raise RevisionError("incomplete_metadata", "checkpoint 没有 head_state", {})
    kind = "full_finetune" if blob.get("backbone_state") else "control_head"
    weight = blob["head_state"]["weight"]
    record = {
        "sha256": _file_sha(path),
        "ckpt_kind": kind,
        "role": blob.get("role", expect.get("role")),
        "backbone_id": blob.get("backbone_id", expect.get("backbone_id")),
        "geometry": blob.get("geometry"),
        "embed_dim": int(weight.shape[1]),
        "depth": int(blob.get("depth", getattr(backbone, "depth", 0))),
        "patch_size": int(blob.get("patch_size", getattr(backbone, "patch_size", patch))),
        "num_classes": int(weight.shape[0]),
        "train_mode": blob.get("train_mode"),
        "epoch": blob.get("epoch"),
        "best_val_dice": blob.get("best_val_dice"),
        "filename": path.name,
    }
    identity = verify_identity(record, expect)
    if int(weight.shape[1]) != int(backbone.embed_dim) or int(weight.shape[0]) != int(head.weight.shape[0]):
        raise RevisionError(
            "architecture_mismatch",
            "head 形状和当前骨干不一致",
            {"head": list(weight.shape), "embed": int(backbone.embed_dim)},
        )
    head.load_state_dict(blob["head_state"])
    if kind == "full_finetune":
        backbone.load_state_dict(blob["backbone_state"])
    device = torch.device(device)
    backbone.to(device).eval()
    head.to(device).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, row in enumerate(rows):
        prepared = prepare_sample(row, input_size, patch, augment=False)
        images = prepared["tensor"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = head(forward_dense(backbone, images))
        side = prepared["crop"]["side"]
        up = F.interpolate(logits.float(), size=(side, side), mode="bilinear", align_corners=False)
        pred = up.argmax(1)[0].cpu().numpy().astype(np.int16)
        gt = prepared["crop_mask"] if row.get("mask") is not None else None
        stem = f"frame{index:04d}"
        write_review(out_dir, stem, prepared["crop_image"], pred, gt)
        height, width = prepared["crop"]["original_hw"]
        x0, y0 = prepared["crop"]["x0"], prepared["crop"]["y0"]
        canvas = np.full((height, width), 255, dtype=np.uint8)
        canvas[y0:y0 + side, x0:x0 + side] = pred.astype(np.uint8)
        Image.fromarray(canvas, mode="L").save(out_dir / f"{stem}_pred_class_id_original.png")
        scored = None if gt is None else score_mask(pred, gt)
        frames.append({
            "index": index,
            "patient": prepared["patient"],
            "coordinates": {
                "original_hw": [height, width],
                "crop_box_original_xyxy": prepared["crop"]["crop_box_original_xyxy"],
                "class_id_crop_space": "square_crop",
                "class_id_crop_hw": [side, side],
                "class_id_original_space": "original_xy_ignore_255_outside_crop",
                "model_input_hw": [input_size, input_size],
            },
            "scores": None if scored is None else {
                "binary_foreground_dice": scored["binary_foreground_dice"],
                "gt_class_dice": scored["gt_class_dice"],
                "pred_name": scored["pred_name"],
                "gt_name": scored["gt_name"],
            },
        })
    (out_dir / "frames.json").write_text(json.dumps({
        "warnings": identity["warnings"],
        "ckpt_kind": kind,
        "frames": frames,
    }, indent=2, ensure_ascii=False) + "\n")
    return {"out_dir": str(out_dir), "warnings": identity["warnings"], "ckpt_kind": kind, "n": len(frames)}


def synthetic_config(base, input_size=16, patch=4, embed=8):
    config = dict(base)
    config["architecture"] = dict(base["architecture"])
    config["architecture"]["embed_dim"] = embed
    config["architecture"]["patch_size"] = patch
    config["input_size"] = input_size
    config["patch_size_runtime"] = patch
    config["synthetic_backbone"] = True
    config["autocast"] = "fp32"
    return config


def _synthetic_rows(n, size, patient, split, cls_name="grasping forceps"):
    rows = []
    for _ in range(n):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.uint8)
        image[4:12, 4:12] = (180, 140, 120)
        mask[4:12, 4:12] = 255
        rows.append({
            "image": image, "mask": mask, "cls": cls_name,
            "patient": patient, "split": split,
        })
    return rows


def run_synthetic_demo(parent, optimizer_steps=2):
    """四组配置各走一遍冻结/累积，并在 frozen 上做 2 step、断点对齐和一轮推理。"""
    parent = Path(parent)
    assert_output_ready(parent, 1)
    reports = []
    for index, config in enumerate(training_matrix(phase="single", seed=7)):
        cfg = synthetic_config(config)
        name = f"fairrun-cfg{index}-{config['train_mode'][:3]}"
        backbone = TinyDino(embed=8, depth=12, patch=4)
        result = train(
            cfg, _synthetic_rows(2, 16, "synth-train", "train"),
            _synthetic_rows(1, 16, "synth-val", "val"),
            backbone, parent, name, device="cpu", max_optimizer_steps=1, val_every="optimizer_step",
        )
        reports.append({
            "run_id": config["run_id"],
            "train_mode": config["train_mode"],
            "backward_passes": result["backward_passes"],
            "optimizer_steps": result["optimizer_steps"],
            "grad_probe": result["grad_probe"],
            "synthetic_backbone": result["synthetic_backbone"],
            "selection_metric": result["selection_metric"],
        })
    resume_ok = _resume_matches(parent, optimizer_steps)
    infer = _infer_demo(parent)
    return {"configs": reports, "resume_match": resume_ok, "infer": infer}


def _clone_tiny(net):
    fresh = TinyDino(embed=net.embed_dim, depth=net.depth, patch=net.patch_size)
    fresh.load_state_dict(net.state_dict())
    return fresh


def _resume_matches(parent, optimizer_steps=2):
    base = training_matrix()[0]
    cfg = synthetic_config(base)
    train_rows = _synthetic_rows(2, 16, "synth-train", "train")
    val_rows = _synthetic_rows(1, 16, "synth-val", "val")
    name_full = "fairrun-resume-full"
    name_part = "fairrun-resume-part"
    torch.manual_seed(int(cfg["seed"]))
    seed_net = TinyDino()
    first = max(1, int(optimizer_steps) - 1)
    rest = int(optimizer_steps) - first
    full = train(
        cfg, train_rows, val_rows, _clone_tiny(seed_net), parent, name_full,
        device="cpu", max_optimizer_steps=int(optimizer_steps), val_every="optimizer_step",
    )
    part = train(
        cfg, train_rows, val_rows, _clone_tiny(seed_net), parent, name_part,
        device="cpu", max_optimizer_steps=first, val_every="optimizer_step",
    )
    marker = Path(part["run_dir"]) / "marker.txt"
    marker.write_text("keep")
    resumed = train(
        cfg, train_rows, val_rows, _clone_tiny(seed_net), parent, name_part,
        device="cpu", max_optimizer_steps=rest, resume=True, val_every="optimizer_step",
    )
    full_blob = torch.load(Path(full["run_dir"]) / "last.pt", map_location="cpu", weights_only=False)
    part_blob = torch.load(Path(resumed["run_dir"]) / "last.pt", map_location="cpu", weights_only=False)
    same = torch.allclose(full_blob["head_state"]["weight"], part_blob["head_state"]["weight"], atol=1e-5)
    kept = marker.read_text() == "keep"
    opt_same = torch.allclose(
        full_blob["opt_state"]["state"][0]["exp_avg"],
        part_blob["opt_state"]["state"][0]["exp_avg"],
        atol=1e-5,
    )
    return bool(
        same and opt_same and kept
        and resumed["optimizer_steps"] == int(optimizer_steps)
        and full_blob["sched_state"]["last_epoch"] == part_blob["sched_state"]["last_epoch"]
    )


def _infer_demo(parent):
    backbone = TinyDino()
    head = make_head(8, 7, seed=0)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[0] = 5.0
    blob = {
        "epoch": 10,
        "best_val_dice": 0.8682456,
        "head_state": {k: v.detach().cpu() for k, v in head.state_dict().items()},
        "train_mode": "frozen_probe",
        "ckpt_kind": "control_head",
        "geometry": "square768",
        "embed_dim": 8,
        "depth": 12,
        "patch_size": 4,
        "num_classes": 7,
        "role": "frozen_backbone_supervised_head",
        "backbone_id": "synthetic_tiny",
    }
    path = parent / "synthetic-control-head.pt"
    torch.save(blob, path)
    image = np.zeros((12, 20, 3), dtype=np.uint8)
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[2:8, 6:12] = 255
    row = {"image": image, "mask": mask, "cls": "bipolar", "patient": "synth-patient", "split": "val"}
    expect = {
        "sha256": _file_sha(path),
        "ckpt_kind": "control_head",
        "embed_dim": 8,
        "num_classes": 7,
        "geometry": "square768",
        "role": "frozen_backbone_supervised_head",
        "backbone_id": "synthetic_tiny",
        "depth": 12,
        "patch_size": 4,
        "train_mode": "frozen_probe",
    }
    out = parent / "fairrun-infer-demo"
    out.mkdir(exist_ok=True)
    return infer_checkpoint(
        path, expect, [row], backbone, make_head(8, 7, seed=1), out,
        device="cpu", input_size=16, patch=4,
    )


def launch_real(*, backbone_id, weight_path, expected_sha, data_root, manifest, out_dir,
                train_mode, device="cpu", seed=7, min_free_bytes=1 << 30, run_name=None):
    """用户本地权重和数据上的真实入口。缺文件、不可写、无 bf16 GPU 时立刻停下。"""
    cuda = torch.cuda.is_available()
    bf16 = bool(cuda and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    plan = runtime_plan(True, cuda, bf16, device=device)
    assert_output_ready(out_dir, min_free_bytes)
    if not expected_sha:
        raise RevisionError(
            "expected_sha_required",
            "把本地核过的 sha256 填进 manifest。仓库不代存私有哈希。",
            {"steps_zh": _STEPS.get(backbone_id, "")},
        )
    resolve_backbone(backbone_id, weight_path, expected_sha=expected_sha)
    root = Path(data_root) if data_root else None
    manifest_path = Path(manifest) if manifest else None
    if root is None or not root.is_dir():
        raise RevisionError("data_missing", "数据根目录不存在", {"path": None if root is None else str(root)})
    if manifest_path is None or not manifest_path.is_file():
        raise RevisionError("manifest_missing", "manifest jsonl 不存在", {"path": None if manifest_path is None else str(manifest_path)})
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["root"] = str(root)
    train_rows = [row for row in rows if row.get("split") == "train"]
    val_rows = [row for row in rows if row.get("split") == "val"]
    if not train_rows or not val_rows:
        raise RevisionError("split_missing", "manifest 需要 train 和 val。test 不会被读取。", {})
    try:
        from surgfm_probe import load_backbone
    except ImportError as exc:
        raise RevisionError(
            "backbone_loader_missing",
            "需要把 InnerLogic-Surg 的 surgfm_probe 放到 PYTHONPATH。"
            "本函数不下载权重，也不用别的结构冒充 ViT-B/16。",
            {"steps_zh": _STEPS.get(backbone_id, "")},
        ) from exc
    backbone = load_backbone(str(weight_path), device=device)
    if not hasattr(backbone, "blocks") or len(list(backbone.blocks)) != 12:
        raise RevisionError("architecture_mismatch", "骨干不是 12-block ViT-B/16 接口", {})
    base = next(item for item in training_matrix(phase="single", seed=seed)
                if item["backbone_id"] == backbone_id and item["train_mode"] == train_mode)
    base = dict(base)
    base["input_size"] = 768
    base["patch_size_runtime"] = 16
    base["synthetic_backbone"] = False
    base["autocast"] = plan["autocast"]
    name = run_name or ("fairrun-real-" + train_mode)
    return train(
        base, train_rows, val_rows, backbone, out_dir, name,
        device=device, val_every="epoch",
    )
