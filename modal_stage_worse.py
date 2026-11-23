"""Stage W: side-by-side BEFORE/AFTER overlays for the 18 worst frames.

Why external Dice got worse after fine-tuning is not uniform: 176 external
frames worsened >0.05 vs only 46 improved. This script renders paired
overlays (LEFT = control / before fine-tune, RIGHT = Stage D / after) for
the 12 worst external frames + 6 worst local frames, so the user can see
directly what changed. INFERENCE ONLY - no training, no script/checkpoint
edits. Reuses the fixed loader + overlay recipe from modal_stage_c_reeval.py
(transpose FIRST by the array's own shape, then NEAREST-resize; green =
corrected GT, red = pred-only).

Checkpoints (volume):
- before: /data/sea/results_b2/best_control.pt (frozen backbone + head)
- after:  /data/sea/results_d/best_finetune.pt (Stage D best = ep0)
Frames: external_test idx [698,597,721,769,712,549,718,710,700,696,599,704],
local_test idx [680,48,809,816,750,800] (positions in the manifest rows,
same indexing as the Stage C/D result JSONs).
Per-frame Dice computed in-script is cross-checked against
results_c/results_reeval_control.json and results_d/results_finetune.json
on the volume (|diff| <= 0.002 required, mismatches printed loudly).
Output: /data/sea/viz_worse/{set}_{idx:04d}_pair.jpg (1920x540, English
labels so no CJK font is needed) + worse_pairs.json manifest.
Run: .venv/bin/modal run --detach modal_stage_worse.py   (nohup-wrap it.)
"""
import json
import time

import modal

app = modal.App("innerlogic-surg-sea")
vol = modal.Volume.from_name("innerlogic-surg", create_if_missing=True)

CKPT_SHA256 = "a516a775805b9093f15e5e7f6fa023663a48d81ca211d6552b4fa87be03c8806"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3",
                 "pillow==10.4.0", "pynrrd==1.1.3")
    .add_local_dir("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/repo",
                   "/root/repo")
)

IN_W, IN_H = 768, 432
NCLS = 7
CLS_NAMES = ["background", "grasping forceps", "bipolar", "drill",
             "punch", "scissor", "dissector"]

EXT_IDX = [698, 597, 721, 769, 712, 549, 718, 710, 700, 696, 599, 704]
LOC_IDX = [680, 48, 809, 816, 750, 800]


@app.function(image=image, volumes={"/data": vol}, gpu="L40S",
              timeout=3600, scaledown_window=10)
def stage_worse():
    import hashlib
    import sys
    from pathlib import Path

    import numpy as np
    import nrrd
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, "/root/repo")
    from surgfm_probe import load_backbone
    from surgfm_probe.transforms import IMAGENET_MEAN, IMAGENET_STD

    res = {"stage": "W"}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    man = base / "manifests_paper"
    out_dir = base / "viz_worse"
    out_dir.mkdir(exist_ok=True)

    ckpt = base / "ckpt" / "dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth"
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    res["ckpt_sha256_ok"] = h.hexdigest() == CKPT_SHA256
    print(f"ckpt sha ok={res['ckpt_sha256_ok']}", flush=True)
    assert res["ckpt_sha256_ok"], "checkpoint SHA mismatch"

    def load_rows(name):
        with open(man / name) as f:
            return [json.loads(line) for line in f]

    local_rows = load_rows("local_test.jsonl")
    ext_rows = load_rows("external_test.jsonl")
    rows_by_set = {"local_test": local_rows, "external_test": ext_rows}

    # Cross-check source: per-frame Dice from the published result JSONs.
    json_dice = {}
    try:
        cj = json.load(open(base / "results_c" / "results_reeval_control.json"))
        dj = json.load(open(base / "results_d" / "results_finetune.json"))
        for tag, j in (("control", cj), ("d", dj)):
            for s in ("local_test", "external_test"):
                for f in j["sets"][s]["frames"]:
                    json_dice[(tag, s, f["idx"])] = f["dice"]
        print(f"loaded {len(json_dice)} reference dice values from volume JSONs", flush=True)
    except Exception as e:
        print(f"WARNING: could not load reference JSONs ({e}); "
              "in-script Dice will not be cross-checked", flush=True)

    device = "cuda"
    torch.manual_seed(7)

    def load_nrrd_mask(path, w=1920, h=1080):
        """FIXED loader (stage C): nrrd axis order is ALWAYS (W, H, 1).
        Transpose FIRST using the array's own shape, then NEAREST-resize."""
        arr, _hdr = nrrd.read(str(path))
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.T  # (W, H) -> (H, W) at native resolution
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST))
        return (m > 0)

    def load_image(r):
        im = Image.open(ext / r["image"]).convert("RGB")
        im = im.resize((IN_W, IN_H), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s_ = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - m) / s_

    def build_model(finetune):
        backbone = load_backbone(str(ckpt), device=device)
        head = torch.nn.Conv2d(768, NCLS, 1).to(device)
        backbone.requires_grad_(False)
        if finetune:
            for blk in backbone.blocks[8:]:
                for p in blk.parameters():
                    p.requires_grad = True
        return backbone, head

    def forward_dense(backbone, x):
        ((fmap, _cls),) = backbone.get_intermediate_layers(
            x, n=1, reshape=True, return_class_token=True)
        return fmap

    @torch.no_grad()
    def predict(backbone, head, r):
        backbone.eval()
        head.eval()
        x = load_image(r).unsqueeze(0).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(forward_dense(backbone, x))
        up = F.interpolate(logits.float(), size=(1080, 1920),
                           mode="bilinear", align_corners=False)
        pm = (up.argmax(1)[0].cpu().numpy() != 0)
        gt = load_nrrd_mask(ext / r["mask"])
        inter = float((pm & gt).sum())
        dice = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
        area = float(pm.sum()) / max(float(gt.sum()), 1)
        return pm, gt, dice, area

    def overlay(im_path, pm, gt, label):
        """One 960x540 overlay half with an English label strip on top."""
        im = Image.open(im_path).convert("RGB").resize((960, 540), Image.BICUBIC)
        ov = np.asarray(im).astype(np.float32)
        gts = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize((960, 540), Image.NEAREST)) > 0
        prs = np.asarray(Image.fromarray((pm.astype(np.uint8)) * 255).resize((960, 540), Image.NEAREST)) > 0
        ov[gts] = ov[gts] * 0.45 + np.array([0, 200, 0]) * 0.55
        only_pred = prs & ~gts
        ov[only_pred] = ov[only_pred] * 0.45 + np.array([220, 0, 0]) * 0.55
        half = Image.fromarray(ov.astype(np.uint8))
        d = ImageDraw.Draw(half, "RGBA")
        d.rectangle([0, 0, 960, 36], fill=(0, 0, 0, 160))
        font = ImageFont.load_default(size=22)
        d.text((10, 6), label, fill=(255, 255, 255), font=font)
        return half

    targets = [("external_test", i) for i in EXT_IDX] + [("local_test", i) for i in LOC_IDX]

    # Load each model once, predict all 18 frames.
    preds = {}  # (tag, set, idx) -> (pm, gt, dice, area)
    for tag, path, has_backbone in [
            ("control", base / "results_b2" / "best_control.pt", False),
            ("d", base / "results_d" / "best_finetune.pt", True)]:
        c = torch.load(str(path), map_location="cpu", weights_only=False)
        backbone, head = build_model(has_backbone)
        if has_backbone:
            backbone.load_state_dict(c["backbone_state"])
        head.load_state_dict(c["head_state"])
        backbone, head = backbone.to(device), head.to(device)
        for s, i in targets:
            pm, gt, dice, area = predict(backbone, head, rows_by_set[s][i])
            preds[(tag, s, i)] = (pm, gt, dice, area)
            ref = json_dice.get((tag, s, i))
            flag = ""
            if ref is not None and abs(dice - ref) > 0.002:
                flag = f"  *** MISMATCH vs JSON {ref:.4f} ***"
            print(f"[{tag}] {s} #{i}: dice={dice:.4f} area={area:.2f}{flag}", flush=True)
        del backbone, head
        torch.cuda.empty_cache()

    manifest = []
    for s, i in targets:
        r = rows_by_set[s][i]
        pm_c, gt, dice_c, area_c = preds[("control", s, i)]
        pm_d, _gt, dice_d, area_d = preds[("d", s, i)]
        # Labels use the published JSON Dice (the numbers the Space cards
        # show); the in-script recompute can differ by ~0.002 on boundary
        # pixels from bf16 batch-size nondeterminism (seen on #680).
        lab_c = json_dice.get(("control", s, i), dice_c)
        lab_d = json_dice.get(("d", s, i), dice_d)
        left = overlay(ext / r["image"], pm_c, gt,
                       f"BEFORE #{i}  Dice {lab_c:.3f}  area {area_c:.2f}")
        right = overlay(ext / r["image"], pm_d, gt,
                        f"AFTER #{i}  Dice {lab_d:.3f}  area {area_d:.2f}")
        pair = Image.new("RGB", (1920, 540), (255, 255, 255))
        pair.paste(left, (0, 0))
        pair.paste(right, (960, 0))
        d = ImageDraw.Draw(pair)
        d.rectangle([958, 0, 962, 540], fill=(255, 255, 255))
        name = f"{s}_{i:04d}_pair.jpg"
        pair.save(out_dir / name, quality=88)
        manifest.append({
            "set": s, "idx": i, "cls": r["cls"], "patient": r["patient"],
            "control_dice": round(float(dice_c), 4), "d_dice": round(float(dice_d), 4),
            "control_dice_json": round(float(lab_c), 4), "d_dice_json": round(float(lab_d), 4),
            "delta_dice": round(float(dice_d - dice_c), 4),
            "control_area": round(float(area_c), 3), "d_area": round(float(area_d), 3),
            "img": f"viz_worse/{name}",
        })
        print(f"[pair] {name} done", flush=True)

    with open(out_dir / "worse_pairs.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    vol.commit()

    mismatches = [m for m in manifest
                  if (("control", m["set"], m["idx"]) in json_dice and
                      abs(m["control_dice"] - json_dice[("control", m["set"], m["idx"])]) > 0.002)
                  or (("d", m["set"], m["idx"]) in json_dice and
                      abs(m["d_dice"] - json_dice[("d", m["set"], m["idx"])]) > 0.002)]
    res["n_pairs"] = len(manifest)
    res["mismatches_vs_json"] = len(mismatches)
    res["total_s"] = round(time.time() - t0, 1)
    print("STAGE_WORSE_RESULT_JSON=" + json.dumps(res, ensure_ascii=False))
    return res


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_worse.remote()
    print("\n=== STAGE W RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/stage_worse_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
