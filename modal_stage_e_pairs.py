"""Stage E pairs: side-by-side BEFORE/AFTER overlays for the same 18 worst
frames as Stage W, but with the Stage E square-crop preprocessing and the
Stage E checkpoints (control = results_e/best_control.pt, after =
results_e/best_finetune.pt). INFERENCE ONLY - no training.

Purpose: direct visual + quantitative check whether the right-side false
positives (Stage W: 87-97% of FP pixels in the right half on the 12 worst
external frames, FP mass 88k-142k px) are relieved by the square crop.
Same 18 frames as Stage W: external_test idx
[698,597,721,769,712,549,718,710,700,696,599,704], local_test idx
[680,48,809,816,750,800] (positions in the manifest rows).
Per-frame Dice cross-checked against results_e/results_control.json and
results_e/results_finetune.json (|diff| <= 0.002 required).
Output: /data/sea/viz_e_pairs/{set}_{idx:04d}_pair.jpg (1440x720, square
720x720 halves, English labels) + e_pairs.json manifest.
Run AFTER modal_stage_e.py finishes: .venv/bin/modal run --detach
modal_stage_e_pairs.py   (nohup-wrap it.)
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

IN_S = 768
NCLS = 7
CLS_NAMES = ["background", "grasping forceps", "bipolar", "drill",
             "punch", "scissor", "dissector"]

EXT_IDX = [698, 597, 721, 769, 712, 549, 718, 710, 700, 696, 599, 704]
LOC_IDX = [680, 48, 809, 816, 750, 800]


@app.function(image=image, volumes={"/data": vol}, gpu="L40S",
              timeout=3600, scaledown_window=10)
def stage_e_pairs():
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

    res = {"stage": "E-pairs"}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    man = base / "manifests_paper"
    out_dir = base / "viz_e_pairs"
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

    json_dice = {}
    try:
        cj = json.load(open(base / "results_e" / "results_control.json"))
        dj = json.load(open(base / "results_e" / "results_finetune.json"))
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

    def load_nrrd_mask(path):
        """FIXED loader (stage C/D/E): nrrd axis order is ALWAYS (W, H, 1).
        Transpose FIRST; native (H, W) bool, no resize."""
        arr, _hdr = nrrd.read(str(path))
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m.T > 0)

    def square_crop_bounds(w, h):
        side = min(w, h)
        return side, (w - side) // 2

    def load_image(r):
        """Center square crop -> 768x768, normalized tensor + native side."""
        im = Image.open(ext / r["image"]).convert("RGB")
        side, x0 = square_crop_bounds(im.width, im.height)
        im = im.crop((x0, 0, x0 + side, side)).resize((IN_S, IN_S), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s_ = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - m) / s_, side

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
        x, side = load_image(r)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(forward_dense(backbone, x.unsqueeze(0).to(device)))
        up = F.interpolate(logits.float(), size=(side, side),
                           mode="bilinear", align_corners=False)
        pm = (up.argmax(1)[0].cpu().numpy() != 0)
        gt_full = load_nrrd_mask(ext / r["mask"])
        _sd, x0 = square_crop_bounds(gt_full.shape[1], gt_full.shape[0])
        gt = gt_full[:, x0:x0 + side]
        inter = float((pm & gt).sum())
        dice = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
        area = float(pm.sum()) / max(float(gt.sum()), 1)
        return pm, gt, dice, area

    def overlay(im_path, pm, gt, label):
        """One 720x720 square overlay half with an English label strip."""
        im = Image.open(im_path).convert("RGB")
        side, x0 = square_crop_bounds(im.width, im.height)
        im = im.crop((x0, 0, x0 + side, side)).resize((720, 720), Image.BICUBIC)
        ov = np.asarray(im).astype(np.float32)
        gts = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize((720, 720), Image.NEAREST)) > 0
        prs = np.asarray(Image.fromarray((pm.astype(np.uint8)) * 255).resize((720, 720), Image.NEAREST)) > 0
        ov[gts] = ov[gts] * 0.45 + np.array([0, 200, 0]) * 0.55
        only_pred = prs & ~gts
        ov[only_pred] = ov[only_pred] * 0.45 + np.array([220, 0, 0]) * 0.55
        half = Image.fromarray(ov.astype(np.uint8))
        d = ImageDraw.Draw(half, "RGBA")
        d.rectangle([0, 0, 720, 36], fill=(0, 0, 0, 160))
        font = ImageFont.load_default(size=22)
        d.text((10, 6), label, fill=(255, 255, 255), font=font)
        return half

    targets = [("external_test", i) for i in EXT_IDX] + [("local_test", i) for i in LOC_IDX]

    preds = {}  # (tag, set, idx) -> (pm, gt, dice, area)
    for tag, path, has_backbone in [
            ("control", base / "results_e" / "best_control.pt", False),
            ("d", base / "results_e" / "best_finetune.pt", True)]:
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
        lab_c = json_dice.get(("control", s, i), dice_c)
        lab_d = json_dice.get(("d", s, i), dice_d)
        left = overlay(ext / r["image"], pm_c, gt,
                       f"BEFORE #{i}  Dice {lab_c:.3f}  area {area_c:.2f}")
        right = overlay(ext / r["image"], pm_d, gt,
                        f"AFTER #{i}  Dice {lab_d:.3f}  area {area_d:.2f}")
        pair = Image.new("RGB", (1440, 720), (255, 255, 255))
        pair.paste(left, (0, 0))
        pair.paste(right, (720, 0))
        d = ImageDraw.Draw(pair)
        d.rectangle([718, 0, 722, 720], fill=(255, 255, 255))
        name = f"{s}_{i:04d}_pair.jpg"
        pair.save(out_dir / name, quality=88)
        manifest.append({
            "set": s, "idx": i, "cls": r["cls"], "patient": r["patient"],
            "control_dice": round(float(dice_c), 4), "d_dice": round(float(dice_d), 4),
            "control_dice_json": round(float(lab_c), 4), "d_dice_json": round(float(lab_d), 4),
            "delta_dice": round(float(dice_d - dice_c), 4),
            "control_area": round(float(area_c), 3), "d_area": round(float(area_d), 3),
            "img": f"viz_e_pairs/{name}",
        })
        print(f"[pair] {name} done", flush=True)

    with open(out_dir / "e_pairs.json", "w") as f:
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
    print("STAGE_E_PAIRS_RESULT_JSON=" + json.dumps(res, ensure_ascii=False))
    return res


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_e_pairs.remote()
    print("\n=== STAGE E PAIRS RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/stage_e_pairs_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
