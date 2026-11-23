"""Stage E: square center-crop preprocessing instead of stretch (user-approved
2026-11-22). The ONLY variable changed vs Stage D is the preprocessing.

Background (user's hypothesis, 2026-11-22): Center 1 is 1920x1080 / 1280x720
(16:9), Center 2 is all 720x576 (5:4). The old pipeline resized image and mask
straight to 768x432 (16:9): Center 1 barely stretched, Center 2 stretched
horizontally x1.42. Training only ever saw Center 1, so at inference Center 2
arrives with a different stretch and the model cannot adapt -- prime suspect
for the post-fine-tune degradation (incl. the right-side false positives).
The endoscopic frame is a circle; all useful information is in the central
square. So: no stretching at all, center-crop in the width direction until
the remaining width equals the height, discarding the black side regions.
Center 1 training and Center 2 inference use the same procedure, so train
and inference are consistent. The alternative (augmentation that deliberately
creates stretch) is NOT used -- square crop is simpler and fits endoscopy.

Preprocessing (this stage):
- Center square crop: image W x H, side = min(W, H), x0 = (W - side) // 2,
  crop (x0, 0, x0 + side, side). Mask .nrrd: transpose to (H, W) first (fixed
  loader, same as C/D), then the SAME square crop. 1920x1080 -> 1080x1080,
  1280x720 -> 720x720, 720x576 -> 576x576.
- Input: square resized to 768x768 (long side stays 768, same as before;
  patch grid 48x48). Image BICUBIC, mask NEAREST.
- Scoring in native square coordinates: logits upsampled to the native square
  side (1080 / 720 / 576) and compared with the cropped native GT. No
  non-uniform stretch anywhere in the pipeline.
- Augmentation: hflip + square random crop/resize, square preserved at every
  step; any aspect-ratio-changing augmentation is forbidden.
- Geometry sanity check (done locally before launch, stage_e_sanity.py):
  GT fraction outside the square crop: train mean 0.007% (max 0.66%),
  external mean 0.05% (max 1.83%); 4 overlays (2 Center 1 + 2 Center 2)
  visually confirmed GT hugs the instrument, no misalignment.

Two groups, sequential, same recipe as Stage D except the preprocessing:
- Control (no fine-tune): backbone frozen, 1x1 head trained from RANDOM init
  (do NOT reuse any old checkpoint -- old heads were trained under stretch
  preprocessing). Max 6 epochs, val-Dice early stop patience 2, head LR 1e-4.
- Fine-tune: last 4 blocks (8-11) + head, backbone LR 1e-5, head LR 1e-4,
  max 6 epochs, patience 2. Starts from the PRETRAINED weights
  dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth, NOT from Stage D's
  checkpoint (stretch-trained weights would mix variables).
- Both: same paper-aligned split (manifests_paper/: train 8057 / val 496 /
  local_test 1040 / external_test 1069 / old100 100; patient-disjoint),
  CE+Dice, AdamW cosine + 1-epoch warmup, bf16 autocast (NEVER fp16).
  Results to results_e/, overlays to viz_e/.
- Expected duration ~1.5-2 h (768x768 = 48x48 patches vs 27x48 before, so
  epochs run slower than Stage D's ~530 s), ~$3-4 metered, billed $0.
Run: .venv/bin/modal run --detach modal_stage_e.py   (nohup-wrap it: local
exec sessions are recycled after ~20 min and kill a piped modal process.)
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

IN_S = 768  # square input side; patch grid 48x48
NCLS = 7
CLS_NAMES = ["background", "grasping forceps", "bipolar", "drill",
             "punch", "scissor", "dissector"]
CLS2ID = {c: i + 1 for i, c in enumerate(CLS_NAMES[1:])}
BATCH = 8
MAX_EPOCHS = 6
PATIENCE = 2
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4


@app.function(image=image, volumes={"/data": vol}, gpu="L40S",
              timeout=14400, scaledown_window=10)
def stage_e():
    import hashlib
    import os
    import random
    import sys
    from pathlib import Path

    import numpy as np
    import nrrd
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/root/repo")
    from surgfm_probe import load_backbone
    from surgfm_probe.transforms import IMAGENET_MEAN, IMAGENET_STD

    res = {"stage": "E"}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    man = base / "manifests_paper"
    out_dir = base / "results_e"
    out_dir.mkdir(exist_ok=True)
    viz_dir = base / "viz_e"
    viz_dir.mkdir(exist_ok=True)

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

    train_rows = load_rows("train.jsonl")
    val_rows = load_rows("val.jsonl")
    local_rows = load_rows("local_test.jsonl")
    ext_rows = load_rows("external_test.jsonl")
    old_rows = load_rows("old100.jsonl")
    train_patients = {r["patient"] for r in train_rows}
    res["n"] = {"train": len(train_rows), "val": len(val_rows),
                "local_test": len(local_rows), "external_test": len(ext_rows),
                "old100": len(old_rows)}
    print("rows:", res["n"], flush=True)

    device = "cuda"
    torch.manual_seed(7)
    rng = random.Random(7)

    def load_nrrd_mask(path):
        """FIXED loader (stage C/D/E): nrrd axis order is ALWAYS (W, H, 1).
        Transpose FIRST using the array's own shape. Returns native (H, W)
        bool -- NO resize here (Stage E scores in native square coordinates)."""
        arr, _hdr = nrrd.read(str(path))
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.T  # (W, H) -> (H, W) at native resolution
        return (m > 0)

    def square_crop_bounds(w, h):
        """Center square crop: side = min(W, H), x0 = (W - side) // 2.
        Keeps the whole endoscopic circle; discards black side regions."""
        side = min(w, h)
        return side, (w - side) // 2

    def cropped_gt(r):
        """Native square-cropped GT bool mask (side, side)."""
        gt = load_nrrd_mask(ext / r["mask"])
        h, w = gt.shape
        side, x0 = square_crop_bounds(w, h)
        return gt[:, x0:x0 + side]

    def mask_to_patch_labels(mask_full, cls_name):
        """Square mask (768,768) -> patch grid (48,48) int64 labels + binary mask."""
        m = mask_full.astype(np.float32)
        gh, gw = IN_S // 16, IN_S // 16
        h, w = m.shape
        m_small = m.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))
        cid = CLS2ID[cls_name]
        lab = np.where(m_small >= 0.5, cid, 0).astype(np.int64)
        return lab, (lab != 0).astype(np.float32)

    def load_sample(r, augment):
        """-> (image tensor (3,768,768) normalized, patch labels (48,48),
        binary patch mask, native square side). Square crop FIRST, then
        augment (flip + random SQUARE crop), then resize to 768x768.
        No non-uniform stretch anywhere."""
        im = Image.open(ext / r["image"]).convert("RGB")
        gt = Image.fromarray((load_nrrd_mask(ext / r["mask"]).astype(np.uint8)) * 255)
        side, x0 = square_crop_bounds(im.width, im.height)
        im = im.crop((x0, 0, x0 + side, side))
        gt = gt.crop((x0, 0, x0 + side, side))
        if augment:
            if rng.random() < 0.5:
                im, gt = im.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)
            s = rng.uniform(0.85, 1.0)
            cw = ch = int(side * s)
            cx0 = rng.randint(0, side - cw)
            cy0 = rng.randint(0, side - ch)
            im = im.crop((cx0, cy0, cx0 + cw, cy0 + ch))
            gt = gt.crop((cx0, cy0, cx0 + cw, cy0 + ch))
        im = im.resize((IN_S, IN_S), Image.BICUBIC)
        gt = gt.resize((IN_S, IN_S), Image.NEAREST)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s_ = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - m) / s_
        gt_np = np.asarray(gt) > 0
        lab, binm = mask_to_patch_labels(gt_np, r["cls"])
        return x, torch.from_numpy(lab), torch.from_numpy(binm), side

    def dice_loss_from_logits(logits, bin_target):
        """logits (B,7,48,48); binary Dice on instrument vs background."""
        p = 1.0 - torch.softmax(logits.float(), dim=1)[:, 0]
        t = bin_target
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        return (1.0 - (2 * inter + 1e-6) / (denom + 1e-6)).mean()

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
    def val_dice(backbone, head):
        backbone.eval()
        head.eval()
        dices = []
        for i in range(0, len(val_rows), 16):
            chunk = val_rows[i:i + 16]
            samples = [load_sample(r, False) for r in chunk]
            xs = torch.stack([s[0] for s in samples]).to(device)
            sides = [s[3] for s in samples]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(forward_dense(backbone, xs))
            for j, r in enumerate(chunk):
                side = sides[j]
                up = F.interpolate(logits[j:j + 1].float(), size=(side, side),
                                   mode="bilinear", align_corners=False)
                pm = (up.argmax(1)[0].cpu().numpy() != 0)
                gt = cropped_gt(r)
                inter = float((pm & gt).sum())
                dices.append(2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9))
        return float(np.mean(dices))

    def train_group(group):
        """group in ('control', 'finetune'). Returns (best_state_info, history)."""
        finetune = group == "finetune"
        backbone, head = build_model(finetune)
        if finetune:
            params = [{"params": [p for blk in backbone.blocks[8:] for p in blk.parameters()],
                       "lr": LR_BACKBONE},
                      {"params": head.parameters(), "lr": LR_HEAD}]
        else:
            params = [{"params": head.parameters(), "lr": LR_HEAD}]
        opt = torch.optim.AdamW(params, weight_decay=1e-4)
        steps_per_epoch = (len(train_rows) + BATCH - 1) // BATCH
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=steps_per_epoch)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps_per_epoch * (MAX_EPOCHS - 1))
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[steps_per_epoch])
        ckpt_path = out_dir / f"checkpoint_{group}.pt"
        start_ep, best_dice, bad, hist = 0, -1.0, 0, []
        if ckpt_path.exists():
            c = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            start_ep = c["epoch"] + 1
            best_dice, bad, hist = c["best_val_dice"], c["bad_epochs"], c["history"]
            if finetune:
                backbone.load_state_dict(c["backbone_state"])
            head.load_state_dict(c["head_state"])
            opt.load_state_dict(c["opt_state"])
            print(f"[{group}] resumed at epoch {start_ep}, best val {best_dice:.4f}", flush=True)
        for ep in range(start_ep, MAX_EPOCHS):
            backbone.train()
            head.train()
            order = list(range(len(train_rows)))
            rng.shuffle(order)
            tot_loss, nb = 0.0, 0
            te = time.time()
            for bi in range(0, len(order), BATCH):
                idxs = order[bi:bi + BATCH]
                xs, ys, bs = [], [], []
                for fi in idxs:
                    x, lab, binm, _sd = load_sample(train_rows[fi], True)
                    xs.append(x)
                    ys.append(lab)
                    bs.append(binm)
                xb = torch.stack(xs).to(device)
                yb = torch.stack(ys).to(device)
                bb = torch.stack(bs).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = head(forward_dense(backbone, xb))
                    loss = F.cross_entropy(logits.float(), yb) + dice_loss_from_logits(logits, bb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in opt.param_groups for p in g["params"]], 1.0)
                opt.step()
                sched.step()
                tot_loss += float(loss.item())
                nb += 1
            vd = val_dice(backbone, head)
            ep_s = round(time.time() - te, 1)
            hist.append({"epoch": ep, "train_loss": round(tot_loss / max(nb, 1), 4),
                         "val_dice": round(vd, 4), "epoch_s": ep_s})
            print(f"[{group}] epoch {ep}: loss={tot_loss/max(nb,1):.4f} "
                  f"val_dice={vd:.4f} ({ep_s}s)", flush=True)
            improved = vd > best_dice
            if improved:
                best_dice, bad = vd, 0
            else:
                bad += 1
            save = {"epoch": ep, "best_val_dice": best_dice, "bad_epochs": bad,
                    "history": hist, "head_state": {k: v.detach().cpu().clone()
                                                    for k, v in head.state_dict().items()},
                    "opt_state": opt.state_dict()}
            if finetune:
                save["backbone_state"] = {k: v.detach().cpu().clone()
                                          for k, v in backbone.state_dict().items()}
            torch.save(save, str(ckpt_path))
            vol.commit()
            if improved:
                torch.save(save, str(out_dir / f"best_{group}.pt"))
                vol.commit()
            if bad >= PATIENCE:
                print(f"[{group}] early stop at epoch {ep}", flush=True)
                break
        best = torch.load(str(out_dir / f"best_{group}.pt"), map_location="cpu",
                          weights_only=False)
        return best, hist

    def bbox_iou(pm, gt):
        def box(m):
            ys, xs = np.where(m)
            if len(ys) == 0:
                return None
            return (xs.min(), ys.min(), xs.max(), ys.max())
        pb, gb = box(pm), box(gt)
        if pb is None or gb is None:
            return 0.0
        ix0, iy0 = max(pb[0], gb[0]), max(pb[1], gb[1])
        ix1, iy1 = min(pb[2], gb[2]), min(pb[3], gb[3])
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        pa = (pb[2] - pb[0]) * (pb[3] - pb[1])
        ga = (gb[2] - gb[0]) * (gb[3] - gb[1])
        return inter / (pa + ga - inter + 1e-9)

    @torch.no_grad()
    def eval_rows(backbone, head, rows, set_name, group, render):
        backbone.eval()
        head.eval()
        frames_out = []
        for k in range(0, len(rows), 16):
            chunk = rows[k:k + 16]
            samples = [load_sample(r, False) for r in chunk]
            xs = torch.stack([s[0] for s in samples]).to(device)
            sides = [s[3] for s in samples]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(forward_dense(backbone, xs))
            for j, r in enumerate(chunk):
                side = sides[j]
                up = F.interpolate(logits[j:j + 1].float(), size=(side, side),
                                   mode="bilinear", align_corners=False)
                pred = up.argmax(1)[0].cpu().numpy()
                gt = cropped_gt(r)
                pm = pred != 0
                inter = float((pm & gt).sum())
                union = float((pm | gt).sum())
                iou = inter / (union + 1e-9)
                dice = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
                biou = bbox_iou(pm, gt)
                counts = [(c, int((pred == c).sum())) for c in range(1, NCLS)]
                counts.sort(key=lambda t: -t[1])
                dom = CLS_NAMES[counts[0][0]] if counts[0][1] > 0 else "none"
                frames_out.append({
                    "idx": k + j, "set": set_name, "cls": r["cls"], "split": r["split"],
                    "patient": r["patient"], "clean_bridge": bool(r["patient"] not in train_patients),
                    "iou": round(float(iou), 4), "dice": round(float(dice), 4),
                    "bbox_iou": round(float(biou), 4),
                    "detected": bool(biou >= 0.5), "detected_mask": bool(iou >= 0.5),
                    "pred_name": dom, "named_correct": bool(dom == r["cls"]),
                    "gt_px": int(gt.sum()), "pred_px": int(pm.sum()),
                    "area_ratio": round(float(pm.sum()) / max(float(gt.sum()), 1), 3),
                })
            if (k // 16) % 20 == 0:
                print(f"  eval {group}/{set_name} {min(k+16,len(rows))}/{len(rows)}", flush=True)
        # overlays: best 3 + worst 3 + 6 seeded-random (square 720x720)
        if render:
            rr = random.Random(11)
            idxs = sorted({f["idx"] for f in sorted(frames_out, key=lambda f: -f["dice"])[:3]} |
                          {f["idx"] for f in sorted(frames_out, key=lambda f: f["dice"])[:3]} |
                          set(rr.sample(range(len(frames_out)), min(6, len(frames_out)))))
            for fi in idxs:
                f = frames_out[fi]
                r = rows[fi]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xx, _lb, _bm, side = load_sample(r, False)
                    lg = head(forward_dense(backbone, xx.unsqueeze(0).to(device)))
                up = F.interpolate(lg.float(), size=(side, side),
                                   mode="bilinear", align_corners=False)
                pm = (up.argmax(1)[0].cpu().numpy() != 0)
                gt = cropped_gt(r)
                im = Image.open(ext / r["image"]).convert("RGB")
                _sd, x0 = square_crop_bounds(im.width, im.height)
                im = im.crop((x0, 0, x0 + side, side)).resize((720, 720), Image.BICUBIC)
                ov = np.asarray(im).astype(np.float32)
                gts = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize((720, 720), Image.NEAREST)) > 0
                prs = np.asarray(Image.fromarray((pm.astype(np.uint8)) * 255).resize((720, 720), Image.NEAREST)) > 0
                ov[gts] = ov[gts] * 0.45 + np.array([0, 200, 0]) * 0.55
                only_pred = prs & ~gts
                ov[only_pred] = ov[only_pred] * 0.45 + np.array([220, 0, 0]) * 0.55
                Image.fromarray(ov.astype(np.uint8)).save(
                    viz_dir / f"{group}_{set_name}_{fi:04d}.jpg", quality=88)
                frames_out[fi]["img"] = f"viz_e/{group}_{set_name}_{fi:04d}.jpg"
        return frames_out

    def summarize(frames):
        d = np.array([f["dice"] for f in frames])
        i = np.array([f["iou"] for f in frames])
        b = np.array([f["bbox_iou"] for f in frames])
        by_cls = {}
        for c in CLS_NAMES[1:]:
            sub = [f for f in frames if f["cls"] == c]
            if sub:
                by_cls[c] = {"n": len(sub),
                             "dice": round(float(np.mean([f["dice"] for f in sub])), 4),
                             "iou": round(float(np.mean([f["iou"] for f in sub])), 4),
                             "det": round(float(np.mean([f["detected"] for f in sub])), 3)}
        return {
            "n": len(frames),
            "mean_dice": round(float(d.mean()), 4),
            "median_dice": round(float(np.median(d)), 4),
            "mean_iou": round(float(i.mean()), 4),
            "mean_bbox_iou": round(float(b.mean()), 4),
            "detection_rate_bbox_iou05": round(float(np.mean([f["detected"] for f in frames])), 4),
            "detection_rate_mask_iou05": round(float(np.mean([f["detected_mask"] for f in frames])), 4),
            "frac_dice_ge05": round(float(np.mean(d >= 0.5)), 4),
            "n_zero_dice": int((d == 0).sum()),
            "naming_accuracy": round(float(np.mean([f["named_correct"] for f in frames])), 4),
            "mean_area_ratio": round(float(np.mean([f["area_ratio"] for f in frames])), 3),
            "by_class": by_cls,
        }

    results = {}
    for group in ("control", "finetune"):
        gt0 = time.time()
        best, hist = train_group(group)
        backbone, head = build_model(group == "finetune")
        if group == "finetune":
            backbone.load_state_dict(best["backbone_state"])
        head.load_state_dict(best["head_state"])
        backbone, head = backbone.to(device), head.to(device)
        per_set = {}
        for set_name, rows in [("local_test", local_rows), ("external_test", ext_rows),
                               ("old100", old_rows)]:
            fr = eval_rows(backbone, head, rows, set_name, group, render=True)
            per_set[set_name] = {"summary": summarize(fr), "frames": fr}
        clean = [f for f in per_set["old100"]["frames"] if f["clean_bridge"]]
        per_set["old100_clean22"] = {"summary": summarize(clean), "frames": clean,
                                     "note": "old100 frames whose patient is NOT in the new "
                                             "train set (clean bridge vs probe); full old100 is "
                                             "contaminated for the fine-tuned model (78/100 frames "
                                             "in train)."}
        results[group] = {"best_val_dice": best["best_val_dice"],
                          "train_history": hist,
                          "train_wall_s": round(time.time() - gt0, 1),
                          "sets": per_set}
        with open(out_dir / f"results_{group}.json", "w") as f:
            json.dump(results[group], f, indent=2, ensure_ascii=False)
        vol.commit()
        print(f"[{group}] done best_val={best['best_val_dice']:.4f} "
              f"local={per_set['local_test']['summary']['mean_dice']:.4f} "
              f"ext={per_set['external_test']['summary']['mean_dice']:.4f}", flush=True)

    res["total_s"] = round(time.time() - t0, 1)
    res["groups"] = {g: {"best_val_dice": results[g]["best_val_dice"],
                         "train_wall_s": results[g]["train_wall_s"],
                         "local_dice": results[g]["sets"]["local_test"]["summary"]["mean_dice"],
                         "external_dice": results[g]["sets"]["external_test"]["summary"]["mean_dice"],
                         "old100_dice": results[g]["sets"]["old100"]["summary"]["mean_dice"],
                         "old100_clean22_dice": results[g]["sets"]["old100_clean22"]["summary"]["mean_dice"]}
                    for g in ("control", "finetune")}
    print("STAGE_E_RESULT_JSON=" + json.dumps(res["groups"], ensure_ascii=False))
    return res


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_e.remote()
    print("\n=== STAGE E RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/stage_e_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
