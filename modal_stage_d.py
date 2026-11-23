"""Stage D: retrain with CORRECTED labels (user-approved 2026-11-21).

The GT-mask orientation bug (§10) corrupted 17.6% of the training labels and
100% of the external-test GT. User hypothesis: the earlier fine-tune got worse
because the wrong gold standard pulled training in the wrong direction, and
mixed right/wrong labels made the model overfit the correct part. This stage
is the clean test of that hypothesis: SAME recipe as B2, the ONLY variable
changed is the fixed label loader.

Time/cost control (user 2026-11-21 21:54): run ~1 hour first, evaluate, then
decide whether to continue from the resulting checkpoint. So this stage runs
the FINE-TUNE group only (the 0-sample baseline comes from Stage C's control
re-eval; a clean control-head retrain is deferred, not dropped). Max 6
epochs, early stop patience 2: fine-tune epochs run ~530 s each, so 6 epochs
+ eval is ~1 h. Per-epoch checkpoints are saved (resume-safe), so training
can continue from the best/last checkpoint afterwards.
- Fine-tune: last 4 blocks (8-11) + head, backbone LR 1e-5, head LR 1e-4,
  AdamW, cosine decay, warmup 1 epoch, max 6 epochs, early stop on val Dice
  (patience 2).
- Splits: manifests_paper/ (seed 7, paper-aligned: train 8057 / val 496 /
  local_test 1040 / external_test 1069 / old100 100; patient-disjoint).
- Loss CE+Dice, augmentation hflip + random resized crop, input 768x432,
  bf16 autocast (NEVER fp16).
- Fixed loader: transpose FIRST using the array's own shape (nrrd axis order
  is always (W, H, 1)), then NEAREST-resize to (1080, 1920) only if needed.
- Eval per frame, same scoring as stage B, on local_test, external_test,
  old100 + clean22. Results to results_d/, overlays to viz_d/.
- Expected duration ~1 h, ~$2 metered, billed $0 (free credits).
Run: .venv/bin/modal run --detach modal_stage_d.py   (nohup-wrap it: local
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

IN_W, IN_H = 768, 432
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
def stage_b2():
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

    res = {"stage": "D"}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    man = base / "manifests_paper"
    out_dir = base / "results_d"
    out_dir.mkdir(exist_ok=True)
    viz_dir = base / "viz_d"
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

    def load_nrrd_mask(path, w=1920, h=1080):
        """FIXED loader (stage C/D): nrrd axis order is ALWAYS (W, H, 1).
        Transpose FIRST using the array's own shape, then NEAREST-resize to
        (h, w) only if needed. The B2 version only transposed when the shape
        was exactly (1920, 1080) and resized the UNTRANSPOSED array otherwise,
        which transposed + stretched every non-1920x1080 mask."""
        arr, _hdr = nrrd.read(str(path))
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.T  # (W, H) -> (H, W) at native resolution
        if m.shape != (h, w):
            m = np.array(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST))
        return (m > 0)

    def mask_to_patch_labels(mask_full, cls_name):
        """Full-res bool mask (H,W) -> patch grid (27,48) int64 labels + binary mask."""
        m = mask_full.astype(np.float32)
        gh, gw = IN_H // 16, IN_W // 16
        h, w = m.shape
        m_small = m.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))
        cid = CLS2ID[cls_name]
        lab = np.where(m_small >= 0.5, cid, 0).astype(np.int64)
        return lab, (lab != 0).astype(np.float32)

    def load_sample(r, augment):
        """-> (image tensor (3,432,768) normalized, patch labels (27,48), binary patch mask)."""
        im = Image.open(ext / r["image"]).convert("RGB")
        gt = Image.fromarray((load_nrrd_mask(ext / r["mask"]).astype(np.uint8)) * 255)
        if augment:
            if rng.random() < 0.5:
                im, gt = im.transpose(Image.FLIP_LEFT_RIGHT), gt.transpose(Image.FLIP_LEFT_RIGHT)
            s = rng.uniform(0.85, 1.0)
            cw, ch = int(im.width * s), int(im.height * s)
            x0 = rng.randint(0, im.width - cw)
            y0 = rng.randint(0, im.height - ch)
            im = im.crop((x0, y0, x0 + cw, y0 + ch))
            gt = gt.crop((x0, y0, x0 + cw, y0 + ch))
        im = im.resize((IN_W, IN_H), Image.BICUBIC)
        gt = gt.resize((IN_W, IN_H), Image.NEAREST)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s_ = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - m) / s_
        gt_np = np.asarray(gt) > 0
        lab, binm = mask_to_patch_labels(gt_np, r["cls"])
        return x, torch.from_numpy(lab), torch.from_numpy(binm)

    def dice_loss_from_logits(logits, bin_target):
        """logits (B,7,27,48); binary Dice on instrument vs background."""
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
            xs = torch.stack([load_sample(r, False)[0] for r in chunk]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(forward_dense(backbone, xs))
            up = F.interpolate(logits.float(), size=(1080, 1920),
                               mode="bilinear", align_corners=False)
            pred = up.argmax(1).cpu().numpy()
            for j, r in enumerate(chunk):
                gt = load_nrrd_mask(ext / r["mask"])
                pm = pred[j] != 0
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
                    x, lab, binm = load_sample(train_rows[fi], True)
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
            xs = torch.stack([load_sample(r, False)[0] for r in chunk]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = head(forward_dense(backbone, xs))
            up = F.interpolate(logits.float(), size=(1080, 1920),
                               mode="bilinear", align_corners=False)
            pred = up.argmax(1).cpu().numpy()
            for j, r in enumerate(chunk):
                gt = load_nrrd_mask(ext / r["mask"])
                pm = pred[j] != 0
                inter = float((pm & gt).sum())
                union = float((pm | gt).sum())
                iou = inter / (union + 1e-9)
                dice = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
                biou = bbox_iou(pm, gt)
                counts = [(c, int((pred[j] == c).sum())) for c in range(1, NCLS)]
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
        # overlays: best 3 + worst 3 + 6 seeded-random
        if render:
            rr = random.Random(11)
            idxs = sorted({f["idx"] for f in sorted(frames_out, key=lambda f: -f["dice"])[:3]} |
                          {f["idx"] for f in sorted(frames_out, key=lambda f: f["dice"])[:3]} |
                          set(rr.sample(range(len(frames_out)), min(6, len(frames_out)))))
            for fi in idxs:
                f = frames_out[fi]
                r = rows[fi]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    xx = load_sample(r, False)[0].unsqueeze(0).to(device)
                    lg = head(forward_dense(backbone, xx))
                up = F.interpolate(lg.float(), size=(1080, 1920),
                                   mode="bilinear", align_corners=False)
                pm = (up.argmax(1)[0].cpu().numpy() != 0)
                gt = load_nrrd_mask(ext / r["mask"])
                im = Image.open(ext / r["image"]).convert("RGB").resize((960, 540), Image.BICUBIC)
                ov = np.asarray(im).astype(np.float32)
                gts = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize((960, 540), Image.NEAREST)) > 0
                prs = np.asarray(Image.fromarray((pm.astype(np.uint8)) * 255).resize((960, 540), Image.NEAREST)) > 0
                ov[gts] = ov[gts] * 0.45 + np.array([0, 200, 0]) * 0.55
                only_pred = prs & ~gts
                ov[only_pred] = ov[only_pred] * 0.45 + np.array([220, 0, 0]) * 0.55
                Image.fromarray(ov.astype(np.uint8)).save(
                    viz_dir / f"{group}_{set_name}_{fi:04d}.jpg", quality=88)
                frames_out[fi]["img"] = f"viz_d/{group}_{set_name}_{fi:04d}.jpg"
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
            "naming_accuracy": round(float(np.mean([f["named_correct"] for f in frames])), 4),
            "mean_area_ratio": round(float(np.mean([f["area_ratio"] for f in frames])), 3),
            "by_class": by_cls,
        }

    results = {}
    for group in ("finetune",):
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
                    for g in ("finetune",)}
    print("STAGE_D_RESULT_JSON=" + json.dumps(res["groups"], ensure_ascii=False))
    return res


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_b2.remote()
    print("\n=== STAGE D RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/stage_d_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
