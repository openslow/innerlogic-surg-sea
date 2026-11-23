"""Stage E continuation: 3 more epochs (ep6-ep8) from the Stage E fine-tune
best checkpoint (ep5), to see whether the metrics keep rising or plateau.

Decision record (LR handling) -- this is the one non-obvious choice:
- Stage E's cosine schedule had decayed to ~0 LR by ep5. Resuming the old
  optimizer/scheduler (or naively extending MAX_EPOCHS) would therefore
  train at ~LR 0 and change nothing, silently. So the continuation builds a
  FRESH optimizer + a FRESH short cosine over exactly 3 epochs, peak LR the
  same as Stage E (backbone 1e-5 / head 1e-4). Rationale: it re-opens a small
  learning window at the already-proven-safe peak, and decays to ~0 again,
  giving a clean endpoint. A constant LR was rejected: no natural stopping
  point, and it risks drifting the converged ep5 weights further than needed.
- Only the fine-tune group continues (the user asked whether the fine-tune
  metrics keep rising; continuing the control would double the cost and is
  not the question). Control untouched.
- Start point: volume results_e/best_finetune.pt (assert epoch == 5).
  Nothing in results_e/ is written; all outputs go to results_e_cont/
  and viz_e_cont/. Best selection by val Dice on the same 496-frame val set
  (P26, P27, P31, p46); all 3 epochs run (no early stop inside the window),
  best checkpoint wins even if that is still ep5.
- Everything else identical to Stage E: same paper-aligned split, CE+Dice,
  768x768 square crop, square-preserving augmentation, bf16 (NEVER fp16).
Run: .venv/bin/modal run --detach modal_stage_e_cont.py (nohup-wrapped:
local exec sessions are recycled after ~20 min and kill a piped modal
process -- B3 and Stage E both hit this.)
Expected: ~30 min train + ~5-10 min eval, ~$1.2 metered, billed $0.
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
CLS2ID = {c: i + 1 for i, c in enumerate(CLS_NAMES[1:])}
BATCH = 8
CONT_EPOCHS = 3          # ep6, ep7, ep8
START_EP = 6
LR_BACKBONE = 1e-5       # same peak as Stage E
LR_HEAD = 1e-4           # same peak as Stage E


@app.function(image=image, volumes={"/data": vol}, gpu="L40S",
              timeout=14400, scaledown_window=10)
def stage_e_cont():
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

    res = {"stage": "E-cont", "cont_epochs": CONT_EPOCHS}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    man = base / "manifests_paper"
    out_dir = base / "results_e_cont"
    out_dir.mkdir(exist_ok=True)
    viz_dir = base / "viz_e_cont"
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
        arr, _hdr = nrrd.read(str(path))
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        m = m.T
        return (m > 0)

    def square_crop_bounds(w, h):
        side = min(w, h)
        return side, (w - side) // 2

    def cropped_gt(r):
        gt = load_nrrd_mask(ext / r["mask"])
        h, w = gt.shape
        side, x0 = square_crop_bounds(w, h)
        return gt[:, x0:x0 + side]

    def mask_to_patch_labels(mask_full, cls_name):
        m = mask_full.astype(np.float32)
        gh, gw = IN_S // 16, IN_S // 16
        h, w = m.shape
        m_small = m.reshape(gh, h // gh, gw, w // gw).mean(axis=(1, 3))
        cid = CLS2ID[cls_name]
        lab = np.where(m_small >= 0.5, cid, 0).astype(np.int64)
        return lab, (lab != 0).astype(np.float32)

    def load_sample(r, augment):
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
        p = 1.0 - torch.softmax(logits.float(), dim=1)[:, 0]
        t = bin_target
        inter = (p * t).sum(dim=(1, 2))
        denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
        return (1.0 - (2 * inter + 1e-6) / (denom + 1e-6)).mean()

    def build_model():
        backbone = load_backbone(str(ckpt), device=device)
        head = torch.nn.Conv2d(768, NCLS, 1).to(device)
        backbone.requires_grad_(False)
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

    # ---- start point: Stage E fine-tune best checkpoint (ep5) ----
    src = base / "results_e" / "best_finetune.pt"
    c0 = torch.load(str(src), map_location="cpu", weights_only=False)
    print(f"start ckpt epoch={c0['epoch']} best_val={c0['best_val_dice']:.4f} "
          f"history={c0['history']}", flush=True)
    assert c0["epoch"] == 5, f"expected ep5 start checkpoint, got epoch {c0['epoch']}"
    assert abs(c0["best_val_dice"] - 0.9659) < 0.001, "unexpected start best_val"
    res["start"] = {"epoch": c0["epoch"], "best_val_dice": round(c0["best_val_dice"], 4)}
    hist = list(c0["history"])
    best_dice = c0["best_val_dice"]
    best_ep = 5

    backbone, head = build_model()
    backbone.load_state_dict(c0["backbone_state"])
    head.load_state_dict(c0["head_state"])
    backbone, head = backbone.to(device), head.to(device)

    # Fresh optimizer + fresh short cosine over the 3 continuation epochs.
    # (Old optimizer LR had decayed to ~0; reusing it would train nothing.)
    opt = torch.optim.AdamW(
        [{"params": [p for blk in backbone.blocks[8:] for p in blk.parameters()],
          "lr": LR_BACKBONE},
         {"params": head.parameters(), "lr": LR_HEAD}],
        weight_decay=1e-4)
    steps_per_epoch = (len(train_rows) + BATCH - 1) // BATCH
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=steps_per_epoch * CONT_EPOCHS)

    # Seed the best file with the ep5 state so eval always has a valid best.
    best_save = {"epoch": 5, "best_val_dice": best_dice, "history": hist,
                 "head_state": {k: v.detach().cpu().clone()
                                for k, v in head.state_dict().items()},
                 "backbone_state": {k: v.detach().cpu().clone()
                                    for k, v in backbone.state_dict().items()},
                 "opt_state": opt.state_dict()}
    torch.save(best_save, str(out_dir / "best_finetune_cont.pt"))
    vol.commit()

    cont_hist = []
    for ep in range(START_EP, START_EP + CONT_EPOCHS):
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
        rec = {"epoch": ep, "train_loss": round(tot_loss / max(nb, 1), 4),
               "val_dice": round(vd, 4), "epoch_s": ep_s}
        cont_hist.append(rec)
        hist.append(rec)
        print(f"[cont] epoch {ep}: loss={tot_loss/max(nb,1):.4f} "
              f"val_dice={vd:.4f} ({ep_s}s)", flush=True)
        save = {"epoch": ep, "best_val_dice": best_dice, "history": hist,
                "head_state": {k: v.detach().cpu().clone()
                               for k, v in head.state_dict().items()},
                "backbone_state": {k: v.detach().cpu().clone()
                                   for k, v in backbone.state_dict().items()},
                "opt_state": opt.state_dict()}
        torch.save(save, str(out_dir / "checkpoint_finetune_cont.pt"))
        vol.commit()
        if vd > best_dice:
            best_dice, best_ep = vd, ep
            save["best_val_dice"] = best_dice
            torch.save(save, str(out_dir / "best_finetune_cont.pt"))
            vol.commit()
            print(f"[cont] new best ep{ep} val {vd:.4f}", flush=True)
    res["cont_history"] = cont_hist
    res["best"] = {"epoch": best_ep, "best_val_dice": round(float(best_dice), 4)}
    train_wall = round(time.time() - t0, 1)
    res["train_wall_s"] = train_wall

    # ---- eval the best checkpoint on the three sets + old100 ----
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

    best = torch.load(str(out_dir / "best_finetune_cont.pt"), map_location="cpu",
                      weights_only=False)
    backbone, head = build_model()
    backbone.load_state_dict(best["backbone_state"])
    head.load_state_dict(best["head_state"])
    backbone, head = backbone.to(device), head.to(device)

    @torch.no_grad()
    def eval_rows(rows, set_name):
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
                print(f"  eval {set_name} {min(k+16,len(rows))}/{len(rows)}", flush=True)
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

    # Stage E fine-tune per-frame Dice, for delta-based overlay picks.
    with open(base / "results_e" / "results_finetune.json") as f:
        e_ft = json.load(f)
    e_dice = {}
    for sname in ("local_test", "external_test", "old100"):
        for fr in e_ft["sets"][sname]["frames"]:
            e_dice[(sname, fr["idx"])] = fr["dice"]

    per_set = {}
    for set_name, rows in [("local_test", local_rows), ("external_test", ext_rows),
                           ("old100", old_rows)]:
        fr = eval_rows(rows, set_name)
        per_set[set_name] = {"summary": summarize(fr), "frames": fr}
    clean = [f for f in per_set["old100"]["frames"] if f["clean_bridge"]]
    per_set["old100_clean22"] = {"summary": summarize(clean), "frames": clean,
                                 "note": "old100 frames whose patient is NOT in the new "
                                         "train set (clean bridge vs probe)."}
    res["sets"] = {s: per_set[s]["summary"] for s in per_set}
    res["best_epoch"] = best_ep

    # ---- overlays: top-2 improved + top-2 worsened per set (vs Stage E ft) ----
    @torch.no_grad()
    def render_one(r, set_name, fi, tag):
        xx, _lb, _bm, side = load_sample(r, False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
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
            viz_dir / f"cont_{set_name}_{fi:04d}_{tag}.jpg", quality=88)
        return f"viz_e_cont/cont_{set_name}_{fi:04d}_{tag}.jpg"

    picks = []
    for set_name, rows in [("local_test", local_rows), ("external_test", ext_rows)]:
        fr = per_set[set_name]["frames"]
        scored = [(f["dice"] - e_dice[(set_name, f["idx"])], f) for f in fr]
        scored.sort(key=lambda t: t[0])
        for tag, f in [("worse", s[1]) for s in scored[:2]] + \
                      [("better", s[1]) for s in scored[-2:]]:
            d = f["dice"] - e_dice[(set_name, f["idx"])]
            img = render_one(rows[f["idx"]], set_name, f["idx"], tag)
            f["img"] = img
            picks.append({"set": set_name, "idx": f["idx"], "tag": tag,
                          "dice_ep5": e_dice[(set_name, f["idx"])],
                          "dice_cont": f["dice"], "delta": round(float(d), 4),
                          "img": img})
    res["overlay_picks"] = picks

    with open(out_dir / "results_finetune_cont.json", "w") as f:
        json.dump({"best_val_dice": res["best"]["best_val_dice"],
                   "best_epoch": best_ep,
                   "cont_history": cont_hist,
                   "train_wall_s": train_wall,
                   "sets": per_set,
                   "overlay_picks": picks}, f, indent=2, ensure_ascii=False)
    vol.commit()
    res["total_s"] = round(time.time() - t0, 1)
    print("STAGE_E_CONT_RESULT_JSON=" + json.dumps(
        {"best": res["best"], "sets": res["sets"], "cont_history": cont_hist},
        ensure_ascii=False))
    return res


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_e_cont.remote()
    print("\n=== STAGE E CONT RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/e_cont_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
