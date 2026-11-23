"""Stage B: InnerLogic-Surg ViT-B/16 linear-probe segmentation on SEA.
- Loads ViT-B checkpoint (SHA-256 verified) on L40S, bf16.
- Input: full-frame resize 768x432 (48x27 patches), frozen backbone, last-block dense features.
- Head: single 1x1 linear (768 -> 7 classes: 6 instruments + background), trained on
  sampled patches from patient-disjoint train split; selected on val by full-res Dice.
- Eval: same 100 frames as Mirage pilot. Metrics: bbox-IoU detection (>=0.5),
  mask IoU, Dice (mean/median), per-class, per-scene, naming accuracy, area ratio.
- Renders overlay images (GT green, pred red) at 960px wide for the space.
Run: .venv/bin/modal run modal_stage_b.py
"""
import json
import time

import modal

app = modal.App("innerlogic-surg-sea")
vol = modal.Volume.from_name("innerlogic-surg", create_if_missing=True)

CKPT_URL = ("https://github.com/semaphor-surgical/Innerlogic-Surg/releases/"
            "download/v0.1.0/dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth")
CKPT_SHA256 = "a516a775805b9093f15e5e7f6fa023663a48d81ca211d6552b4fa87be03c8806"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3",
                 "pillow==10.4.0", "pynrrd==1.1.3")
    .add_local_dir("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/repo",
                   "/root/repo")
)

IN_W, IN_H = 768, 432          # full-frame input; patch grid 48x27
NCLS = 7                        # 0=background, 1..6 instruments
CLS_NAMES = ["background", "grasping forceps", "bipolar", "drill",
             "punch", "scissor", "dissector"]
CLS2ID = {c: i + 1 for i, c in enumerate(CLS_NAMES[1:])}
BATCH = 16
TRAIN_EPOCHS = 12
PATCHES_PER_CLASS = 64          # sampled patches per class per frame per epoch


@app.function(image=image, volumes={"/data": vol}, gpu="L40S",
              timeout=3600, scaledown_window=10)
def stage_b():
    import hashlib
    import os
    import random
    import sys
    import urllib.request
    from pathlib import Path

    import numpy as np
    import nrrd
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/root/repo")
    from surgfm_probe import FeatureExtractor, load_backbone
    from surgfm_probe.transforms import IMAGENET_MEAN, IMAGENET_STD

    res = {"stage": "B"}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    ckpt = base / "ckpt" / "dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if not ckpt.exists():
        print("downloading ViT-B checkpoint ...", flush=True)
        urllib.request.urlretrieve(CKPT_URL, str(ckpt))
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    res["ckpt_sha256"] = h.hexdigest()
    res["ckpt_ok"] = h.hexdigest() == CKPT_SHA256
    res["ckpt_bytes"] = int(ckpt.stat().st_size)
    print(f"ckpt sha ok={res['ckpt_ok']} bytes={res['ckpt_bytes']}", flush=True)
    if not res["ckpt_ok"]:
        return res

    device = "cuda"
    backbone = load_backbone(str(ckpt), device=device)
    extractor = FeatureExtractor(backbone, autocast_dtype="bfloat16").to(device)
    print(f"backbone: embed_dim={backbone.embed_dim} patch={backbone.patch_size} "
          f"blocks={backbone.n_blocks}", flush=True)

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def load_rows(name):
        rows = []
        with open(base / "manifests" / name) as f:
            for line in f:
                rows.append(json.loads(line))
        return rows

    train_rows = load_rows("train.jsonl")
    val_rows = load_rows("val.jsonl")
    test_rows = load_rows("test.jsonl")
    res["n_train"] = len(train_rows)
    res["n_val"] = len(val_rows)
    res["n_test"] = len(test_rows)
    print(f"rows train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}", flush=True)

    def load_image_tensor(path):
        im = Image.open(path).convert("RGB").resize((IN_W, IN_H), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
        m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (x - m) / s

    def load_nrrd_mask(path, w=1920, h=1080):
        """Mirage pilot loader: nrrd axis order is (W,H,1), transpose to (H,W);
        resize NEAREST to image size when dimensions differ. -> (H,W) bool."""
        arr, _hdr = nrrd.read(str(path))  # pynrrd returns (data, header)
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        if m.shape == (w, h):
            m = m.T
        elif m.shape != (h, w):
            m = np.array(Image.fromarray(m.astype(np.uint8)).resize((w, h), Image.NEAREST))
        return (m > 0)

    def load_mask_grid(path, cls_name):
        """-> (27, 48) int64 patch labels: instrument class id if >=50% instrument px else 0."""
        m = load_nrrd_mask(path).astype(np.float32)  # (1080, 1920)
        gh, gw = IN_H // 16, IN_W // 16
        m_small = m.reshape(gh, 1080 // gh, gw, 1920 // gw).mean(axis=(1, 3))
        cid = CLS2ID[cls_name]
        return np.where(m_small >= 0.5, cid, 0).astype(np.int64)

    def load_mask_full(path):
        return load_nrrd_mask(path)

    @torch.no_grad()
    def extract_features(rows):
        """-> list of (feat bf16 (768,27,48) np, patch-label (27,48) np or None)."""
        feats = []
        for i in range(0, len(rows), BATCH):
            chunk = rows[i:i + BATCH]
            xs = torch.stack([load_image_tensor(ext / r["image"]) for r in chunk]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dense = extractor.dense(xs)  # (B,768,27,48)
            dense = dense.float().cpu().numpy()
            for j, r in enumerate(chunk):
                lab = load_mask_grid(ext / r["mask"], r["cls"])
                feats.append((dense[j].astype(np.float16), lab))
            if (i // BATCH) % 20 == 0:
                print(f"  extract {i+len(chunk)}/{len(rows)}", flush=True)
        return feats

    print("extracting train features ...", flush=True)
    te = time.time()
    train_feats = extract_features(train_rows)
    res["extract_train_s"] = round(time.time() - te, 1)
    print("extracting val features ...", flush=True)
    val_feats = extract_features(val_rows)
    print("extracting test features ...", flush=True)
    test_feats = extract_features(test_rows)
    res["extract_total_s"] = round(time.time() - te, 1)
    print(f"extraction done in {res['extract_total_s']}s", flush=True)

    # cache features on volume (for audit / re-training without re-extraction)
    feat_dir = base / "features_vitb_768x432"
    feat_dir.mkdir(exist_ok=True)
    for name, rows, fs in [("train", train_rows, train_feats),
                           ("val", val_rows, val_feats),
                           ("test", test_rows, test_feats)]:
        np.savez_compressed(feat_dir / f"{name}.npz",
                            feats=np.stack([f[0] for f in fs]),
                            labels=np.stack([f[1] for f in fs]),
                            meta=np.array([json.dumps(r, ensure_ascii=False) for r in rows]))
    vol.commit()
    print("features cached on volume", flush=True)

    # ---- train 1x1 linear head on sampled patches ----
    head = torch.nn.Conv2d(768, NCLS, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=TRAIN_EPOCHS)
    rng = random.Random(7)
    # per-frame per-class patch index lists (train only)
    train_idx = []
    for f, lab in train_feats:
        d = {}
        for c in range(NCLS):
            ys, xs = np.where(lab == c)
            if len(ys):
                d[c] = list(zip(ys.tolist(), xs.tolist()))
        train_idx.append(d)

    def predict_full(feat_bf16):
        """feat (768,27,48) bf16 -> full-res (1080,1920) int64 class map."""
        with torch.no_grad():
            x = torch.from_numpy(feat_bf16.astype(np.float32)).unsqueeze(0).to(device)
            logits = head(x)  # (1,7,27,48)
            up = F.interpolate(logits, size=(1080, 1920), mode="bilinear", align_corners=False)
            return up.argmax(1)[0].cpu().numpy()

    def val_dice():
        dices = []
        for (f, _lab), r in zip(val_feats, val_rows):
            pred = predict_full(f)
            gt = load_mask_full(ext / r["mask"])
            pm = pred != 0
            inter = float((pm & gt).sum())
            d = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
            dices.append(d)
        return float(np.mean(dices))

    best_dice, best_state, hist = -1.0, None, []
    for ep in range(TRAIN_EPOCHS):
        head.train()
        tot_loss, nb = 0.0, 0
        order = list(range(len(train_feats)))
        rng.shuffle(order)
        for bi in range(0, len(order), BATCH):
            idxs = order[bi:bi + BATCH]
            xs_list, ys_list = [], []
            for fi in idxs:
                f = train_feats[fi][0]
                for c, cells in train_idx[fi].items():
                    if not cells:
                        continue
                    take = rng.sample(cells, min(PATCHES_PER_CLASS, len(cells)))
                    for (y, x) in take:
                        xs_list.append(f[:, y, x])
                        ys_list.append(c)
            if not xs_list:
                continue
            xb = torch.from_numpy(np.stack(xs_list).astype(np.float32)).to(device)
            yb = torch.tensor(ys_list, dtype=torch.long, device=device)
            logits = head.weight.squeeze(-1).squeeze(-1) @ xb.T  # (7, N)
            logits = logits.T + head.bias
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())
            nb += 1
        sched.step()
        vd = val_dice()
        hist.append({"epoch": ep, "train_loss": round(tot_loss / max(nb, 1), 4),
                     "val_dice": round(vd, 4)})
        print(f"epoch {ep}: loss={tot_loss/max(nb,1):.4f} val_dice={vd:.4f}", flush=True)
        if vd > best_dice:
            best_dice = vd
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    res["train_history"] = hist
    res["best_val_dice"] = round(best_dice, 4)
    head.load_state_dict(best_state)
    torch.save({"head": best_state, "cls_names": CLS_NAMES, "input": [IN_W, IN_H]},
               str(base / "probe_head_vitb.pt"))
    print(f"best val Dice={best_dice:.4f}", flush=True)

    # ---- eval on 100 test frames ----
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

    frames_out = []
    viz_dir = base / "viz_vitb"
    viz_dir.mkdir(exist_ok=True)
    for k, ((f, _lab), r) in enumerate(zip(test_feats, test_rows)):
        pred = predict_full(f)
        gt = load_mask_full(ext / r["mask"])
        pm = pred != 0
        inter = float((pm & gt).sum())
        union = float((pm | gt).sum())
        iou = inter / (union + 1e-9)
        dice = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
        biou = bbox_iou(pm, gt)
        # naming: dominant predicted instrument class by pixel count
        counts = [(c, int((pred == c).sum())) for c in range(1, NCLS)]
        counts.sort(key=lambda t: -t[1])
        dom_cls = CLS_NAMES[counts[0][0]] if counts[0][1] > 0 else "none"
        frames_out.append({
            "idx": k, "cls": r["cls"], "split": r["split"], "patient": r["patient"],
            "iou": round(float(iou), 4), "dice": round(float(dice), 4),
            "bbox_iou": round(float(biou), 4),
            "detected": bool(biou >= 0.5),
            "detected_mask": bool(iou >= 0.5),
            "pred_name": dom_cls, "named_correct": bool(dom_cls == r["cls"]),
            "gt_px": int(gt.sum()), "pred_px": int(pm.sum()),
            "area_ratio": round(float(pm.sum()) / max(float(gt.sum()), 1), 3),
            "img": f"viz/{k:03d}.jpg",
        })
        # overlay render at 960px wide
        im = Image.open(ext / r["image"]).convert("RGB")
        im = im.resize((960, 540), Image.BICUBIC)
        ov = np.asarray(im).astype(np.float32)
        gts = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize((960, 540), Image.NEAREST)) > 0
        prs = np.asarray(Image.fromarray((pm.astype(np.uint8)) * 255).resize((960, 540), Image.NEAREST)) > 0
        ov[gts] = ov[gts] * 0.45 + np.array([0, 200, 0]) * 0.55
        only_pred = prs & ~gts
        ov[only_pred] = ov[only_pred] * 0.45 + np.array([220, 0, 0]) * 0.55
        Image.fromarray(ov.astype(np.uint8)).save(viz_dir / f"{k:03d}.jpg", quality=88)
        if (k + 1) % 25 == 0:
            print(f"  eval {k+1}/100", flush=True)

    import collections
    dices = np.array([f["dice"] for f in frames_out])
    ious = np.array([f["iou"] for f in frames_out])
    bious = np.array([f["bbox_iou"] for f in frames_out])
    summary = {
        "n": len(frames_out),
        "detection_rate_bbox_iou05": round(float(np.mean([f["detected"] for f in frames_out])), 4),
        "detection_rate_mask_iou05": round(float(np.mean([f["detected_mask"] for f in frames_out])), 4),
        "mean_best_iou_mask": round(float(ious.mean()), 4),
        "mean_bbox_iou": round(float(bious.mean()), 4),
        "mean_dice": round(float(dices.mean()), 4),
        "median_dice": round(float(np.median(dices)), 4),
        "frac_dice_ge05": round(float(np.mean(dices >= 0.5)), 4),
        "frac_dice_lt02": round(float(np.mean(dices < 0.2)), 4),
        "naming_accuracy": round(float(np.mean([f["named_correct"] for f in frames_out])), 4),
        "mean_area_ratio": round(float(np.mean([f["area_ratio"] for f in frames_out])), 3),
    }
    by_cls = {}
    for c in CLS_NAMES[1:]:
        sub = [f for f in frames_out if f["cls"] == c]
        by_cls[c] = {"n": len(sub),
                     "det": round(float(np.mean([f["detected"] for f in sub])), 3),
                     "iou": round(float(np.mean([f["iou"] for f in sub])), 3),
                     "dice": round(float(np.mean([f["dice"] for f in sub])), 3),
                     "named": round(float(np.mean([f["named_correct"] for f in sub])), 3)}
    by_scene = {}
    for s in ["Difficult scenario", "Normal scenario"]:
        sub = [f for f in frames_out if f["split"] == s]
        by_scene[s] = {"n": len(sub),
                       "det": round(float(np.mean([f["detected"] for f in sub])), 3),
                       "iou": round(float(np.mean([f["iou"] for f in sub])), 3),
                       "dice": round(float(np.mean([f["dice"] for f in sub])), 3)}
    res["summary"] = summary
    res["by_class"] = by_cls
    res["by_scene"] = by_scene
    res["frames"] = frames_out
    res["total_s"] = round(time.time() - t0, 1)
    with open(base / "results_vitb.json", "w") as f:
        json.dump({"summary": summary, "by_class": by_cls, "by_scene": by_scene,
                   "frames": frames_out, "train_history": hist,
                   "best_val_dice": res["best_val_dice"]}, f, indent=2, ensure_ascii=False)
    vol.commit()
    print("STAGE_B_RESULT_JSON=" + json.dumps(
        {"summary": summary, "by_class": by_cls, "by_scene": by_scene,
         "best_val_dice": res["best_val_dice"], "total_s": res["total_s"]},
        ensure_ascii=False))
    return {"summary": summary, "by_class": by_cls, "by_scene": by_scene,
            "best_val_dice": float(best_dice), "total_s": float(res["total_s"]),
            "ckpt_ok": bool(res["ckpt_ok"]),
            "n_train": int(len(train_rows)), "n_val": int(len(val_rows)),
            "train_history": hist}


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_b.remote()
    print("\n=== STAGE B RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/stage_b_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
