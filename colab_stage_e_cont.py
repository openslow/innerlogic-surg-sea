"""Colab version of Stage E continuation (modal_stage_e_cont2.py with the
Modal parts stripped). Runs on a Colab L4 session.

Modes:
  --mode eval   : load the ep10 best checkpoint, eval the first --eval-n
                  rows of --eval-set (default external_test, 100) and compare
                  with the Modal-recorded ep10 numbers. No training.
  --mode train  : continue fine-tuning from the ep10 checkpoint.
                  --train-steps N runs only N optimizer steps (smoke test:
                  forward/backward/optimizer/checkpoint save + timing/VRAM);
                  without --train-steps it runs full epochs (--epochs,
                  default 5, early stop patience 3) exactly like the Modal
                  continuation rounds, saving to --out-dir. After every
                  epoch it writes results_e_cont3/progress.json (epoch,
                  val_dice, train_loss, best epoch, status) so a local
                  poller can mirror it without waiting for the run to end.
  --mode smoke  : eval 100 external frames + 30 training steps, both.

Paths (Colab session):
  /content/sea/sea.zip            dataset zip (figshare, verified size)
  /content/sea/extracted/         selectively extracted frames+masks
  /content/ils/manifests_paper/   paper-aligned split (seed 7)
  /content/ils/repo/              InnerLogic-Surg official code (surgfm_probe)
  /content/ils/best_finetune_ep10.pt
  /content/ils/dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth

Start checkpoint: --start-ckpt (default best_finetune_ep10.pt), with
--start-epoch-expect / --start-best-val-expect assertions (default
ep10 / 0.9676). A resume from a later best (e.g. a downloaded
best_finetune_epXX.pt) passes those three flags explicitly.

Recipe (identical to Stage E / cont rounds): square centre crop 768x768,
CE+Dice, bf16 (NEVER fp16), backbone LR 1e-5 / head LR 1e-4, fresh AdamW +
fresh cosine over the continuation window, unfreeze blocks 8-11 + 1x1 head,
val set = same 496 frames (P26, P27, P31, p46), best by val Dice.

L4 note: 23.7 GB VRAM vs L40S 48 GB on Modal. Default BATCH=8 (same as
Modal). If it OOMs, rerun with --batch 4 --accum 2 (same effective batch);
the smoke mode retries this automatically once and records which config
was actually used.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import nrrd
import torch
import torch.nn.functional as F
from PIL import Image

BASE = Path("/content/ils")
SEA = Path("/content/sea")
EXT = SEA / "extracted"
MAN = BASE / "manifests_paper"
REPO = BASE / "repo"
CKPT_PRE = BASE / "dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth"
CKPT_PRE_SHA256 = "a516a775805b9093f15e5e7f6fa023663a48d81ca211d6552b4fa87be03c8806"


def load_start_ckpt(path, expect_epoch, expect_best_val):
    c0 = torch.load(str(path), map_location="cpu", weights_only=False)
    print(f"start ckpt epoch={c0['epoch']} best_val={c0['best_val_dice']:.4f}",
          flush=True)
    assert c0["epoch"] == expect_epoch, (
        f"expected ep{expect_epoch} start checkpoint, got {c0['epoch']}")
    assert abs(c0["best_val_dice"] - expect_best_val) < 0.001, (
        "unexpected start best_val")
    return c0

sys.path.insert(0, str(REPO))
from surgfm_probe import load_backbone
from surgfm_probe.transforms import IMAGENET_MEAN, IMAGENET_STD

IN_S = 768
NCLS = 7
CLS_NAMES = ["background", "grasping forceps", "bipolar", "drill",
             "punch", "scissor", "dissector"]
CLS2ID = {c: i + 1 for i, c in enumerate(CLS_NAMES[1:])}
LR_BACKBONE = 1e-5
LR_HEAD = 1e-4


def load_rows(name):
    with open(MAN / name) as f:
        return [json.loads(line) for line in f]


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
    gt = load_nrrd_mask(EXT / r["mask"])
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


def make_loader(rng):
    def load_sample(r, augment):
        im = Image.open(EXT / r["image"]).convert("RGB")
        gt = Image.fromarray((load_nrrd_mask(EXT / r["mask"]).astype(np.uint8)) * 255)
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
    return load_sample


def dice_loss_from_logits(logits, bin_target):
    p = 1.0 - torch.softmax(logits.float(), dim=1)[:, 0]
    t = bin_target
    inter = (p * t).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
    return (1.0 - (2 * inter + 1e-6) / (denom + 1e-6)).mean()


def build_model(device):
    backbone = load_backbone(str(CKPT_PRE), device=device)
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
def eval_rows(backbone, head, rows, set_name, load_sample, device, limit=None):
    backbone.eval()
    head.eval()
    rows = rows[:limit] if limit else rows
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
                "idx": k + j, "set": set_name, "cls": r["cls"],
                "patient": r["patient"],
                "iou": round(float(iou), 4), "dice": round(float(dice), 4),
                "bbox_iou": round(float(biou), 4),
                "detected": bool(biou >= 0.5), "detected_mask": bool(iou >= 0.5),
                "pred_name": dom, "named_correct": bool(dom == r["cls"]),
                "gt_px": int(gt.sum()), "pred_px": int(pm.sum()),
                "area_ratio": round(float(pm.sum()) / max(float(gt.sum()), 1), 3),
            })
    return frames_out


@torch.no_grad()
def val_dice(backbone, head, val_rows, load_sample, device):
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


def check_pretrain_sha():
    h = hashlib.sha256()
    with open(CKPT_PRE, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    ok = h.hexdigest() == CKPT_PRE_SHA256
    print(f"pretrain sha ok={ok}", flush=True)
    return ok


def train_steps(backbone, head, train_rows, load_sample, device, opt, sched,
                batch, accum, n_opt_steps, rng, t_report=None):
    """Run n_opt_steps optimizer steps; returns (seconds, peak_vram_GB,
    mean_loss, micro_steps_done)."""
    backbone.train()
    head.train()
    torch.cuda.reset_peak_memory_stats(device)
    order = list(range(len(train_rows)))
    rng.shuffle(order)
    oi = 0  # index into order
    tot_loss, nb = 0.0, 0
    t0 = time.time()
    step = 0
    micro = 0
    while step < n_opt_steps:
        opt.zero_grad()
        acc_loss = 0.0
        for _ in range(accum):
            idxs = []
            for _ in range(batch):
                idxs.append(order[oi % len(order)])
                oi += 1
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
                loss = (F.cross_entropy(logits.float(), yb)
                        + dice_loss_from_logits(logits, bb)) / accum
            loss.backward()
            acc_loss += float(loss.item()) * accum
            micro += 1
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"]], 1.0)
        opt.step()
        sched.step()
        tot_loss += acc_loss
        nb += 1
        step += 1
        if t_report and step % t_report == 0:
            print(f"  step {step}/{n_opt_steps} loss={acc_loss:.4f} "
                  f"elapsed={time.time()-t0:.1f}s", flush=True)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    return dt, peak, tot_loss / max(nb, 1), micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "train", "smoke"], default="smoke")
    ap.add_argument("--eval-set", default="external_test")
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--train-steps", type=int, default=0,
                    help="smoke: run only this many optimizer steps (0 = full epochs)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stop patience in epochs (val not beating best)")
    ap.add_argument("--start-ep", type=int, default=11)
    ap.add_argument("--start-ckpt", default="best_finetune_ep10.pt",
                    help="start checkpoint filename under /content/ils")
    ap.add_argument("--start-epoch-expect", type=int, default=10)
    ap.add_argument("--start-best-val-expect", type=float, default=0.9676)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--out-dir", default="results_e_cont3")
    ap.add_argument("--smoke-steps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(7)
    rng = random.Random(7)
    load_sample = make_loader(rng)
    t00 = time.time()

    start_ckpt = BASE / args.start_ckpt
    assert CKPT_PRE.exists(), f"missing {CKPT_PRE}"
    assert start_ckpt.exists(), f"missing {start_ckpt}"
    assert check_pretrain_sha(), "pretrain checkpoint SHA mismatch"
    c0 = load_start_ckpt(start_ckpt, args.start_epoch_expect,
                         args.start_best_val_expect)

    train_rows = load_rows("train.jsonl")
    val_rows = load_rows("val.jsonl")
    local_rows = load_rows("local_test.jsonl")
    ext_rows = load_rows("external_test.jsonl")
    old_rows = load_rows("old100.jsonl")
    print("rows:", {"train": len(train_rows), "val": len(val_rows),
                    "local_test": len(local_rows),
                    "external_test": len(ext_rows), "old100": len(old_rows)},
          flush=True)
    print(f"gpu: {torch.cuda.get_device_name(0)} "
          f"vram={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB",
          flush=True)

    backbone, head = build_model(device)
    backbone.load_state_dict(c0["backbone_state"])
    head.load_state_dict(c0["head_state"])
    backbone, head = backbone.to(device), head.to(device)

    report = {"mode": args.mode, "batch": args.batch, "accum": args.accum,
              "start_epoch": c0["epoch"], "start_best_val": c0["best_val_dice"]}

    if args.mode in ("eval", "smoke"):
        rows = {"local_test": local_rows, "external_test": ext_rows,
                "old100": old_rows}[args.eval_set]
        t0 = time.time()
        fr = eval_rows(backbone, head, rows, args.eval_set, load_sample,
                       device, limit=args.eval_n)
        md = float(np.mean([f["dice"] for f in fr]))
        report["eval"] = {"set": args.eval_set, "n": len(fr),
                          "mean_dice": round(md, 4),
                          "eval_s": round(time.time() - t0, 1),
                          "frames": fr}
        print(f"[eval] {args.eval_set} first {len(fr)} frames "
              f"mean Dice={md:.4f} ({time.time()-t0:.1f}s)", flush=True)

    if args.mode in ("train", "smoke"):
        n_steps = args.train_steps or args.smoke_steps
        opt_steps_per_epoch = -(-len(train_rows) // (args.batch * args.accum))
        opt = torch.optim.AdamW(
            [{"params": [p for blk in backbone.blocks[8:] for p in blk.parameters()],
              "lr": LR_BACKBONE},
             {"params": head.parameters(), "lr": LR_HEAD}],
            weight_decay=1e-4)
        total_opt_steps = (opt_steps_per_epoch * args.epochs
                           if not args.train_steps and args.mode == "train"
                           else max(n_steps, 1))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_opt_steps)
        try:
            dt, peak, mloss, micro = train_steps(
                backbone, head, train_rows, load_sample, device, opt, sched,
                args.batch, args.accum, n_steps, rng, t_report=10)
            used = {"batch": args.batch, "accum": args.accum}
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[train] OOM at batch={args.batch}; retrying batch="
                  f"{args.batch // 2} accum={args.accum * 2}", flush=True)
            args.batch //= 2
            args.accum *= 2
            opt_steps_per_epoch = -(-len(train_rows) // (args.batch * args.accum))
            opt = torch.optim.AdamW(
                [{"params": [p for blk in backbone.blocks[8:] for p in blk.parameters()],
                  "lr": LR_BACKBONE},
                 {"params": head.parameters(), "lr": LR_HEAD}],
                weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(n_steps, 1))
            dt, peak, mloss, micro = train_steps(
                backbone, head, train_rows, load_sample, device, opt, sched,
                args.batch, args.accum, n_steps, rng, t_report=10)
            used = {"batch": args.batch, "accum": args.accum,
                    "oom_fallback": True}
        est_epoch_s = dt / n_steps * opt_steps_per_epoch
        report["train_smoke"] = {
            "opt_steps": n_steps, "micro_batches": micro,
            "seconds": round(dt, 1), "sec_per_opt_step": round(dt / n_steps, 2),
            "peak_vram_GB": round(peak, 2),
            "mean_loss": round(mloss, 4),
            "est_epoch_s": round(est_epoch_s, 1),
            "est_epoch_min": round(est_epoch_s / 60, 1),
            "opt_steps_per_epoch": opt_steps_per_epoch,
            "effective_batch": used["batch"] * used["accum"],
            **used}
        print(f"[train-smoke] {n_steps} opt steps in {dt:.1f}s "
              f"({dt/n_steps:.2f}s/step), peak VRAM {peak:.2f}GB, "
              f"mean loss {mloss:.4f}, est epoch {est_epoch_s/60:.1f} min",
              flush=True)
        # quick val to confirm the pipeline end to end
        vd = val_dice(backbone, head, val_rows, load_sample, device)
        report["train_smoke"]["val_dice_after_steps"] = round(vd, 4)
        print(f"[train-smoke] val Dice after {n_steps} steps: {vd:.4f}",
              flush=True)
        # save + reload a checkpoint to prove the save path works
        ck = BASE / "smoke_ckpt.pt"
        torch.save({"epoch": c0["epoch"], "best_val_dice": c0["best_val_dice"],
                    "head_state": {k: v.detach().cpu().clone()
                                   for k, v in head.state_dict().items()},
                    "backbone_state": {k: v.detach().cpu().clone()
                                       for k, v in backbone.state_dict().items()}},
                   str(ck))
        back = torch.load(str(ck), map_location="cpu", weights_only=False)
        assert back["epoch"] == c0["epoch"]
        report["train_smoke"]["ckpt_save_reload_ok"] = True
        report["train_smoke"]["ckpt_path"] = str(ck)
        print(f"[train-smoke] checkpoint saved + reloaded OK: {ck}", flush=True)

    if args.mode == "train" and not args.train_steps:
        # full continuation run: epochs from --start-ep, early stop
        # patience args.patience (default 3)
        out_dir = BASE / args.out_dir
        out_dir.mkdir(exist_ok=True)

        def write_progress(status, cont_hist, best_ep, best_dice,
                           no_improve, early_stopped, sets=None):
            prog = {"status": status, "start_epoch": c0["epoch"],
                    "start_best_val": c0["best_val_dice"],
                    "best_epoch": best_ep,
                    "best_val_dice": round(float(best_dice), 4),
                    "no_improve": no_improve,
                    "patience": args.patience,
                    "max_epoch": args.start_ep + args.epochs - 1,
                    "history": cont_hist,
                    "updated_at_utc": time.strftime("%Y-%m-%d %H:%M:%S",
                                                    time.gmtime()),
                    "early_stopped": early_stopped}
            if sets is not None:
                prog["sets"] = sets
            tmp = out_dir / "progress.json.tmp"
            with open(tmp, "w") as pf:
                json.dump(prog, pf, indent=2, ensure_ascii=False)
            os.replace(tmp, out_dir / "progress.json")

        hist = list(c0["history"])
        best_dice = c0["best_val_dice"]
        best_ep = c0["epoch"]
        opt = torch.optim.AdamW(
            [{"params": [p for blk in backbone.blocks[8:] for p in blk.parameters()],
              "lr": LR_BACKBONE},
             {"params": head.parameters(), "lr": LR_HEAD}],
            weight_decay=1e-4)
        opt_steps_per_epoch = -(-len(train_rows) // (args.batch * args.accum))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=opt_steps_per_epoch * args.epochs)
        best_save = {"epoch": best_ep, "best_val_dice": best_dice,
                     "history": hist,
                     "head_state": {k: v.detach().cpu().clone()
                                    for k, v in head.state_dict().items()},
                     "backbone_state": {k: v.detach().cpu().clone()
                                        for k, v in backbone.state_dict().items()},
                     "opt_state": opt.state_dict()}
        torch.save(best_save, str(out_dir / "best_finetune_cont.pt"))
        cont_hist = []
        no_improve = 0
        early_stopped = False
        for ep in range(args.start_ep, args.start_ep + args.epochs):
            dt, peak, mloss, _micro = train_steps(
                backbone, head, train_rows, load_sample, device, opt, sched,
                args.batch, args.accum, opt_steps_per_epoch, rng)
            vd = val_dice(backbone, head, val_rows, load_sample, device)
            rec = {"epoch": ep, "train_loss": round(mloss, 4),
                   "val_dice": round(vd, 4), "epoch_s": round(dt, 1)}
            cont_hist.append(rec)
            hist.append(rec)
            print(f"[cont] epoch {ep}: loss={mloss:.4f} val_dice={vd:.4f} "
                  f"({dt:.1f}s)", flush=True)
            save = {"epoch": ep, "best_val_dice": best_dice, "history": hist,
                    "head_state": {k: v.detach().cpu().clone()
                                   for k, v in head.state_dict().items()},
                    "backbone_state": {k: v.detach().cpu().clone()
                                       for k, v in backbone.state_dict().items()},
                    "opt_state": opt.state_dict()}
            torch.save(save, str(out_dir / "checkpoint_finetune_cont.pt"))
            if vd > best_dice:
                best_dice, best_ep = vd, ep
                no_improve = 0
                save["best_val_dice"] = best_dice
                torch.save(save, str(out_dir / "best_finetune_cont.pt"))
                print(f"[cont] new best ep{ep} val {vd:.4f}", flush=True)
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    early_stopped = True
                    print(f"[cont] early stop at ep{ep} "
                          f"(patience {args.patience})", flush=True)
                    write_progress("running", cont_hist, best_ep, best_dice,
                                   no_improve, early_stopped)
                    break
            write_progress("running", cont_hist, best_ep, best_dice,
                           no_improve, early_stopped)
        write_progress("evaluating", cont_hist, best_ep, best_dice,
                       no_improve, early_stopped)
        report["cont_history"] = cont_hist
        report["best"] = {"epoch": best_ep, "best_val_dice": round(float(best_dice), 4)}
        # full eval of the best checkpoint on the three sets + old100
        best = torch.load(str(out_dir / "best_finetune_cont.pt"),
                          map_location="cpu", weights_only=False)
        backbone, head = build_model(device)
        backbone.load_state_dict(best["backbone_state"])
        head.load_state_dict(best["head_state"])
        backbone, head = backbone.to(device), head.to(device)
        per_set = {}
        for set_name, rows in [("local_test", local_rows),
                               ("external_test", ext_rows),
                               ("old100", old_rows)]:
            fr = eval_rows(backbone, head, rows, set_name, load_sample, device)
            per_set[set_name] = {
                "n": len(fr),
                "mean_dice": round(float(np.mean([f["dice"] for f in fr])), 4),
                "frames": fr}
        clean = [f for f in per_set["old100"]["frames"]
                 if f["patient"] not in {r["patient"] for r in train_rows}]
        per_set["old100_clean22"] = {
            "n": len(clean),
            "mean_dice": round(float(np.mean([f["dice"] for f in clean])), 4),
            "frames": clean}
        report["sets"] = {s: {"n": per_set[s]["n"],
                              "mean_dice": per_set[s]["mean_dice"]}
                          for s in per_set}
        with open(out_dir / "results_finetune_cont3.json", "w") as f:
            json.dump({"best_val_dice": report["best"]["best_val_dice"],
                       "best_epoch": best_ep, "cont_history": cont_hist,
                       "sets": per_set}, f, indent=2, ensure_ascii=False)
        print("FULL_EVAL=" + json.dumps(report["sets"], ensure_ascii=False),
              flush=True)
        write_progress("done", cont_hist, best_ep, best_dice,
                       no_improve, early_stopped, sets=report["sets"])

    report["total_s"] = round(time.time() - t00, 1)
    print("COLAB_RESULT_JSON=" + json.dumps(report, ensure_ascii=False))
    with open(BASE / "colab_smoke_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
