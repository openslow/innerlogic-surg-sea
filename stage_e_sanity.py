"""Stage E sanity check (LOCAL, no Modal, no training):
1. GT fraction outside the center square crop (instrument is inside the
   endoscopic circle; the square crop keeps the whole circle, so the
   fraction should be ~0; if large, STOP and report).
2. Render 4 overlays: square-cropped image + GT (green) at 720x720,
   2 Center 1 (one 1920x1080, one 1280x720) + 2 Center 2 (720x576).
Run: ~/workspace/mirage-smoke/.venv/bin/python stage_e_sanity.py
"""
import io
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import nrrd
from PIL import Image, ImageDraw

BASE = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg")
Z = zipfile.ZipFile(BASE / "sea" / "sea.zip")


def load_mask(zip_path):
    data = Z.read(zip_path)
    open("/tmp/_se.nrrd", "wb").write(data)
    arr, _ = nrrd.read("/tmp/_se.nrrd")
    m = np.asarray(arr)
    if m.ndim == 3:
        m = m[:, :, 0]
    return (m.T > 0)  # (H, W) native, correct orientation


def square_crop_bounds(w, h):
    side = min(w, h)
    x0 = (w - side) // 2
    return side, x0


def main():
    rng = random.Random(7)
    for name in ["train.jsonl", "external_test.jsonl"]:
        rows = [json.loads(l) for l in open(BASE / "manifests_paper" / name)]
        samp = rng.sample(rows, 100)
        fracs, cover = [], []
        for r in samp:
            gt = load_mask(r["mask"])
            h, w = gt.shape
            side, x0 = square_crop_bounds(w, h)
            outside = gt[:, :x0].sum() + gt[:, x0 + side:].sum()
            tot = gt.sum()
            fracs.append(outside / max(tot, 1))
            cover.append(tot / gt.size)
        fracs = np.array(fracs)
        print(f"{name}: n=100 GT-outside-crop frac mean={fracs.mean():.5f} "
              f"max={fracs.max():.5f} n>0.01={(fracs > 0.01).sum()} "
              f"mean GT coverage={np.mean(cover):.3f}")

    # 4 overlays
    out = BASE / "center_compare"
    out.mkdir(exist_ok=True)
    picks = []
    train = [json.loads(l) for l in open(BASE / "manifests_paper" / "train.jsonl")]
    ext = [json.loads(l) for l in open(BASE / "manifests_paper" / "external_test.jsonl")]
    for r in train:
        if r["image"].endswith(".png") and "P1-" in r["image"]:
            picks.append(("C1-1080p", r))
            break
    for r in train:
        if "P45" in r["patient"]:
            picks.append(("C1-720p", r))
            break
    for r in ext:
        if r["patient"] == "T121":
            picks.append(("C2-576", r))
            break
    for r in ext:
        if r["patient"] == "T147":
            picks.append(("C2-576b", r))
            break
    for tag, r in picks:
        im = Image.open(io.BytesIO(Z.read(r["image"]))).convert("RGB")
        gt = load_mask(r["mask"])
        w, h = im.size
        side, x0 = square_crop_bounds(w, h)
        im2 = im.crop((x0, 0, x0 + side, side)).resize((720, 720), Image.BICUBIC)
        g2 = np.asarray(Image.fromarray((gt[:, x0:x0 + side].astype(np.uint8)) * 255)
                        .resize((720, 720), Image.NEAREST)) > 0
        ov = np.asarray(im2).astype(np.float32)
        ov[g2] = ov[g2] * 0.45 + np.array([0, 200, 0]) * 0.55
        canvas = Image.fromarray(ov.astype(np.uint8))
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, 720, 30], fill=(0, 0, 0))
        d.text((10, 8), f"{tag} | {r['cls']} | {r['patient']} | {w}x{h} -> square {side}",
               fill=(255, 255, 255))
        p = out / f"stage_e_sanity_{tag}.jpg"
        canvas.save(p, quality=88)
        print("saved", p)


if __name__ == "__main__":
    main()
