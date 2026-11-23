"""YOLOv11 zero-shot on SEA (Stage Y) — Colab L4 run script.

Run from the extracted yolo_job/ dir:  python3 yolo_run.py
Layout expected next to this file:
  repo/            (yolo_repo: inference_demo.py, weights/, test_images/, data/)
  sample/          (sample/{local,external}/{images,masks}/, masks as PNG)
  sample_list.json

Geometry (explicit, GT-bug lesson): Ultralytics letterboxes the input to
imgsz=640 (pad, NO stretch) and returns masks in ORIGINAL image coordinates
(result.masks.data is (n, H, W) of the input image — asserted below).
GT is the native .nrrd mask converted with the FIXED loader (transpose first,
then no resize) to PNG. Dice/IoU are computed in NATIVE image coordinates.
No resize, no stretch anywhere in scoring.
"""
import argparse
import importlib.machinery
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def pip_install():
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics", "cryptography"]
    )


def load_loader_module(repo_dir: Path):
    pyc = repo_dir / "weights" / "loader.pyc"
    if not pyc.exists():
        raise FileNotFoundError(f"loader.pyc not found: {pyc}")
    loader = importlib.machinery.SourcelessFileLoader("weights.loader", str(pyc))
    spec = importlib.util.spec_from_loader("weights.loader", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def dice_iou(pred: np.ndarray, gt: np.ndarray):
    inter = int((pred & gt).sum())
    ps, gs = int(pred.sum()), int(gt.sum())
    dice = 2 * inter / (ps + gs) if (ps + gs) > 0 else 0.0
    union = ps + gs - inter
    iou = inter / union if union > 0 else 0.0
    return dice, iou


def run_variant(model, frames, instrument_idx, names, conf=0.25, score=True):
    """frames: list of dicts with img_path, gt (bool array or None).
    Returns (per_frame list, mean_ms, median_ms)."""
    out = []
    times = []
    for i, fr in enumerate(frames):
        im = Image.open(fr["img_path"]).convert("RGB")
        W, H = im.size
        t0 = time.perf_counter()
        res = model.predict(source=str(fr["img_path"]), conf=conf, verbose=False)[0]
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        if i == 0:
            # geometry assertion: masks must come back in native coords
            print("  [geom] orig_shape:", res.orig_shape, "img:", (H, W),
                  "masks:", None if res.masks is None else tuple(res.masks.data.shape),
                  "boxes:", None if res.boxes is None else tuple(res.boxes.data.shape))
        rec = {"idx": fr["idx"], "tag": fr["tag"], "cls": fr["cls"],
               "patient": fr["patient"], "W": W, "H": H,
               "infer_ms": round(times[-1], 2)}
        # all-class prediction counts (qualitative)
        counts = {}
        if res.boxes is not None and len(res.boxes) > 0:
            for c in res.boxes.cls.tolist():
                counts[names[int(c)]] = counts.get(names[int(c)], 0) + 1
        rec["pred_classes"] = counts
        if score and fr.get("gt") is not None:
            gt = fr["gt"]
            pred = np.zeros(gt.shape, dtype=bool)
            if res.masks is not None and len(res.masks) > 0:
                m = res.masks.data.cpu().numpy()  # (n, H, W) native coords
                assert m.shape[1:] == gt.shape, f"mask shape {m.shape} != gt {gt.shape}"
                cls_ids = res.boxes.cls.cpu().numpy().astype(int)
                sel = m[cls_ids == instrument_idx]
                if len(sel) > 0:
                    pred = sel.max(axis=0) > 0.5
            d, iou = dice_iou(pred, gt)
            rec.update({"dice": round(d, 4), "iou": round(iou, 4),
                        "gt_px": int(gt.sum()), "pred_px": int(pred.sum()),
                        "area_ratio": round(int(pred.sum()) / max(int(gt.sum()), 1), 3),
                        "detected": bool(pred.any())})
        out.append(rec)
    ms = np.array(times)
    return out, float(ms.mean()), float(np.median(ms))


def render_overlay(img_path, gt, pred, label, out_path, size=(960, 540)):
    im = Image.open(img_path).convert("RGB").resize(size, Image.BICUBIC)
    g = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize(size, Image.NEAREST)) > 0
    p = np.asarray(Image.fromarray((pred.astype(np.uint8)) * 255).resize(size, Image.NEAREST)) > 0
    ov = im.copy()
    green = Image.new("RGB", size, (0, 200, 0))
    red = Image.new("RGB", size, (220, 0, 0))
    ov.paste(green, (0, 0), Image.fromarray(((g & p).astype(np.uint8)) * 110))
    ov.paste(red, (0, 0), Image.fromarray(((p & ~g).astype(np.uint8)) * 170))
    d = ImageDraw.Draw(ov)
    d.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
    d.text((8, 6), label, fill=(255, 255, 255))
    ov.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    root = Path(args.root)
    repo = root / "repo"
    sample = root / "sample"
    out_dir = root / "results_yolo"
    viz_dir = root / "viz_yolo"
    out_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    pip_install()
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
          "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("gpu mem GB:", round(p.total_memory / 1e9, 1))

    loader_mod = load_loader_module(repo)
    listing = json.loads((root / "sample_list.json").read_text())

    frames = []
    for tag in ["local", "external"]:
        for r in listing[tag]:
            img_path = sample / tag / "images" / r["img_file"]
            msk_path = sample / tag / "masks" / r["mask_file"].replace(".nrrd", ".png")
            gt = np.asarray(Image.open(msk_path).convert("L")) > 0
            frames.append({"tag": tag, "idx": r["sample_idx"], "cls": r["cls"],
                           "patient": r["patient"], "img_path": img_path, "gt": gt})
    print("frames:", len(frames))

    # ---- main test: variant l ----
    print("[l] loading ...")
    model_l = loader_mod.load_yolov11_model(variant="l", device="cuda")
    names = model_l.names
    print("[l] class names:", names)
    inst_idx = [i for i, n in names.items() if n.lower().startswith("instrument")]
    assert len(inst_idx) == 1, f"instrument class not unique: {names}"
    inst_idx = inst_idx[0]
    print(f"[l] instrument class idx={inst_idx} name={names[inst_idx]}")
    print("[l] running 200 frames ...")
    t0 = time.perf_counter()
    recs, mean_ms, med_ms = run_variant(model_l, frames, inst_idx, names, conf=args.conf)
    wall = time.perf_counter() - t0
    print(f"[l] done: mean {mean_ms:.1f} ms/frame, median {med_ms:.1f} ms, "
          f"wall {wall:.1f}s for {len(frames)} frames")

    # ---- speed only: n and s ----
    speed = {}
    for v in ["n", "s"]:
        print(f"[{v}] loading ...")
        mv = loader_mod.load_yolov11_model(variant=v, device="cuda")
        print(f"[{v}] timing 200 frames ...")
        _, m_ms, md_ms = run_variant(mv, frames, inst_idx, names, conf=args.conf, score=False)
        speed[v] = {"mean_ms": round(m_ms, 2), "median_ms": round(md_ms, 2),
                    "fps_mean": round(1000.0 / m_ms, 1)}
        print(f"[{v}] mean {m_ms:.2f} ms/frame ({1000.0/m_ms:.1f} fps)")
        del mv
    speed["l"] = {"mean_ms": round(mean_ms, 2), "median_ms": round(med_ms, 2),
                  "fps_mean": round(1000.0 / mean_ms, 1)}

    # ---- summary ----
    def summ(recs, tag):
        d = [r["dice"] for r in recs if r["tag"] == tag]
        i = [r["iou"] for r in recs if r["tag"] == tag]
        det = [r["detected"] for r in recs if r["tag"] == tag]
        return {"n": len(d), "dice_mean": round(float(np.mean(d)), 4),
                "dice_median": round(float(np.median(d)), 4),
                "iou_mean": round(float(np.mean(i)), 4),
                "iou_median": round(float(np.median(i)), 4),
                "detected_rate": round(float(np.mean(det)), 4),
                "dice_zero": int(sum(1 for x in d if x == 0.0)),
                "dice_ge_05": round(float(np.mean([x >= 0.5 for x in d])), 4)}
    by_cls = {}
    for tag in ["local", "external"]:
        for cls in sorted(set(r["cls"] for r in recs if r["tag"] == tag)):
            d = [r["dice"] for r in recs if r["tag"] == tag and r["cls"] == cls]
            by_cls[f"{tag}/{cls}"] = {"n": len(d), "dice_mean": round(float(np.mean(d)), 4)}
    result = {"variant": "l", "conf": args.conf, "imgsz": "640 (default, letterbox, no stretch)",
              "instrument_idx": inst_idx, "class_names": {str(k): v for k, v in names.items()},
              "geometry": "Dice/IoU in NATIVE image coords; YOLO masks returned in orig coords (asserted); GT native .nrrd via fixed loader (transpose first)",
              "speed": speed, "summary": {t: summ(recs, t) for t in ["local", "external"]},
              "by_class": by_cls, "frames": recs}
    (out_dir / "results_yolo_l.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"summary": result["summary"], "by_class": by_cls, "speed": speed}, indent=1))

    # ---- overlays: per set best-3 / worst-3 + interesting non-instrument preds ----
    # recompute pred masks for selected frames only (cheap: re-run those frames)
    def pred_mask(model, img_path, inst_idx):
        res = model.predict(source=str(img_path), conf=args.conf, verbose=False)[0]
        im = Image.open(img_path).convert("RGB")
        pred = np.zeros((im.size[1], im.size[0]), dtype=bool)
        others = {}
        if res.boxes is not None and len(res.boxes) > 0:
            for c, cf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                others[names[int(c)]] = round(float(cf), 2)
        if res.masks is not None and len(res.masks) > 0:
            m = res.masks.data.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            sel = m[cls_ids == inst_idx]
            if len(sel) > 0:
                pred = sel.max(axis=0) > 0.5
        return pred, others

    picks = []
    for tag in ["local", "external"]:
        sub = [r for r in recs if r["tag"] == tag]
        best = sorted(sub, key=lambda r: r["dice"], reverse=True)[:3]
        worst = sorted([r for r in sub if r["detected"]], key=lambda r: r["dice"])[:3]
        # interesting: frames where a non-instrument class was predicted
        other = [r for r in sub if any(k != names[inst_idx] for k in r["pred_classes"])]
        other = sorted(other, key=lambda r: r["dice"])[:4]
        for r in best + worst + other:
            if r not in picks:
                picks.append(r)
    for r in picks:
        fr = next(f for f in frames if f["tag"] == r["tag"] and f["idx"] == r["idx"])
        pred, others = pred_mask(model_l, fr["img_path"], inst_idx)
        label = (f"{r['tag']} #{r['idx']:03d} {r['cls']} | Dice {r['dice']:.3f} IoU {r['iou']:.3f} "
                 f"| pred: {others}")
        render_overlay(fr["img_path"], fr["gt"], pred, label[:160],
                       viz_dir / f"{r['tag']}_{r['idx']:03d}.jpg")
    print(f"overlays: {len(picks)} saved to viz_yolo/")
    print("DONE")


if __name__ == "__main__":
    main()
