"""Stage E ep10 viz: render ep10 overlays + control-vs-ep10 pairs for the
space update (best round only). INFERENCE ONLY - no training.

Context: the space's three tabs switch to the best round -- BEFORE = Stage E
control (results_e/best_control.pt, viz_e/control_* 36 overlays already
exist), AFTER = ep10 (results_e_cont2/best_finetune_cont.pt, local 0.9643 /
external 0.9677 / clean22 0.9754). Local result JSONs have per-frame metrics
but no predicted masks, and ep10 has only 8 local overlays, so the 36
singles + 18 pairs must be re-rendered on Modal.

Outputs (volume /data/sea, then pulled back locally):
- viz_ep10/ep10_<set>_<idx:04d>.jpg (36, 720x720, EXACT same style as
  modal_stage_e.py eval_rows: square center crop, GT green, pred-only red,
  no boxes/labels -- matches the existing viz_e/control_* files).
- viz_ep10_pairs/<set>_<idx:04d>_pair.jpg (18, 1440x720, EXACT same style
  as modal_stage_e_pairs.py: two 720x720 halves + label strips
  "BEFORE #idx Dice x.xxx area x.xx" / "AFTER ..."; the existing
  viz_e_pairs_local/ files are 1440x720, NOT 1920x540).
- viz_ep10_data.json: one entry per new image (54): set/idx/cls/patient,
  control dice + area, ep10 dice + area, pred names + naming correctness
  for both sides, img path. For singles the control numbers come from the
  reference JSON; for pairs both sides are freshly rendered.
- viz_ep10_bundle.zip: the 54 jpgs in one file (volume dir-get concatenates
  files; single-file get is the reliable path).

Cross-checks (same bar as the pairs script, |diff| <= 0.002 flags):
fresh ep10 dice vs results_e_cont2/results_finetune_cont2.json (volume copy),
fresh control dice (pairs) vs results_e/results_control.json.
Any mismatch is reported, never papered over.

Run: nohup .venv/bin/modal run --detach modal_stage_e_ep10_viz.py
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

# 36 single frames = viz_e/control_* filenames (set, idx)
SINGLES = ([("external_test", i) for i in
            [56, 149, 184, 378, 388, 465, 753, 925, 926, 953, 1040, 1060]] +
           [("local_test", i) for i in
            [1, 301, 378, 388, 705, 755, 799, 837, 925, 926, 953, 974]] +
           [("old100", i) for i in
            [0, 22, 28, 57, 59, 65, 68, 71, 75, 78, 95, 99]])
# 18 pair frames = viz_e_pairs_local/e_pairs.json (set, idx)
PAIRS = ([("external_test", i) for i in
          [698, 597, 721, 769, 712, 549, 718, 710, 700, 696, 599, 704]] +
         [("local_test", i) for i in [680, 48, 809, 816, 750, 800]])


@app.function(image=image, volumes={"/data": vol}, gpu="L40S",
              timeout=3600, scaledown_window=10)
def stage_e_ep10_viz():
    import hashlib
    import sys
    import zipfile
    from pathlib import Path

    import numpy as np
    import nrrd
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageDraw, ImageFont

    sys.path.insert(0, "/root/repo")
    from surgfm_probe import load_backbone
    from surgfm_probe.transforms import IMAGENET_MEAN, IMAGENET_STD

    res = {"stage": "E-ep10-viz"}
    t0 = time.time()
    base = Path("/data/sea")
    ext = base / "extracted"
    man = base / "manifests_paper"
    single_dir = base / "viz_ep10"
    pair_dir = base / "viz_ep10_pairs"
    single_dir.mkdir(exist_ok=True)
    pair_dir.mkdir(exist_ok=True)

    ckpt = base / "ckpt" / "dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth"
    h = hashlib.sha256()
    with open(ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    res["ckpt_sha256_ok"] = h.hexdigest() == CKPT_SHA256
    print(f"ckpt sha ok={res['ckpt_sha256_ok']}", flush=True)
    assert res["ckpt_sha256_ok"], "checkpoint SHA mismatch"
    ctrl_path = base / "results_e" / "best_control.pt"
    ep10_path = base / "results_e_cont2" / "best_finetune_cont.pt"
    assert ctrl_path.exists(), f"missing control checkpoint {ctrl_path}"
    assert ep10_path.exists(), f"missing ep10 checkpoint {ep10_path}"
    print("checkpoints in place: best_control.pt, best_finetune_cont.pt", flush=True)

    def load_rows(name):
        with open(man / name) as f:
            return [json.loads(line) for line in f]

    local_rows = load_rows("local_test.jsonl")
    ext_rows = load_rows("external_test.jsonl")
    old_rows = load_rows("old100.jsonl")
    rows_by_set = {"local_test": local_rows, "external_test": ext_rows,
                   "old100": old_rows}

    # Reference per-frame metrics from the volume JSONs.
    ref = {}  # (tag, set, idx) -> frame dict
    try:
        cj = json.load(open(base / "results_e" / "results_control.json"))
        ej = json.load(open(base / "results_e_cont2" / "results_finetune_cont2.json"))
        for tag, j in (("control", cj), ("ep10", ej)):
            for s in ("local_test", "external_test", "old100"):
                for f in j["sets"][s]["frames"]:
                    ref[(tag, s, f["idx"])] = f
        print(f"loaded {len(ref)} reference frames from volume JSONs", flush=True)
    except Exception as e:
        print(f"WARNING: could not load reference JSONs ({e})", flush=True)

    device = "cuda"
    torch.manual_seed(7)

    def load_nrrd_mask(path):
        """FIXED loader: nrrd axis order is ALWAYS (W, H, 1). Transpose
        FIRST; native (H, W) bool, no resize."""
        arr, _hdr = nrrd.read(str(path))
        m = np.asarray(arr)
        if m.ndim == 3:
            m = m[:, :, 0]
        return (m.T > 0)

    def square_crop_bounds(w, h):
        side = min(w, h)
        return side, (w - side) // 2

    def cropped_gt(r):
        gt = load_nrrd_mask(ext / r["mask"])
        side, x0 = square_crop_bounds(gt.shape[1], gt.shape[0])
        return gt[:, x0:x0 + side]

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
    def predict_full(backbone, head, r):
        """Full prediction: argmax map, binary mask, GT, dice, area,
        dominant predicted name + whether it matches the GT class."""
        backbone.eval()
        head.eval()
        x, side = load_image(r)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = head(forward_dense(backbone, x.unsqueeze(0).to(device)))
        up = F.interpolate(logits.float(), size=(side, side),
                           mode="bilinear", align_corners=False)
        pred = up.argmax(1)[0].cpu().numpy()
        pm = pred != 0
        gt = cropped_gt(r)
        inter = float((pm & gt).sum())
        dice = 2 * inter / (float(pm.sum()) + float(gt.sum()) + 1e-9)
        area = float(pm.sum()) / max(float(gt.sum()), 1)
        counts = [(c, int((pred == c).sum())) for c in range(1, NCLS)]
        counts.sort(key=lambda t: -t[1])
        dom = CLS_NAMES[counts[0][0]] if counts[0][1] > 0 else "none"
        return pred, pm, gt, dice, area, dom, bool(dom == r["cls"])

    def render_single(r, pm, gt, side):
        """EXACT style of modal_stage_e.py eval_rows: 720x720, GT green,
        pred-only red, 0.45/0.55 blend, no boxes, no labels."""
        im = Image.open(ext / r["image"]).convert("RGB")
        _sd, x0 = square_crop_bounds(im.width, im.height)
        im = im.crop((x0, 0, x0 + side, side)).resize((720, 720), Image.BICUBIC)
        ov = np.asarray(im).astype(np.float32)
        gts = np.asarray(Image.fromarray((gt.astype(np.uint8)) * 255).resize((720, 720), Image.NEAREST)) > 0
        prs = np.asarray(Image.fromarray((pm.astype(np.uint8)) * 255).resize((720, 720), Image.NEAREST)) > 0
        ov[gts] = ov[gts] * 0.45 + np.array([0, 200, 0]) * 0.55
        only_pred = prs & ~gts
        ov[only_pred] = ov[only_pred] * 0.45 + np.array([220, 0, 0]) * 0.55
        return Image.fromarray(ov.astype(np.uint8))

    def render_half(r, pm, gt, label):
        """EXACT style of modal_stage_e_pairs.py overlay(): 720x720 half
        with a label strip on top."""
        im = Image.open(ext / r["image"]).convert("RGB")
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

    def ref_entry(tag, s, i):
        return ref.get((tag, s, i))

    mismatches = []
    data_entries = []

    # ---- Part A: 36 ep10 singles ----
    c = torch.load(str(ep10_path), map_location="cpu", weights_only=False)
    backbone, head = build_model(True)
    backbone.load_state_dict(c["backbone_state"])
    head.load_state_dict(c["head_state"])
    backbone, head = backbone.to(device), head.to(device)
    for s, i in SINGLES:
        r = rows_by_set[s][i]
        _pred, pm, gt, dice, area, dom, named_ok = predict_full(backbone, head, r)
        side = gt.shape[0]
        img = render_single(r, pm, gt, side)
        name = f"ep10_{s}_{i:04d}.jpg"
        img.save(single_dir / name, quality=88)
        rj = ref_entry("ep10", s, i)
        rc = ref_entry("control", s, i)
        flag = ""
        if rj is not None and abs(dice - rj["dice"]) > 0.002:
            flag = f"  *** MISMATCH vs ep10 JSON {rj['dice']:.4f} ***"
            mismatches.append(("ep10-single", s, i, round(float(dice), 4), rj["dice"]))
        print(f"[ep10] {s} #{i}: dice={dice:.4f} area={area:.2f} "
              f"pred={dom} named_ok={named_ok}{flag}", flush=True)
        data_entries.append({
            "kind": "single", "set": s, "idx": i, "cls": r["cls"],
            "patient": r["patient"], "split": r.get("split"),
            "control_dice": round(float(rc["dice"]), 4) if rc else None,
            "control_area": round(float(rc["area_ratio"]), 3) if rc else None,
            "control_pred_name": rc["pred_name"] if rc else None,
            "control_named_correct": bool(rc["named_correct"]) if rc else None,
            "ep10_dice": round(float(dice), 4),
            "ep10_area": round(float(area), 3),
            "ep10_dice_json": round(float(rj["dice"]), 4) if rj else None,
            "pred_name": dom, "named_correct": named_ok,
            "img": f"viz_ep10/{name}",
        })
    del backbone, head
    torch.cuda.empty_cache()

    # ---- Part B: 18 control-vs-ep10 pairs ----
    preds = {}
    for tag, path, has_backbone in [
            ("control", ctrl_path, False),
            ("ep10", ep10_path, True)]:
        c = torch.load(str(path), map_location="cpu", weights_only=False)
        backbone, head = build_model(has_backbone)
        if has_backbone:
            backbone.load_state_dict(c["backbone_state"])
        head.load_state_dict(c["head_state"])
        backbone, head = backbone.to(device), head.to(device)
        for s, i in PAIRS:
            _pred, pm, gt, dice, area, dom, named_ok = predict_full(
                backbone, head, rows_by_set[s][i])
            preds[(tag, s, i)] = (pm, gt, dice, area, dom, named_ok)
            rj = ref_entry(tag, s, i)
            flag = ""
            if rj is not None and abs(dice - rj["dice"]) > 0.002:
                flag = f"  *** MISMATCH vs {tag} JSON {rj['dice']:.4f} ***"
                mismatches.append((f"{tag}-pair", s, i, round(float(dice), 4), rj["dice"]))
            print(f"[{tag}] {s} #{i}: dice={dice:.4f} area={area:.2f} "
                  f"pred={dom}{flag}", flush=True)
        del backbone, head
        torch.cuda.empty_cache()

    for s, i in PAIRS:
        r = rows_by_set[s][i]
        pm_c, gt, dice_c, area_c, dom_c, named_c = preds[("control", s, i)]
        pm_e, _gt, dice_e, area_e, dom_e, named_e = preds[("ep10", s, i)]
        lab_c = ref_entry("control", s, i)
        lab_e = ref_entry("ep10", s, i)
        lc = lab_c["dice"] if lab_c else dice_c
        le = lab_e["dice"] if lab_e else dice_e
        left = render_half(r, pm_c, gt, f"BEFORE #{i}  Dice {lc:.3f}  area {area_c:.2f}")
        right = render_half(r, pm_e, gt, f"AFTER #{i}  Dice {le:.3f}  area {area_e:.2f}")
        pair = Image.new("RGB", (1440, 720), (255, 255, 255))
        pair.paste(left, (0, 0))
        pair.paste(right, (720, 0))
        d = ImageDraw.Draw(pair)
        d.rectangle([718, 0, 722, 720], fill=(255, 255, 255))
        name = f"{s}_{i:04d}_pair.jpg"
        pair.save(pair_dir / name, quality=88)
        data_entries.append({
            "kind": "pair", "set": s, "idx": i, "cls": r["cls"],
            "patient": r["patient"], "split": r.get("split"),
            "control_dice": round(float(dice_c), 4),
            "control_area": round(float(area_c), 3),
            "control_pred_name": dom_c,
            "control_named_correct": named_c,
            "ep10_dice": round(float(dice_e), 4),
            "ep10_area": round(float(area_e), 3),
            "ep10_dice_json": round(float(le), 4),
            "pred_name": dom_e, "named_correct": named_e,
            "img": f"viz_ep10_pairs/{name}",
        })
        print(f"[pair] {name} done", flush=True)

    # ---- Manifest + bundle ----
    with open(base / "viz_ep10_data.json", "w") as f:
        json.dump(data_entries, f, indent=2, ensure_ascii=False)
    vol.commit()
    bundle = base / "viz_ep10_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(single_dir.glob("*.jpg")):
            z.write(p, f"viz_ep10/{p.name}")
        for p in sorted(pair_dir.glob("*.jpg")):
            z.write(p, f"viz_ep10_pairs/{p.name}")
    vol.commit()

    res["n_singles"] = len(list(single_dir.glob("*.jpg")))
    res["n_pairs"] = len(list(pair_dir.glob("*.jpg")))
    res["n_data_entries"] = len(data_entries)
    res["mismatches"] = mismatches
    res["total_s"] = round(time.time() - t0, 1)
    print("STAGE_E_EP10_VIZ_RESULT_JSON=" + json.dumps(res, ensure_ascii=False))
    return res


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_e_ep10_viz.remote()
    print("\n=== STAGE E EP10 VIZ RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/ep10_viz_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
