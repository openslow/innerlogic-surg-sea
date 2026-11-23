"""Stage D verification: recompute every summary from per-frame data and
compare against the stored summaries + Stage C baselines (corrected GT)."""
import json
import numpy as np
from pathlib import Path

root = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg")
d = json.load(open(root / "results_d_local/results_d/results_finetune.json"))
c_control = json.load(open(root / "results_c_local/results_reeval_control.json"))
c_b2ft = json.load(open(root / "results_c_local/results_reeval_b2ft.json"))

# Stage C corrected-GT baselines (local / external / clean22), verified in §11
C = {
    "control": (0.9056, 0.9460, 0.9392),
    "b2ft_old": (0.9234, 0.9177, 0.9312),
    "b3best": (0.8814, 0.6205, 0.5975),
}


def summarize(frames):
    dd = np.array([f["dice"] for f in frames])
    i = np.array([f["iou"] for f in frames])
    b = np.array([f["bbox_iou"] for f in frames])
    by_cls = {}
    for c in ["grasping forceps", "bipolar", "drill", "punch", "scissor", "dissector"]:
        sub = [f for f in frames if f["cls"] == c]
        if sub:
            by_cls[c] = {"n": len(sub),
                         "dice": round(float(np.mean([f["dice"] for f in sub])), 4)}
    return {
        "n": len(frames),
        "mean_dice": round(float(dd.mean()), 4),
        "median_dice": round(float(np.median(dd)), 4),
        "mean_iou": round(float(i.mean()), 4),
        "mean_bbox_iou": round(float(b.mean()), 4),
        "detection_rate_bbox_iou05": round(float(np.mean([f["detected"] for f in frames])), 4),
        "detection_rate_mask_iou05": round(float(np.mean([f["detected_mask"] for f in frames])), 4),
        "frac_dice_ge05": round(float(np.mean(dd >= 0.5)), 4),
        "naming_accuracy": round(float(np.mean([f["named_correct"] for f in frames])), 4),
        "mean_area_ratio": round(float(np.mean([f["area_ratio"] for f in frames])), 3),
        "by_class": by_cls,
    }


print("=== Stage D verification (recompute from per-frame data) ===")
print(f"best_val_dice={d['best_val_dice']} train_wall_s={d['train_wall_s']}")
print(f"train_history={[(h['epoch'], h['val_dice'], round(h['train_loss'], 4)) for h in d['train_history']]}")
all_ok = True
table = {}
for s in ("local_test", "external_test", "old100", "old100_clean22"):
    fr = d["sets"][s]["frames"]
    stored = d["sets"][s]["summary"]
    recomputed = summarize(fr)
    mism = [k for k in stored if k != "by_class" and stored[k] != recomputed[k]]
    for c, v in stored["by_class"].items():
        if v["dice"] != recomputed["by_class"][c]["dice"]:
            mism.append(f"by_class.{c}.dice")
    ok = not mism
    all_ok &= ok
    n_img = sum(1 for f in fr if "img" in f)
    bad_img = [f["img"] for f in fr if "img" in f and not f["img"].startswith("viz_d/")]
    missing = [f["img"] for f in fr if "img" in f and not (root / f["img"]).exists()]
    print(f"  {s}: n={len(fr)} mean_dice stored={stored['mean_dice']} recomputed={recomputed['mean_dice']} "
          f"match={ok} {mism if mism else ''} imgs={n_img} bad_prefix={len(bad_img)} missing_files={len(missing)}")
    table[s] = recomputed

print(f"\nALL SUMMARIES MATCH: {all_ok}")

print("\n=== Stage D vs Stage C baselines (local / external / clean22) ===")
print(f"{'group':<12}{'local':>8}{'ext':>8}{'c22':>8}")
for tag, vals in [("control(0-sample)", C["control"]), ("b2ft_old", C["b2ft_old"]),
                  ("b3best", C["b3best"])]:
    print(f"{tag:<12}{vals[0]:>8.4f}{vals[1]:>8.4f}{vals[2]:>8.4f}")
sd = (table["local_test"]["mean_dice"], table["external_test"]["mean_dice"],
      table["old100_clean22"]["mean_dice"])
print(f"{'STAGE D':<12}{sd[0]:>8.4f}{sd[1]:>8.4f}{sd[2]:>8.4f}")
print(f"{'D - control':<12}{sd[0]-C['control'][0]:>+8.4f}{sd[1]-C['control'][1]:>+8.4f}{sd[2]-C['control'][2]:>+8.4f}")
print(f"{'D - b2ft_old':<12}{sd[0]-C['b2ft_old'][0]:>+8.4f}{sd[1]-C['b2ft_old'][1]:>+8.4f}{sd[2]-C['b2ft_old'][2]:>+8.4f}")
print(f"old100 full: Stage D {table['old100']['mean_dice']:.4f} (contaminated, 78/100 in train)")

print("\n=== Stage D other metrics vs Stage C control / b2ft ===")
for s, sname in (("local_test", "local"), ("external_test", "external")):
    r = table[s]
    cc = c_control["sets"][s]["summary"]
    bb = c_b2ft["sets"][s]["summary"]
    print(f"{sname}: IoU D {r['mean_iou']:.4f} (ctrl {cc['mean_iou']:.4f}, b2ft {bb['mean_iou']:.4f}) | "
          f"det D {r['detection_rate_bbox_iou05']:.4f} (ctrl {cc['detection_rate_bbox_iou05']:.4f}, b2ft {bb['detection_rate_bbox_iou05']:.4f}) | "
          f"naming D {r['naming_accuracy']:.4f} (ctrl {cc['naming_accuracy']:.4f}, b2ft {bb['naming_accuracy']:.4f}) | "
          f"area D {r['mean_area_ratio']} (ctrl {cc['mean_area_ratio']}, b2ft {bb['mean_area_ratio']}) | "
          f"Dice>=0.5 D {r['frac_dice_ge05']:.4f} (ctrl {cc['frac_dice_ge05']:.4f}, b2ft {bb['frac_dice_ge05']:.4f}) | "
          f"median D {r['median_dice']:.4f}")

print("\n=== Stage D per-class Dice (local / external) vs control / b2ft ===")
for c in ["grasping forceps", "bipolar", "drill", "punch", "scissor", "dissector"]:
    ld = table["local_test"]["by_class"].get(c, {}).get("dice")
    ed = table["external_test"]["by_class"].get(c, {}).get("dice")
    cl = c_control["sets"]["local_test"]["summary"]["by_class"].get(c, {}).get("dice")
    ce = c_control["sets"]["external_test"]["summary"]["by_class"].get(c, {}).get("dice")
    bl = c_b2ft["sets"]["local_test"]["summary"]["by_class"].get(c, {}).get("dice")
    be = c_b2ft["sets"]["external_test"]["summary"]["by_class"].get(c, {}).get("dice")
    print(f"{c}: local D {ld} (ctrl {cl}, b2ft {bl}) | external D {ed} (ctrl {ce}, b2ft {be})")

print("\n=== Dice == 0.0 frames ===")
for s in ("local_test", "external_test", "old100"):
    z = sum(1 for f in d["sets"][s]["frames"] if f["dice"] == 0.0)
    print(f"Stage D {s}: {z} frames with Dice 0.0")
for tag, dd in (("control", c_control), ("b2ft", c_b2ft)):
    for s in ("local_test", "external_test"):
        z = sum(1 for f in dd["sets"][s]["frames"] if f["dice"] == 0.0)
        print(f"Stage C {tag} {s}: {z} frames with Dice 0.0")

print("\n=== Worst / best frames (Stage D) ===")
for s in ("local_test", "external_test"):
    fr = sorted(d["sets"][s]["frames"], key=lambda f: f["dice"])
    print(f"{s} worst: " + ", ".join(f"#{f['idx']} {f['cls']} {f['dice']:.3f} ({f['patient']})" for f in fr[:5]))
    print(f"{s} best: " + ", ".join(f"#{f['idx']} {f['cls']} {f['dice']:.3f}" for f in fr[-3:]))

with open(root / "d_verification.md", "w") as f:
    f.write(f"Stage D verification: ALL SUMMARIES MATCH = {all_ok}\n"
            f"Stage D Dice local/external/clean22 = {sd[0]}/{sd[1]}/{sd[2]}\n"
            f"D - control = {sd[0]-C['control'][0]:+.4f}/{sd[1]-C['control'][1]:+.4f}/{sd[2]-C['control'][2]:+.4f}\n"
            f"D - b2ft_old = {sd[0]-C['b2ft_old'][0]:+.4f}/{sd[1]-C['b2ft_old'][1]:+.4f}/{sd[2]-C['b2ft_old'][2]:+.4f}\n")
print("\nwrote d_verification.md")
