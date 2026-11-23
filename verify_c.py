"""Stage C verification: recompute every summary from per-frame data and
compare against the stored summaries + the buggy B2/B3 numbers."""
import json
import numpy as np
from pathlib import Path

base = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/results_c_local/results_c")

# Buggy numbers (B2/B3, misaligned GT) from run record §6/§9: local / external / clean22
BUGGY = {
    "control": (0.8259, 0.4293, 0.6502),
    "b2ft": (0.8502, 0.4185, 0.6466),
    "b3best": (0.8425, 0.3624, 0.5949),
}
# B2/B3 buggy per-set extras for reference (local IoU/detection/naming, external)
BUGGY_EXTRA = {
    "control": {"local": {"mean_iou": 0.7556, "detection_rate_bbox_iou05": 0.8365,
                          "naming_accuracy": 0.7654, "mean_area_ratio": 0.923},
                "external": {"mean_iou": 0.2934, "detection_rate_bbox_iou05": 0.3246,
                             "naming_accuracy": 0.6268, "mean_area_ratio": 0.995}},
    "b2ft": {"local": {"mean_iou": 0.7938, "detection_rate_bbox_iou05": 0.8471,
                       "naming_accuracy": 0.9144, "mean_area_ratio": 0.980},
             "external": {"mean_iou": 0.2851, "detection_rate_bbox_iou05": 0.3059,
                          "naming_accuracy": 0.7493, "mean_area_ratio": 0.933}},
    "b3best": {"local": {"mean_iou": 0.7932, "detection_rate_bbox_iou05": 0.8298,
                         "naming_accuracy": 0.9519, "mean_area_ratio": 1.105},
               "external": {"mean_iou": 0.2424, "detection_rate_bbox_iou05": 0.2385,
                            "naming_accuracy": 0.7446, "mean_area_ratio": None}},
}


def summarize(frames):
    d = np.array([f["dice"] for f in frames])
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


print("=== Stage C verification (recompute from per-frame data) ===")
all_ok = True
table = {}
for tag in ("control", "b2ft", "b3best"):
    d = json.load(open(base / f"results_reeval_{tag}.json"))
    print(f"--- {tag} (ckpt: {d['ckpt']}, eval_wall_s: {d['eval_wall_s']}) ---")
    row = {}
    for s in ("local_test", "external_test", "old100", "old100_clean22"):
        fr = d["sets"][s]["frames"]
        stored = d["sets"][s]["summary"]
        recomputed = summarize(fr)
        mism = [k for k in stored if k != "by_class" and stored[k] != recomputed[k]]
        # by_class dice check
        for c, v in stored["by_class"].items():
            if v["dice"] != recomputed["by_class"][c]["dice"]:
                mism.append(f"by_class.{c}.dice")
        ok = not mism
        all_ok &= ok
        n_img = sum(1 for f in fr if "img" in f)
        bad_img = [f["img"] for f in fr if "img" in f and not f["img"].startswith("viz_c/")]
        print(f"  {s}: n={len(fr)} mean_dice stored={stored['mean_dice']} recomputed={recomputed['mean_dice']} "
              f"match={ok} {mism if mism else ''} imgs={n_img} bad_img_prefix={len(bad_img)}")
        row[s] = recomputed
    table[tag] = row

print(f"\nALL SUMMARIES MATCH: {all_ok}")

print("\n=== Corrected Dice: bug vs fixed (local / external / clean22) ===")
print(f"{'group':<10}{'local bug':>10}{'local fix':>10}{'ext bug':>9}{'ext fix':>9}{'c22 bug':>9}{'c22 fix':>9}")
for tag in ("control", "b2ft", "b3best"):
    b = BUGGY[tag]
    f = (table[tag]["local_test"]["mean_dice"], table[tag]["external_test"]["mean_dice"],
         table[tag]["old100_clean22"]["mean_dice"])
    print(f"{tag:<10}{b[0]:>10.4f}{f[0]:>10.4f}{b[1]:>9.4f}{f[1]:>9.4f}{b[2]:>9.4f}{f[2]:>9.4f}")

print("\n=== Fixed: other metrics (local / external) ===")
for tag in ("control", "b2ft", "b3best"):
    for s, sname in (("local_test", "local"), ("external_test", "external")):
        r = table[tag][s]
        b = BUGGY_EXTRA[tag][sname]
        print(f"{tag} {sname}: IoU {b['mean_iou']:.4f} -> {r['mean_iou']:.4f} | "
              f"det {b['detection_rate_bbox_iou05']:.4f} -> {r['detection_rate_bbox_iou05']:.4f} | "
              f"naming {b['naming_accuracy']:.4f} -> {r['naming_accuracy']:.4f} | "
              f"area {b['mean_area_ratio']} -> {r['mean_area_ratio']} | "
              f"Dice>=0.5 {r['frac_dice_ge05']:.4f} | median {r['median_dice']:.4f}")

print("\n=== Fixed: per-class Dice (local / external) ===")
for tag in ("control", "b2ft", "b3best"):
    for c in ["grasping forceps", "bipolar", "drill", "punch", "scissor", "dissector"]:
        ld = table[tag]["local_test"]["by_class"].get(c, {}).get("dice")
        ed = table[tag]["external_test"]["by_class"].get(c, {}).get("dice")
        print(f"{tag} {c}: local {ld} external {ed}")

print("\n=== Cross-center gap (fixed) ===")
for tag in ("control", "b2ft", "b3best"):
    l = table[tag]["local_test"]["mean_dice"]
    e = table[tag]["external_test"]["mean_dice"]
    print(f"{tag}: local {l:.4f} vs external {e:.4f}, gap {l - e:+.4f} (bug gap: "
          f"{BUGGY[tag][0] - BUGGY[tag][1]:+.4f})")

print("\n=== Fine-tune effect AFTER fix (vs control) ===")
for tag in ("b2ft", "b3best"):
    for s in ("local_test", "external_test", "old100_clean22"):
        print(f"{tag} {s}: {table[tag][s]['mean_dice'] - table['control'][s]['mean_dice']:+.4f}")

# Dice==0 frames after fix (sanity: the 0037-type artifact should be gone)
print("\n=== Dice == 0.0 frames after fix ===")
for tag in ("control", "b2ft", "b3best"):
    for s in ("local_test", "external_test"):
        z = sum(1 for f in table[tag][s]["frames"] if f["dice"] == 0.0) if "frames" in table[tag][s] else None
    # frames not kept in table; reload counts
    d = json.load(open(base / f"results_reeval_{tag}.json"))
    for s in ("local_test", "external_test", "old100"):
        z = sum(1 for f in d["sets"][s]["frames"] if f["dice"] == 0.0)
        print(f"{tag} {s}: {z} frames with Dice 0.0")

with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/c_verification.md", "w") as f:
    f.write(f"Stage C verification: ALL SUMMARIES MATCH = {all_ok}\n")
print("\nwrote c_verification.md")
