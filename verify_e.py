"""Stage E verification: recompute every summary from per-frame data and
compare against the stored summaries + Stage C/D baselines (stretch
preprocessing). Also computes the control->finetune per-frame delta counts
(worsened/improved) for the right-side-FP analysis."""
import json
import numpy as np
from pathlib import Path

root = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg")
ec = json.load(open(root / "results_e_local/results_e/results_control.json"))
ed = json.load(open(root / "results_e_local/results_e/results_finetune.json"))
c_control = json.load(open(root / "results_c_local/results_reeval_control.json"))
d_ft = json.load(open(root / "results_d_local/results_d/results_finetune.json"))

# Old (stretch) baselines, verified in §11/§12
C_CTRL = (0.9056, 0.9460, 0.9392)   # Stage C control, stretch
D_FT = (0.9101, 0.9314, 0.9172)     # Stage D finetune, stretch


def summarize(frames):
    dd = np.array([f["dice"] for f in frames])
    i = np.array([f["iou"] for f in frames])
    b = np.array([f["bbox_iou"] for f in frames])
    by_cls = {}
    for c in ["grasping forceps", "bipolar", "drill", "punch", "scissor", "dissector"]:
        sub = [f for f in frames if f["cls"] == c]
        if sub:
            by_cls[c] = {"n": len(sub),
                         "dice": round(float(np.mean([f["dice"] for f in sub])), 4),
                         "iou": round(float(np.mean([f["iou"] for f in sub])), 4),
                         "det": round(float(np.mean([f["detected"] for f in sub])), 3)}
    return {
        "n": len(frames),
        "mean_dice": round(float(dd.mean()), 4),
        "median_dice": round(float(np.median(dd)), 4),
        "mean_iou": round(float(i.mean()), 4),
        "mean_bbox_iou": round(float(b.mean()), 4),
        "detection_rate_bbox_iou05": round(float(np.mean([f["detected"] for f in frames])), 4),
        "detection_rate_mask_iou05": round(float(np.mean([f["detected_mask"] for f in frames])), 4),
        "frac_dice_ge05": round(float(np.mean(dd >= 0.5)), 4),
        "n_zero_dice": int((dd == 0).sum()),
        "naming_accuracy": round(float(np.mean([f["named_correct"] for f in frames])), 4),
        "mean_area_ratio": round(float(np.mean([f["area_ratio"] for f in frames])), 3),
        "by_class": by_cls,
    }


print("=== Stage E verification (recompute from per-frame data) ===")
all_ok = True
table = {}
for tag, d in (("control", ec), ("finetune", ed)):
    print(f"--- {tag}: best_val={d['best_val_dice']} train_wall_s={d['train_wall_s']}")
    print(f"    history={[(h['epoch'], h['val_dice'], h['train_loss'], h['epoch_s']) for h in d['train_history']]}")
    table[tag] = {}
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
        bad_img = [f["img"] for f in fr if "img" in f and not f["img"].startswith("viz_e/")]
        missing = [f["img"] for f in fr if "img" in f and not (root / f["img"]).exists()]
        print(f"  {s}: n={len(fr)} mean_dice stored={stored['mean_dice']} recomputed={recomputed['mean_dice']} "
              f"match={ok} {mism if mism else ''} imgs={n_img} bad_prefix={len(bad_img)} missing_files={len(missing)}")
        table[tag][s] = recomputed

print(f"\nALL SUMMARIES MATCH: {all_ok}")

print("\n=== Stage E vs stretch baselines (local / external / clean22) ===")
print(f"{'group':<22}{'local':>8}{'ext':>8}{'c22':>8}")
print(f"{'C control (stretch)':<22}{C_CTRL[0]:>8.4f}{C_CTRL[1]:>8.4f}{C_CTRL[2]:>8.4f}")
print(f"{'D finetune (stretch)':<22}{D_FT[0]:>8.4f}{D_FT[1]:>8.4f}{D_FT[2]:>8.4f}")
for tag in ("control", "finetune"):
    v = (table[tag]["local_test"]["mean_dice"], table[tag]["external_test"]["mean_dice"],
         table[tag]["old100_clean22"]["mean_dice"])
    print(f"{'E ' + tag:<22}{v[0]:>8.4f}{v[1]:>8.4f}{v[2]:>8.4f}")
ec_v = (table["control"]["local_test"]["mean_dice"], table["control"]["external_test"]["mean_dice"],
        table["control"]["old100_clean22"]["mean_dice"])
ed_v = (table["finetune"]["local_test"]["mean_dice"], table["finetune"]["external_test"]["mean_dice"],
        table["finetune"]["old100_clean22"]["mean_dice"])
print(f"{'E control - C control':<22}{ec_v[0]-C_CTRL[0]:>+8.4f}{ec_v[1]-C_CTRL[1]:>+8.4f}{ec_v[2]-C_CTRL[2]:>+8.4f}")
print(f"{'E finetune - D finetune':<22}{ed_v[0]-D_FT[0]:>+8.4f}{ed_v[1]-D_FT[1]:>+8.4f}{ed_v[2]-D_FT[2]:>+8.4f}")
print(f"{'E finetune - E control':<22}{ed_v[0]-ec_v[0]:>+8.4f}{ed_v[1]-ec_v[1]:>+8.4f}{ed_v[2]-ec_v[2]:>+8.4f}")
print(f"E local-external gap control: {ec_v[0]-ec_v[1]:+.4f} (C control: {C_CTRL[0]-C_CTRL[1]:+.4f})")
print(f"E local-external gap finetune: {ed_v[0]-ed_v[1]:+.4f} (D finetune: {D_FT[0]-D_FT[1]:+.4f})")
print(f"old100 full: E control {table['control']['old100']['mean_dice']:.4f}, "
      f"E finetune {table['finetune']['old100']['mean_dice']:.4f} (contaminated, 78/100 in train)")

print("\n=== Stage E other metrics (control / finetune) vs C control / D ===")
for s, sname in (("local_test", "local"), ("external_test", "external")):
    rc, rf = table["control"][s], table["finetune"][s]
    cc = c_control["sets"][s]["summary"]
    dd = d_ft["sets"][s]["summary"]
    print(f"{sname}: IoU E-ctrl {rc['mean_iou']:.4f} / E-ft {rf['mean_iou']:.4f} (C {cc['mean_iou']:.4f}, D {dd['mean_iou']:.4f}) | "
          f"det E-ctrl {rc['detection_rate_bbox_iou05']:.4f} / E-ft {rf['detection_rate_bbox_iou05']:.4f} (C {cc['detection_rate_bbox_iou05']:.4f}, D {dd['detection_rate_bbox_iou05']:.4f}) | "
          f"naming E-ctrl {rc['naming_accuracy']:.4f} / E-ft {rf['naming_accuracy']:.4f} (C {cc['naming_accuracy']:.4f}, D {dd['naming_accuracy']:.4f}) | "
          f"area E-ctrl {rc['mean_area_ratio']} / E-ft {rf['mean_area_ratio']} (C {cc['mean_area_ratio']}, D {dd['mean_area_ratio']}) | "
          f"Dice>=0.5 E-ctrl {rc['frac_dice_ge05']:.4f} / E-ft {rf['frac_dice_ge05']:.4f} | "
          f"median E-ctrl {rc['median_dice']:.4f} / E-ft {rf['median_dice']:.4f} | "
          f"zero E-ctrl {rc['n_zero_dice']} / E-ft {rf['n_zero_dice']}")

print("\n=== Stage E per-class Dice (local / external) ===")
for c in ["grasping forceps", "bipolar", "drill", "punch", "scissor", "dissector"]:
    cl = table["control"]["local_test"]["by_class"].get(c, {}).get("dice")
    ce = table["control"]["external_test"]["by_class"].get(c, {}).get("dice")
    fl = table["finetune"]["local_test"]["by_class"].get(c, {}).get("dice")
    fe = table["finetune"]["external_test"]["by_class"].get(c, {}).get("dice")
    print(f"{c}: local ctrl {cl} / ft {fl} | external ctrl {ce} / ft {fe}")

print("\n=== Control -> finetune per-frame delta (Stage E) vs Stage D ===")
for s in ("local_test", "external_test"):
    fc = {f["idx"]: f["dice"] for f in ec["sets"][s]["frames"]}
    ff = {f["idx"]: f["dice"] for f in ed["sets"][s]["frames"]}
    deltas = np.array([ff[i] - fc[i] for i in sorted(fc)])
    worse = int((deltas < -0.05).sum())
    better = int((deltas > 0.05).sum())
    print(f"{s}: mean dDice={deltas.mean():+.4f} worsened>0.05: {worse} improved>0.05: {better} "
          f"(Stage D: external 176 worsened / 46 improved; local 95 / 109)")
    worst = sorted(((ff[i] - fc[i], i) for i in fc))[:5]
    print(f"   worst deltas: " + ", ".join(f"#{i} {d:+.3f}" for d, i in worst))

print("\n=== Dice == 0.0 frames ===")
for tag, d in (("E control", ec), ("E finetune", ed)):
    for s in ("local_test", "external_test", "old100"):
        z = sum(1 for f in d["sets"][s]["frames"] if f["dice"] == 0.0)
        print(f"{tag} {s}: {z} frames with Dice 0.0")

with open(root / "e_verification.md", "w") as f:
    f.write(f"Stage E verification: ALL SUMMARIES MATCH = {all_ok}\n"
            f"E control Dice local/external/clean22 = {ec_v[0]}/{ec_v[1]}/{ec_v[2]}\n"
            f"E finetune Dice local/external/clean22 = {ed_v[0]}/{ed_v[1]}/{ed_v[2]}\n"
            f"E control - C control = {ec_v[0]-C_CTRL[0]:+.4f}/{ec_v[1]-C_CTRL[1]:+.4f}/{ec_v[2]-C_CTRL[2]:+.4f}\n"
            f"E finetune - D finetune = {ed_v[0]-D_FT[0]:+.4f}/{ed_v[1]-D_FT[1]:+.4f}/{ed_v[2]-D_FT[2]:+.4f}\n"
            f"E finetune - E control = {ed_v[0]-ec_v[0]:+.4f}/{ed_v[1]-ec_v[1]:+.4f}/{ed_v[2]-ec_v[2]:+.4f}\n"
            f"E local-external gap control = {ec_v[0]-ec_v[1]:+.4f} (C: {C_CTRL[0]-C_CTRL[1]:+.4f})\n"
            f"E local-external gap finetune = {ed_v[0]-ed_v[1]:+.4f} (D: {D_FT[0]-D_FT[1]:+.4f})\n")
print("\nwrote e_verification.md")
