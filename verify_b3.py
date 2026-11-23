"""B3 verification: pull results_b3 + viz_b3 from the Modal volume, recompute
summaries from per-frame data, compare against the JSONs' own summaries, and
tabulate B3-best / B3-last / B2-ep4-reeval vs B2 control and B2 finetune.

Usage: python3 verify_b3.py
Requires: ~/workspace/modal-smoke/.venv/bin/modal on PATH (called via subprocess).
Writes: results_b3_local/ (downloaded JSONs), viz_b3/ (downloaded overlays),
  b3_verification.md (comparison table + recompute check).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg")
MODAL = "/home/hatch/workspace/modal-smoke/.venv/bin/modal"
VOL = "innerlogic-surg"

# B2 baselines (from results_probe_paper_split.json / results_finetune.json, verified)
B2 = {
    "control": {"local": 0.8259, "external": 0.4293, "clean22": 0.6502, "old100": 0.7338},
    "finetune": {"local": 0.8502, "external": 0.4185, "clean22": 0.6466, "old100": 0.7564},
}


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"command failed: {cmd[0]}")
    return r.stdout


def recompute_summary(frames):
    d = np.array([f["dice"] for f in frames])
    i = np.array([f["iou"] for f in frames])
    b = np.array([f["bbox_iou"] for f in frames])
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
    }


def check_file(path, label):
    d = json.loads(Path(path).read_text())
    print(f"--- {label}: {path} ---")
    problems = []
    sets = d["sets"]
    for s in ("local_test", "external_test", "old100", "old100_clean22"):
        summ = sets[s]["summary"]
        re = recompute_summary(sets[s]["frames"])
        for k, v in re.items():
            if summ.get(k) != v:
                problems.append(f"{s}.{k}: json={summ.get(k)} recomputed={v}")
        print(f"  {s}: n={summ['n']} dice={summ['mean_dice']} "
              f"iou={summ['mean_iou']} naming={summ['naming_accuracy']} "
              f"area={summ['mean_area_ratio']}")
    if problems:
        print("  MISMATCHES:")
        for p in problems:
            print("   -", p)
    else:
        print("  recompute: all summary fields match per-frame data")
    return d, problems


def main():
    out = BASE / "results_b3_local"
    out.mkdir(exist_ok=True)
    # The whole-directory get also pulls two ~544 MB .pt files (optimizer state
    # included); the JSONs are already local from the first attempt, so only
    # fetch the three JSONs individually if missing.
    src = out / "results_b3"
    if not (src / "results_finetune_b3_best.json").exists():
        src = out
        for name in ("results_finetune_b3_best.json", "results_finetune_b3_last.json",
                     "results_b2ep4_reeval.json"):
            run([MODAL, "volume", "get", VOL, f"/sea/results_b3/{name}", str(out / name)])
    viz = BASE / "viz_b3"
    viz.mkdir(exist_ok=True)
    run([MODAL, "volume", "get", VOL, "/sea/viz_b3/", str(viz)])
    print("downloaded:", sorted(p.name for p in src.iterdir()))
    print("viz images:", len(list(viz.glob("*.jpg"))))

    # NOTE: B3's eval_rows writes img fields as viz_b2/... (copied from B2) while
    # the files actually land in viz_b3/. Fix the paths in the local copies.
    fixed = 0
    for jf in src.glob("results_finetune_b3_*.json"):
        d = json.loads(jf.read_text())
        for s in d["sets"].values():
            for f in s["frames"]:
                if f.get("img", "").startswith("viz_b2/"):
                    f["img"] = "viz_b3/" + f["img"][len("viz_b2/"):]
                    fixed += 1
        jf.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"fixed {fixed} img paths viz_b2/ -> viz_b3/")

    all_problems = []
    data = {}
    for name, label in [("results_finetune_b3_best.json", "B3-best"),
                        ("results_finetune_b3_last.json", "B3-last"),
                        ("results_b2ep4_reeval.json", "B2-ep4 re-eval")]:
        d, probs = check_file(src / name, label)
        all_problems += probs
        data[label] = d

    # comparison table: local / external / clean22 / old100 Dice
    lines = []
    lines.append("# B3 verification (recomputed from per-frame data)")
    lines.append("")
    lines.append("| group | local_test | external_test | clean22 | old100 |")
    lines.append("|---|---|---|---|---|")
    for label, key in [("B2 control", "control"), ("B2 finetune", "finetune")]:
        b = B2[key]
        lines.append(f"| {label} | {b['local']:.4f} | {b['external']:.4f} | "
                     f"{b['clean22']:.4f} | {b['old100']:.4f} |")
    for label in ("B3-best", "B3-last", "B2-ep4 re-eval"):
        s = data[label]["sets"]
        lines.append(f"| {label} | {s['local_test']['summary']['mean_dice']:.4f} | "
                     f"{s['external_test']['summary']['mean_dice']:.4f} | "
                     f"{s['old100_clean22']['summary']['mean_dice']:.4f} | "
                     f"{s['old100']['summary']['mean_dice']:.4f} |")
    lines.append("")
    # deltas vs B2 finetune and vs control
    for label in ("B3-best", "B3-last", "B2-ep4 re-eval"):
        s = data[label]["sets"]
        dl = s["local_test"]["summary"]["mean_dice"] - B2["finetune"]["local"]
        de = s["external_test"]["summary"]["mean_dice"] - B2["finetune"]["external"]
        dc = s["old100_clean22"]["summary"]["mean_dice"] - B2["finetune"]["clean22"]
        dl0 = s["local_test"]["summary"]["mean_dice"] - B2["control"]["local"]
        de0 = s["external_test"]["summary"]["mean_dice"] - B2["control"]["external"]
        lines.append(f"- {label} vs B2-finetune: local {dl:+.4f}, external {de:+.4f}, "
                     f"clean22 {dc:+.4f}; vs B2-control: local {dl0:+.4f}, external {de0:+.4f}")
    lines.append("")
    # training history (from B3-best file)
    hist = data["B3-best"].get("train_history", [])
    lines.append("## B3 training history (val Dice)")
    for h in hist:
        lines.append(f"- ep{h['epoch']}: loss={h['train_loss']} val_dice={h['val_dice']} "
                     f"({h['epoch_s']}s)")
    lines.append(f"- best_val={data['B3-best'].get('best_val_dice')} "
                 f"best_ckpt_epoch={data['B3-best'].get('ckpt_epoch')} "
                 f"last_ckpt_epoch={data['B3-last'].get('ckpt_epoch')}")
    lines.append(f"- B2-ep4 re-eval ckpt_epoch={data['B2-ep4 re-eval'].get('ckpt_epoch')}")
    lines.append("")
    if all_problems:
        lines.append("## RECOMPUTE MISMATCHES (must investigate before reporting)")
        lines += [f"- {p}" for p in all_problems]
    else:
        lines.append("Recompute check: all summary fields match per-frame data in all three files.")
    (BASE / "b3_verification.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nwrote b3_verification.md")


if __name__ == "__main__":
    main()
