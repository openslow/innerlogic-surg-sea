"""Independent verification for Stage W (worse-frames pairs).

Checks, from the LOCAL copies only (no trust in the Modal run's own logs):
1. All 18 pair JPGs exist in viz_worse_local/ and are 1920x540.
2. Manifest JSON-dice fields match the published Stage C / Stage D result
   JSONs per frame (exact, after the same rounding).
3. delta_dice == d_dice_json - control_dice_json.
4. Manifest area ratios match the published JSONs (within 0.01).
5. Left/right halves of each pair are genuinely different images
   (mean abs pixel diff > 1.0) and neither half is blank.
6. Manifest order: 12 external frames first, then 6 local frames.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

base = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg")
man = json.load(open(base / "viz_worse_local" / "worse_pairs.json"))
cj = json.load(open(base / "results_c_local" / "results_reeval_control.json"))
dj = json.load(open(base / "results_d_local" / "results_d" / "results_finetune.json"))

def frame(j, s, idx):
    for f in j["sets"][s]["frames"]:
        if f["idx"] == idx:
            return f
    raise KeyError((s, idx))

errors = []
assert len(man) == 18, f"manifest has {len(man)} entries, expected 18"
ext = man[:12]
loc = man[12:]
assert all(m["set"] == "external_test" for m in ext), "first 12 must be external"
assert all(m["set"] == "local_test" for m in loc), "last 6 must be local"
print(f"order OK: 12 external + 6 local; sets in manifest: "
      f"{[m['idx'] for m in ext]} / {[m['idx'] for m in loc]}")

for m in man:
    s, i = m["set"], m["idx"]
    cf, df = frame(cj, s, i), frame(dj, s, i)
    if m["control_dice_json"] != cf["dice"]:
        errors.append(f"{s}#{i} control dice {m['control_dice_json']} != JSON {cf['dice']}")
    if m["d_dice_json"] != df["dice"]:
        errors.append(f"{s}#{i} D dice {m['d_dice_json']} != JSON {df['dice']}")
    if abs(m["delta_dice"] - round(df["dice"] - cf["dice"], 4)) > 1e-9:
        errors.append(f"{s}#{i} delta {m['delta_dice']} != {round(df['dice'] - cf['dice'], 4)}")
    if abs(m["control_area"] - cf["area_ratio"]) > 0.01:
        errors.append(f"{s}#{i} control area {m['control_area']} != JSON {cf['area_ratio']}")
    if abs(m["d_area"] - df["area_ratio"]) > 0.01:
        errors.append(f"{s}#{i} D area {m['d_area']} != JSON {df['area_ratio']}")
    if m["cls"] != cf["cls"] or m["patient"] != cf["patient"]:
        errors.append(f"{s}#{i} cls/patient mismatch")
    p = base / "viz_worse_local" / f"{s}_{i:04d}_pair.jpg"
    if not p.exists():
        errors.append(f"missing image {p.name}")
        continue
    im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    if im.shape != (540, 1920, 3):
        errors.append(f"{p.name} shape {im.shape}, expected (540,1920,3)")
    left, right = im[:, :960], im[:, 960:]
    diff = float(np.abs(left - right).mean())
    if diff < 1.0:
        errors.append(f"{p.name} halves nearly identical (mean|diff|={diff:.2f})")
    for name, half in (("left", left), ("right", right)):
        if float(half.std()) < 5:
            errors.append(f"{p.name} {name} half looks blank (std={half.std():.1f})")
    print(f"  {s}#{i} {m['cls'][:12]:12s} {m['patient']} "
          f"Dice {cf['dice']:.3f}->{df['dice']:.3f} (d={df['dice']-cf['dice']:+.3f}) "
          f"area {cf['area_ratio']:.2f}->{df['area_ratio']:.2f} halves-diff={diff:.1f}")

print("\nERRORS:", len(errors))
for e in errors:
    print(" -", e)
print("VERIFY_WORSE:", "PASS" if not errors else "FAIL")
