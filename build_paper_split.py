"""Build paper-aligned SEA splits locally from sea.zip (no repo code, stdlib only).

Paper protocol (Sci Data 2025, DOI 10.1038/s41597-025-05897-7):
train 80% (all Center 1), local test 10%, external test 10% (all Center 2).
Exact patient IDs not published -> protocol-aligned reproduction, seed 7, patient-disjoint.
"""
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path

BASE = Path("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg")
ZIP = BASE / "sea" / "sea.zip"
OUT = BASE / "manifests_paper"
OUT.mkdir(exist_ok=True)

CLS_CANON = {
    "grasping forceps": "grasping forceps", "grasping_forceps": "grasping forceps",
    "bipolar": "bipolar", "drill": "drill", "punch": "punch",
    "scissor": "scissor", "scissors": "scissor", "dissector": "dissector",
}
IMG_EXTS = {".png", ".jpg", ".jpeg"}

z = zipfile.ZipFile(ZIP)
names = set(z.namelist())
print("zip entries:", len(names))

# annotated = image with sibling seg/<stem>.nrrd
annotated, anomalies = [], []
img_re = re.compile(r"^Spine endoscopic atlas/(.+)\.(png|jpg|jpeg)$", re.I)
for n in sorted(names):
    m = img_re.match(n)
    if not m:
        continue
    stem_path = n.rsplit(".", 1)[0]
    seg = stem_path.rsplit("/", 1)[0] + "/seg/" + stem_path.rsplit("/", 1)[1] + ".nrrd"
    if seg not in names:
        continue
    rel = n[len("Spine endoscopic atlas/"):]
    parts = rel.split("/")
    # expected: classified/<channel>/<region>/<patient>/<difficulty>/<cls>/<file>
    if len(parts) != 7 or parts[0] != "classified":
        anomalies.append({"image": n, "mask": seg, "parts": parts})
        continue
    _top, channel, region, patient, difficulty, cls_raw, _fn = parts
    cls = CLS_CANON.get(cls_raw.lower(), cls_raw.lower())
    # center by patient prefix, case-insensitive: dataset has one lowercase
    # folder "p46" (filenames P46-1-xxxx) -> Center 1 like all P-patients
    center = "C1" if patient.upper().startswith("P") else ("C2" if patient.upper().startswith("T") else "??")
    annotated.append({"image": n, "mask": seg, "cls": cls, "patient": patient,
                      "split": difficulty, "center": center,
                      "channel": channel, "region": region})

print("annotated:", len(annotated), "| anomalies:", len(anomalies))
print("by center:", dict(Counter(a["center"] for a in annotated)))
print("by cls:", dict(Counter(a["cls"] for a in annotated)))
pat = Counter(a["patient"] for a in annotated)
print("n patients:", len(pat))
p_pats = sorted({a["patient"] for a in annotated if a["center"] == "C1"})
t_pats = sorted({a["patient"] for a in annotated if a["center"] == "C2"})
unk = sorted({a["patient"] for a in annotated if a["center"] == "??"})
print("C1 patients:", len(p_pats), "frames:", sum(1 for a in annotated if a["center"] == "C1"))
print("C2 patients:", len(t_pats), "frames:", sum(1 for a in annotated if a["center"] == "C2"))
print("unknown-center patients:", unk)
print("anomaly examples:", json.dumps(anomalies[:8], ensure_ascii=False, indent=1))
# where do anomalies live?
print("anomaly path prefixes:", dict(Counter("/".join(a["parts"][:3]) for a in anomalies)))

# ---- splits, seed 7 ----
rng = random.Random(7)
by_pat = {}
for a in annotated:
    by_pat.setdefault(a["patient"], []).append(a)

external = [a for a in annotated if a["center"] == "C2"]

# local test: shuffle C1 patients, greedy fill to 1020..1066 frames
c1 = p_pats[:]
rng.shuffle(c1)
local_test_pats, lt_n = [], 0
for p in c1:
    n = len(by_pat[p])
    if lt_n >= 1020:
        break
    if lt_n + n > 1066:
        continue
    local_test_pats.append(p)
    lt_n += n
print("local_test patients:", len(local_test_pats), "frames:", lt_n)

# val: ~5 patients, 300..500 frames, from remaining C1 patients (same shuffled order)
remaining = [p for p in c1 if p not in set(local_test_pats)]
val_pats, v_n = [], 0
for p in remaining:
    n = len(by_pat[p])
    if len(val_pats) >= 5 and v_n >= 300:
        break
    if v_n + n > 500:
        continue
    val_pats.append(p)
    v_n += n
print("val patients:", val_pats, "frames:", v_n)

train_pats = [p for p in remaining if p not in set(val_pats)]
train = [a for p in train_pats for a in by_pat[p]]
val = [a for p in val_pats for a in by_pat[p]]
local_test = [a for p in local_test_pats for a in by_pat[p]]

# sanity: patient disjointness
def pset(rows):
    return {r["patient"] for r in rows}
assert not (pset(train) & pset(val)), "train/val patient overlap"
assert not (pset(train) & pset(local_test)), "train/local_test overlap"
assert not (pset(train) & pset(external)), "train/external overlap"
assert not (pset(val) & pset(local_test)), "val/local_test overlap"
assert not (pset(val) & pset(external)), "val/external overlap"
assert not (pset(local_test) & pset(external)), "local/external overlap"
# frame disjointness
def fset(rows):
    return {r["image"] for r in rows}
allf = fset(train) | fset(val) | fset(local_test) | fset(external)
print("frames covered:", len(allf), "of", len(annotated),
      "| uncovered:", len(annotated) - len(allf))
uncovered = [a for a in annotated if a["image"] not in allf]
print("uncovered by center:", dict(Counter(a["center"] for a in uncovered)))

# old100 bridge manifest + overlap analysis
sample = json.load(open("/home/hatch/workspace/mirage-sea-pilot/space_data/sample_list.json"))
old100 = []
for s in sample:
    old100.append({"image": s["zip_path"], "mask": s["mask_zip_path"],
                   "cls": s["cls"], "patient": s["patient"], "split": s["split"]})
o_pat = {r["patient"] for r in old100}
o_img = {r["image"] for r in old100}
overlap = {
    "old100_patients": len(o_pat),
    "old100_patients_in_train": sorted(o_pat & pset(train)),
    "old100_patients_in_val": sorted(o_pat & pset(val)),
    "old100_patients_in_local_test": sorted(o_pat & pset(local_test)),
    "old100_patients_in_external": sorted(o_pat & pset(external)),
    "old100_frames_in_train": len(o_img & fset(train)),
    "old100_frames_in_val": len(o_img & fset(val)),
    "old100_frames_in_local_test": len(o_img & fset(local_test)),
    "old100_frames_in_external": len(o_img & fset(external)),
    "old100_frames_not_in_new_splits": len(o_img - allf),
}
print("overlap:", json.dumps({k: (len(v) if isinstance(v, list) else v)
                              for k, v in overlap.items()}, ensure_ascii=False))

def dump(name, rows):
    rows = sorted(rows, key=lambda r: (r["patient"], r["image"]))
    with open(OUT / name, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("image", "mask", "cls", "patient", "split")},
                               ensure_ascii=False) + "\n")
    return len(rows)

counts = {
    "train": dump("train.jsonl", train),
    "val": dump("val.jsonl", val),
    "local_test": dump("local_test.jsonl", local_test),
    "external_test": dump("external_test.jsonl", external),
    "old100": dump("old100.jsonl", old100),
}
summary = {
    "seed": 7,
    "protocol": "paper-aligned reproduction (Sci Data 2025, DOI 10.1038/s41597-025-05897-7): "
                "train 80% all Center 1, local test ~10% Center 1, external test ~10% all Center 2; "
                "exact paper patient IDs not published, so splits are protocol-aligned, not identical frames.",
    "n_annotated": len(annotated),
    "n_anomalies_excluded": len(anomalies),
    "counts": counts,
    "counts_by_cls": {k: dict(Counter(r["cls"] for r in rows))
                      for k, rows in [("train", train), ("val", val),
                                      ("local_test", local_test), ("external_test", external)]},
    "train_patients": sorted(pset(train)),
    "val_patients": sorted(pset(val)),
    "local_test_patients": sorted(pset(local_test)),
    "external_test_patients": sorted(pset(external)),
    "n_train_patients": len(pset(train)),
    "n_val_patients": len(pset(val)),
    "n_local_test_patients": len(pset(local_test)),
    "n_external_patients": len(pset(external)),
    "overlap_old100": overlap,
    "anomalies": anomalies,
}
with open(OUT / "split_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("counts:", counts)
print("wrote manifests to", OUT)
