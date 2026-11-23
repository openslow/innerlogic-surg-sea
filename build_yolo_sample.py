"""YOLOv11 zero-shot sample builder (Stage Y).

Stratified (by instrument class, proportional, seed 7) sample of 100 frames
each from local_test (Center 1) and external_test (Center 2).
Frames + GT masks are extracted from the local sea/sea.zip (read-only).
Output: yolo_sample/{local,external}/{images,masks}/ + sample_list.json
"""
import json
import random
import zipfile
from pathlib import Path

WORK = Path(__file__).resolve().parent
random.seed(7)
N_PER_SET = 100
MIN_PER_CLASS = 2


def alloc(counts, n):
    """Largest-remainder proportional allocation with a per-class floor."""
    total = sum(counts.values())
    out = {}
    for k, c in counts.items():
        out[k] = max(MIN_PER_CLASS, int(n * c / total))
    # fix total by trimming the largest classes / topping up the largest
    while sum(out.values()) > n:
        k = max(out, key=lambda kk: out[kk])
        out[k] -= 1
    while sum(out.values()) < n:
        k = max(counts, key=lambda kk: counts[kk] / max(out[kk], 1))
        out[k] += 1
    return out


def sample_split(manifest_name, tag):
    rows = [json.loads(l) for l in open(WORK / "manifests_paper" / manifest_name)]
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r)
    counts = {k: len(v) for k, v in by_cls.items()}
    plan = alloc(counts, N_PER_SET)
    picked = []
    for cls, m in plan.items():
        pool = by_cls[cls][:]
        random.shuffle(pool)
        for r in pool[:m]:
            picked.append(r)
    random.shuffle(picked)
    assert len(picked) == N_PER_SET, (tag, len(picked))
    return picked, plan, counts


def main():
    z = zipfile.ZipFile(WORK / "sea" / "sea.zip")
    out_root = WORK / "yolo_sample"
    listing = {}
    for manifest_name, tag in [("local_test.jsonl", "local"), ("external_test.jsonl", "external")]:
        picked, plan, counts = sample_split(manifest_name, tag)
        print(tag, "plan:", plan, "pool:", counts)
        d = out_root / tag
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(parents=True, exist_ok=True)
        rows_out = []
        for i, r in enumerate(picked):
            ext = Path(r["image"]).suffix
            img_name = f"{i:03d}{ext}"
            msk_name = f"{i:03d}.nrrd"
            (d / "images" / img_name).write_bytes(z.read(r["image"]))
            (d / "masks" / msk_name).write_bytes(z.read(r["mask"]))
            rows_out.append({**r, "sample_idx": i, "img_file": img_name, "mask_file": msk_name})
        listing[tag] = rows_out
    (out_root / "sample_list.json").write_text(json.dumps(listing, indent=1))
    n_files = sum(1 for _ in out_root.rglob("*") if _.is_file())
    print("wrote", n_files, "files to", out_root)


if __name__ == "__main__":
    main()
