"""Stage A: download SEA zip inside Modal, extract, enumerate annotated frames,
build patient-disjoint train/val/test manifests. CPU-only function.
Run: .venv/bin/modal run modal_stage_a.py
"""
import json
import time

import modal

app = modal.App("innerlogic-surg-sea")
vol = modal.Volume.from_name("innerlogic-surg", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.1.3", "pillow==10.4.0", "pynrrd==1.1.3", "requests==2.32.3")
    .add_local_file("/home/hatch/workspace/mirage-sea-pilot/space_data/sample_list.json",
                    "/root/sample_list.json")
)

FIGSHARE_URL = "https://ndownloader.figshare.com/files/56202515"
CLS_CANON = {
    "grasping forceps": "grasping forceps", "grasping_forceps": "grasping forceps",
    "bipolar": "bipolar", "drill": "drill", "punch": "punch",
    "scissor": "scissor", "scissors": "scissor", "dissector": "dissector",
}


@app.function(image=image, volumes={"/data": vol}, timeout=1800, scaledown_window=10)
def stage_a():
    import os
    import random
    import urllib.request
    import zipfile
    from pathlib import Path

    import nrrd

    res = {"stage": "A"}
    t0 = time.time()
    base = Path("/data/sea")
    base.mkdir(parents=True, exist_ok=True)
    zipp = base / "sea.zip"

    if not zipp.exists() or zipp.stat().st_size < 9_000_000_000:
        print(f"downloading figshare zip -> {zipp}", flush=True)
        td = time.time()
        urllib.request.urlretrieve(FIGSHARE_URL, str(zipp))
        res["download_s"] = round(time.time() - td, 1)
        res["zip_bytes"] = int(zipp.stat().st_size)
    else:
        res["download_s"] = 0
        res["zip_bytes"] = int(zipp.stat().st_size)
        print(f"zip already present: {res['zip_bytes']} bytes", flush=True)
    print(f"zip size {res['zip_bytes']}, download {res['download_s']}s", flush=True)
    vol.commit()
    print("volume committed after download", flush=True)

    ext = base / "extracted"
    if not (ext / "Spine endoscopic atlas").exists():
        print("extracting zip ...", flush=True)
        te = time.time()
        with zipfile.ZipFile(zipp) as zf:
            zf.extractall(ext)
        res["extract_s"] = round(time.time() - te, 1)
        print(f"extracted in {res['extract_s']}s", flush=True)
    else:
        res["extract_s"] = 0
        print("already extracted", flush=True)
    vol.commit()
    print("volume committed after extraction", flush=True)

    root = ext / "Spine endoscopic atlas"
    res["top_level"] = sorted(os.listdir(root))[:20]
    print("top level:", res["top_level"], flush=True)

    # enumerate annotated frames: image with sibling seg/*.nrrd
    sample = json.load(open("/root/sample_list.json"))
    test_zip_paths = {s["zip_path"] for s in sample}
    test_patients = {s["patient"] for s in sample}
    res["n_test_frames"] = len(sample)
    res["n_test_patients"] = len(test_patients)

    annotated = []
    missing_test = []
    img_exts = {".png", ".jpg", ".jpeg"}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            p = Path(dirpath) / fn
            if p.suffix.lower() not in img_exts:
                continue
            rel = str(p.relative_to(ext))
            segdir = p.parent / "seg"
            stem = p.stem
            cands = list(segdir.glob(stem + ".nrrd")) if segdir.is_dir() else []
            if not cands:
                continue
            parts = p.relative_to(root).parts
            # expected: classified/<channel>/<region>/<patient>/<difficulty>/<cls>/<file>
            try:
                cls_raw = parts[5].lower()
                patient = parts[3]
                difficulty = parts[4]
            except IndexError:
                cls_raw, patient, difficulty = "?", "?", "?"
            annotated.append({
                "image": rel, "mask": str(cands[0].relative_to(ext)),
                "cls": CLS_CANON.get(cls_raw, cls_raw),
                "patient": patient, "split": difficulty,
            })
    res["n_annotated"] = len(annotated)
    from collections import Counter
    res["annotated_by_cls"] = dict(Counter(a["cls"] for a in annotated))
    res["annotated_by_patient_top"] = dict(Counter(a["patient"] for a in annotated).most_common(5))
    res["n_patients_annotated"] = len({a["patient"] for a in annotated})
    print(f"annotated={len(annotated)} patients={res['n_patients_annotated']}", flush=True)
    print("by cls:", res["annotated_by_cls"], flush=True)

    # verify the 100 test frames exist in the zip with masks
    by_img = {a["image"]: a for a in annotated}
    for s in sample:
        if s["zip_path"] not in by_img:
            missing_test.append(s["zip_path"])
    res["test_missing_in_zip"] = missing_test
    print(f"test frames missing in zip: {len(missing_test)}", flush=True)

    # verify mask Segment0_Name matches folder class for test frames (spot check all 100)
    name_mismatch = []
    for s in sample:
        a = by_img.get(s["zip_path"])
        if not a:
            continue
        arr, hdr = nrrd.read(str(ext / a["mask"]))  # pynrrd returns (data, header)
        seg_name = hdr.get("Segment0_Name", "?")
        if CLS_CANON.get(str(seg_name).lower(), str(seg_name).lower()) != CLS_CANON.get(s["cls"].lower(), s["cls"].lower()):
            name_mismatch.append((s["zip_path"], seg_name, s["cls"]))
    res["mask_name_mismatches"] = name_mismatch
    print(f"mask name mismatches: {len(name_mismatch)}", flush=True)

    # patient-disjoint split: train pool = annotated, patient not in test, image not in test
    pool = [a for a in annotated
            if a["patient"] not in test_patients and a["image"] not in test_zip_paths]
    res["pool_n"] = len(pool)
    res["pool_by_cls"] = dict(Counter(a["cls"] for a in pool))
    res["pool_patients"] = len({a["patient"] for a in pool})
    print(f"train pool={len(pool)} patients={res['pool_patients']}", flush=True)

    # val: hold out 2 patients per class-ish -> simplest: hold out 10% of pool patients
    rng = random.Random(1234)
    pool_patients = sorted({a["patient"] for a in pool})
    rng.shuffle(pool_patients)
    n_val_pat = max(3, len(pool_patients) // 10)
    val_patients = set(pool_patients[:n_val_pat])
    val = [a for a in pool if a["patient"] in val_patients]
    train_pool = [a for a in pool if a["patient"] not in val_patients]

    # stratified cap for training: max 500 per class (keeps extraction time sane)
    rng.shuffle(train_pool)
    per_cls = Counter()
    train = []
    for a in train_pool:
        if per_cls[a["cls"]] < 500:
            train.append(a)
            per_cls[a["cls"]] += 1
    res["train_n"] = len(train)
    res["train_by_cls"] = dict(Counter(a["cls"] for a in train))
    res["val_n"] = len(val)
    res["val_patients"] = sorted(val_patients)
    res["val_by_cls"] = dict(Counter(a["cls"] for a in val))

    test = []
    for s in sample:
        a = by_img.get(s["zip_path"])
        test.append({
            "image": s["zip_path"],
            "mask": a["mask"] if a else s["mask_zip_path"],
            "cls": s["cls"], "patient": s["patient"], "split": s["split"],
        })
    res["test_n"] = len(test)

    man_dir = base / "manifests"
    man_dir.mkdir(exist_ok=True)
    for name, rows in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)]:
        with open(man_dir / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res["manifests"] = str(man_dir)
    res["total_s"] = round(time.time() - t0, 1)
    print("STAGE_A_RESULT_JSON=" + json.dumps(res, ensure_ascii=False))
    vol.commit()
    return {k: (str(v) if not isinstance(v, (int, float, bool, list, dict)) else v)
            for k, v in res.items()}


@app.local_entrypoint()
def main():
    t0 = time.time()
    res = stage_a.remote()
    print("\n=== STAGE A RESULT ===")
    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"=== local wall: {time.time()-t0:.1f}s ===")
    with open("/home/hatch/workspace/goals/goal-14/hidden_files/innerlogic-surg/stage_a_result.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
