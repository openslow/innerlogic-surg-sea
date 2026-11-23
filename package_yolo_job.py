"""Package the YOLOv11 zero-shot Colab job.

- repo/ : copy of yolo_repo (weights, loader, demo, data)
- sample/: frames copied as-is; masks converted .nrrd -> PNG with the FIXED
  loader (transpose first, then NO resize — native coords).
- sample_list.json: mask_file rewritten to .png
- yolo_run.py copied in.
Then: zip -r yolo_job.zip yolo_job
"""
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import nrrd

WORK = Path(__file__).resolve().parent
JOB = WORK / "yolo_job"


def mask_to_png(nrrd_path: Path, out_path: Path):
    arr, _ = nrrd.read(str(nrrd_path))
    m = arr[:, :, 0].astype(bool).T  # fixed loader: transpose FIRST
    Image.fromarray((m.astype(np.uint8)) * 255).save(out_path)


def main():
    if JOB.exists():
        shutil.rmtree(JOB)
    (JOB / "repo").mkdir(parents=True)
    shutil.copytree(WORK / "yolo_repo", JOB / "repo", dirs_exist_ok=True)
    shutil.copy(WORK / "yolo_run.py", JOB / "yolo_run.py")
    listing = json.loads((WORK / "yolo_sample" / "sample_list.json").read_text())
    for tag in ["local", "external"]:
        (JOB / "sample" / tag / "images").mkdir(parents=True)
        (JOB / "sample" / tag / "masks").mkdir(parents=True)
        for r in listing[tag]:
            src_img = WORK / "yolo_sample" / tag / "images" / r["img_file"]
            shutil.copy(src_img, JOB / "sample" / tag / "images" / r["img_file"])
            src_msk = WORK / "yolo_sample" / tag / "masks" / r["mask_file"]
            png_name = r["mask_file"].replace(".nrrd", ".png")
            mask_to_png(src_msk, JOB / "sample" / tag / "masks" / png_name)
            r["mask_file"] = png_name
    (JOB / "sample_list.json").write_text(json.dumps(listing, indent=1))
    print("packaged", JOB)


if __name__ == "__main__":
    main()
