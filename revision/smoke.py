"""CPU 合成 smoke。不读医学图像，不下载权重，不训练。"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from revision.fair_config import training_matrix
from revision.identity import (
    B2_CONTROL_BINARY_VAL,
    RevisionError,
    create_run_dir,
    verify_identity,
)
from revision.metrics import score_mask
from revision.overlays import write_review
from revision.preflight import environment_report
from revision.pretrain import resolve_backbone


def main(argv=None):
    parser = argparse.ArgumentParser(description="CPU synthetic smoke")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def emit(text):
        print(text)
        lines.append(text)

    report = environment_report(out)
    emit(
        f"hostname={report['hostname']} disk_free_bytes={report['disk_free_bytes']} "
        f"cuda={report['cuda']} gpu={report['gpu_name']}"
    )
    gt = np.zeros((8, 8), dtype=np.int16)
    pred = np.zeros((8, 8), dtype=np.int16)
    gt[2:6, 2:6] = 1
    pred[2:6, 2:6] = 2
    scored = score_mask(pred, gt)
    emit(f"binary_foreground_dice={scored['binary_foreground_dice']}")
    emit(f"gt_class_dice={scored['gt_class_dice']}")
    emit(f"named_correct={scored['named_correct']}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_name = f"smoke-{stamp}"
    run = create_run_dir(out, run_name, config={"kind": "cpu-synthetic", "matrix": len(training_matrix())})
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    written = write_review(run, "wrongclass", image, pred, gt)
    emit(f"review={written['panel'].name}")
    try:
        create_run_dir(out, run_name)
    except RevisionError as exc:
        emit(exc.code)
    try:
        verify_identity(
            {
                "filename": "best_control.pt",
                "epoch": 3,
                "best_val_dice": B2_CONTROL_BINARY_VAL,
                "role": "control_head",
            },
            {
                "role": "stage_e_control",
                "backbone_id": "innerlogic_surg_vitb16",
                "geometry": "square768",
                "ckpt_kind": "control_head",
            },
        )
    except RevisionError as exc:
        emit(exc.code)
        for warning in exc.details.get("warnings", []):
            emit(warning)
    try:
        resolve_backbone("innerlogic_surg_vitb16", out / "missing-ils.pth")
    except RevisionError as exc:
        emit(exc.code)
    emit("weights_loaded=False")
    emit("trainer_launched=False")
    (out / "smoke.log").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
