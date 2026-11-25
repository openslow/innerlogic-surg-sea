"""Binary foreground、class-aware 与命名指标。三者不互相填数。"""
from collections import defaultdict

from revision.identity import RevisionError

CLS_NAMES = [
    "background", "grasping forceps", "bipolar", "drill",
    "punch", "scissor", "dissector",
]
N_FOREGROUND = 6


def _dice_iou(tp, fp, fn):
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, True
    denom_d = 2 * tp + fp + fn
    denom_i = tp + fp + fn
    return (2 * tp / denom_d if denom_d else 0.0,
            tp / denom_i if denom_i else 0.0,
            False)


def score_mask(pred, gt, ignore_label=255):
    """像素级评分。历史 `dice` 键不在返回值里，避免和 class-aware 混用。"""
    pred = _as_2d(pred)
    gt = _as_2d(gt)
    if pred.shape != gt.shape:
        raise ValueError(f"pred {pred.shape} 与 gt {gt.shape} 形状不同")
    valid = gt != ignore_label
    pred_fg = (pred > 0) & valid
    gt_fg = (gt > 0) & valid
    binary_tp = int((pred_fg & gt_fg).sum())
    binary_fp = int((pred_fg & ~gt_fg).sum())
    binary_fn = int((~pred_fg & gt_fg).sum())
    binary_dice, binary_iou, empty = _dice_iou(binary_tp, binary_fp, binary_fn)

    per_class = {}
    for cls_id in range(1, N_FOREGROUND + 1):
        p = (pred == cls_id) & valid
        g = (gt == cls_id) & valid
        tp = int((p & g).sum())
        fp = int((p & ~g).sum())
        fn = int((~p & g).sum())
        if tp == 0 and fp == 0 and fn == 0:
            dice = iou = None
        else:
            dice, iou, _ = _dice_iou(tp, fp, fn)
        per_class[cls_id] = {"tp": tp, "fp": fp, "fn": fn, "dice": dice, "iou": iou}

    present = [cls_id for cls_id in range(1, N_FOREGROUND + 1)
               if int(((gt == cls_id) & valid).sum()) > 0]
    if len(present) == 1:
        gt_name = CLS_NAMES[present[0]]
        gt_class_dice = per_class[present[0]]["dice"]
        naming_defined = True
    else:
        gt_name = None
        gt_class_dice = None
        naming_defined = False
    pred_name = _dominant(pred, valid)
    return {
        "binary_foreground_dice": binary_dice,
        "binary_foreground_iou": binary_iou,
        "binary_tp": binary_tp,
        "binary_fp": binary_fp,
        "binary_fn": binary_fn,
        "empty_agreement": empty,
        "gt_class_dice": gt_class_dice,
        "gt_name": gt_name,
        "pred_name": pred_name,
        "naming_defined": naming_defined,
        "named_correct": bool(naming_defined and pred_name == gt_name),
        "class_fn_only": int((valid & (gt > 0) & (pred == 0)).sum()),
        "class_fp_only": int((valid & (gt == 0) & (pred > 0)).sum()),
        "class_wrong": int((valid & (gt > 0) & (pred > 0) & (pred != gt)).sum()),
        "class_tp": int((valid & (gt > 0) & (pred == gt)).sum()),
        "per_class": per_class,
    }


def _as_2d(array):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("需要 numpy") from exc
    out = np.asarray(array)
    if out.ndim != 2:
        raise ValueError("mask 必须是二维 class-id")
    return out


def _dominant(pred, valid):
    import numpy as np
    best_id, best_n = None, 0
    for cls_id in range(1, N_FOREGROUND + 1):
        n = int(((pred == cls_id) & valid).sum())
        if n > best_n:
            best_id, best_n = cls_id, n
    if best_id is None:
        return "none"
    return CLS_NAMES[best_id]


def weighting_means(rows, key):
    if not rows:
        raise ValueError("没有帧")
    frame_vals = [float(row[key]) for row in rows]
    by_patient = defaultdict(list)
    for row in rows:
        by_patient[row["patient"]].append(float(row[key]))
    patient_vals = [sum(vals) / len(vals) for vals in by_patient.values()]
    return {
        "frame_equal": sum(frame_vals) / len(frame_vals),
        "patient_equal": sum(patient_vals) / len(patient_vals),
        "n_frames": len(rows),
        "n_patients": len(by_patient),
    }


def _macro_from_counts(per_class):
    dices, ious = [], []
    for item in per_class.values():
        if item["dice"] is None:
            continue
        dices.append(item["dice"])
        ious.append(item["iou"])
    if not dices:
        return None, None
    return sum(dices) / len(dices), sum(ious) / len(ious)


def _pool(frames):
    pooled = {cls_id: {"tp": 0, "fp": 0, "fn": 0} for cls_id in range(1, N_FOREGROUND + 1)}
    for frame in frames:
        for cls_id, item in frame["per_class"].items():
            pooled[cls_id]["tp"] += item["tp"]
            pooled[cls_id]["fp"] += item["fp"]
            pooled[cls_id]["fn"] += item["fn"]
    out = {}
    for cls_id, item in pooled.items():
        if item["tp"] == 0 and item["fp"] == 0 and item["fn"] == 0:
            dice = iou = None
        else:
            dice, iou, _ = _dice_iou(item["tp"], item["fp"], item["fn"])
        out[cls_id] = {**item, "dice": dice, "iou": iou, "name": CLS_NAMES[cls_id]}
    return out


def _naming(frames):
    usable = [frame for frame in frames if frame.get("naming_defined")]
    if not usable:
        return {"accuracy_frame_equal": None, "accuracy_patient_equal": None,
                "macro_f1": None, "n": 0}
    acc = weighting_means(
        [{"patient": frame["patient"], "hit": 1.0 if frame["named_correct"] else 0.0}
         for frame in usable],
        "hit",
    )
    labels = []
    for frame in usable:
        if frame["gt_name"] not in labels:
            labels.append(frame["gt_name"])
    f1s = []
    for label in labels:
        tp = sum(frame["gt_name"] == label and frame["pred_name"] == label for frame in usable)
        fp = sum(frame["gt_name"] != label and frame["pred_name"] == label for frame in usable)
        fn = sum(frame["gt_name"] == label and frame["pred_name"] != label for frame in usable)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    confusion = {}
    for frame in usable:
        row = confusion.setdefault(frame["gt_name"], {})
        row[frame["pred_name"]] = row.get(frame["pred_name"], 0) + 1
    return {
        "accuracy_frame_equal": acc["frame_equal"],
        "accuracy_patient_equal": acc["patient_equal"],
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "confusion": confusion,
        "labels": labels,
        "n": len(usable),
    }


def summarize_frames(frames):
    if any("patient" not in frame for frame in frames):
        raise ValueError("每帧都需要 patient")
    binary = weighting_means(frames, "binary_foreground_dice")
    semantic_rows = [frame for frame in frames if frame["gt_class_dice"] is not None]
    semantic = weighting_means(semantic_rows, "gt_class_dice") if semantic_rows else None
    pooled = _pool(frames)
    macro_dice, macro_iou = _macro_from_counts(pooled)
    by_patient = defaultdict(list)
    for frame in frames:
        by_patient[frame["patient"]].append(frame)
    patient_macros = []
    for group in by_patient.values():
        dice, _iou = _macro_from_counts(_pool(group))
        if dice is not None:
            patient_macros.append(dice)
    return {
        "binary_foreground": binary,
        "gt_class_dice": semantic,
        "class_aware_pooled": {
            "per_class": {CLS_NAMES[k]: v for k, v in pooled.items()},
            "pixel_pooled_macro_dice": macro_dice,
            "pixel_pooled_macro_iou": macro_iou,
            "patient_equal_macro_dice": (
                sum(patient_macros) / len(patient_macros) if patient_macros else None
            ),
        },
        "naming": _naming(frames),
        "n_frames": len(frames),
        "n_semantic_undefined": sum(frame["gt_class_dice"] is None for frame in frames),
    }


def paired_patient_bootstrap(rows_a, rows_b, key, n_boot, seed):
    """按患者簇配对抽样。区间是百分位区间，并扩到包含点估计。"""
    import numpy as np

    def _group(rows):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["patient"]].append(float(row[key]))
        return grouped

    grouped_a, grouped_b = _group(rows_a), _group(rows_b)
    common = sorted(set(grouped_a) & set(grouped_b))
    if not common:
        raise RevisionError("bootstrap_no_overlap", "两组没有共同患者，不能做配对 bootstrap", {})
    deltas = {
        patient: (sum(grouped_a[patient]) / len(grouped_a[patient])
                  - sum(grouped_b[patient]) / len(grouped_b[patient]))
        for patient in common
    }
    point = sum(deltas.values()) / len(deltas)
    rng = np.random.default_rng(seed)
    names = np.array(common)
    samples = []
    for _ in range(int(n_boot)):
        draw = rng.choice(names, size=len(names), replace=True)
        samples.append(float(sum(deltas[name] for name in draw) / len(draw)))
    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    return {
        "point_patient_equal_delta": float(point),
        "ci_low": float(min(low, point)),
        "ci_high": float(max(high, point)),
        "cluster": "patient",
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_patients": len(common),
        "score": key,
    }


_UNRECOVERABLE = [
    "class_aware_pixel_dice",
    "class_aware_per_class_dice",
    "class_aware_miou",
    "tp_fp_fn_maps",
]


def describe_recoverable(payload):
    """历史 JSON 能重算什么。binary 分数不会写进 class-aware 字段。"""
    recoverable = {
        "class_aware_pixel_dice": None,
        "by_class_is_class_aware": False,
    }
    notes = []
    frames = _historical_frames(payload)
    if frames is not None:
        notes.append(
            "历史字段 dice/iou 是 binary foreground（argmax 后 pred!=0 对二值 GT），"
            "不是 class-aware Dice。"
        )
        notes.append(
            "named_correct 只表示面积最大的类别名是否碰上帧标签；"
            "少数像素仍可能是别的类，所以不能由它反推像素级 class Dice。"
        )
        dices = [float(frame["dice"]) for frame in frames if "dice" in frame]
        recoverable["binary_foreground_dice_frame_equal"] = (
            sum(dices) / len(dices) if dices else None
        )
        if dices and all("patient" in frame for frame in frames):
            recoverable["weighting_available"] = ["frame", "patient"]
            recoverable["binary_foreground_dice_patient_equal"] = weighting_means(
                [{"patient": frame["patient"], "binary_foreground_dice": float(frame["dice"])}
                 for frame in frames],
                "binary_foreground_dice",
            )["patient_equal"]
        else:
            recoverable["weighting_available"] = ["frame"] if dices else []
        if frames and all("named_correct" in frame for frame in frames):
            recoverable["naming_accuracy_frame_equal"] = (
                sum(bool(frame["named_correct"]) for frame in frames) / len(frames)
            )
        if frames and all("pred_name" in frame and "cls" in frame for frame in frames):
            confusion = {}
            for frame in frames:
                row = confusion.setdefault(frame["cls"], {})
                row[frame["pred_name"]] = row.get(frame["pred_name"], 0) + 1
            recoverable["naming_confusion"] = confusion
            notes.append("有 pred_name 和 cls 时可以重算命名混淆；这仍然不是像素级 class Dice。")
        return {"recoverable": recoverable, "unrecoverable": list(_UNRECOVERABLE), "notes": notes}
    if "mean_dice" in payload:
        notes.append(
            "只有标量 mean_dice。按历史 eval_rows，它是帧等权 binary foreground Dice。"
            "没有逐帧记录就做不了患者等权、命名、bootstrap 或 class-aware。"
        )
        if "by_class" in payload:
            notes.append(
                "by_class.*.dice 是按帧标签分组后的 binary foreground Dice，不是该类的像素 Dice。"
            )
        recoverable["binary_foreground_dice_frame_equal"] = float(payload["mean_dice"])
        recoverable["weighting_available"] = ["frame"]
        extra = list(_UNRECOVERABLE) + ["patient_equal", "naming", "patient_cluster_bootstrap"]
        return {"recoverable": recoverable, "unrecoverable": extra, "notes": notes}
    raise RevisionError(
        "unreadable_result",
        "认不出的结果 JSON：需要 mean_dice、frames，或 sets.*.frames",
        {},
    )


def _historical_frames(payload):
    if isinstance(payload, dict) and "frames" in payload:
        return list(payload["frames"])
    if isinstance(payload, dict) and "sets" in payload:
        frames = []
        for item in payload["sets"].values():
            frames.extend(item.get("frames", []))
        return frames
    return None
