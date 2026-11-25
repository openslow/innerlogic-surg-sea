"""原图 / 仅预测 / GT / TP-FP-FN。无 GT 时不写 Dice。"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from revision.identity import RevisionError
from revision.metrics import score_mask

# 误差图用纯色，方便复核；不把 FN 画成 GT 的绿色。
TP_COLOR = (0, 170, 0)
FP_COLOR = (220, 30, 30)
FN_COLOR = (20, 80, 240)
WRONG_COLOR = (210, 0, 200)
_GAP = 4
_CLASS_COLOR = {
    1: (255, 200, 0),
    2: (0, 200, 255),
    3: (255, 120, 0),
    4: (180, 0, 255),
    5: (0, 255, 180),
    6: (255, 0, 120),
}


def write_review(out_dir, stem, image_rgb, pred, gt=None, ignore_label=255):
    out_dir = Path(out_dir)
    image = np.asarray(image_rgb)
    pred_arr = np.asarray(pred)
    if image.ndim != 3 or image.shape[:2] != pred_arr.shape:
        raise ValueError("image 必须是与 mask 同高宽的 RGB")
    gt_arr = None if gt is None else np.asarray(gt)
    panel_path = out_dir / f"{stem}_panel.png"
    pred_path = out_dir / f"{stem}_pred_class_id.png"
    meta_path = out_dir / f"{stem}_meta.json"
    targets = [panel_path, pred_path, meta_path]
    error_path = gt_path = None
    if gt_arr is not None:
        error_path = out_dir / f"{stem}_error_map.png"
        gt_path = out_dir / f"{stem}_gt_class_id.png"
        targets.extend([error_path, gt_path])
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise RevisionError("nonoverwrite", "复核文件已存在", {"paths": existing})

    pred_tile = _paint_classes(image, pred_arr)
    if gt_arr is None:
        tiles = [image, pred_tile]
        scores = None
        dice_displayed = False
        error = None
    else:
        error = _error_map(pred_arr, gt_arr, ignore_label)
        tiles = [image, pred_tile, _paint_classes(image, gt_arr), error]
        scored = score_mask(pred_arr, gt_arr, ignore_label=ignore_label)
        scores = {
            "binary_foreground_dice": scored["binary_foreground_dice"],
            "binary_foreground_iou": scored["binary_foreground_iou"],
            "gt_class_dice": scored["gt_class_dice"],
            "pred_name": scored["pred_name"],
            "gt_name": scored["gt_name"],
            "named_correct": scored["named_correct"],
        }
        dice_displayed = True
    panel = _row(tiles)
    Image.fromarray(panel, mode="RGB").save(panel_path)
    Image.fromarray(pred_arr.astype(np.uint8), mode="L").save(pred_path)
    if error is not None:
        Image.fromarray(error, mode="RGB").save(error_path)
        Image.fromarray(gt_arr.astype(np.uint8), mode="L").save(gt_path)
    meta = {
        "stem": stem,
        "dice_displayed": dice_displayed,
        "displayed_scores": scores,
        "native_class_id": True,
        "legend_zh": _legend(dice_displayed),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    _write_index(out_dir)
    result = {"panel": panel_path, "pred_class_id": pred_path, "meta": meta_path}
    if error_path is not None:
        result["error_map"] = error_path
        result["gt_class_id"] = gt_path
    return result


def _paint_classes(image, labels):
    out = np.array(image, copy=True)
    for cls_id, color in _CLASS_COLOR.items():
        out[labels == cls_id] = color
    return out


def _error_map(pred, gt, ignore_label):
    err = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != ignore_label
    tp = valid & (gt > 0) & (pred == gt)
    wrong = valid & (gt > 0) & (pred > 0) & (pred != gt)
    fp = valid & (gt == 0) & (pred > 0)
    fn = valid & (gt > 0) & (pred == 0)
    err[tp] = TP_COLOR
    err[fp] = FP_COLOR
    err[fn] = FN_COLOR
    err[wrong] = WRONG_COLOR
    return err


def _row(tiles):
    height, width = tiles[0].shape[:2]
    total = len(tiles) * width + (len(tiles) - 1) * _GAP
    panel = np.full((height, total, 3), 255, dtype=np.uint8)
    x = 0
    for tile in tiles:
        panel[:, x:x + width] = tile
        x += width + _GAP
    return panel


def _legend(dice_displayed):
    rows = [
        "原图 | 仅预测（类别色，不是分数）",
    ]
    if dice_displayed:
        rows.append("GT | 误差图：绿 TP，红 FP（GT 为背景），蓝 FN（预测为背景），品红 错类")
        rows.append("图上的两个分数分别是 binary foreground Dice 和 GT 类别的 class-aware Dice")
    else:
        rows.append("无 GT：不显示 Dice，也不画 TP/FP/FN")
    return rows


def _write_index(out_dir):
    cards = []
    for meta_path in sorted(out_dir.glob("*_meta.json")):
        meta = json.loads(meta_path.read_text())
        stem = meta["stem"]
        scores = meta["displayed_scores"]
        if scores is None:
            caption = "无 GT，不显示 Dice"
        else:
            caption = (
                f"binary foreground Dice {scores['binary_foreground_dice']:.4f} | "
                f"gt-class Dice {scores['gt_class_dice']}"
            )
        cards.append(
            f"<figure><img src=\"{stem}_panel.png\" alt=\"{stem}\"/>"
            f"<figcaption>{stem}：{caption}</figcaption>"
            f"<p>原生 class-id：{stem}_pred_class_id.png</p></figure>"
        )
    legend = "<ul><li>绿 TP：预测类别与 GT 相同，且不是背景</li>" \
             "<li>红 FP：预测为前景，GT 为背景</li>" \
             "<li>蓝 FN：GT 为前景，预测为背景</li>" \
             "<li>品红：两边都是前景但类别不同，不算 TP</li></ul>"
    html = (
        "<!DOCTYPE html><meta charset=\"utf-8\"><title>复核</title>"
        "<h1>合成复核样例</h1>" + legend + "\n".join(cards)
    )
    (out_dir / "index.html").write_text(html)
