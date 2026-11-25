"""读取 manifests_paper 里已经写好的 jsonl。划分看文件名，不看 split 字段。"""
import json
from pathlib import Path

from revision.identity import RevisionError
from revision.metrics import CLS_NAMES

OFFICIAL_ROLES = ("train", "val", "local_test", "external_test")
TRAINING_ROLES = ("train", "val")
REQUIRED_FIELDS = ("image", "mask", "cls", "patient")
_FOREGROUND = set(CLS_NAMES[1:])


def load_official_splits(manifest_dir, roles=TRAINING_ROLES):
    """按文件名装入划分。行内的 split 是难度（Difficult/Normal scenario），原样保留。

    不打乱、不按帧重分。缺文件或关键字段立刻失败。
    """
    root = Path(manifest_dir) if manifest_dir else None
    if root is None or not root.is_dir():
        raise RevisionError(
            "manifest_missing",
            "需要 manifests_paper 这种目录，里面已有 train.jsonl 和 val.jsonl。",
            {"path": None if root is None else str(root)},
        )
    unknown = [role for role in roles if role not in OFFICIAL_ROLES]
    if unknown:
        raise RevisionError("manifest_role", "未知划分文件", {"roles": unknown})
    loaded = {}
    for role in roles:
        path = root / f"{role}.jsonl"
        if not path.is_file():
            raise RevisionError(
                "manifest_missing",
                f"缺少 {path.name}。不要另写一份把 split 改成 train/val 的 jsonl。",
                {"path": str(path), "role": role},
            )
        loaded[role] = _read_role(path, role)
        if not loaded[role]:
            raise RevisionError("manifest_empty", f"{path.name} 没有数据行", {"path": str(path)})
    return loaded


def _read_role(path, role):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RevisionError(
                "manifest_field_missing",
                f"{path.name}:{number} 不是 json",
                {"path": str(path), "line": number},
            ) from exc
        missing = [key for key in REQUIRED_FIELDS if raw.get(key) in (None, "")]
        if missing:
            raise RevisionError(
                "manifest_field_missing",
                f"{path.name}:{number} 缺少 {', '.join(missing)}",
                {"path": str(path), "line": number, "missing": missing},
            )
        if raw["cls"] not in _FOREGROUND:
            raise RevisionError(
                "manifest_field_missing",
                f"{path.name}:{number} 的 cls 不在六类器械里",
                {"path": str(path), "line": number, "cls": raw["cls"]},
            )
        row = dict(raw)
        row["difficulty"] = raw.get("split")
        row["partition"] = role
        rows.append(row)
    return rows
