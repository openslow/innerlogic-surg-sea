"""Checkpoint 身份与新运行目录。文件名不是身份。"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# 这两个标量都是历史 binary foreground val Dice，不是 class-aware。
# Stage E control 的正确权重字节尚未钉死，因此即使标量吻合也不能接受。
B2_CONTROL_BINARY_VAL = 0.8682456
STAGE_E_CONTROL_BINARY_VAL = 0.9394298

_REQUIRED = (
    "role", "backbone_id", "geometry", "epoch", "best_val_dice",
    "metric_name", "sha256", "head_init",
)


class RevisionError(Exception):
    def __init__(self, code, message, details=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.details = details or {}


def _close(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def verify_identity(record, expect):
    """对照 manifest 期望检查 checkpoint 记录。`filename` 不参与认定。"""
    found = record.get("best_val_dice")
    expected_val = expect.get("best_val_dice", STAGE_E_CONTROL_BINARY_VAL)
    if _close(found, B2_CONTROL_BINARY_VAL) and (
        expect.get("role") == "stage_e_control"
        or _close(expected_val, STAGE_E_CONTROL_BINARY_VAL)
    ):
        raise RevisionError(
            "known_wrong_head",
            "best_val_dice 0.8682456 是旧 B2（epoch 3，stretch 768x432）的头，"
            "不是 Stage E control（binary val 0.9394298）。文件名 best_control.pt 不能当作身份。",
            {
                "found_best_val_dice": float(found),
                "expected_best_val_dice": float(expected_val),
                "filename_ignored": record.get("filename"),
            },
        )
    missing = [key for key in _REQUIRED if key not in record or record[key] in (None, "")]
    if missing:
        raise RevisionError(
            "incomplete_metadata",
            "checkpoint 缺少身份字段，拒绝按文件名加载：" + ", ".join(missing),
            {"missing": missing, "filename_ignored": record.get("filename")},
        )
    if record.get("role") == "stage_e_control" or expect.get("role") == "stage_e_control":
        raise RevisionError(
            "unverified_historical_head",
            "Stage E control 的正确 head 还没有可核对的 sha256。标量吻合不够，拒绝加载。",
            {"best_val_dice": record.get("best_val_dice"), "filename_ignored": record.get("filename")},
        )
    if record.get("geometry") != expect.get("geometry"):
        raise RevisionError(
            "geometry_mismatch",
            f"几何不一致：checkpoint={record.get('geometry')} expect={expect.get('geometry')}",
            {"found": record.get("geometry"), "expected": expect.get("geometry")},
        )
    if record.get("sha256") != expect.get("sha256"):
        raise RevisionError(
            "hash_mismatch",
            "checkpoint sha256 与 manifest 不一致",
            {"found": record.get("sha256"), "expected": expect.get("sha256")},
        )
    for key in ("role", "backbone_id", "metric_name", "head_init"):
        if record.get(key) != expect.get(key):
            raise RevisionError(
                "metadata_mismatch",
                f"{key} 不一致",
                {"key": key, "found": record.get(key), "expected": expect.get(key)},
            )
    if "epoch" in expect and record.get("epoch") != expect.get("epoch"):
        raise RevisionError("metadata_mismatch", "epoch 不一致", {"key": "epoch"})
    if not _close(record.get("best_val_dice"), expect.get("best_val_dice")):
        raise RevisionError("metadata_mismatch", "best_val_dice 不一致", {"key": "best_val_dice"})
    accepted = {key: record[key] for key in _REQUIRED}
    accepted["accepted"] = True
    accepted["filename_ignored"] = record.get("filename")
    return accepted


def plan_resume(record, config):
    """公平配置的断点：不从 ep10 续，只在同一配置、本地 val 选择下继续。"""
    epoch = int(record.get("epoch", -1))
    if record.get("continued_from_ep10") or epoch >= 10:
        raise RevisionError(
            "resume_forbidden",
            "禁止从 ep10 或更晚的历史微调续训。公平对比从随机 head 重新开始。",
            {"epoch": epoch, "continued_from_ep10": record.get("continued_from_ep10")},
        )
    if record.get("selection_split", "val") != "val" or config.get("selection_split", "val") != "val":
        raise RevisionError("resume_forbidden", "checkpoint 只能按本地 val 选择", {})
    for key in ("backbone_id", "geometry", "seed", "train_mode", "role"):
        if key in config and record.get(key) != config.get(key):
            raise RevisionError(
                "resume_forbidden",
                f"续训配置 {key} 与 checkpoint 不一致",
                {"key": key},
            )
    if "sha256" in record and "sha256" in config and record["sha256"] != config["sha256"]:
        raise RevisionError("hash_mismatch", "续训 sha256 不一致", {})
    budget = int(config.get("epochs", 6))
    next_epoch = epoch + 1
    if next_epoch >= budget:
        raise RevisionError("resume_complete", "已经跑满 epoch 预算", {"next_epoch": next_epoch})
    return {"next_epoch": next_epoch, "selection_split": "val"}


def _historical_part(part):
    return part.startswith("results_") or part.startswith("viz_")


def revision_source_sha256():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def create_run_dir(parent, name, config=None, manifest_path=None, checkpoint_path=None):
    parent = Path(parent)
    full = parent / name
    for part in full.parts:
        if _historical_part(part):
            raise RevisionError(
                "historical_protected",
                f"路径含历史结果目录 {part}，拒绝写入",
                {"path": str(full)},
            )
    if not (name.startswith("smoke-") or name.startswith("fairrun-")):
        raise RevisionError("run_name", "新运行目录必须以 smoke- 或 fairrun- 开头", {"name": name})
    parent.mkdir(parents=True, exist_ok=True)
    try:
        full.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise RevisionError("nonoverwrite", f"运行目录已存在：{full}", {"path": str(full)}) from exc
    record = {
        "run_id": name,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_rev": git_rev(),
        "revision_sources_sha256": revision_source_sha256(),
        "config": config,
        "manifest_sha256": _sha_path(manifest_path) if manifest_path else None,
        "checkpoint_sha256": _sha_path(checkpoint_path) if checkpoint_path else None,
        "continued_from_ep10": False,
    }
    (full / "RUN_RECORD.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    return full


def _sha_path(path):
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(child.relative_to(path).as_posix().encode())
            digest.update(hashlib.sha256(child.read_bytes()).hexdigest().encode())
        return digest.hexdigest()
    raise RevisionError("missing_path", f"找不到 {path}", {"path": str(path)})
