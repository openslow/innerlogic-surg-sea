"""Checkpoint 身份与新运行目录。文件名不是身份。"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# 这两个标量都是历史 binary foreground val Dice，不是 class-aware，也不是身份。
# 接近 B2 标量时只警告。身份看调用方本地记录的 sha256、架构和 ckpt 种类。
B2_CONTROL_BINARY_VAL = 0.8682456
STAGE_E_CONTROL_BINARY_VAL = 0.9394298

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


_COMPARE_KEYS = (
    "role", "backbone_id", "geometry", "embed_dim", "depth",
    "patch_size", "num_classes", "train_mode",
)


def verify_identity(record, expect):
    """身份是期望 sha256、张量架构和 ckpt 种类。best_val 标量只作历史警告。"""
    warnings = []
    if _close(record.get("best_val_dice"), B2_CONTROL_BINARY_VAL):
        warnings.append("historical_b2_scalar")
    if not expect.get("sha256"):
        raise RevisionError(
            "expected_sha_required",
            "调用方要在本地 manifest 写上合法取得的权重 sha256。仓库不预存私有文件哈希，"
            "也不用 best_val 标量认定身份。",
            {"warnings": warnings, "filename_ignored": record.get("filename")},
        )
    if not record.get("sha256"):
        raise RevisionError(
            "incomplete_metadata",
            "checkpoint 没有 sha256，拒绝按文件名或 best_val 加载。",
            {"warnings": warnings, "missing": ["sha256"], "filename_ignored": record.get("filename")},
        )
    if str(record["sha256"]).lower() != str(expect["sha256"]).lower():
        raise RevisionError(
            "hash_mismatch",
            "checkpoint sha256 与本地 manifest 的期望值不一致",
            {"warnings": warnings, "found": record.get("sha256"), "expected": expect.get("sha256")},
        )
    if "ckpt_kind" in expect and record.get("ckpt_kind") != expect.get("ckpt_kind"):
        raise RevisionError(
            "ckpt_kind_mismatch",
            "full finetune（含 backbone_state）和 control head（只有 head_state）不能混用。",
            {
                "warnings": warnings,
                "found": record.get("ckpt_kind"),
                "expected": expect.get("ckpt_kind"),
            },
        )
    for key in _COMPARE_KEYS:
        if expect.get(key) in (None, ""):
            continue
        if record.get(key) in (None, ""):
            if key == "geometry":
                warnings.append("geometry_not_in_checkpoint")
                continue
            raise RevisionError(
                "incomplete_metadata",
                f"checkpoint 缺少 {key}",
                {"warnings": warnings, "missing": [key]},
            )
        if record.get(key) != expect.get(key):
            code = "geometry_mismatch" if key == "geometry" else "metadata_mismatch"
            raise RevisionError(
                code,
                f"{key} 不一致",
                {"warnings": warnings, "key": key, "found": record.get(key), "expected": expect.get(key)},
            )
    return {
        "accepted": True,
        "warnings": warnings,
        "sha256": str(record["sha256"]).lower(),
        "ckpt_kind": record.get("ckpt_kind"),
        "filename_ignored": record.get("filename"),
    }


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
