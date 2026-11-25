"""行为测试：指标口径、误差图、checkpoint 身份、运行目录、公平配置、smoke。"""
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from revision.fair_config import (  # noqa: E402
    prepared_geometries,
    runtime_plan,
    select_epoch,
    training_matrix,
)
from revision.identity import (  # noqa: E402
    B2_CONTROL_BINARY_VAL,
    STAGE_E_CONTROL_BINARY_VAL,
    RevisionError,
    create_run_dir,
    plan_resume,
    verify_identity,
)
from revision.metrics import (  # noqa: E402
    describe_recoverable,
    paired_patient_bootstrap,
    score_mask,
    summarize_frames,
    weighting_means,
)
from revision.overlays import (  # noqa: E402
    FN_COLOR,
    FP_COLOR,
    TP_COLOR,
    WRONG_COLOR,
    write_review,
)
from revision.engine import (  # noqa: E402
    TinyDino,
    infer_checkpoint,
    launch_real,
    make_head,
    run_synthetic_demo,
    train,
)
from revision.preflight import assert_output_ready  # noqa: E402
from revision.pretrain import ILS_SHA256, decide_weight, resolve_backbone  # noqa: E402


def _box(value, r0, r1, c0, c1, shape=(8, 8)):
    m = np.zeros(shape, dtype=np.int16)
    m[r0:r1, c0:c1] = value
    return m


class MetricsTests(unittest.TestCase):
    def test_wrong_class_silhouette_is_perfect_binary_and_zero_semantic(self):
        gt = _box(1, 2, 6, 2, 6)
        pred = _box(2, 2, 6, 2, 6)
        s = score_mask(pred, gt)
        self.assertAlmostEqual(s["binary_foreground_dice"], 1.0)
        self.assertAlmostEqual(s["binary_foreground_iou"], 1.0)
        self.assertAlmostEqual(s["gt_class_dice"], 0.0)
        self.assertFalse(s["named_correct"])
        self.assertNotEqual(s["pred_name"], s["gt_name"])
        self.assertNotEqual(s["binary_foreground_dice"], s["gt_class_dice"])
        pooled = summarize_frames([{**s, "patient": "A"}])["class_aware_pooled"]
        self.assertAlmostEqual(pooled["pixel_pooled_macro_dice"], 0.0)
        self.assertNotIn("dice", s)

    def test_fp_and_fn_counts(self):
        gt = _box(1, 1, 5, 1, 2)  # 4 px
        pred = gt.copy()
        pred[1:3, 2] = 1  # 2 extra
        s = score_mask(pred, gt)
        self.assertEqual(s["binary_tp"], 4)
        self.assertEqual(s["binary_fp"], 2)
        self.assertEqual(s["binary_fn"], 0)
        self.assertAlmostEqual(s["binary_foreground_dice"], 8 / 10)

        pred2 = np.zeros_like(gt)
        pred2[1:3, 1] = 1  # 2 of 4
        s2 = score_mask(pred2, gt)
        self.assertEqual(s2["binary_tp"], 2)
        self.assertEqual(s2["binary_fn"], 2)
        self.assertAlmostEqual(s2["binary_foreground_dice"], 4 / 6)
        self.assertEqual(s2["class_fn_only"], 2)
        self.assertEqual(s2["class_wrong"], 0)

    def test_empty_and_ignore(self):
        z = np.zeros((4, 4), dtype=np.int16)
        empty = score_mask(z, z)
        self.assertTrue(empty["empty_agreement"])
        self.assertAlmostEqual(empty["binary_foreground_dice"], 1.0)
        self.assertIsNone(empty["gt_class_dice"])
        self.assertFalse(empty["naming_defined"])

        pred = _box(1, 0, 1, 0, 1, shape=(4, 4))
        miss = score_mask(z, pred)
        self.assertAlmostEqual(miss["binary_foreground_dice"], 0.0)
        self.assertEqual(miss["binary_fn"], 1)
        fp = score_mask(pred, z)
        self.assertAlmostEqual(fp["binary_foreground_dice"], 0.0)
        self.assertEqual(fp["binary_fp"], 1)
        self.assertFalse(fp["naming_defined"])

        gt = np.zeros((4, 4), dtype=np.int16)
        pr = np.zeros((4, 4), dtype=np.int16)
        gt[0, 0] = 255
        pr[0, 0] = 3
        gt[1, 1] = 1
        pr[1, 1] = 1
        ign = score_mask(pr, gt, ignore_label=255)
        self.assertEqual(ign["binary_tp"], 1)
        self.assertEqual(ign["binary_fp"], 0)
        self.assertAlmostEqual(ign["binary_foreground_dice"], 1.0)

    def test_patient_weight_and_pooled_macro_differ(self):
        rows = [
            {"patient": "A", "binary_foreground_dice": 1.0},
            {"patient": "B", "binary_foreground_dice": 0.0},
            {"patient": "B", "binary_foreground_dice": 0.0},
            {"patient": "B", "binary_foreground_dice": 0.0},
        ]
        w = weighting_means(rows, "binary_foreground_dice")
        self.assertAlmostEqual(w["frame_equal"], 0.25)
        self.assertAlmostEqual(w["patient_equal"], 0.5)

        perfect = score_mask(_box(1, 0, 2, 0, 2), _box(1, 0, 2, 0, 2))
        wrong = score_mask(_box(2, 0, 2, 0, 2), _box(1, 0, 2, 0, 2))
        summary = summarize_frames(
            [{**perfect, "patient": "A"}, {**wrong, "patient": "B"}]
        )
        pooled = summary["class_aware_pooled"]
        self.assertAlmostEqual(pooled["pixel_pooled_macro_dice"], 1 / 3)
        self.assertAlmostEqual(pooled["patient_equal_macro_dice"], 0.5)
        self.assertAlmostEqual(summary["gt_class_dice"]["frame_equal"], 0.5)
        self.assertNotAlmostEqual(
            pooled["pixel_pooled_macro_dice"],
            summary["binary_foreground"]["frame_equal"],
        )
        self.assertAlmostEqual(summary["naming"]["accuracy_frame_equal"], 0.5)
        self.assertEqual(
            summary["naming"]["confusion"]["grasping forceps"],
            {"grasping forceps": 1, "bipolar": 1},
        )

    def test_paired_patient_bootstrap_is_reproducible(self):
        a, b = [], []
        for i, (sa, sb) in enumerate([(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]):
            a.append({"patient": f"P{i}", "binary_foreground_dice": sa})
            b.append({"patient": f"P{i}", "binary_foreground_dice": sb})
        r1 = paired_patient_bootstrap(a, b, "binary_foreground_dice", n_boot=40, seed=3)
        r2 = paired_patient_bootstrap(a, b, "binary_foreground_dice", n_boot=40, seed=3)
        self.assertEqual(r1, r2)
        self.assertAlmostEqual(r1["point_patient_equal_delta"], 0.75)
        self.assertLessEqual(r1["ci_low"], r1["point_patient_equal_delta"])
        self.assertGreaterEqual(r1["ci_high"], r1["point_patient_equal_delta"])
        self.assertEqual(r1["cluster"], "patient")
        with self.assertRaises(RevisionError) as ctx:
            paired_patient_bootstrap(a, [{"patient": "Z", "binary_foreground_dice": 1}],
                                     "binary_foreground_dice", n_boot=10, seed=0)
        self.assertEqual(ctx.exception.code, "bootstrap_no_overlap")

    def test_scalar_and_historical_frames_do_not_invent_semantic(self):
        scalar = describe_recoverable({"mean_dice": 0.9754, "by_class": {"bipolar": {"dice": 0.9}}})
        self.assertAlmostEqual(scalar["recoverable"]["binary_foreground_dice_frame_equal"], 0.9754)
        self.assertFalse(scalar["recoverable"]["by_class_is_class_aware"])
        self.assertIn("class_aware_per_class_dice", scalar["unrecoverable"])
        self.assertIsNone(scalar["recoverable"]["class_aware_pixel_dice"])
        dumped = json.dumps(scalar)
        self.assertNotIn('"class_aware_pixel_dice": 0.9754', dumped)
        self.assertNotIn('"class_aware_pixel_dice": 0.9', dumped)

        frames = describe_recoverable({
            "frames": [{
                "dice": 1.0, "iou": 1.0, "pred_name": "drill", "cls": "bipolar",
                "patient": "P1", "named_correct": False,
            }]
        })
        self.assertAlmostEqual(frames["recoverable"]["binary_foreground_dice_frame_equal"], 1.0)
        self.assertAlmostEqual(frames["recoverable"]["naming_accuracy_frame_equal"], 0.0)
        self.assertIsNone(frames["recoverable"]["class_aware_pixel_dice"])
        self.assertIn("class_aware_pixel_dice", frames["unrecoverable"])
        self.assertIn("patient", frames["recoverable"]["weighting_available"])
        self.assertEqual(frames["recoverable"]["naming_confusion"], {"bipolar": {"drill": 1}})
        self.assertTrue(frames["notes"])


class OverlayTests(unittest.TestCase):
    def test_error_map_shows_fn_fp_and_wrong_class(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:] = (40, 40, 40)
        gt = np.zeros((8, 8), dtype=np.int16)
        pred = np.zeros((8, 8), dtype=np.int16)
        gt[1:3, 1:3] = 1
        pred[1:3, 1:3] = 1  # TP
        gt[5, 1] = 1        # FN
        pred[1, 6] = 1      # FP
        gt[6, 6] = 1
        pred[6, 6] = 2      # wrong class
        with tempfile.TemporaryDirectory() as td:
            out = write_review(Path(td), "case", image, pred, gt)
            err = np.array(Image.open(out["error_map"]))
            self.assertTrue(np.all(err[1, 1] == TP_COLOR))
            self.assertTrue(np.all(err[5, 1] == FN_COLOR))
            self.assertTrue(np.all(err[1, 6] == FP_COLOR))
            self.assertTrue(np.all(err[6, 6] == WRONG_COLOR))
            panel = np.array(Image.open(out["panel"]))
            tile = panel[:, 3 * (8 + 4): 3 * (8 + 4) + 8]
            self.assertTrue(np.array_equal(tile, err))
            meta = json.loads(Path(out["meta"]).read_text())
            self.assertTrue(meta["dice_displayed"])
            self.assertIn("binary_foreground_dice", meta["displayed_scores"])
            self.assertIn("gt_class_dice", meta["displayed_scores"])
            ids = np.array(Image.open(out["pred_class_id"]))
            self.assertTrue(np.array_equal(ids, pred.astype(np.uint8)))
            self.assertEqual(ids.shape, pred.shape)

    def test_no_gt_does_not_display_dice(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        pred = _box(3, 2, 5, 2, 5)
        with tempfile.TemporaryDirectory() as td:
            out = write_review(Path(td), "nogt", image, pred, None)
            meta = json.loads(Path(out["meta"]).read_text())
            self.assertFalse(meta["dice_displayed"])
            self.assertIsNone(meta["displayed_scores"])
            self.assertNotIn("error_map", out)
            text = json.dumps(meta)
            self.assertNotIn("0.", text)
            panel = np.array(Image.open(out["panel"]))
            self.assertFalse(np.any(np.all(panel == FN_COLOR, axis=-1)))


class IdentityTests(unittest.TestCase):
    def test_b2_scalar_warns_and_sha_decides(self):
        record = {
            "filename": "best_control.pt",
            "epoch": 3,
            "best_val_dice": B2_CONTROL_BINARY_VAL,
            "role": "control_head",
            "ckpt_kind": "control_head",
            "backbone_id": "innerlogic_surg_vitb16",
            "geometry": "square768",
            "sha256": "user-recorded-sha",
            "embed_dim": 768,
            "num_classes": 7,
            "train_mode": "frozen_probe",
        }
        accepted = verify_identity(record, dict(record))
        self.assertTrue(accepted["accepted"])
        self.assertIn("historical_b2_scalar", accepted["warnings"])
        with self.assertRaises(RevisionError) as ctx:
            verify_identity(dict(record, sha256="other"), record)
        self.assertEqual(ctx.exception.code, "hash_mismatch")
        self.assertIn("historical_b2_scalar", ctx.exception.details["warnings"])
        with self.assertRaises(RevisionError) as ctx:
            verify_identity(dict(record, ckpt_kind="full_finetune"), record)
        self.assertEqual(ctx.exception.code, "ckpt_kind_mismatch")

    def test_user_recorded_sha_does_not_need_a_repo_pin(self):
        record = {
            "filename": "best_control.pt",
            "role": "stage_e_control",
            "ckpt_kind": "control_head",
            "backbone_id": "innerlogic_surg_vitb16",
            "geometry": "square768",
            "epoch": 3,
            "best_val_dice": STAGE_E_CONTROL_BINARY_VAL,
            "sha256": "recorded-locally",
            "embed_dim": 768,
            "num_classes": 7,
            "train_mode": "frozen_probe",
        }
        accepted = verify_identity(record, dict(record))
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["warnings"], [])
        with self.assertRaises(RevisionError) as ctx:
            verify_identity(record, {"ckpt_kind": "control_head"})
        self.assertEqual(ctx.exception.code, "expected_sha_required")

    def test_fair_identity_and_resume(self):
        good = {
            "role": "fair_frozen_probe",
            "backbone_id": "innerlogic_surg_vitb16",
            "geometry": "square768",
            "epoch": 2,
            "best_val_dice": 0.5,
            "metric_name": "binary_foreground_dice_frame_equal",
            "sha256": "deadbeef",
            "head_init": "random",
            "seed": 7,
            "train_mode": "frozen_probe",
            "selection_split": "val",
            "continued_from_ep10": False,
        }
        self.assertTrue(verify_identity(good, dict(good))["accepted"])
        bad = dict(good, sha256="other")
        with self.assertRaises(RevisionError) as ctx:
            verify_identity(bad, good)
        self.assertEqual(ctx.exception.code, "hash_mismatch")
        stretch = dict(good, geometry="stretch_768x432")
        with self.assertRaises(RevisionError) as ctx:
            verify_identity(stretch, good)
        self.assertEqual(ctx.exception.code, "geometry_mismatch")
        incomplete = {"best_val_dice": 0.5, "filename": "best_control.pt"}
        with self.assertRaises(RevisionError) as ctx:
            verify_identity(incomplete, good)
        self.assertEqual(ctx.exception.code, "incomplete_metadata")

        plan = plan_resume(good, good)
        self.assertEqual(plan["next_epoch"], 3)
        self.assertEqual(plan["selection_split"], "val")
        ep10 = dict(good, epoch=10, continued_from_ep10=True)
        with self.assertRaises(RevisionError) as ctx:
            plan_resume(ep10, good)
        self.assertEqual(ctx.exception.code, "resume_forbidden")

    def test_run_dir_does_not_overwrite_or_touch_history(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            first = create_run_dir(parent, "smoke-keep", config={"id": "x"})
            (first / "marker.txt").write_text("keep")
            with self.assertRaises(RevisionError) as ctx:
                create_run_dir(parent, "smoke-keep")
            self.assertEqual(ctx.exception.code, "nonoverwrite")
            self.assertEqual((first / "marker.txt").read_text(), "keep")
            with self.assertRaises(RevisionError) as ctx:
                create_run_dir(parent, "results_e")
            self.assertEqual(ctx.exception.code, "historical_protected")
            with self.assertRaises(RevisionError) as ctx:
                create_run_dir(parent / "results_e_cont2", "smoke-new")
            self.assertEqual(ctx.exception.code, "historical_protected")
            rec = json.loads((first / "RUN_RECORD.json").read_text())
            self.assertTrue(rec["revision_sources_sha256"])
            self.assertIn("git_rev", rec)


class PretrainAndConfigTests(unittest.TestCase):
    def test_weight_decisions_do_not_substitute_backbones(self):
        ok = decide_weight("innerlogic_surg_vitb16", ILS_SHA256)
        self.assertTrue(ok["hash_accepted"])
        self.assertFalse(ok["weights_loaded"])
        with self.assertRaises(RevisionError) as ctx:
            decide_weight("dinov3_vitb16_lvd1689m", ILS_SHA256)
        self.assertEqual(ctx.exception.code, "backbone_substitution")
        with self.assertRaises(RevisionError) as ctx:
            decide_weight("dinov3_vitb16_lvd1689m", ILS_SHA256, expected_sha=ILS_SHA256)
        self.assertEqual(ctx.exception.code, "backbone_substitution")
        with self.assertRaises(RevisionError) as ctx:
            decide_weight("dinov3_vitb16_lvd1689m", "f" * 64)
        self.assertEqual(ctx.exception.code, "weight_unverified")
        with self.assertRaises(RevisionError) as ctx:
            decide_weight("innerlogic_surg_vitb16", "ab")
        self.assertEqual(ctx.exception.code, "hash_mismatch")
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nope.pth"
            with self.assertRaises(RevisionError) as ctx:
                resolve_backbone("innerlogic_surg_vitb16", missing)
            self.assertEqual(ctx.exception.code, "weight_missing")
            self.assertIn("获取", ctx.exception.details["steps_zh"])
            blob = Path(td) / "generic.pth"
            blob.write_bytes(b"not a real backbone")
            with self.assertRaises(RevisionError) as ctx:
                resolve_backbone("dinov3_vitb16_lvd1689m", blob)
            self.assertEqual(ctx.exception.code, "weight_unverified")
            self.assertFalse(ctx.exception.details.get("weights_loaded", False))
            digest = __import__("hashlib").sha256(blob.read_bytes()).hexdigest()
            accepted = resolve_backbone(
                "dinov3_vitb16_lvd1689m", blob, expected_sha=digest)
            self.assertTrue(accepted["hash_accepted"])
            self.assertFalse(accepted["weights_loaded"])
            self.assertEqual(accepted["source"], "user_manifest")

    def test_matrix_is_four_square768_runs_until_triple_seed(self):
        runs = training_matrix(phase="single", seed=7)
        self.assertEqual(len(runs), 4)
        self.assertEqual({r["geometry"] for r in runs}, {"square768"})
        self.assertEqual({r["backbone_id"] for r in runs}, {
            "dinov3_vitb16_lvd1689m", "innerlogic_surg_vitb16"})
        self.assertEqual({r["train_mode"] for r in runs}, {"frozen_probe", "last4_finetune"})
        for r in runs:
            self.assertFalse(r["zero_shot"])
            self.assertFalse(r["resume_from_ep10"])
            self.assertEqual(r["epochs"], 6)
            self.assertEqual(r["effective_batch"], 8)
            self.assertEqual(r["microbatch"], 1)
            self.assertEqual(r["grad_accum"], 8)
            self.assertEqual(r["selection_split"], "val")
            self.assertEqual(r["selection_score"], "binary_foreground_dice_frame_equal")
            self.assertIn("gt_class_dice", r["reported_but_not_used_for_selection"])
            self.assertEqual(r["head_init"], "random")
            self.assertFalse(r["early_stop"])
            self.assertEqual(r["seeds"], [7])
        self.assertEqual(len(training_matrix(phase="triple", seed=7)), 12)
        geos = prepared_geometries()
        self.assertEqual(
            [name for name, spec in geos.items() if spec["in_training_matrix"]],
            ["square768"],
        )
        self.assertIn("stretch_768x432", geos)
        self.assertIn("letterbox_768", geos)
        self.assertEqual(select_epoch({1: 0.2, 3: 0.8, 2: 0.8}), 2)
        import inspect
        self.assertEqual(list(inspect.signature(select_epoch).parameters), ["val_scores"])
        plan = runtime_plan(real=False, cuda=False, bf16=False)
        self.assertIsNone(plan["assumed_gpu"])
        self.assertEqual(plan["effective_batch"], 8)
        self.assertEqual(plan["launch"], "cpu_smoke")
        self.assertEqual(plan["device"], "cpu")
        cpu_real = runtime_plan(real=True, cuda=True, bf16=True)
        self.assertEqual(cpu_real["device"], "cpu")
        self.assertEqual(cpu_real["launch"], "train")
        self.assertEqual(cpu_real["autocast"], "fp32")
        with self.assertRaises(RevisionError) as ctx:
            runtime_plan(real=True, cuda=True, bf16=False, device="cuda")
        self.assertEqual(ctx.exception.code, "bf16_required")
        real = runtime_plan(real=True, cuda=True, bf16=True, device="cuda")
        self.assertEqual(real["autocast"], "bf16")
        self.assertEqual(real["launch"], "train")
        self.assertIsNone(real["assumed_gpu"])

    def test_historical_eval_script_is_unchanged_evidence(self):
        text = (ROOT / "modal_stage_e_cont2.py").read_text()
        self.assertIn("pm = pred != 0", text)
        self.assertIn("only_pred = prs & ~gts", text)
        self.assertLess(text.index("ov[gts]"), text.index("only_pred = prs & ~gts"))


class SmokeAndNotebookTests(unittest.TestCase):
    def test_cpu_smoke_and_notebook(self):
        nb_path = ROOT / "notebooks" / "fair_comparison_entry.ipynb"
        nb = json.loads(nb_path.read_text())
        self.assertGreaterEqual(len(nb["cells"]), 4)
        sources = []
        for cell in nb["cells"]:
            src = cell["source"]
            if isinstance(src, list):
                src = "".join(src)
            sources.append(src)
            if cell["cell_type"] == "code":
                ast.parse(src)
        joined = "\n".join(sources)
        self.assertIn("MOUNT_DRIVE = False", joined)
        self.assertIn("RUN_REAL = False", joined)
        self.assertNotIn("token", joined.lower())
        self.assertNotIn("modal", joined.lower())

        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, "-m", "revision.smoke", "--out", td],
                cwd=ROOT, capture_output=True, text=True, check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr + proc.stdout)
            log = proc.stdout + "\n" + (Path(td).joinpath("smoke.log").read_text())
            self.assertIn("expected_sha_required", log)
            self.assertIn("historical_b2_scalar", log)
            self.assertIn("nonoverwrite", log)
            self.assertIn("weight_missing", log)
            self.assertIn("binary_foreground_dice", log)
            self.assertIn("gt_class_dice", log)
            self.assertNotIn("weights_loaded=True", log)


class TrainTests(unittest.TestCase):
    def test_synthetic_four_configs_resume_and_infer(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            summary = run_synthetic_demo(parent, optimizer_steps=2)
            self.assertEqual(len(summary["configs"]), 4)
            modes = {item["train_mode"] for item in summary["configs"]}
            self.assertEqual(modes, {"frozen_probe", "last4_finetune"})
            for item in summary["configs"]:
                self.assertEqual(item["backward_passes"], 8)
                self.assertEqual(item["optimizer_steps"], 1)
                self.assertTrue(item["synthetic_backbone"])
                self.assertEqual(item["selection_metric"], "binary_foreground_dice_frame_equal")
                probe = item["grad_probe"]
                self.assertTrue(probe["head_has_grad"])
                self.assertFalse(probe["early_requires_grad"])
                self.assertFalse(probe["early_blocks_have_grad"])
                if item["train_mode"] == "frozen_probe":
                    self.assertFalse(probe["last4_requires_grad"])
                    self.assertFalse(probe["last4_have_grad"])
                else:
                    self.assertTrue(probe["last4_requires_grad"])
                    self.assertTrue(probe["last4_have_grad"])
            self.assertTrue(summary["resume_match"])
            infer = summary["infer"]
            self.assertEqual(infer["ckpt_kind"], "control_head")
            self.assertIn("historical_b2_scalar", infer["warnings"])
            payload = json.loads((Path(infer["out_dir"]) / "frames.json").read_text())
            frame = payload["frames"][0]
            self.assertEqual(frame["patient"], "synth-patient")
            self.assertEqual(frame["coordinates"]["crop_box_original_xyxy"], [4, 0, 16, 12])
            self.assertEqual(frame["coordinates"]["original_hw"], [12, 20])
            self.assertEqual(frame["coordinates"]["class_id_crop_hw"], [12, 12])
            self.assertAlmostEqual(frame["scores"]["binary_foreground_dice"], 0.0)
            err = np.array(Image.open(Path(infer["out_dir"]) / "frame0000_error_map.png"))
            self.assertTrue(np.any(np.all(err == FN_COLOR, axis=-1)))
            canvas = np.array(Image.open(Path(infer["out_dir"]) / "frame0000_pred_class_id_original.png"))
            self.assertEqual(canvas.shape, (12, 20))
            self.assertEqual(int(canvas[0, 0]), 255)
            self.assertEqual(int(canvas[0, 4]), 0)

    def test_val_selection_ignores_semantic_and_test(self):
        scores = iter([
            {"binary_foreground_dice": 0.2, "gt_class_dice": 0.99, "frames": []},
            {"binary_foreground_dice": 0.85, "gt_class_dice": 0.01, "frames": []},
        ])

        def stub(backbone, head, rows, split, **_kwargs):
            self.assertEqual(split, "val")
            return next(scores)

        base = training_matrix_one()
        with tempfile.TemporaryDirectory() as td:
            result = train(
                base,
                _rows("train"),
                _rows("val"),
                TinyDino(),
                Path(td),
                "fairrun-select",
                device="cpu",
                max_optimizer_steps=2,
                val_every="optimizer_step",
                evaluate_fn=stub,
            )
            self.assertEqual(result["best_step"], 2)
            self.assertAlmostEqual(result["best_val_binary_foreground_dice"], 0.85)
            with self.assertRaises(RevisionError) as ctx:
                train(
                    base, _rows("local_test"), _rows("val"), TinyDino(),
                    Path(td), "fairrun-leak", device="cpu", max_optimizer_steps=1,
                )
            self.assertEqual(ctx.exception.code, "test_leak")

    def test_real_entry_fail_fast(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(RevisionError) as ctx:
                assert_output_ready(root / "out", min_free_bytes=10 ** 18)
            self.assertEqual(ctx.exception.code, "disk_insufficient")
            with self.assertRaises(RevisionError) as ctx:
                assert_output_ready(root / "missing-parent" / "out", min_free_bytes=1)
            self.assertEqual(ctx.exception.code, "not_writable")
            with self.assertRaises(RevisionError) as ctx:
                launch_real(
                    backbone_id="innerlogic_surg_vitb16",
                    weight_path=root / "no-weight.pth",
                    expected_sha="abc",
                    data_root=root,
                    manifest=root / "m.jsonl",
                    out_dir=root,
                    train_mode="frozen_probe",
                    device="cpu",
                    min_free_bytes=1,
                )
            self.assertEqual(ctx.exception.code, "weight_missing")
            blob = root / "fake.pth"
            blob.write_bytes(b"not-a-backbone")
            digest = __import__("hashlib").sha256(blob.read_bytes()).hexdigest()
            data = root / "data"
            data.mkdir()
            manifest = root / "m.jsonl"
            manifest.write_text(
                '{"image":"a.png","mask":"a.png","cls":"drill","patient":"p","split":"train"}\n'
                '{"image":"b.png","mask":"b.png","cls":"drill","patient":"q","split":"val"}\n'
            )
            with self.assertRaises(RevisionError) as ctx:
                launch_real(
                    backbone_id="dinov3_vitb16_lvd1689m",
                    weight_path=blob,
                    expected_sha=digest,
                    data_root=data,
                    manifest=manifest,
                    out_dir=root,
                    train_mode="last4_finetune",
                    device="cpu",
                    min_free_bytes=1,
                )
            self.assertEqual(ctx.exception.code, "backbone_loader_missing")
            self.assertIn("surgfm_probe", str(ctx.exception))
            with self.assertRaises(RevisionError) as ctx:
                launch_real(
                    backbone_id="innerlogic_surg_vitb16",
                    weight_path=blob,
                    expected_sha=digest,
                    data_root=data,
                    manifest=manifest,
                    out_dir=root,
                    train_mode="frozen_probe",
                    device="cuda",
                    min_free_bytes=1,
                )
            self.assertEqual(ctx.exception.code, "gpu_unavailable")

            head = make_head(8, 7, seed=0)
            net = TinyDino()
            path = root / "full.pt"
            torch_save(path, {
                "head_state": head.state_dict(),
                "backbone_state": net.state_dict(),
                "best_val_dice": 0.2,
                "embed_dim": 8,
                "num_classes": 7,
                "depth": 12,
                "patch_size": 4,
                "geometry": "square768",
                "train_mode": "last4_finetune",
                "role": "last4",
                "backbone_id": "synthetic_tiny",
            })
            expect = {
                "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "ckpt_kind": "control_head",
                "embed_dim": 8,
                "num_classes": 7,
            }
            with self.assertRaises(RevisionError) as ctx:
                infer_checkpoint(
                    path, expect, _rows("val"), net, make_head(8, 7, seed=1),
                    root / "fairrun-bad-infer", device="cpu", input_size=16, patch=4,
                )
            self.assertEqual(ctx.exception.code, "ckpt_kind_mismatch")


def training_matrix_one():
    from revision.fair_config import training_matrix
    from revision.engine import synthetic_config
    return synthetic_config(training_matrix()[0])


def _rows(split):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 255
    return [{
        "image": image, "mask": mask, "cls": "grasping forceps",
        "patient": "synth", "split": split,
    }]


def torch_save(path, blob):
    import torch
    torch.save(blob, path)


if __name__ == "__main__":
    unittest.main()
