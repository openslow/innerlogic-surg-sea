# InnerLogic-Surg × SEA Spinal Endoscopy Segmentation

Training, evaluation and visualization scripts for testing
[InnerLogic-Surg](https://github.com/semaphor-surgical/Innerlogic-Surg)
(ViT-B/16, Release v0.1.0) on the SEA spinal endoscopy dataset
(Scientific Data 2025, DOI 10.1038/s41597-025-05897-7).

**Status (2026-11-23): stage complete, at a plateau.** Same recipe (square crop +
last-4-block fine-tune) was continued for 3 rounds; validation stopped improving.
Best checkpoint is **ep10**: local test Dice **0.9643**, external test **0.9677**,
clean22 **0.9754** (paper's nnU-Net: 0.9753 / 0.9376 — ours beats it on external,
trails on local; protocol-aligned reproduction, patient IDs differ from the paper).

Data, checkpoints, results and figures are **not** in this repo — they live in the
project Drive folder (see below). This repo is code only.

## Layout

```
├── modal_stage_*.py      training / eval / viz (Modal L40S, bf16)
├── verify_*.py           independent re-computation from per-frame JSON
├── build_paper_split.py  builds the paper-aligned split from sea.zip (seed 7)
├── check_weight_diff.py  proves backbone weights actually moved in fine-tune
├── yolo_run.py / yolo_decrypt.py / build_yolo_sample.py
│                         YOLOv11 zero-shot comparison (see below)
├── colab_stage_e_cont.py / run_train.py   Colab continuation entry points
├── manifests_paper/     the paper-aligned split (train/val/local/external/old100)
└── docs/COLAB_PLAYBOOK.md  Colab done right: Drive as volume, session as compute
```

## Setup

1. Clone the official model repo next to this one (Release v0.1.0; ViT-B/16 and
   ViT-L/16 released, ViT-S/16 never released):
   `git clone https://github.com/semaphor-surgical/Innerlogic-Surg repo`
2. Data + checkpoints come from the project Drive folder
   (`Projects/spine-endoscopy-segmentation/`): `data/sea.zip`
   (9,877,132,195 bytes, md5 `bde2014aa786181140552ab248fad0b2`, figshare 27109312,
   CC0) and `checkpoints/` (pretrain weight + control/ep5/ep8/ep10).
3. Python deps: torch (bf16 capable), numpy, pynrrd, pillow. Modal runs use
   `modal_stage_*.py` (`--detach` for anything long); Colab runs use
   `run_train.py` — read `docs/COLAB_PLAYBOOK.md` first.

## Path notes (read before running)

- The `modal_stage_*.py` scripts were written for Modal: they import the model
  code from `/root/repo` and read/write a Modal volume (`sea/...`). The Colab
  entry points use a relative `REPO` path instead. Adapt paths when running
  elsewhere; do not assume `/root` or `/home/hatch` exists.
- `verify_*.py` scripts read per-frame result JSONs by absolute local path;
  point them at the `results/` files from the Drive folder.
- `best_control.pt` (0-shot control) is **head-only** (68 KB). Rebuilding the
  control model requires the pretrain weight — see `checkpoints/MANIFEST.md`
  in the Drive folder.
- Never fp16; always bf16 autocast.

## Recipe (final, Stage E)

Backbone ViT-B/16 (86M), frozen except blocks 8–11 + 1×1 linear head + bilinear
upsampling. LR backbone 1e-5 / head 1e-4, AdamW cosine, CE+Dice, early stop
patience 2 on the 496-frame val set. Preprocessing: **center square crop**
(side = min(W,H)), train and inference the same way — this replaced the old
resize-to-768×432 stretch. Input 768×768.

## YOLOv11 zero-shot comparison: blocked

`yolo_run.py` scores the YOLOv11-seg weights on the sampled frames
(`manifests_paper/` has the split; the sample list is in the Drive folder).
It cannot run end to end yet: the repo's R2 weights cannot be decrypted with
the shipped loader (password mismatch; the loader only opens the first
release). An email was sent 2026-11-22 to the corresponding author
(Tae-Shin Kim) requesting the R2 password or an updated loader; awaiting reply.
Do not retry decryption with the shipped loader — it is proven to fail.
`yolo_decrypt.py` reproduces the failure with the loader's own default password
(already public in the upstream repo).

## License

To be decided by the owner before making this repo public. The base model
carries the DINOv3 license; read it before commercial use or redistribution.
