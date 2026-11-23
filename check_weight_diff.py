"""One-off diagnostic: how much did the fine-tuned backbone actually move?

Compares best_finetune.pt (ep2, the evaluated checkpoint) and
checkpoint_finetune.pt (ep4, last epoch) against the original pretrained
checkpoint, per block (relative L2). Blocks 0-7 were frozen and must be
bit-identical to the original; blocks 8-11 were unfrozen.
Run: .venv/bin/modal run check_weight_diff.py  (CPU is enough)
"""
import modal

app = modal.App("innerlogic-surg-sea")
vol = modal.Volume.from_name("innerlogic-surg", create_if_missing=True)

# Same image definition as modal_stage_b2.py so Modal reuses the cached image.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy==2.1.3",
                 "pillow==10.4.0", "pynrrd==1.1.3")
)


def unwrap_state_dict(obj):
    sd = obj
    for key in ("teacher", "student", "model", "state_dict"):
        if key in sd and isinstance(sd[key], dict):
            sd = sd[key]
            break
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    if any(k.startswith("backbone.") for k in sd):
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    return sd


@app.function(image=image, volumes={"/data": vol}, timeout=1200)
def weight_diff():
    import numpy as np
    import torch

    orig = unwrap_state_dict(torch.load(
        "/data/sea/ckpt/dinov3_vitb16_pretrain_innerlogic_surg-a516a775.pth",
        map_location="cpu", weights_only=True))
    print("orig keys:", len(orig), "sample:", list(orig)[:3], flush=True)

    for name in ("best_finetune.pt", "checkpoint_finetune.pt"):
        ft = torch.load(f"/data/sea/results_b2/{name}", map_location="cpu",
                        weights_only=False)
        bs = ft["backbone_state"]
        print(f"=== {name} (epoch {ft['epoch']}, best_val {ft['best_val_dice']:.4f}) ===", flush=True)
        # frozen blocks 0-7 must be identical
        for i in range(12):
            keys = [k for k in orig if k.startswith(f"blocks.{i}.")]
            num = sum((bs[k].float() - orig[k].float()).pow(2).sum().item() for k in keys)
            den = sum(orig[k].float().pow(2).sum().item() for k in keys)
            print(f"  block {i:2d}: rel_L2={np.sqrt(num / den):.5f} "
                  f"(abs_L2={np.sqrt(num):.4f})", flush=True)
        # non-block params
        other = [k for k in orig if not k.startswith("blocks.")]
        num = sum((bs[k].float() - orig[k].float()).pow(2).sum().item() for k in other)
        den = sum(orig[k].float().pow(2).sum().item() for k in other)
        print(f"  non-block params: rel_L2={np.sqrt(num / den):.6f}", flush=True)
        # head norm comparison is not meaningful across groups (different init);
        # report head weight norm only
        hs = ft["head_state"]
        print(f"  head weight L2={hs['weight'].float().pow(2).sum().sqrt().item():.4f} "
              f"bias L2={hs['bias'].float().pow(2).sum().sqrt().item():.4f}", flush=True)
    return "done"


@app.local_entrypoint()
def main():
    print(weight_diff.remote())
