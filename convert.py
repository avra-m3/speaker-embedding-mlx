"""Convert an official ReDimNet2 PyTorch checkpoint (.pt) to MLX.

    python convert.py --model b3 --train-type lm --dataset vox2 --out mlx_models/b3-vox2-lm
    python convert.py --checkpoint b3-vox2-lm.pt --out mlx_models/b3-vox2-lm

Writes <out>/config.json and <out>/weights.safetensors. Needs torch (CPU is fine).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import mlx.core as mx

URL = "https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{name}"


def torch_to_mlx(state_dict):
    """Rename-free conversion: only conv weight layouts change."""
    out = {}
    for k, v in state_dict.items():
        if k.endswith("num_batches_tracked"):
            continue
        a = v.detach().cpu().float().numpy()
        if a.ndim == 4 and not k.endswith(".w"):      # Conv2d (O, I, H, W) -> (O, H, W, I)
            a = a.transpose(0, 2, 3, 1)
        elif a.ndim == 3:                              # Conv1d (O, I, K) -> (O, K, I)
            a = a.transpose(0, 2, 1)
        out[k] = mx.array(np.ascontiguousarray(a))
    return out


def load_checkpoint(args):
    import torch
    if args.checkpoint:
        return torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    name = f"{args.model}-{args.dataset}-{args.train_type}.pt"
    return torch.hub.load_state_dict_from_url(URL.format(name=name.replace("+", "%2B")),
                                              map_location="cpu", weights_only=False,
                                              file_name=name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", help="local .pt file (otherwise downloaded)")
    p.add_argument("--model", default="b3")
    p.add_argument("--train-type", default="lm")
    p.add_argument("--dataset", default="vox2")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ckpt = load_checkpoint(args)
    weights = torch_to_mlx(ckpt["state_dict"])

    # Build the MLX model to check every key and shape lines up before saving.
    from redimnet2_mlx.model import ReDimNet2Wrap
    from mlx.utils import tree_flatten
    model = ReDimNet2Wrap(**ckpt["model_config"])
    expected = dict(tree_flatten(model.parameters()))
    missing = sorted(set(expected) - set(weights))
    unexpected = sorted(set(weights) - set(expected))
    bad = [k for k in expected if k in weights and expected[k].shape != weights[k].shape]
    if missing or unexpected or bad:
        raise SystemExit(f"mismatch: missing={missing} unexpected={unexpected} shape={bad}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "weights.safetensors"), weights)
    (out / "config.json").write_text(json.dumps(ckpt["model_config"], indent=2))
    n = sum(v.size for v in weights.values())
    print(f"wrote {out} ({len(weights)} tensors, {n/1e6:.2f}M values)")


if __name__ == "__main__":
    main()
