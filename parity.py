"""Compare the MLX port against the official PyTorch ReDimNet2 on the same audio.

    python parity.py --torch-repo path/to/redimnet2 --checkpoint b3-vox2-lm.pt \
        --mlx-model mlx_models/b3-vox2-lm [--wav speech.wav]

Reports max abs / relative error for the log-mel features, the backbone output and the
final embedding, plus embedding cosine similarity. The float64 PyTorch run gives the
float32 rounding floor that neither framework can beat.
"""

import argparse
import sys

import mlx.core as mx
import numpy as np


def stats(name, ref, got):
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    err = np.abs(ref - got)
    rel = err.max() / (np.abs(ref).max() + 1e-12)
    line = f"{name:<26} max_abs={err.max():.3e}  rel_to_max={rel:.3e}"
    if ref.ndim == 2:
        cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
        line += f"  min_cos={cos.min():.9f}"
    print(line)
    return err.max(), rel


def torch_forward_fp64(m, x):
    """ReDimNet2Wrap.forward in float64 (the torch feature code hard-casts to float32)."""
    import torch
    import torch.nn.functional as F

    fb = m.spec.torchfbank
    x = x.unsqueeze(1)
    if not isinstance(fb[0], torch.nn.Identity):
        x = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True, unbiased=False) + fb[0].eps)
    if not isinstance(fb[1], torch.nn.Identity):
        x = F.conv1d(F.pad(x, (1, 0), "reflect"), fb[1].flipped_filter)
    sp = fb[2]
    re = F.conv1d(x, sp.real_kernel_pt, stride=sp.shift, padding=sp.shift // 2)
    im = F.conv1d(x, sp.image_kernel_pt, stride=sp.shift, padding=sp.shift // 2)
    p = (re.square() + im.square()).clip(sp.eps, 1 / sp.eps)
    mel = F.conv1d(p, sp.melbanks_pt).clip(sp.eps, 1 / sp.eps)
    s = (mel + m.spec.eps).log()
    s = s - s.mean(-1, keepdim=True)
    out = m.backbone(s.unsqueeze(1))
    if out.ndim == 4:
        out = out.reshape(out.shape[0], -1, out.shape[-1])
    out = m.linear(m.bn(m.pool(out)))
    return m.bn2(out) if m.bn2 is not None else out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--torch-repo", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mlx-model", required=True)
    p.add_argument("--wav", action="append", default=[])
    p.add_argument("--seconds", type=float, default=[1.0, 3.3, 8.0], nargs="+")
    p.add_argument("--tol", type=float, default=1e-4, help="max relative embedding error")
    args = p.parse_args()

    import torch

    sys.path.insert(0, args.torch_repo)
    from redimnet2.redimnet2 import ReDimNet2Wrap as TorchWrap

    from redimnet2_mlx import load_model

    torch.manual_seed(0)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tm = TorchWrap(**ckpt["model_config"])
    tm.load_state_dict(ckpt["state_dict"])
    tm.eval()
    tm64 = TorchWrap(**ckpt["model_config"]).double()
    tm64.load_state_dict(ckpt["state_dict"])
    tm64.eval()
    mm = load_model(args.mlx_model)

    inputs = []
    rng = np.random.default_rng(0)
    for s in args.seconds:
        n = int(16000 * s)
        t = np.arange(n) / 16000
        # batch of 2: noisy chirp-like signal and white noise
        a = 0.3 * np.sin(2 * np.pi * (200 + 300 * t) * t) + 0.05 * rng.standard_normal(n)
        b = 0.1 * rng.standard_normal(n)
        inputs.append((f"synthetic {s:g}s x2", np.stack([a, b]).astype(np.float32)))
    for w in args.wav:
        import soundfile as sf

        x, sr = sf.read(w, dtype="float32")
        assert sr == 16000 and x.ndim == 1, "need mono 16 kHz audio"
        inputs.append((w.split("/")[-1], x[None]))

    worst = 0.0
    for name, x in inputs:
        print(f"\n== {name}: shape {x.shape}")
        with torch.no_grad():
            xt = torch.from_numpy(x)
            spec_t = tm.spec(xt)
            back_t = tm.backbone(spec_t.unsqueeze(1))
            emb_t = tm(xt).numpy()
            emb_64 = torch_forward_fp64(tm64, xt.double()).numpy()
        spec_m = mm.spec(mx.array(x))
        back_m = mm.backbone(spec_m)
        emb_m = mm(mx.array(x))
        mx.eval(spec_m, back_m, emb_m)

        stats("log-mel features", spec_t.numpy(), np.array(spec_m))
        bt = back_t.numpy()
        bm = np.array(back_m)
        bm = bm.transpose(0, 3, 1, 2) if bm.ndim == 4 else bm.transpose(0, 2, 1)
        stats("backbone output", bt, bm)
        stats("embedding (torch fp32 floor)", emb_64, emb_t)
        _, rel = stats("embedding (MLX vs torch)", emb_t, np.array(emb_m))
        stats("embedding (MLX vs fp64)", emb_64, np.array(emb_m))
        worst = max(worst, rel)

    print(f"\nworst embedding rel error: {worst:.3e} (tolerance {args.tol:g})")
    print("PASS" if worst <= args.tol else "FAIL")
    sys.exit(0 if worst <= args.tol else 1)


if __name__ == "__main__":
    main()
