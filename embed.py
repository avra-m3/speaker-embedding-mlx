"""Compute a ReDimNet2 speaker embedding with MLX.

python embed.py causal/redimnet2-b3-vox2-lm-mlx a.wav [b.wav]   # prints cosine score for two files
"""

import sys

import mlx.core as mx
import numpy as np
import soundfile as sf

from redimnet2_mlx import load_model


def embed(model, path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 16000:
        raise SystemExit(f"{path}: expected 16 kHz audio, got {sr}")
    e = model(mx.array(wav.mean(1))[None])[0]
    return e / mx.linalg.norm(e)


def main():
    model = load_model(sys.argv[1])
    embs = [embed(model, p) for p in sys.argv[2:]]
    for p, e in zip(sys.argv[2:], embs):
        print(p, np.array(e)[:5], "...")
    if len(embs) == 2:
        print("cosine:", (embs[0] * embs[1]).sum().item())


if __name__ == "__main__":
    main()
