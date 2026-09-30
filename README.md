# speaker-embedding-mlx

## ReDimNet2 on MLX (b3, inference)

MLX port of [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) (MIT), checked against
upstream commit `c5bbe0b`. It covers everything the released **b3** checkpoints use: TF-style
log-mel front end, `basic_resnet` 2D blocks, `conv+att` 1D blocks, `agg_gnorm`, the 2D output
head and ASTP pooling. Other config options raise `NotImplementedError`.

## Files

- `redimnet2_mlx/model.py` holds the model. Parameter names mirror the PyTorch `state_dict`, so
  conversion only transposes conv weights.
- `convert.py` turns an official `.pt` checkpoint into `config.json` + `weights.safetensors`.
- `parity.py` runs PyTorch and MLX on the same audio and compares features, backbone output and
  embeddings, with a float64 PyTorch run as the rounding floor.
- `embed.py` embeds wav files and prints the cosine score for a pair.
- `mlx_models/` has the three b3 checkpoints already converted: `b3-vox2-lm`,
  `b3-vox2-ptn` and `b3-vb2+vox2+cnc2_v0-lm`.
- `results/` holds the parity logs.

## Usage

```bash
pip install mlx numpy soundfile            # Apple Silicon
python embed.py mlx_models/b3-vox2-lm a.wav b.wav
```

```python
import mlx.core as mx
from redimnet2_mlx import load_model
model = load_model("mlx_models/b3-vox2-lm")
emb = model(mx.array(wav_16k_float32)[None])   # (1, 192)
```

The input is mono 16 kHz float audio shaped (B, samples), the same as the PyTorch model.

Convert another checkpoint (downloads from the upstream release when `--checkpoint` is omitted):

```bash
pip install torch scipy
python convert.py --model b3 --train-type lm --dataset vox2 --out mlx_models/b3-vox2-lm
```

Re-run parity (needs a clone of the upstream repo and the `.pt` file):

```bash
git clone https://github.com/PalabraAI/redimnet2 && git -C redimnet2 checkout c5bbe0b
PYTHONPATH=. python parity.py --torch-repo redimnet2 --checkpoint b3-vox2-lm.pt \
    --mlx-model mlx_models/b3-vox2-lm --wav some_speech.wav
```

## Parity (float32, MLX CPU backend, Linux)

Embeddings were checked on synthetic 1 s, 3.3 s and 8 s batches plus one real speech clip. The
worst embedding error relative to max |value|:

| checkpoint | MLX vs PyTorch fp32 | PyTorch fp32 vs fp64 (floor) |
|---|---|---|
| b3-vox2-lm | 4.4e-6 | ~1e-6 |
| b3-vox2-ptn | 4.4e-6 | ~1e-6 |
| b3-vb2+vox2+cnc2_v0-lm | 3.2e-6 | ~1e-6 |

Cosine similarity between the MLX and PyTorch embeddings rounds to 1.000000000 in every case.
Bit-exact equality isn't achievable because the two frameworks accumulate conv and matmul sums
in a different order. The remaining difference is the same size as PyTorch's own float32
rounding error.

## Not verified here

This was built and tested in a Linux container on the MLX **CPU** backend. The following
still needs checking on a Mac:
- Metal/GPU numerics. `parity.py` should show similar numbers there, but that hasn't been run.
- Speed. The Linux `mlx[cpu]` wheel uses a very slow reference BLAS (about 100x slower than numpy
  on a plain matmul). A 10 s clip took about 12 s here, while PyTorch CPU took 0.5 s. That
  says nothing about Apple Silicon speed.
- float16/bfloat16 inference hasn't been tried.

## License

The model code is ported from ReDimNet2, which is MIT licensed; its license is in `LICENSE-redimnet2`. The converted weights come from the upstream v1.0.0 release.
