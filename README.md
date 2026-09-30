# speaker-embedding-mlx

## ReDimNet2 on MLX (b0 to b4, inference)

MLX port of [PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) (MIT), checked against
upstream commit `c5bbe0b`. It covers everything the released **b0 to b4** checkpoints use: TF-style
log-mel front end, `basic_resnet` 2D blocks, `conv+att` 1D blocks, `agg_gnorm`, the 2D output
head and ASTP pooling. Other config options raise `NotImplementedError`.

## Files

- `redimnet2_mlx/model.py` holds the model. Parameter names mirror the PyTorch `state_dict`, so
  conversion only transposes conv weights.
- `convert.py` turns an official `.pt` checkpoint into `config.json` + `weights.safetensors`.
- `parity.py` runs PyTorch and MLX on the same audio and compares features, backbone output and
  embeddings, with a float64 PyTorch run as the rounding floor.
- `publish_hf.py` uploads converted models to Hugging Face with a model card.
- `embed.py` embeds wav files and prints the cosine score for a pair.
- `mlx_models/` has the three b3 checkpoints already converted: `b3-vox2-lm`,
  `b3-vox2-ptn` and `b3-vb2+vox2+cnc2_v0-lm`.
- `results/` holds the parity logs.

## Usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv sync` installs MLX
(the `mlx[cpu]` build on Linux), numpy and soundfile from `uv.lock`.

```bash
uv sync
uv run python embed.py mlx_models/b3-vox2-lm a.wav b.wav
```

```python
import mlx.core as mx
from redimnet2_mlx import load_model

model = load_model("mlx_models/b3-vox2-lm")
emb = model(mx.array(wav_16k_float32)[None])  # (1, 192)
```

`load_model` also accepts a Hugging Face repo id and downloads it into the HF cache. The b0 to b4
checkpoints are being moved to Hugging Face; until that's done, use the local `mlx_models/` paths.

Publish converted models (one HF repo per checkpoint, with a generated model card).
`publish_all.sh` converts every released b0 to b4 checkpoint from the upstream release and
uploads them all:

```bash
HF_TOKEN=... ./publish_all.sh avra-m3
```

The input is mono 16 kHz float audio shaped (B, samples), the same as the PyTorch model.

Convert another checkpoint (downloads from the upstream release when `--checkpoint` is omitted):

```bash
uv sync --group convert
uv run python convert.py --model b3 --train-type lm --dataset vox2 --out mlx_models/b3-vox2-lm
```

Re-run parity (needs a clone of the upstream repo and the `.pt` file):

```bash
git clone https://github.com/PalabraAI/redimnet2 && git -C redimnet2 checkout c5bbe0b
uv run --group convert python parity.py --torch-repo redimnet2 --checkpoint b3-vox2-lm.pt \
    --mlx-model mlx_models/b3-vox2-lm --wav some_speech.wav
```

## Development

```bash
uv run --group dev ruff format --check .
uv run --group dev ruff check .
```

## Parity (float32, MLX CPU backend, Linux)

Every released checkpoint from b0 to b4 was checked on synthetic 1 s, 3.3 s and 8 s batches
plus one real speech clip. The worst embedding error relative to max |value| (logs in
`results/`):

| checkpoint | lm | ptn |
|---|---|---|
| b0 vox2 | 2.9e-6 | 3.5e-6 |
| b1 vox2 | 3.4e-6 | 3.9e-6 |
| b2 vox2 | 3.9e-6 | 3.6e-6 |
| b3 vox2 | 4.4e-6 | 4.4e-6 |
| b3 vb2+vox2+cnc2 | 3.2e-6 | not released |
| b4 vox2 | 4.1e-6 | 3.1e-6 |

PyTorch's own float32 output is about 1e-6 away from a float64 run, which is the practical
floor. Cosine similarity between the MLX and PyTorch embeddings rounds to 1.000000000 in every
case. Bit-exact equality isn't achievable because the two frameworks accumulate conv and matmul
sums in a different order.

## Not verified here

This was built and tested in a Linux container on the MLX **CPU** backend. The following
still needs checking on a Mac:
- Metal/GPU numerics. `parity.py` should show similar numbers there, but that hasn't been run.
- Speed. The Linux `mlx[cpu]` wheel uses a very slow reference BLAS (about 100x slower than numpy
  on a plain matmul). A 10 s clip took about 12 s here, while PyTorch CPU took 0.5 s. That
  says nothing about Apple Silicon speed.
- float16/bfloat16 inference hasn't been tried.

## License

The model code is ported from ReDimNet2 (MIT) and parts of ReDimNet (MIT) and WeSpeaker (Apache-2.0). The converted weights come from the ReDimNet2 v1.0.0 release, which is published under the same MIT license. See `LICENSE-redimnet2` and `THIRD_PARTY_NOTICES.md`.
