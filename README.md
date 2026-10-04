# speaker-embedding-mlx

Tell voices apart on a Mac. This repo runs the ReDimNet2 speaker-recognition models on Apple
Silicon through [MLX](https://github.com/ml-explore/mlx), Apple's machine-learning framework.
You give it a short clip of someone talking and it returns 192 numbers that describe their
voice, called a speaker embedding. Two clips of the same person give similar embeddings. Two
different people give different ones. It does not care what was said, only who said it.

## Who this is for

You might want this if you are building something that needs to know which person is talking.
Common uses are checking whether a voice matches an enrolled user, labelling speakers in a
meeting recording, grouping a pile of clips by speaker, or flagging when a new voice joins a
call. It runs locally, so audio never leaves the machine.

It is also for anyone who already uses ReDimNet2 in PyTorch and wants the same numbers on a
Mac without PyTorch. The MLX models match the original PyTorch output to about 4 parts in a
million.

This repo does not transcribe speech and does not train models. It only runs the released
ReDimNet2 checkpoints.

## Get it running

You need Python 3.10 or newer and [uv](https://docs.astral.sh/uv/), a Python package manager.
If you don't have uv, install it with `curl -LsSf https://astral.sh/uv/install.sh | sh` or
`brew install uv`.

Clone the repo and install its dependencies:

```bash
git clone https://github.com/avra-m3/speaker-embedding-mlx
cd speaker-embedding-mlx
uv sync
```

`uv sync` creates a private environment in `.venv/` with MLX, numpy, soundfile and the
Hugging Face downloader. It does not touch your system Python. On Linux it installs the CPU
build of MLX, which works but runs slowly.

## Your first comparison

The repo ships six short test clips: three sentences from one speaker and three from another.
Compare two clips from the same speaker:

```bash
uv run python embed.py causal/redimnet2-b3-vox2-lm-mlx demo/test_audio/spk1_snt1.wav demo/test_audio/spk1_snt2.wav
```

The first run downloads the model, about 17 MB, from Hugging Face into its local cache. Later
runs reuse it. The last line prints `cosine: 0.767`. Now compare two different speakers:

```bash
uv run python embed.py causal/redimnet2-b3-vox2-lm-mlx demo/test_audio/spk1_snt1.wav demo/test_audio/spk2_snt1.wav
```

This time it prints `cosine: 0.030`.

The cosine score runs from -1 to 1. Higher means the voices are more alike. Same-speaker pairs
usually land well above 0.5 and different speakers near 0, but the right cutoff depends on
your microphones, room and clip length. Record a few pairs of your own and pick a threshold
between the two groups. Clips of 3 seconds or more give steadier scores than 1 second ones.

To try your own recordings, `embed.py` needs WAV files at 16 kHz. Stereo files are mixed down
to mono. Convert anything else with ffmpeg:

```bash
ffmpeg -i voice_memo.m4a -ac 1 -ar 16000 voice_memo.wav
```

## Use it from Python

```python
import mlx.core as mx
import soundfile as sf

from redimnet2_mlx import load_model

model = load_model("causal/redimnet2-b3-vox2-lm-mlx")


def embed(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 16000, "resample to 16 kHz first"
    e = model(mx.array(wav.mean(axis=1))[None])[0]
    return e / mx.linalg.norm(e)


score = (embed("a.wav") * embed("b.wav")).sum().item()
print(f"cosine {score:.3f}")
```

`load_model` takes either a local folder or a Hugging Face repo id. A repo id downloads the
weights once into the Hugging Face cache. The model expects mono 16 kHz float32 audio shaped
`(batch, samples)` and returns embeddings shaped `(batch, 192)`.

## Pick a model

All eleven released checkpoints are on Hugging Face as `causal/redimnet2-<name>-mlx`. The
sizes go from b0, the smallest and fastest, to b4, the largest and most accurate. Start with
[`causal/redimnet2-b3-vox2-lm-mlx`](https://huggingface.co/causal/redimnet2-b3-vox2-lm-mlx).
It balances accuracy and speed, and it is the model the examples above use.

| size | weights | names |
|---|---|---|
| b0 | 5 MB | `b0-vox2-lm`, `b0-vox2-ptn` |
| b1 | 9 MB | `b1-vox2-lm`, `b1-vox2-ptn` |
| b2 | 15 MB | `b2-vox2-lm`, `b2-vox2-ptn` |
| b3 | 17 MB | `b3-vox2-lm`, `b3-vox2-ptn`, `b3-vb2-vox2-cnc2-lm` |
| b4 | 26 MB | `b4-vox2-lm`, `b4-vox2-ptn` |

The suffix says how the model was trained. `ptn` is the pretrained model. `lm` adds a
large-margin fine-tuning pass on top, which separates speakers better, so use `lm` unless you
have a reason not to. `vox2` means trained on VoxCeleb2. `b3-vb2-vox2-cnc2-lm` adds VoxBlink2
and CN-Celeb2 to the training data. Those datasets have their own terms of use, which may
restrict use to research, so check them before you ship it in a product.

## Demo: a speaker match light

`demo/` holds a small Mac app that enrolls your voice and then listens to the microphone. A
dot turns green while you talk, red for anyone else, and grey during silence. It is the
quickest way to get a feel for how the scores behave live.

```bash
uv sync --group demo
uv run --group demo demo/speaker_light.py --save-embedding me.npy
```

See [demo/README.md](demo/README.md) for the options and the microphone permission prompt.

## What has been checked

Every checkpoint was compared against the original PyTorch model on the same audio. The worst
embedding error, relative to the largest value, was 4.4e-6, and cosine similarity between the
MLX and PyTorch embeddings rounds to 1.000000000. The full table is below.

All of that ran on Linux with the MLX CPU backend. Apple Silicon GPU (Metal) results and speed
have not been measured yet. If you run `parity.py` or time it on a Mac, an issue with the
numbers is welcome. The Linux CPU build uses a slow reference BLAS, so a 10 s clip took about
12 s there. That says nothing about Mac speed. float16 and bfloat16 inference have not been
tried.

## For contributors

The rest of this file covers converting checkpoints, checking parity and publishing weights.
You don't need any of it to use the models.

### Files

`redimnet2_mlx/model.py` holds the model. It ports
[PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) at commit `c5bbe0b` and covers
everything the released b0 to b4 checkpoints use: the TF-style log-mel front end,
`basic_resnet` 2D blocks, `conv+att` 1D blocks, `agg_gnorm`, the 2D output head and ASTP
pooling. Other config options raise `NotImplementedError`. Parameter names mirror the PyTorch
`state_dict`, so conversion only transposes conv weights.

`convert.py` turns an official `.pt` checkpoint into `config.json` plus `weights.safetensors`.
`parity.py` runs PyTorch and MLX on the same audio and compares features, backbone output and
embeddings, with a float64 PyTorch run as the rounding floor. `publish_hf.py` uploads a
converted model to Hugging Face with a model card, and `publish_all.sh` converts and uploads
every released checkpoint. `results/` holds the parity logs.

### Convert a checkpoint

`convert.py` downloads from the upstream release when `--checkpoint` is omitted:

```bash
uv sync --group convert
uv run python convert.py --model b3 --train-type lm --dataset vox2 --out build/b3-vox2-lm
```

### Check parity

This needs a clone of the upstream repo and the `.pt` file:

```bash
git clone https://github.com/PalabraAI/redimnet2 && git -C redimnet2 checkout c5bbe0b
uv run --group convert python parity.py --torch-repo redimnet2 --checkpoint b3-vox2-lm.pt \
    --mlx-model build/b3-vox2-lm --wav some_speech.wav
```

Each checkpoint was checked on synthetic 1 s, 3.3 s and 8 s batches plus one real speech clip
(float32, MLX CPU backend, Linux). Worst embedding error relative to max |value|:

| checkpoint | lm | ptn |
|---|---|---|
| b0 vox2 | 2.9e-6 | 3.5e-6 |
| b1 vox2 | 3.4e-6 | 3.9e-6 |
| b2 vox2 | 3.9e-6 | 3.6e-6 |
| b3 vox2 | 4.4e-6 | 4.4e-6 |
| b3 vb2+vox2+cnc2 | 3.2e-6 | not released |
| b4 vox2 | 4.1e-6 | 3.1e-6 |

PyTorch's own float32 output sits about 1e-6 from a float64 run, which is the practical floor.
Bit-exact equality isn't possible because the two frameworks sum conv and matmul terms in a
different order.

### Publish to Hugging Face

`publish_all.sh` converts every released checkpoint and uploads each to its own repo with a
generated model card. `HF_TOKEN` comes from the environment:

```bash
HF_TOKEN=... ./publish_all.sh causal
```

### Lint

```bash
uv run --group dev ruff format --check .
uv run --group dev ruff check .
```

## License

The model code is ported from ReDimNet2 (MIT) and parts of ReDimNet (MIT) and WeSpeaker
(Apache-2.0). The converted weights come from the ReDimNet2 v1.0.0 release under the same MIT
license. The demo clips come from SpeechBrain (Apache-2.0). See `LICENSE-redimnet2` and
`THIRD_PARTY_NOTICES.md`.
