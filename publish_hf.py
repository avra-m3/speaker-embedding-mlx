"""Upload converted models to Hugging Face, one repo per checkpoint, with a model card.

    HF_TOKEN=... uv run python publish_hf.py --namespace avra-m3 mlx_models/b3-vox2-lm ...

Each directory becomes <namespace>/redimnet2-<name>-mlx (override with --repo-name for a
single directory). Use --dry-run to write the model cards locally without uploading.
"""

import argparse
import json
import re
from pathlib import Path

UPSTREAM = "https://github.com/PalabraAI/redimnet2"
CODE = "https://github.com/avra-m3/speaker-embedding-mlx"

DATASETS = {
    "vox2": "VoxCeleb2-dev",
    "vb2+vox2+cnc2_v0": "VoxBlink2 + VoxCeleb2 + CN-Celeb2",
}
TRAIN = {
    "lm": "large-margin fine-tuned (recommended)",
    "ptn": "pretrained, before large-margin fine-tuning",
}

DATASET_CAVEAT = """
## Training data terms

This checkpoint was trained on VoxBlink2 and CN-Celeb in addition to VoxCeleb2. The model
code and weights are MIT licensed upstream, but those datasets have their own terms of use,
which may restrict use to research. Check them before any commercial use.
"""


def repo_name(model_dir):
    slug = re.sub(r"[^a-z0-9]+", "-", model_dir.name.lower()).strip("-")
    slug = slug.replace("-v0-", "-")
    return f"redimnet2-{slug}-mlx"


RESULTS = Path(__file__).parent / "results"


def parity_line(name):
    """Per-checkpoint parity sentence from results/parity_<name>.txt, if it exists."""
    log = RESULTS / f"parity_{name}.txt"
    m = log.exists() and re.search(r"worst embedding rel error: ([0-9.e+-]+)", log.read_text())
    if not m:
        return "Parity against PyTorch has not been recorded for this checkpoint."
    return (
        f"Embeddings match the PyTorch reference to {float(m.group(1)):.1e} relative error "
        "(worst case, float32, MLX CPU backend)."
    )


def model_card(model_dir, repo_id):
    size, rest = model_dir.name.split("-", 1)
    dataset, train = rest.rsplit("-", 1)
    cfg = json.loads((model_dir / "config.json").read_text())
    caveat = DATASET_CAVEAT if dataset != "vox2" else ""
    return f"""---
license: mit
library_name: mlx
pipeline_tag: audio-classification
tags:
  - mlx
  - speaker-verification
  - speaker-embedding
  - redimnet2
---

# ReDimNet2 {size.upper()} ({DATASETS.get(dataset, dataset)}, {train}) for MLX

MLX conversion of the official [ReDimNet2]({UPSTREAM}) `{model_dir.name}.pt` checkpoint
({TRAIN.get(train, train)}). It produces {cfg.get("embed_dim", 192)}-dimensional speaker
embeddings from 16 kHz mono audio.

## Usage

```bash
uv add git+{CODE}
```

```python
import mlx.core as mx
from redimnet2_mlx import load_model

model = load_model("{repo_id}")
emb = model(mx.array(wav_16k_float32)[None])  # (1, {cfg.get("embed_dim", 192)})
```

## Parity

{parity_line(model_dir.name)} See the
[parity log]({CODE}/blob/main/results/parity_{model_dir.name.replace("+", "%2B")}.txt).
{caveat}
## License and citation

MIT, following the upstream ReDimNet2 release (Copyright (c) 2026 Palabra.ai). See
[third-party notices]({CODE}/blob/main/THIRD_PARTY_NOTICES.md).

```bibtex
@inproceedings{{redimnet2_2026,
  title={{ReDimNet2: Scaling Speaker Verification via Time-Pooled Dimension Reshaping}},
  author={{Yakovlev, Ivan and Okhotnikov, Anton}},
  booktitle={{Proceedings of Interspeech 2026}},
  year={{2026}},
  eprint={{2603.11841}},
  archivePrefix={{arXiv}},
}}
```
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model_dirs", nargs="+", type=Path)
    p.add_argument("--namespace", required=True, help="HF user or org")
    p.add_argument("--repo-name", help="override repo name (single directory only)")
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.repo_name and len(args.model_dirs) != 1:
        raise SystemExit("--repo-name needs exactly one model directory")

    for d in args.model_dirs:
        repo_id = f"{args.namespace}/{args.repo_name or repo_name(d)}"
        card = model_card(d, repo_id)
        (d / "README.md").write_text(card)
        if args.dry_run:
            print(f"[dry run] {d} -> {repo_id}")
            continue
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(d),
            allow_patterns=["config.json", "weights.safetensors", "README.md"],
            commit_message=f"Upload {d.name} converted to MLX",
        )
        print(f"uploaded {d} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
