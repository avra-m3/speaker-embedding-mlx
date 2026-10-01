# AGENTS.md

Instructions for AI coding agents working in this repository. Write for the agent, not for
humans. Keep this file under 300 lines. `CLAUDE.md` imports it.

## What this repo is

An MLX port of the ReDimNet2 speaker-embedding models, b0 to b4. Upstream is
github.com/PalabraAI/redimnet2, pinned at `c5bbe0b`. Inference only. Parameter names mirror the PyTorch `state_dict`, so
conversion only transposes conv weights. Keep that property when you add model code.

## Commands

```bash
uv sync                                  # runtime deps
uv sync --group convert                  # torch, torchaudio, scipy for convert.py and parity.py
uv run --group dev ruff format --check .
uv run --group dev ruff check .
```

Rerun `parity.py` against the PyTorch reference after any change under `redimnet2_mlx/`, and
update the matching log in `results/`. Aim for exact parity. Float32 rounding differences of
about 1e-5 relative are acceptable. Anything larger is a bug.

## Code conventions

- Manage Python with uv (`pyproject.toml`, `uv.lock`). Never use pip or requirements.txt.
- Code must pass `ruff format --check` and `ruff check` with the repo config (line length 100).
- Converted weights are published to Hugging Face under `causal`, one repo per checkpoint.
  `HF_TOKEN` comes from the environment only.

## Tone of voice

This is the most important section. It applies to all generated text: comments, docstrings,
commit messages, README and model cards, PR descriptions, error messages.

- Write like a staff engineer talking to a peer. Direct, specific, no theatre.
- Use short sentences. If a sentence needs a semicolon, split it.
- Prefer active voice. Say what the thing does, not what it aims to do.
- Never use em-dashes. Rewrite as two sentences or use a comma.
- No filler openers (Certainly, Great question, Sure), no throat-clearing (It's worth
  noting, As previously mentioned), no hedge stacking (could potentially), no vague
  intensifiers (very, really, quite, rather, extremely).
- Use verbs, not nominalisations: decide, not make a decision.
- No bullet lists for things that read fine as prose. No decorative bold.

## Do not

- Never perform git operations (commit, push, branch, PR) without explicit instruction.
- Never commit secrets, tokens or credentials.
- Never leave TODO comments without an issue reference.
- Never add a dependency without checking whether an existing one covers it.
- Never write a comment that restates the code.
