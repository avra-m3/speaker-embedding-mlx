# Speaker match light

Enroll one speaker, then listen to the mic. A small always-on-top window shows a green dot
while the enrolled speaker talks, red for anyone else, and grey during silence. It runs on
Apple Silicon with the b3 model from this repo.

## Setup

```bash
uv sync --group demo
```

The `demo` group adds `sounddevice` for mic input and `scipy`, which resamples when a mic
refuses to open at 16 kHz. The window uses tkinter. uv's managed Python ships it. With
Homebrew Python, run `brew install python-tk@3.12` and match your version.

The first mic run makes macOS ask for microphone access for your terminal app. If you missed
the prompt, enable it under System Settings > Privacy & Security > Microphone and restart the
terminal.

## Run

```bash
# Enroll from 8 s of mic audio after a 3 s countdown, save it, then listen
uv run --group demo demo/speaker_light.py --save-embedding me.npy

# Reuse the saved embedding
uv run --group demo demo/speaker_light.py --enroll me.npy

# Enroll from 16 kHz mono wav files. Several files are averaged.
uv run --group demo demo/speaker_light.py --enroll a.wav b.wav

# Score a recording at real-time speed instead of the mic
uv run --group demo demo/speaker_light.py --enroll me.npy --file meeting.wav

# Print one line per score instead of opening a window
uv run --group demo demo/speaker_light.py --enroll me.npy --no-gui
```

Drag the threshold slider to tune the cutoff live. Re-enroll captures a fresh enrollment from
the running mic and writes it to `--save-embedding` when that flag is set.

The model defaults to `mlx_models/b3-vox2-lm` when that directory exists and to
`causal/redimnet2-b3-vox2-lm-mlx` on Hugging Face otherwise. Pass `--model` to pick another.

## How it scores

Every `--hop` seconds (default 0.5) the scorer embeds the last `--window` seconds (default 2)
and takes the cosine with the enrolled embedding. A running average smooths the score
(`--smooth 0.5`, set 1 to disable). Windows quieter than `--silence-db` (default -45 dBFS)
skip the model and show grey. Mic enrollment drops 100 ms frames below that level so pauses
don't dilute the voiceprint.

The default threshold is 0.45. On the bundled clips, 2 s windows of the enrolled speaker
score 0.62 to 0.72 and the other speaker scores 0.00 to 0.22. A real mic and room shift both,
so expect to tune it. A longer window gives steadier scores and slower reactions. If one
window takes longer than `--hop` to score, the app skips hops instead of falling behind.

## Tests

```bash
uv run demo/test_scoring.py
```

The test enrolls on one clip and checks whole-clip scores, the `.npy` round trip, and a
simulated conversation streamed through the live scorer in 100 ms chunks. `test_audio/` holds
two speakers from SpeechBrain's test samples (Apache-2.0). `screenshots/` shows the window
rendering a match and a non-match.

`scoring.py` holds enrollment, the sliding-window scorer and the decision rule with no mic or
GUI code, so the test exercises the same logic as the app. `speaker_light.py` adds the mic and
file sources, mic enrollment, the window and the headless mode.

## Not verified yet

The test and the file mode ran in a Linux container on the MLX CPU backend. The window was
checked on a virtual X display. Live mic capture, the macOS permission prompt, mic
enrollment, Re-enroll and Metal speed have not run on a Mac yet. The app prints its warm-up
inference time at start.
