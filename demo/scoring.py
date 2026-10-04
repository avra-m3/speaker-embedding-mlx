"""Speaker enrollment and sliding-window scoring, with no mic or GUI dependencies.

Everything here runs on plain numpy arrays of 16 kHz mono float32 audio, so the same code
drives the live mic demo and the file-based tests.
"""

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
import soundfile as sf

from redimnet2_mlx import load_model  # noqa: F401  (re-exported for the scripts)

SR = 16000
DEFAULT_MODEL = "causal/redimnet2-b3-vox2-lm-mlx"


def read_wav(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        raise SystemExit(f"{path}: expected 16 kHz audio, got {sr} Hz (resample it first)")
    return wav.mean(1)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-10)


def embed(model, wav):
    """Unit-length 192-dim embedding of one clip."""
    e = np.array(model(mx.array(np.asarray(wav, dtype=np.float32))[None])[0])
    return e / np.linalg.norm(e)


def enroll(model, clips):
    """Average the unit embeddings of one or more clips of the target speaker."""
    e = np.mean([embed(model, c) for c in clips], axis=0)
    return e / np.linalg.norm(e)


def load_reference(model, paths):
    """Build the enrolled embedding from .wav files and/or saved .npy embeddings."""
    embs = []
    for p in paths:
        if p.endswith(".npy"):
            e = np.load(p).astype(np.float32).ravel()
            embs.append(e / np.linalg.norm(e))
        else:
            embs.append(embed(model, read_wav(p)))
    e = np.mean(embs, axis=0)
    return e / np.linalg.norm(e)


@dataclass
class Result:
    t: float  # stream time (s) at the end of the scored window
    score: float  # raw cosine for this window (nan when silent)
    smoothed: float  # exponential moving average of voiced scores (nan until the first one)
    level_db: float  # window RMS in dBFS
    voiced: bool


class LiveScorer:
    """Keeps the last `window_s` seconds of audio and scores it every `hop_s` seconds.

    Silent windows (RMS below `silence_db`) are not embedded; they report voiced=False so the
    UI can show a neutral state instead of a spurious red.
    """

    def __init__(self, model, reference, window_s=2.0, hop_s=0.5, silence_db=-45.0, smooth=0.5):
        self.model = model
        self.ref = reference
        self.win = int(window_s * SR)
        self.hop = int(hop_s * SR)
        self.silence_db = silence_db
        self.smooth = smooth  # weight of the newest score in the EMA; 1.0 = no smoothing
        self.buf = np.zeros(0, np.float32)
        self.total = 0  # samples pushed so far
        self.last_scored = None  # value of self.total at the last score
        self.ema = float("nan")

    def push(self, samples):
        samples = np.asarray(samples, np.float32).ravel()
        self.total += len(samples)
        self.buf = np.concatenate([self.buf, samples])[-self.win :]

    def ready(self):
        if len(self.buf) < self.win:
            return False
        return self.last_scored is None or self.total - self.last_scored >= self.hop

    def score(self):
        """Score the current window. Call when ready(); if the caller fell behind, this just
        scores the newest audio, so slow inference drops hops instead of building a backlog."""
        self.last_scored = self.total
        window = self.buf
        level = rms_db(window)
        t = self.total / SR
        if level < self.silence_db:
            return Result(t, float("nan"), self.ema, level, False)
        s = float(embed(self.model, window) @ self.ref)
        self.ema = s if np.isnan(self.ema) else self.smooth * s + (1 - self.smooth) * self.ema
        return Result(t, s, self.ema, level, True)


def decide(result, threshold):
    """'match', 'no-match', or 'silence' for a Result."""
    if not result.voiced or np.isnan(result.smoothed):
        return "silence"
    return "match" if result.smoothed >= threshold else "no-match"


def score_file(scorer, wav, chunk_s=0.1):
    """Feed a whole clip through a LiveScorer in mic-sized chunks; returns every Result."""
    step = int(chunk_s * SR)
    out = []
    for i in range(0, len(wav), step):
        scorer.push(wav[i : i + step])
        if scorer.ready():
            out.append(scorer.score())
    return out
