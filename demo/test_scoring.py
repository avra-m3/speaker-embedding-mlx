"""Checks the enroll + sliding-window scoring logic on the bundled two-speaker clips.

    uv run demo/test_scoring.py [--model mlx_models/b3-vox2-lm]

The clips in test_audio/ are from SpeechBrain's test samples (Apache-2.0): spk1_* and spk2_*
are two different speakers, three short utterances each.
"""

import argparse
import os
import tempfile

import numpy as np
from scoring import (
    DEFAULT_MODEL,
    SR,
    LiveScorer,
    decide,
    embed,
    enroll,
    load_model,
    load_reference,
    read_wav,
    score_file,
)

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIO = os.path.join(HERE, "test_audio")


def clip(name):
    return read_wav(os.path.join(AUDIO, name + ".wav"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--threshold", type=float, default=0.45)
    args = ap.parse_args()
    model = load_model(args.model)
    th = args.threshold
    failures = []

    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            failures.append(msg)

    # 1. Whole-clip scores: enroll on spk1_snt1, the other spk1 clips must beat every spk2 clip.
    print("1. whole-clip scores, enrolled on spk1_snt1")
    ref = enroll(model, [clip("spk1_snt1")])
    same = {n: float(embed(model, clip(n)) @ ref) for n in ("spk1_snt2", "spk1_snt3")}
    diff = {n: float(embed(model, clip(n)) @ ref) for n in ("spk2_snt1", "spk2_snt2", "spk2_snt3")}
    for n, s in {**same, **diff}.items():
        print(f"     {n}: {s:+.3f}")
    check(
        min(same.values()) > th > max(diff.values()),
        f"same speaker > {th} > different speaker",
    )

    # 2. Saved embedding round trip.
    print("2. .npy embedding round trip")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ref.npy")
        np.save(p, ref)
        check(
            np.allclose(load_reference(model, [p]), ref),
            "load_reference(.npy) == enrolled embedding",
        )

    # 3. A simulated conversation streamed through the live scorer in 100 ms chunks.
    print("3. streaming: spk1 | silence | spk2 | spk1 | spk2, 2 s window, 0.5 s hop")
    segs = [
        ("spk1", clip("spk1_snt2")),
        ("silence", np.zeros(int(1.5 * SR), np.float32)),
        ("spk2", clip("spk2_snt1")),
        ("spk1", clip("spk1_snt3")),
        ("spk2", clip("spk2_snt2")),
    ]
    labels = np.concatenate([np.full(len(w), i) for i, (_, w) in enumerate(segs)])
    wav = np.concatenate([w for _, w in segs])
    rng = np.random.default_rng(0)
    wav = wav + rng.normal(0, 1e-4, len(wav)).astype(np.float32)  # -80 dBFS noise floor

    scorer = LiveScorer(
        model, ref, window_s=2.0, hop_s=0.5, smooth=1.0
    )  # no smoothing: judge each window
    results = score_file(scorer, wav)
    expect = {"spk1": "match", "spk2": "no-match", "silence": "silence"}
    pure, correct = 0, 0
    print(f"     {'t':>5}  {'window':<16} {'score':>6}  state")
    for r in results:
        end = round(r.t * SR)
        w = labels[max(end - scorer.win, 0) : end]
        ids = np.unique(w)
        name = "+".join(segs[i][0] for i in ids)
        state = decide(r, th)
        mark = ""
        if len(ids) == 1:
            pure += 1
            ok = state == expect[segs[ids[0]][0]]
            correct += ok
            mark = "" if ok else "  <-- wrong"
        raw = "  --" if np.isnan(r.score) else f"{r.score:+.2f}"
        print(f"     {r.t:5.1f}  {name:<16} {raw:>6}  {state}{mark}")
    check(
        pure > 0 and correct == pure,
        f"{correct}/{pure} single-segment windows decided correctly",
    )

    # 4. Smoothing: with the default EMA the state must still flip within two hops of a change.
    print("4. default smoothing (0.5) still switches")
    scorer = LiveScorer(model, ref, window_s=2.0, hop_s=0.5)
    states = [decide(r, th) for r in score_file(scorer, wav)]
    check(
        "match" in states and "no-match" in states,
        "both match and no-match appear with smoothing on",
    )

    print("\nPASS" if not failures else f"\n{len(failures)} check(s) failed")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
