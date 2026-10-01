"""Live "is this the enrolled speaker?" light, using ReDimNet2 on MLX.

Enroll a speaker, then listen to the microphone: the dot turns green while the enrolled
speaker is talking, red for anyone else, and grey during silence.

    # enroll from 8 s of mic audio, save the embedding, then listen
    uv run --group demo demo/speaker_light.py --save-embedding me.npy

    # reuse a saved embedding (or enroll from wav files; several are averaged)
    uv run --group demo demo/speaker_light.py --enroll me.npy
    uv run --group demo demo/speaker_light.py --enroll a.wav b.wav

    # replay a file instead of the mic (real time, same window as the mic)
    uv run --group demo demo/speaker_light.py --enroll a.wav --file conversation.wav

    # no window: print one line per scored hop
    uv run --group demo demo/speaker_light.py --enroll me.npy --no-gui
"""

import argparse
import queue
import sys
import time
from math import gcd

import mlx.core as mx
import numpy as np
from scoring import (
    DEFAULT_MODEL,
    SR,
    LiveScorer,
    decide,
    enroll,
    load_model,
    load_reference,
    read_wav,
    rms_db,
)

COLORS = {
    "match": "#22c55e",
    "no-match": "#ef4444",
    "silence": "#6b7280",
    "waiting": "#374151",
}


# ---------------------------------------------------------------- audio sources


class MicSource:
    """Pulls 16 kHz mono chunks from the default input device via sounddevice."""

    def __init__(self, device=None, block_s=0.1):
        import sounddevice as sd

        self.q = queue.Queue()
        self.resample = None
        try:
            sd.check_input_settings(device=device, samplerate=SR, channels=1)
            rate = SR
        except (sd.PortAudioError, ValueError):
            # Device can't do 16 kHz directly: capture at its native rate and resample here.
            rate = int(sd.query_devices(device, "input")["default_samplerate"])
            from scipy.signal import resample_poly

            g = gcd(SR, rate)
            self.resample = lambda x: resample_poly(x, SR // g, rate // g).astype(np.float32)
        self.stream = sd.InputStream(
            device=device,
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=int(block_s * rate),
            callback=self._cb,
        )
        self.rate = rate

    def _cb(self, indata, frames, time_info, status):
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def read(self):
        chunks = []
        while True:
            try:
                chunks.append(self.q.get_nowait())
            except queue.Empty:
                break
        if not chunks:
            return np.zeros(0, np.float32)
        x = np.concatenate(chunks)
        return self.resample(x) if self.resample else x


class FileSource:
    """Plays a wav file into the scorer at real-time speed (no sound output)."""

    def __init__(self, path, loop=False):
        self.wav = read_wav(path)
        self.loop = loop
        self.pos = 0

    def start(self):
        self.t0 = time.monotonic()

    def stop(self):
        pass

    def done(self):
        return not self.loop and self.pos >= len(self.wav)

    def read(self):
        due = int((time.monotonic() - self.t0) * SR)
        n = due - self.pos
        if n <= 0:
            return np.zeros(0, np.float32)
        idx = np.arange(self.pos, self.pos + n)
        self.pos = due
        if self.loop:
            return self.wav[idx % len(self.wav)]
        return self.wav[idx[idx < len(self.wav)]]


# ---------------------------------------------------------------- enrollment


def speech_only(wav, silence_db, frame_s=0.1):
    """Drop 100 ms frames quieter than silence_db, so pauses don't dilute the enrollment."""
    f = int(frame_s * SR)
    frames = [wav[i : i + f] for i in range(0, len(wav) - f + 1, f)]
    kept = [x for x in frames if rms_db(x) >= silence_db]
    return np.concatenate(kept) if kept else wav


def record_enrollment(seconds, device, silence_db):
    print(f"Enrollment: speak normally for {seconds:.0f} s after the countdown.")
    for i in (3, 2, 1):
        print(f"  {i}...")
        time.sleep(1)
    print("  Recording.")
    src = MicSource(device)
    src.start()
    got, t_end = [], time.monotonic() + seconds
    while time.monotonic() < t_end:
        time.sleep(0.05)
        got.append(src.read())
    src.stop()
    wav = np.concatenate(got)
    voiced = speech_only(wav, silence_db)
    print(
        f"  Done: {len(wav) / SR:.1f} s recorded, {len(voiced) / SR:.1f} s above {silence_db:.0f} dBFS."
    )
    if len(voiced) < SR:
        print("  Warning: less than 1 s of speech captured; check the mic level or --silence-db.")
    return voiced


# ---------------------------------------------------------------- UI


class LightWindow:
    def __init__(self, scorer, source, threshold, allow_reenroll, enroll_s, save_path):
        import tkinter as tk

        self.tk = tk
        self.scorer, self.source = scorer, source
        self.enroll_s, self.save_path = enroll_s, save_path
        self.capture = None  # list of chunks while re-enrolling from the live stream
        self.capture_until = 0

        self.root = tk.Tk()
        self.root.title("Speaker match")
        self.root.configure(bg="#111827")
        self.root.attributes("-topmost", True)

        self.canvas = tk.Canvas(
            self.root, width=200, height=200, bg="#111827", highlightthickness=0
        )
        self.canvas.pack(padx=20, pady=(20, 8))
        self.dot = self.canvas.create_oval(20, 20, 180, 180, fill=COLORS["waiting"], outline="")

        self.label = tk.Label(
            self.root,
            text="warming up…",
            fg="#e5e7eb",
            bg="#111827",
            font=("Helvetica", 16),
        )
        self.label.pack()
        self.detail = tk.Label(self.root, text="", fg="#9ca3af", bg="#111827", font=("Menlo", 11))
        self.detail.pack(pady=(2, 8))

        self.threshold = tk.DoubleVar(value=threshold)
        tk.Scale(
            self.root,
            variable=self.threshold,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient="horizontal",
            length=220,
            label="threshold",
            fg="#e5e7eb",
            bg="#111827",
            highlightthickness=0,
            troughcolor="#374151",
        ).pack(padx=20)

        if allow_reenroll:
            self.btn = tk.Button(
                self.root,
                text=f"Re-enroll ({enroll_s:.0f} s)",
                command=self.start_reenroll,
            )
            self.btn.pack(pady=(8, 20))
        else:
            tk.Label(self.root, text="", bg="#111827").pack(pady=6)

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def start_reenroll(self):
        self.capture = []
        self.capture_until = time.monotonic() + self.enroll_s
        self.btn.configure(state="disabled")

    def finish_reenroll(self):
        wav = speech_only(np.concatenate(self.capture), self.scorer.silence_db)
        self.capture = None
        self.btn.configure(state="normal")
        if len(wav) < SR:
            self.label.configure(text="re-enroll: too little speech")
            return
        self.scorer.ref = enroll(self.scorer.model, [wav])
        self.scorer.ema = float("nan")
        if self.save_path:
            np.save(self.save_path, self.scorer.ref)
        print(f"Re-enrolled from {len(wav) / SR:.1f} s of speech.")

    def tick(self):
        x = self.source.read()
        if len(x):
            if self.capture is not None:
                self.capture.append(x)
            self.scorer.push(x)
        if self.capture is not None:
            left = self.capture_until - time.monotonic()
            self.label.configure(text=f"re-enrolling… {max(left, 0):.1f} s")
            self.canvas.itemconfigure(self.dot, fill="#3b82f6")
            if left <= 0:
                self.finish_reenroll()
        elif self.scorer.ready():
            r = self.scorer.score()
            state = decide(r, self.threshold.get())
            self.canvas.itemconfigure(self.dot, fill=COLORS[state])
            self.label.configure(
                text={
                    "match": "enrolled speaker",
                    "no-match": "someone else",
                    "silence": "silence",
                }[state]
            )
            raw = "  --  " if np.isnan(r.score) else f"{r.score:+.2f}"
            sm = "  --  " if np.isnan(r.smoothed) else f"{r.smoothed:+.2f}"
            self.detail.configure(text=f"score {raw}  avg {sm}  {r.level_db:5.0f} dB")
        if isinstance(self.source, FileSource) and self.source.done():
            self.label.configure(text=self.label.cget("text") + " (file ended)")
            return
        self.root.after(20, self.tick)

    def run(self):
        self.source.start()
        self.root.after(20, self.tick)
        self.root.mainloop()

    def close(self):
        self.source.stop()
        self.root.destroy()


def run_headless(scorer, source, threshold):
    source.start()
    print(f"{'time':>6}  {'score':>6}  {'avg':>6}  {'level':>6}  state")
    try:
        while not (isinstance(source, FileSource) and source.done()):
            x = source.read()
            if len(x):
                scorer.push(x)
            if scorer.ready():
                r = scorer.score()
                raw = "--" if np.isnan(r.score) else f"{r.score:+.2f}"
                sm = "--" if np.isnan(r.smoothed) else f"{r.smoothed:+.2f}"
                print(
                    f"{r.t:6.1f}  {raw:>6}  {sm:>6}  {r.level_db:6.0f}  {decide(r, threshold)}",
                    flush=True,
                )
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="converted MLX model dir (default: b3-vox2-lm)",
    )
    ap.add_argument(
        "--enroll",
        nargs="+",
        metavar="WAV_OR_NPY",
        help="enrollment clips (.wav) or a saved embedding (.npy); omit to record from the mic",
    )
    ap.add_argument(
        "--enroll-seconds",
        type=float,
        default=8.0,
        help="mic enrollment length (default 8)",
    )
    ap.add_argument("--save-embedding", metavar="NPY", help="write the enrolled embedding here")
    ap.add_argument("--file", help="score this wav at real-time speed instead of the mic")
    ap.add_argument("--loop", action="store_true", help="loop --file")
    ap.add_argument("--device", default=None, help="sounddevice input device name or index")
    ap.add_argument("--threshold", type=float, default=0.45, help="cosine threshold (default 0.45)")
    ap.add_argument(
        "--window",
        type=float,
        default=2.0,
        help="seconds of audio per score (default 2)",
    )
    ap.add_argument("--hop", type=float, default=0.5, help="seconds between scores (default 0.5)")
    ap.add_argument(
        "--smooth",
        type=float,
        default=0.5,
        help="weight of the newest score in the running average; 1 = none (default 0.5)",
    )
    ap.add_argument(
        "--silence-db",
        type=float,
        default=-45.0,
        help="windows quieter than this (dBFS) show grey (default -45)",
    )
    ap.add_argument(
        "--no-gui", action="store_true", help="print scores instead of opening a window"
    )
    args = ap.parse_args()
    device = int(args.device) if args.device and args.device.isdigit() else args.device

    print(f"Loading {args.model}")
    model = load_model(args.model)
    if args.enroll:
        ref = load_reference(model, args.enroll)
    else:
        ref = enroll(model, [record_enrollment(args.enroll_seconds, device, args.silence_db)])
    if args.save_embedding:
        np.save(args.save_embedding, ref)
        print(f"Saved embedding to {args.save_embedding}")

    scorer = LiveScorer(model, ref, args.window, args.hop, args.silence_db, args.smooth)
    t = time.perf_counter()
    mx.eval(scorer.model(mx.zeros((1, scorer.win))))  # warm up kernels
    print(
        f"Warm-up inference on a {args.window:.1f} s window: {1000 * (time.perf_counter() - t):.0f} ms"
    )

    source = FileSource(args.file, args.loop) if args.file else MicSource(device)
    if args.no_gui:
        run_headless(scorer, source, args.threshold)
    else:
        LightWindow(
            scorer,
            source,
            args.threshold,
            allow_reenroll=not args.file,
            enroll_s=args.enroll_seconds,
            save_path=args.save_embedding,
        ).run()


if __name__ == "__main__":
    main()
