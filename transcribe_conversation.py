#!/usr/bin/env python3
"""
Two-microphone conversation transcriber
(RØDE Wireless Micro in Split mode, recorded in stereo with RØDE Reporter)

Left channel  = person on transmitter 1
Right channel = person on transmitter 2

What it does, 100% locally on your Mac:
  1. Reads the stereo recording (WAV, M4A, AAC, MP3...)
  2. Transcribes it with Whisper (large-v3-turbo, optimised for Apple Silicon)
  3. For every word, checks which microphone was louder at that moment
     -> that tells us who said it
  4. Groups the words into turns and writes a transcript:
       [00:03:12] Alice: ...
       [00:03:20] Bob: ...

Usage:
  python transcribe_conversation.py "recording.m4a" --left "Alice" --right "Bob" --language pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
WINDOW_PAD = 0.05          # seconds added around each word when measuring loudness
DEFAULT_MARGIN_DB = 3.0    # how much louder one mic must be to be sure who spoke


# ---------------------------------------------------------------- audio input

def load_stereo(path: Path):
    """Decode any audio file into two float32 arrays (left, right) at 16 kHz."""
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-nostdin", "-v", "error", "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(SAMPLE_RATE), "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        sys.exit("Could not read the audio file:\n"
                 + result.stderr.decode(errors="ignore"))

    audio = np.frombuffer(result.stdout, dtype=np.float32)
    if audio.size < SAMPLE_RATE * 2:
        sys.exit("The audio file is empty or too short.")
    audio = audio[: audio.size // 2 * 2].reshape(-1, 2)
    return audio[:, 0].copy(), audio[:, 1].copy()


def check_really_split(left, right):
    """Stop early if both channels are (almost) identical = not recorded in Split/stereo."""
    diff = np.mean(np.abs(left - right))
    level = np.mean(np.abs(left) + np.abs(right)) + 1e-9
    if diff / level < 0.01:
        sys.exit(
            "Both channels contain the same audio, so the speakers can't be told apart.\n"
            "Check that the mics are in Split mode and RØDE Reporter records in STEREO."
        )


def speech_level(x):
    """Typical loudness of someone speaking into this mic (95th percentile of 20 ms frames)."""
    frame = int(0.02 * SAMPLE_RATE)
    n = len(x) // frame
    if n == 0:
        return 1e-6
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1))
    return max(float(np.percentile(rms, 95)), 1e-6)


def level_db(x, start, end):
    a = max(0, int((start - WINDOW_PAD) * SAMPLE_RATE))
    b = min(len(x), int((end + WINDOW_PAD) * SAMPLE_RATE))
    if b <= a:
        return -120.0
    return 20 * np.log10(np.sqrt(np.mean(x[a:b] ** 2)) + 1e-9)


# ------------------------------------------------------------- attribution

def attribute_words(words, left, right, left_name, right_name, margin_db):
    """Give each word a speaker based on which mic was louder."""
    labelled = []
    previous = None
    for w in words:
        diff = level_db(left, w["start"], w["end"]) - level_db(right, w["start"], w["end"])
        if diff > margin_db:
            speaker = left_name
        elif diff < -margin_db:
            speaker = right_name
        else:
            # Unclear (e.g. very short word): keep the current speaker
            speaker = previous or (left_name if diff >= 0 else right_name)
        previous = speaker
        labelled.append({**w, "speaker": speaker})
    return labelled


def group_into_turns(labelled):
    turns = []
    for w in labelled:
        if turns and turns[-1]["speaker"] == w["speaker"]:
            turns[-1]["text"] += w["word"]
            turns[-1]["end"] = w["end"]
        else:
            turns.append({"speaker": w["speaker"], "start": w["start"],
                          "end": w["end"], "text": w["word"]})
    for t in turns:
        t["text"] = " ".join(t["text"].split())
    return [t for t in turns if t["text"]]


def fmt_time(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="Transcribe a two-mic stereo recording with speaker names.")
    p.add_argument("audio", help="The recording (stereo file from RØDE Reporter)")
    p.add_argument("--left", default="Speaker 1", help="Name of the person on the LEFT channel")
    p.add_argument("--right", default="Speaker 2", help="Name of the person on the RIGHT channel")
    p.add_argument("--language", default=None,
                   help="Language code, e.g. pt or en. Leave out to auto-detect.")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model to use")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN_DB,
                   help="dB difference needed to switch speaker (default 3)")
    args = p.parse_args()

    path = Path(args.audio).expanduser()
    if not path.exists():
        sys.exit(f"File not found: {path}")

    print("1/3  Reading audio…")
    left, right = load_stereo(path)
    check_really_split(left, right)

    # Balance the two mics so a mic with higher gain doesn't "win" unfairly
    left = left / speech_level(left)
    right = right / speech_level(right)

    mix = (left + right) / 2
    peak = float(np.max(np.abs(mix)))
    if peak > 0:
        mix = mix / peak * 0.9
    mix = mix.astype(np.float32)

    print("2/3  Transcribing locally (first run downloads the model, ~1.6 GB)…")
    import mlx_whisper

    result = mlx_whisper.transcribe(
        mix,
        path_or_hf_repo=args.model,
        language=args.language,
        word_timestamps=True,
        condition_on_previous_text=False,   # reduces repeated-text glitches
        verbose=False,                       # shows a progress bar
    )

    words = [w for seg in result.get("segments", []) for w in seg.get("words", [])]
    if not words:
        sys.exit("No speech was found in the recording.")

    print("3/3  Working out who said what…")
    labelled = attribute_words(words, left, right, args.left, args.right, args.margin)
    turns = group_into_turns(labelled)

    out_path = path.with_name(path.stem + "_transcript.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for t in turns:
            f.write(f"[{fmt_time(t['start'])}] {t['speaker']}: {t['text']}\n\n")

    print(f"\nDone. Transcript saved to:\n{out_path}")


if __name__ == "__main__":
    main()
