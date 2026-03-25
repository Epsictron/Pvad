#!/usr/bin/env python3
"""
Script: 11_visualize_session.py
Shared — Visualize a single simulated session (waveform + RTTM speaker segments).

Usage:
    python scripts/11_visualize_session.py \
        --wav /data/simulated/hindi_conv/train_2spk/multispeaker_session_0.wav \
        --rttm /data/simulated/hindi_conv/train_2spk/multispeaker_session_0.rttm \
        --output_png /data/viz/session_0.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


SPEAKER_COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]


def parse_rttm(rttm_path: str) -> list:
    segments = []
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            segments.append({
                "speaker": parts[7],
                "start": float(parts[3]),
                "duration": float(parts[4]),
            })
    return segments


def main():
    parser = argparse.ArgumentParser(description="Visualize a simulated session")
    parser.add_argument("--wav", type=str, required=True, help="Path to WAV file")
    parser.add_argument("--rttm", type=str, required=True, help="Path to RTTM file")
    parser.add_argument("--output_png", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    # Load audio
    audio, sr = sf.read(args.wav)
    duration = len(audio) / sr
    time = np.arange(len(audio)) / sr

    # Parse RTTM
    segments = parse_rttm(args.rttm)
    speakers = sorted(set(s["speaker"] for s in segments))
    speaker_colors = {spk: SPEAKER_COLORS[i % len(SPEAKER_COLORS)] for i, spk in enumerate(speakers)}

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(20, 6), gridspec_kw={"height_ratios": [2, 1]})

    # Waveform
    ax_wav = axes[0]
    ax_wav.plot(time, audio, linewidth=0.3, color="gray", alpha=0.7)
    ax_wav.set_xlim(0, duration)
    ax_wav.set_ylabel("Amplitude")
    ax_wav.set_title(f"Session: {os.path.basename(args.wav)} | {len(speakers)} speaker(s)")

    # Speaker segments
    ax_spk = axes[1]
    for seg in segments:
        spk_idx = speakers.index(seg["speaker"])
        color = speaker_colors[seg["speaker"]]
        ax_spk.barh(
            spk_idx,
            seg["duration"],
            left=seg["start"],
            height=0.6,
            color=color,
            alpha=0.8,
        )

    ax_spk.set_yticks(range(len(speakers)))
    ax_spk.set_yticklabels(speakers)
    ax_spk.set_xlim(0, duration)
    ax_spk.set_xlabel("Time (s)")
    ax_spk.set_ylabel("Speaker")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png), exist_ok=True)
    plt.savefig(args.output_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output_png}")


if __name__ == "__main__":
    main()
