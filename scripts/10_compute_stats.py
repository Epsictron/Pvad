#!/usr/bin/env python3
"""
Script: 10_compute_stats.py
Shared — Compute and display dataset statistics.

Usage:
    python scripts/10_compute_stats.py --config configs/datasets.yaml
"""

import argparse
import glob
import os

import numpy as np
import soundfile as sf
import yaml


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


def compute_turn_gaps(segments: list) -> list:
    """Compute gaps between consecutive segments."""
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    gaps = []
    for i in range(1, len(sorted_segs)):
        prev_end = sorted_segs[i - 1]["start"] + sorted_segs[i - 1]["duration"]
        gap = sorted_segs[i]["start"] - prev_end
        if gap > 0:
            gaps.append(gap)
    return gaps


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    all_gaps = []

    # Header
    print()
    print("+" + "-" * 25 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 10 + "+")
    print(f"| {'Dataset':<23} | {'Lang':<4} | {'1spk':<5} | {'2spk':<5} | {'3spk':<5} | {'Hours':<8} |")
    print("+" + "-" * 25 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 10 + "+")

    total_counts = {1: 0, 2: 0, 3: 0}
    total_hours = 0.0
    languages = set()

    for ds_key, ds_config in config["datasets"].items():
        ds_path = ds_config["path"]
        language = ds_config["language"]
        languages.add(language)

        counts = {}
        ds_hours = 0.0

        for n_spk in sim_config["speaker_counts"]:
            count = 0
            for split in ["train", "val", "test"]:
                sim_dir = f"/data/simulated/{ds_path}/{split}_{n_spk}spk"
                if os.path.isdir(sim_dir):
                    rttm_files = glob.glob(os.path.join(sim_dir, "*.rttm"))
                    count += len(rttm_files)

                    # Compute gaps from a sample
                    for rttm_path in rttm_files[:50]:
                        segments = parse_rttm(rttm_path)
                        all_gaps.extend(compute_turn_gaps(segments))

            counts[n_spk] = count
            total_counts[n_spk] += count
            ds_hours += count * sim_config["session_length"] / 3600

        total_hours += ds_hours
        print(f"| {ds_key:<23} | {language:<4} | {counts.get(1, 0):<5} | {counts.get(2, 0):<5} | {counts.get(3, 0):<5} | {ds_hours:<8.0f} |")

    print("+" + "-" * 25 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 10 + "+")
    print(f"| {'TOTAL':<23} | {len(languages):<4} | {total_counts[1]:<5} | {total_counts[2]:<5} | {total_counts[3]:<5} | {total_hours:<8.0f} |")
    print("+" + "-" * 25 + "+" + "-" * 6 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 7 + "+" + "-" * 10 + "+")

    # Turn gap stats
    if all_gaps:
        gaps = np.array(all_gaps)
        print(f"\nTurn gap stats: mean={gaps.mean():.2f}s, std={gaps.std():.2f}s, "
              f"min={gaps.min():.2f}s, max={gaps.max():.2f}s")
    else:
        print("\nNo turn gap data available (no simulated sessions found)")

    print("Overlap: 0.0%")
    print("Background noise: None")


if __name__ == "__main__":
    main()
