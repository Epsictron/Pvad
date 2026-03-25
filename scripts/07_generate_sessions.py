#!/usr/bin/env python3
"""
Script: 07_generate_sessions.py
Shared — Run NeMo multi-speaker simulator per dataset x per speaker count (1, 2, 3).

No augmentation, no overlap. Clean speech + silence only.
Auto-scales session count per dataset based on available speakers.

Usage:
    python scripts/07_generate_sessions.py --config configs/datasets.yaml --split train
    python scripts/07_generate_sessions.py --config configs/datasets.yaml --split val
"""

import argparse
import json
import os
import subprocess
from math import comb

import yaml


def get_aligned_manifest(ds_key: str, ds_config: dict) -> str:
    """Get the path to the aligned manifest for a dataset."""
    manifest = ds_config["manifest"]
    return manifest.replace(".json", "_aligned.json")


def compute_max_sessions(num_speakers: int, speaker_count: int, target_sessions: int) -> int:
    """Auto-scale sessions to avoid speaker repetition."""
    if num_speakers < speaker_count:
        print(f"    WARNING: Only {num_speakers} speakers available, need {speaker_count}. Skipping.")
        return 0
    max_combinations = comb(num_speakers, speaker_count) * 5
    return min(target_sessions, max_combinations)


def run_simulator(
    manifest_path: str,
    num_sessions: int,
    num_speakers: int,
    output_dir: str,
    seed: int,
    session_length: int,
    turn_gap: dict,
):
    """Run NeMo multispeaker_simulator for one dataset x one speaker count."""
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "NeMo/tools/speech_data_simulator/multispeaker_simulator.py",
        "--config-path=configs",
        "--config-name=data_simulator",
        f"data_simulator.manifest_filepath={manifest_path}",
        f"data_simulator.session_config.num_sessions={num_sessions}",
        f"data_simulator.session_config.session_length={session_length}",
        f"data_simulator.session_params.num_speakers={num_speakers}",
        f"data_simulator.outputs.output_dir={output_dir}",
        f"data_simulator.random_seed={seed}",
        f"data_simulator.session_params.mean_silence={turn_gap['mean']}",
        f"data_simulator.session_params.mean_silence_var={turn_gap['variance']}",
        f"data_simulator.session_params.per_silence_min={turn_gap['min']}",
        f"data_simulator.session_params.per_silence_max={turn_gap['max']}",
    ]

    print(f"    Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-speaker sessions")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    speaker_counts = sim_config["speaker_counts"]
    target_sessions = sim_config["sessions_per_count"][args.split]
    session_length = sim_config["session_length"]
    turn_gap = sim_config["turn_gap"]

    for ds_key, ds_config in config["datasets"].items():
        aligned_manifest = get_aligned_manifest(ds_key, ds_config)
        if not os.path.isfile(aligned_manifest):
            print(f"WARNING: Aligned manifest not found for {ds_key}: {aligned_manifest}. Skipping.")
            continue

        num_speakers_corpus = ds_config.get("num_speakers_in_corpus", 100)
        ds_path = ds_config["path"]

        print(f"\n{'='*60}")
        print(f"Generating sessions for: {ds_config['name']} ({ds_key})")
        print(f"  Split: {args.split}")
        print(f"{'='*60}")

        for n_spk in speaker_counts:
            max_sessions = compute_max_sessions(num_speakers_corpus, n_spk, target_sessions)
            if max_sessions == 0:
                continue

            output_dir = f"/data/simulated/{ds_path}/{args.split}_{n_spk}spk"
            seed = 42 + n_spk + (hash(args.split) % 1000)

            print(f"\n  {n_spk}-speaker: {max_sessions} sessions → {output_dir}")

            run_simulator(
                manifest_path=aligned_manifest,
                num_sessions=max_sessions,
                num_speakers=n_spk,
                output_dir=output_dir,
                seed=seed,
                session_length=session_length,
                turn_gap=turn_gap,
            )

    print(f"\n=== All session generation complete for split={args.split} ===")


if __name__ == "__main__":
    main()
