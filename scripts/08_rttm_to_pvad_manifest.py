#!/usr/bin/env python3
"""
Script: 08_rttm_to_pvad_manifest.py
Shared — Convert RTTM + WAV to PVAD training manifests.

Per session:
  - 1-speaker → 1 PVAD sample
  - 2-speaker → 2 PVAD samples
  - 3-speaker → 3 PVAD samples

Supports enrollment audio extraction (random clip from source).

Usage:
    python scripts/08_rttm_to_pvad_manifest.py \
        --config configs/datasets.yaml \
        --split train \
        --enrollment_duration 5.0 \
        --enrollment_strategy random_clip
"""

import argparse
import glob
import json
import os
import random

import yaml
from tqdm import tqdm


def parse_rttm(rttm_path: str) -> dict:
    """Parse RTTM file and return dict of speaker → list of (start, duration) segments."""
    speakers = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9 or parts[0] != "SPEAKER":
                continue
            speaker = parts[7]
            start = float(parts[3])
            duration = float(parts[4])
            if speaker not in speakers:
                speakers[speaker] = []
            speakers[speaker].append((start, duration))
    return speakers


def find_enrollment_audio(
    speaker_id: str,
    ds_config: dict,
    enrollment_duration: float,
) -> str:
    """Find enrollment audio for a speaker from the source dataset."""
    # Load the aligned manifest and find utterances for this speaker
    manifest_path = ds_config["manifest"].replace(".json", "_aligned.json")
    if not os.path.isfile(manifest_path):
        manifest_path = ds_config["manifest"]

    if not os.path.isfile(manifest_path):
        return ""

    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("speaker_id") == speaker_id:
                if entry.get("duration", 0) >= enrollment_duration:
                    return entry["audio_filepath"]

    return ""


def process_split(config: dict, split: str, enrollment_duration: float, enrollment_strategy: str):
    """Process all datasets for a given split."""
    all_entries = []

    for ds_key, ds_config in config["datasets"].items():
        ds_path = ds_config["path"]
        language = ds_config["language"]
        speaker_counts = config["simulation"]["speaker_counts"]

        for n_spk in speaker_counts:
            sim_dir = f"/data/simulated/{ds_path}/{split}_{n_spk}spk"
            if not os.path.isdir(sim_dir):
                print(f"  Skipping {sim_dir} (not found)")
                continue

            rttm_files = sorted(glob.glob(os.path.join(sim_dir, "*.rttm")))
            wav_files = sorted(glob.glob(os.path.join(sim_dir, "*.wav")))

            print(f"  {ds_key}/{split}_{n_spk}spk: {len(rttm_files)} sessions")

            for rttm_path in tqdm(rttm_files, desc=f"    {ds_key} {n_spk}spk"):
                session_name = os.path.splitext(os.path.basename(rttm_path))[0]
                wav_path = os.path.join(sim_dir, f"{session_name}.wav")

                if not os.path.isfile(wav_path):
                    continue

                speakers = parse_rttm(rttm_path)

                for target_speaker in speakers:
                    enrollment_audio = find_enrollment_audio(
                        target_speaker, ds_config, enrollment_duration
                    )

                    entry = {
                        "audio_filepath": wav_path,
                        "duration": config["simulation"]["session_length"],
                        "rttm_filepath": rttm_path,
                        "target_speaker": target_speaker,
                        "enrollment_audio": enrollment_audio,
                        "num_speakers": n_spk,
                        "language": language,
                        "source_dataset": ds_path,
                    }
                    all_entries.append(entry)

        # Write per-dataset per-speaker-count manifests
        for n_spk in speaker_counts:
            ds_entries = [
                e for e in all_entries
                if e["source_dataset"] == ds_path and e["num_speakers"] == n_spk
            ]
            if ds_entries:
                out_path = f"/data/pvad_manifests/{ds_path}_{split}_{n_spk}spk.json"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, "w") as f:
                    for e in ds_entries:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                print(f"  Written: {out_path} ({len(ds_entries)} entries)")

    # Write merged manifest
    random.shuffle(all_entries)
    merged_path = f"/data/pvad_manifests/{split}_merged.json"
    os.makedirs(os.path.dirname(merged_path), exist_ok=True)
    with open(merged_path, "w") as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n  Merged manifest: {merged_path} ({len(all_entries)} entries)")

    return all_entries


def main():
    parser = argparse.ArgumentParser(description="Convert RTTM to PVAD training manifests")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--enrollment_duration", type=float, default=5.0)
    parser.add_argument("--enrollment_strategy", type=str, default="random_clip",
                        choices=["random_clip"])
    parser.add_argument("--frame_hop_ms", type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"=== Converting RTTM → PVAD Manifest (split={args.split}) ===")
    entries = process_split(config, args.split, args.enrollment_duration, args.enrollment_strategy)
    print(f"\n=== Done. Total PVAD samples: {len(entries)} ===")


if __name__ == "__main__":
    main()
