#!/usr/bin/env python3
"""
Script: 03b_run_silero_vad.py
PATH B — Step 3: Run Silero VAD on custom dataset utterances to get precise speech boundaries.

Per utterance:
  1. Load audio
  2. Run Silero VAD → frame-level speech probabilities (~32ms resolution)
  3. Save per-utterance VAD segments

Usage:
    python scripts/03b_run_silero_vad.py --config configs/datasets.yaml
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

SILERO_SAMPLE_RATE = 16000


def load_manifest(manifest_path: str) -> list:
    """Load NeMo-format manifest (JSONL)."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_vad_on_dataset(ds_key: str, ds_config: dict, alignment_cfg: dict):
    """Run Silero VAD on all utterances of a dataset."""
    import torchaudio

    manifest_path = ds_config["manifest"]
    ds_path = ds_config["path"]
    output_dir = f"/data/vad_output/{ds_path}"

    threshold = alignment_cfg.get("silero_vad_threshold", 0.5)
    window_ms = alignment_cfg.get("silero_window_ms", 32)
    window_samples = int(SILERO_SAMPLE_RATE * window_ms / 1000)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running Silero VAD for: {ds_config['name']} ({ds_key})")
    print(f"  Manifest: {manifest_path}")
    print(f"  Output: {output_dir}")
    print(f"  Threshold: {threshold}")
    print(f"  Window: {window_ms}ms ({window_samples} samples)")
    print(f"{'='*60}")

    # Load Silero VAD model
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    (get_speech_timestamps, _, read_audio, _, _) = utils

    entries = load_manifest(manifest_path)
    print(f"  Processing {len(entries)} utterances...")

    stats = {"processed": 0, "errors": 0, "no_speech": 0}

    for idx, entry in enumerate(tqdm(entries, desc=f"  {ds_key} VAD")):
        audio_path = entry["audio_filepath"]
        utt_id = f"utt_{idx:06d}"
        vad_path = os.path.join(output_dir, f"{utt_id}.vad")

        try:
            wav = read_audio(audio_path, sampling_rate=SILERO_SAMPLE_RATE)

            speech_timestamps = get_speech_timestamps(
                wav,
                model,
                threshold=threshold,
                sampling_rate=SILERO_SAMPLE_RATE,
                min_speech_duration_ms=50,
                min_silence_duration_ms=50,
                window_size_samples=window_samples,
                return_seconds=False,
            )

            if not speech_timestamps:
                stats["no_speech"] += 1

            # Write VAD segments
            with open(vad_path, "w") as f:
                for seg in speech_timestamps:
                    start_sec = seg["start"] / SILERO_SAMPLE_RATE
                    end_sec = seg["end"] / SILERO_SAMPLE_RATE
                    f.write(f"{start_sec:.3f}  {end_sec:.3f}  1.00\n")

            stats["processed"] += 1

        except Exception as e:
            print(f"  WARNING: Error processing {audio_path}: {e}")
            stats["errors"] += 1

        # Reset model state between utterances
        model.reset_states()

    print(f"  Processed: {stats['processed']}")
    print(f"  No speech detected: {stats['no_speech']}")
    print(f"  Errors: {stats['errors']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run Silero VAD on custom datasets")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    alignment_cfg = config.get("alignment", {})

    for ds_key, ds_config in config["datasets"].items():
        if ds_config.get("has_alignments", False):
            print(f"Skipping {ds_key} (has_alignments=true)")
            continue
        run_vad_on_dataset(ds_key, ds_config, alignment_cfg)

    print("\n=== All Silero VAD runs complete ===")


if __name__ == "__main__":
    main()
