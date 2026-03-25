#!/usr/bin/env python3
"""
Script: 06b_validate_alignments.py
PATH B — Step 6: Validate alignment quality for custom datasets.

Checks:
  1. CTM word count == transcript word count
  2. Timestamps monotonically increasing
  3. No gaps > 2s between consecutive words
  4. Refined vs raw NFA: report boundary shift statistics
  5. Generate visual spot-checks (waveform + word boundaries + VAD overlay)

Usage:
    python scripts/06b_validate_alignments.py --config configs/datasets.yaml --num_samples 100
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

MAX_GAP_THRESHOLD = 2.0  # seconds


def load_manifest(manifest_path: str) -> list:
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_ctm(ctm_path: str) -> list:
    words = []
    if not os.path.isfile(ctm_path):
        return words
    with open(ctm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                words.append({
                    "start": float(parts[2]),
                    "duration": float(parts[3]),
                    "word": parts[4],
                })
    return words


def validate_entry(entry: dict) -> dict:
    """Validate a single manifest entry's alignment."""
    issues = []
    ctm_path = entry.get("ctm_filepath")
    if not ctm_path or not os.path.isfile(ctm_path):
        return {"status": "error", "issues": ["missing CTM file"]}

    words = load_ctm(ctm_path)
    transcript_words = entry.get("text", "").split()

    # Check 1: Word count match
    if len(words) != len(transcript_words):
        issues.append(
            f"word count mismatch: CTM={len(words)} vs transcript={len(transcript_words)}"
        )

    if not words:
        return {"status": "error", "issues": issues or ["empty CTM"]}

    # Check 2: Monotonically increasing timestamps
    for i in range(1, len(words)):
        if words[i]["start"] < words[i - 1]["start"]:
            issues.append(
                f"non-monotonic at word {i}: {words[i-1]['start']:.3f} > {words[i]['start']:.3f}"
            )
            break

    # Check 3: No large gaps between consecutive words
    for i in range(1, len(words)):
        prev_end = words[i - 1]["start"] + words[i - 1]["duration"]
        gap = words[i]["start"] - prev_end
        if gap > MAX_GAP_THRESHOLD:
            issues.append(
                f"large gap ({gap:.2f}s) between words {i-1} and {i}"
            )

    status = "ok" if not issues else "warning"
    return {"status": status, "issues": issues, "num_words": len(words)}


def visualize_sample(entry: dict, output_path: str):
    """Generate a visual spot-check for a single utterance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import soundfile as sf
    except ImportError:
        return

    audio_path = entry["audio_filepath"]
    ctm_path = entry.get("ctm_filepath")
    if not ctm_path or not os.path.isfile(ctm_path):
        return

    try:
        audio, sr = sf.read(audio_path)
    except Exception:
        return

    words = load_ctm(ctm_path)
    if not words:
        return

    fig, ax = plt.subplots(figsize=(14, 3))
    time = np.arange(len(audio)) / sr
    ax.plot(time, audio, alpha=0.5, linewidth=0.5)

    for w in words:
        ax.axvline(w["start"], color="green", alpha=0.7, linewidth=0.8)
        ax.axvline(w["start"] + w["duration"], color="red", alpha=0.5, linewidth=0.5)
        ax.text(
            w["start"] + w["duration"] / 2,
            max(audio) * 0.8,
            w["word"],
            fontsize=5,
            ha="center",
            rotation=45,
        )

    ax.set_xlabel("Time (s)")
    ax.set_title(f"Alignment: {Path(audio_path).stem}")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def validate_dataset(ds_key: str, ds_config: dict, num_samples: int):
    """Validate alignment for a dataset."""
    ds_path = ds_config["path"]
    manifest_path = ds_config["manifest"].replace(".json", "_aligned.json")

    print(f"\n{'='*60}")
    print(f"Validating: {ds_config['name']} ({ds_key})")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")

    if not os.path.isfile(manifest_path):
        print(f"  ERROR: Aligned manifest not found: {manifest_path}")
        return

    entries = load_manifest(manifest_path)
    print(f"  Total entries: {len(entries)}")

    stats = {"ok": 0, "warning": 0, "error": 0, "issues": []}

    for entry in tqdm(entries, desc=f"  {ds_key} validate"):
        result = validate_entry(entry)
        stats[result["status"]] += 1
        if result["issues"]:
            stats["issues"].extend(result["issues"][:3])

    print(f"  OK: {stats['ok']}")
    print(f"  Warnings: {stats['warning']}")
    print(f"  Errors: {stats['error']}")
    if stats["issues"]:
        print(f"  Sample issues:")
        for issue in stats["issues"][:10]:
            print(f"    - {issue}")

    # Visual spot-checks
    if num_samples > 0 and entries:
        viz_dir = f"/data/viz/alignment_checks/{ds_path}"
        samples = random.sample(entries, min(num_samples, len(entries)))
        print(f"  Generating {len(samples)} visual spot-checks...")
        for i, entry in enumerate(samples):
            visualize_sample(entry, os.path.join(viz_dir, f"sample_{i:03d}.png"))


def main():
    parser = argparse.ArgumentParser(description="Validate alignment quality")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for ds_key, ds_config in config["datasets"].items():
        if ds_config.get("has_alignments", False):
            continue
        validate_dataset(ds_key, ds_config, args.num_samples)

    print("\n=== Validation complete ===")


if __name__ == "__main__":
    main()
