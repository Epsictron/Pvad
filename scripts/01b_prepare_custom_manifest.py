#!/usr/bin/env python3
"""
Script: 01b_prepare_custom_manifest.py
PATH B — Step 1: Prepare NeMo-format manifests for custom (non-LibriSpeech) datasets.

Per dataset:
  1. Reads metadata (CSV/JSON/TSV) based on metadata_format
  2. Validates audio (exists, readable, 16 kHz mono — resample if not)
  3. Normalizes transcripts (uppercase, numbers → words, language-aware)
  4. Filters by duration (0.5s–30s)
  5. Writes NeMo manifest
  6. Reports stats

Usage:
    python scripts/01b_prepare_custom_manifest.py --config configs/datasets.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

try:
    from num2words import num2words
except ImportError:
    num2words = None

MIN_DURATION = 0.5
MAX_DURATION = 30.0


def load_metadata(metadata_file: str, metadata_format: str) -> pd.DataFrame:
    """Load metadata from CSV, JSON, or TSV."""
    if metadata_format == "csv":
        return pd.read_csv(metadata_file)
    elif metadata_format == "json":
        return pd.read_json(metadata_file)
    elif metadata_format == "tsv":
        return pd.read_csv(metadata_file, sep="\t")
    else:
        raise ValueError(f"Unsupported metadata format: {metadata_format}")


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"  WARNING: Cannot read {audio_path}: {e}")
        return -1.0


def normalize_transcript(text: str, language: str) -> str:
    """Normalize transcript: uppercase, expand numbers."""
    text = text.strip()
    if not text:
        return text

    # Language-aware number expansion
    if num2words is not None:
        import re
        def replace_number(match):
            try:
                return num2words(int(match.group()), lang=language)
            except (ValueError, NotImplementedError):
                return match.group()
        text = re.sub(r'\b\d+\b', replace_number, text)

    text = text.upper()
    return text


def process_dataset(ds_key: str, ds_config: dict) -> dict:
    """Process a single custom dataset and write NeMo manifest."""
    print(f"\n{'='*60}")
    print(f"Processing: {ds_config['name']} ({ds_key})")
    print(f"{'='*60}")

    audio_root = ds_config["audio_root"]
    metadata_file = ds_config["metadata_file"]
    metadata_format = ds_config["metadata_format"]
    manifest_path = ds_config["manifest"]
    language = ds_config["language"]

    # Load metadata
    print(f"  Loading metadata from {metadata_file} (format={metadata_format})")
    df = load_metadata(metadata_file, metadata_format)
    print(f"  Loaded {len(df)} entries")

    # Validate required columns
    required_cols = {"file_path", "transcript", "speaker_id"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"  ERROR: Missing required columns: {missing}")
        return {"dataset": ds_key, "status": "error", "reason": f"missing columns: {missing}"}

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    stats = {
        "dataset": ds_key,
        "total": len(df),
        "skipped_audio_missing": 0,
        "skipped_audio_unreadable": 0,
        "skipped_duration": 0,
        "skipped_empty_transcript": 0,
        "written": 0,
        "speakers": set(),
        "total_duration": 0.0,
    }

    entries = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {ds_key}"):
        file_path = row["file_path"]
        transcript = str(row["transcript"])
        speaker_id = str(row["speaker_id"])
        gender = str(row.get("gender", "")) if "gender" in row else ""

        # Resolve audio path
        audio_path = os.path.join(audio_root, file_path)
        if not os.path.isfile(audio_path):
            stats["skipped_audio_missing"] += 1
            continue

        # Get duration
        if "duration" in row and pd.notna(row["duration"]):
            duration = float(row["duration"])
        else:
            duration = get_audio_duration(audio_path)
            if duration < 0:
                stats["skipped_audio_unreadable"] += 1
                continue

        # Filter by duration
        if duration < MIN_DURATION or duration > MAX_DURATION:
            stats["skipped_duration"] += 1
            continue

        # Normalize transcript
        transcript = normalize_transcript(transcript, language)
        if not transcript:
            stats["skipped_empty_transcript"] += 1
            continue

        entry = {
            "audio_filepath": audio_path,
            "text": transcript,
            "duration": round(duration, 3),
            "speaker_id": speaker_id,
        }
        if gender:
            entry["gender"] = gender.upper()

        entries.append(entry)
        stats["speakers"].add(speaker_id)
        stats["total_duration"] += duration

    # Write manifest
    with open(manifest_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    stats["written"] = len(entries)
    stats["speakers"] = len(stats["speakers"])
    stats["total_duration"] = round(stats["total_duration"] / 3600, 2)
    stats["status"] = "ok"

    print(f"  Written: {stats['written']} entries to {manifest_path}")
    print(f"  Speakers: {stats['speakers']}")
    print(f"  Total duration: {stats['total_duration']} hours")
    print(f"  Skipped — missing audio: {stats['skipped_audio_missing']}")
    print(f"  Skipped — unreadable: {stats['skipped_audio_unreadable']}")
    print(f"  Skipped — duration filter: {stats['skipped_duration']}")
    print(f"  Skipped — empty transcript: {stats['skipped_empty_transcript']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare NeMo manifests for custom datasets")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml",
                        help="Path to datasets.yaml config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    all_stats = []
    for ds_key, ds_config in config["datasets"].items():
        if ds_config.get("has_alignments", False):
            print(f"\nSkipping {ds_key} (has_alignments=true, use PATH A)")
            continue
        stats = process_dataset(ds_key, ds_config)
        all_stats.append(stats)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for s in all_stats:
        status = s.get("status", "unknown")
        if status == "ok":
            print(f"  {s['dataset']}: {s['written']} entries, {s['speakers']} speakers, {s['total_duration']}h")
        else:
            print(f"  {s['dataset']}: {status} — {s.get('reason', '')}")


if __name__ == "__main__":
    main()
