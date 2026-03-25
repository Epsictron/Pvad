#!/usr/bin/env python3
"""
Script: 04b_refine_ctm_with_vad.py
PATH B — Step 4: Refine NFA CTM word boundaries using Silero VAD output.

Algorithm per utterance:
  For each word in NFA CTM:
    - Get VAD-active regions within [nfa_start - margin, nfa_end + margin]
    - If VAD active: trim to precise speech boundary
    - If no VAD activity: keep NFA boundary, flag as low-confidence

Usage:
    python scripts/04b_refine_ctm_with_vad.py --config configs/datasets.yaml
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import yaml
from tqdm import tqdm

SEARCH_MARGIN = 0.05  # 50ms search margin


def load_manifest(manifest_path: str) -> list:
    """Load NeMo-format manifest (JSONL)."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_vad_segments(vad_path: str) -> list:
    """Load VAD segments from .vad file."""
    segments = []
    if not os.path.isfile(vad_path):
        return segments
    with open(vad_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                segments.append((float(parts[0]), float(parts[1])))
    return segments


def load_ctm(ctm_path: str) -> list:
    """Load CTM file. Format: <file> <channel> <start> <duration> <word> [confidence]"""
    words = []
    if not os.path.isfile(ctm_path):
        return words
    with open(ctm_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            words.append({
                "file": parts[0],
                "channel": parts[1],
                "start": float(parts[2]),
                "duration": float(parts[3]),
                "word": parts[4],
                "confidence": float(parts[5]) if len(parts) > 5 else 1.0,
            })
    return words


def get_vad_active_in_range(vad_segments: list, start: float, end: float) -> list:
    """Get VAD-active regions that overlap with [start, end]."""
    active = []
    for seg_start, seg_end in vad_segments:
        if seg_end <= start or seg_start >= end:
            continue
        active.append((max(seg_start, start), min(seg_end, end)))
    return active


def refine_word(word: dict, vad_segments: list, min_word_duration: float) -> dict:
    """Refine a single word's boundaries using VAD."""
    nfa_start = word["start"]
    nfa_end = nfa_start + word["duration"]

    search_start = nfa_start - SEARCH_MARGIN
    search_end = nfa_end + SEARCH_MARGIN

    vad_active = get_vad_active_in_range(vad_segments, search_start, search_end)

    if vad_active:
        refined_start = max(nfa_start, vad_active[0][0])
        refined_end = min(nfa_end, vad_active[-1][1])

        # Ensure minimum duration
        if refined_end - refined_start < min_word_duration:
            refined_start = nfa_start
            refined_end = nfa_end
    else:
        refined_start = nfa_start
        refined_end = nfa_end
        word["low_confidence"] = True

    word["start"] = refined_start
    word["duration"] = max(refined_end - refined_start, min_word_duration)
    return word


def write_ctm(words: list, output_path: str):
    """Write refined CTM file."""
    with open(output_path, "w") as f:
        for w in words:
            f.write(f"{w['file']} {w['channel']} {w['start']:.4f} {w['duration']:.4f} {w['word']}")
            if "confidence" in w:
                f.write(f" {w['confidence']:.2f}")
            f.write("\n")


def process_dataset(ds_key: str, ds_config: dict, alignment_cfg: dict):
    """Refine CTM with VAD for all utterances in a dataset."""
    ds_path = ds_config["path"]
    manifest_path = ds_config["manifest"]
    nfa_dir = f"/data/nfa_output/{ds_path}/ctm/words"
    vad_dir = f"/data/vad_output/{ds_path}"
    output_dir = f"/data/refined_ctm/{ds_path}"
    min_word_duration = alignment_cfg.get("min_word_duration", 0.02)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Refining CTM for: {ds_config['name']} ({ds_key})")
    print(f"  NFA CTM dir: {nfa_dir}")
    print(f"  VAD dir: {vad_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    entries = load_manifest(manifest_path)
    stats = {"refined": 0, "low_confidence_words": 0, "total_words": 0, "errors": 0}

    for idx, entry in enumerate(tqdm(entries, desc=f"  {ds_key} refine")):
        utt_id = f"utt_{idx:06d}"

        # Find matching CTM and VAD files
        ctm_path = os.path.join(nfa_dir, f"{utt_id}.ctm")
        vad_path = os.path.join(vad_dir, f"{utt_id}.vad")
        output_path = os.path.join(output_dir, f"{utt_id}.ctm")

        # Try alternate CTM naming (NFA may use audio filename)
        if not os.path.isfile(ctm_path):
            audio_stem = Path(entry["audio_filepath"]).stem
            ctm_path = os.path.join(nfa_dir, f"{audio_stem}.ctm")

        words = load_ctm(ctm_path)
        if not words:
            stats["errors"] += 1
            continue

        vad_segments = load_vad_segments(vad_path)

        for word in words:
            refine_word(word, vad_segments, min_word_duration)
            stats["total_words"] += 1
            if word.get("low_confidence"):
                stats["low_confidence_words"] += 1

        write_ctm(words, output_path)
        stats["refined"] += 1

    print(f"  Refined: {stats['refined']} utterances")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Low-confidence words: {stats['low_confidence_words']}")
    print(f"  Errors (missing CTM): {stats['errors']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Refine CTM boundaries with Silero VAD")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    alignment_cfg = config.get("alignment", {})

    for ds_key, ds_config in config["datasets"].items():
        if ds_config.get("has_alignments", False):
            continue
        process_dataset(ds_key, ds_config, alignment_cfg)

    print("\n=== All CTM refinement complete ===")


if __name__ == "__main__":
    main()
