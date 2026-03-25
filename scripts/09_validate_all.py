#!/usr/bin/env python3
"""
Script: 09_validate_all.py
Shared — Validate all generated sessions (WAV + RTTM consistency).

Checks:
  - WAV exists and is readable for every RTTM
  - RTTM has no overlapping segments
  - RTTM speaker count matches expected
  - All turn gaps are within configured bounds
  - Session durations are within tolerance

Usage:
    python scripts/09_validate_all.py --config configs/datasets.yaml
"""

import argparse
import glob
import json
import os

import soundfile as sf
import yaml
from tqdm import tqdm


def parse_rttm(rttm_path: str) -> list:
    """Parse RTTM into list of (speaker, start, duration) tuples."""
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


def check_overlap(segments: list) -> list:
    """Check for overlapping segments."""
    issues = []
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    for i in range(1, len(sorted_segs)):
        prev_end = sorted_segs[i - 1]["start"] + sorted_segs[i - 1]["duration"]
        curr_start = sorted_segs[i]["start"]
        if curr_start < prev_end - 0.001:  # 1ms tolerance
            overlap = prev_end - curr_start
            issues.append(f"overlap of {overlap:.3f}s at t={curr_start:.3f}")
    return issues


def check_turn_gaps(segments: list, turn_gap_config: dict) -> list:
    """Check that turn gaps are within configured bounds."""
    issues = []
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    for i in range(1, len(sorted_segs)):
        prev_end = sorted_segs[i - 1]["start"] + sorted_segs[i - 1]["duration"]
        curr_start = sorted_segs[i]["start"]
        gap = curr_start - prev_end
        if gap < -0.001:  # overlap (handled separately)
            continue
        min_gap = turn_gap_config.get("min", 0.0)
        max_gap = turn_gap_config.get("max", float("inf"))
        if gap < min_gap - 0.05:  # 50ms tolerance
            issues.append(f"gap too short: {gap:.3f}s at t={curr_start:.3f}")
        if gap > max_gap + 0.5:  # 500ms tolerance for edge cases
            issues.append(f"gap too long: {gap:.3f}s at t={curr_start:.3f}")
    return issues


def validate_session(wav_path: str, rttm_path: str, expected_speakers: int,
                     session_length: int, turn_gap_config: dict) -> dict:
    """Validate a single session."""
    issues = []

    # Check WAV
    try:
        info = sf.info(wav_path)
        wav_duration = info.duration
        if abs(wav_duration - session_length) > 5.0:  # 5s tolerance
            issues.append(f"duration mismatch: WAV={wav_duration:.1f}s, expected={session_length}s")
    except Exception as e:
        issues.append(f"WAV unreadable: {e}")
        return {"status": "error", "issues": issues}

    # Check RTTM
    segments = parse_rttm(rttm_path)
    if not segments:
        issues.append("empty RTTM")
        return {"status": "error", "issues": issues}

    speakers = set(s["speaker"] for s in segments)
    if len(speakers) != expected_speakers:
        issues.append(f"speaker count: found {len(speakers)}, expected {expected_speakers}")

    # Check overlaps
    overlap_issues = check_overlap(segments)
    issues.extend(overlap_issues)

    # Check turn gaps
    gap_issues = check_turn_gaps(segments, turn_gap_config)
    issues.extend(gap_issues[:5])  # Limit to 5

    status = "ok" if not issues else "warning"
    return {"status": status, "issues": issues}


def main():
    parser = argparse.ArgumentParser(description="Validate all generated sessions")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    sim_config = config["simulation"]
    turn_gap = sim_config["turn_gap"]
    session_length = sim_config["session_length"]
    total_stats = {"ok": 0, "warning": 0, "error": 0}

    for ds_key, ds_config in config["datasets"].items():
        ds_path = ds_config["path"]
        print(f"\n{'='*60}")
        print(f"Validating: {ds_config['name']} ({ds_key})")
        print(f"{'='*60}")

        for split in ["train", "val", "test"]:
            for n_spk in sim_config["speaker_counts"]:
                sim_dir = f"/data/simulated/{ds_path}/{split}_{n_spk}spk"
                if not os.path.isdir(sim_dir):
                    continue

                rttm_files = sorted(glob.glob(os.path.join(sim_dir, "*.rttm")))
                stats = {"ok": 0, "warning": 0, "error": 0, "sample_issues": []}

                for rttm_path in tqdm(rttm_files, desc=f"  {split}_{n_spk}spk"):
                    session_name = os.path.splitext(os.path.basename(rttm_path))[0]
                    wav_path = os.path.join(sim_dir, f"{session_name}.wav")

                    result = validate_session(
                        wav_path, rttm_path, n_spk, session_length, turn_gap
                    )
                    stats[result["status"]] += 1
                    total_stats[result["status"]] += 1
                    if result["issues"]:
                        stats["sample_issues"].extend(result["issues"][:2])

                total = stats["ok"] + stats["warning"] + stats["error"]
                print(f"  {split}_{n_spk}spk: {total} sessions — "
                      f"OK={stats['ok']}, WARN={stats['warning']}, ERR={stats['error']}")
                if stats["sample_issues"]:
                    for issue in stats["sample_issues"][:5]:
                        print(f"    - {issue}")

    print(f"\n{'='*60}")
    print(f"TOTAL: OK={total_stats['ok']}, WARN={total_stats['warning']}, ERR={total_stats['error']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
