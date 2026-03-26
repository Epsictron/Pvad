#!/usr/bin/env python
"""Validate a speaker manifest file."""

import argparse
import logging
import sys

from speaker_embedding.src.data.manifest import load_manifest, validate_manifest, get_manifest_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a speaker manifest file.")
    parser.add_argument("manifest_path", type=str, help="Path to JSONL manifest file.")
    parser.add_argument("--min_duration", type=float, default=0.0, help="Minimum duration filter (seconds).")
    parser.add_argument("--max_duration", type=float, default=float("inf"), help="Maximum duration filter (seconds).")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("Loading manifest: %s", args.manifest_path)
    try:
        entries = load_manifest(
            args.manifest_path,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        )
    except (FileNotFoundError, KeyError) as exc:
        logger.error("Failed to load manifest: %s", exc)
        sys.exit(1)

    # Validate entries
    is_valid = validate_manifest(entries)

    # Print statistics
    stats = get_manifest_stats(entries)
    print("\n=== Manifest Statistics ===")
    print(f"  File:             {args.manifest_path}")
    print(f"  Total utterances: {stats['num_utterances']}")
    print(f"  Total speakers:   {stats['num_speakers']}")
    print(f"  Duration (min):   {stats['duration_stats']['min']:.2f}s")
    print(f"  Duration (max):   {stats['duration_stats']['max']:.2f}s")
    print(f"  Duration (mean):  {stats['duration_stats']['mean']:.2f}s")
    print(f"  Duration (total): {stats['duration_stats']['total']:.2f}s ({stats['duration_stats']['total'] / 3600:.2f}h)")
    print(f"  Gender dist:      {stats['gender_distribution']}")
    print(f"  Valid:            {is_valid}")

    if args.min_duration > 0.0 or args.max_duration < float("inf"):
        print(f"\n  Duration filter:  [{args.min_duration:.2f}s, {args.max_duration:.2f}s]")

    if not is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()
