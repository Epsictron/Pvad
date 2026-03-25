#!/usr/bin/env python3
"""
Script: 05b_create_alignment_manifest.py
PATH B — Step 5: Merge NeMo manifest with refined CTM paths to create alignment manifests.

Usage:
    python scripts/05b_create_alignment_manifest.py --config configs/datasets.yaml
"""

import argparse
import json
import os

import yaml
from tqdm import tqdm


def load_manifest(manifest_path: str) -> list:
    """Load NeMo-format manifest (JSONL)."""
    entries = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def process_dataset(ds_key: str, ds_config: dict):
    """Create alignment manifest for a dataset by adding ctm_filepath."""
    ds_path = ds_config["path"]
    manifest_path = ds_config["manifest"]
    ctm_dir = f"/data/refined_ctm/{ds_path}"
    output_path = manifest_path.replace(".json", "_aligned.json")

    print(f"\n{'='*60}")
    print(f"Creating alignment manifest for: {ds_config['name']} ({ds_key})")
    print(f"  Input manifest: {manifest_path}")
    print(f"  CTM dir: {ctm_dir}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    entries = load_manifest(manifest_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    stats = {"total": len(entries), "with_ctm": 0, "missing_ctm": 0}

    with open(output_path, "w") as f:
        for idx, entry in enumerate(tqdm(entries, desc=f"  {ds_key}")):
            utt_id = f"utt_{idx:06d}"
            ctm_path = os.path.join(ctm_dir, f"{utt_id}.ctm")

            if os.path.isfile(ctm_path):
                entry["ctm_filepath"] = ctm_path
                stats["with_ctm"] += 1
            else:
                stats["missing_ctm"] += 1
                continue  # Skip entries without CTM

            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Total entries: {stats['total']}")
    print(f"  With CTM: {stats['with_ctm']}")
    print(f"  Missing CTM (skipped): {stats['missing_ctm']}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create alignment manifests for custom datasets")
    parser.add_argument("--config", type=str, default="configs/datasets.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    for ds_key, ds_config in config["datasets"].items():
        if ds_config.get("has_alignments", False):
            continue
        process_dataset(ds_key, ds_config)

    print("\n=== All alignment manifests created ===")


if __name__ == "__main__":
    main()
