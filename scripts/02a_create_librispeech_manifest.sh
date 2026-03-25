#!/bin/bash
# Script: 02a_create_librispeech_manifest.sh
# Creates alignment manifests for LibriSpeech splits using pre-computed alignments.
# PATH A — Step 2
#
# Usage: bash scripts/02a_create_librispeech_manifest.sh [--config configs/datasets.yaml]

set -euo pipefail

CONFIG="${1:-configs/datasets.yaml}"

# Parse config with Python to extract LibriSpeech dataset info
python3 - "${CONFIG}" <<'PYEOF'
import sys
import yaml
import subprocess
import os

config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)

for ds_key, ds in config["datasets"].items():
    if not ds.get("has_alignments", False):
        continue

    splits = ds.get("splits", [])
    manifest = ds["manifest"]
    alignment_path = ds["alignment_path"]
    ds_path = ds["path"]

    for split in splits:
        split_underscore = split.replace("-", "_")
        output_manifest = manifest.replace(".json", f"_aligned.json")
        ctm_output = f"/data/ctm_out/{ds_path}"

        os.makedirs(os.path.dirname(output_manifest), exist_ok=True)
        os.makedirs(ctm_output, exist_ok=True)

        input_manifest = manifest
        print(f"Processing {ds_key} split={split}")
        print(f"  Input:  {input_manifest}")
        print(f"  Output: {output_manifest}")
        print(f"  CTM:    {ctm_output}")

        cmd = [
            "python", "NeMo/scripts/speaker_tasks/create_alignment_manifest.py",
            f"--input_manifest_filepath={input_manifest}",
            f"--base_alignment_path={alignment_path}",
            f"--output_manifest_filepath={output_manifest}",
            f"--ctm_output_directory={ctm_output}",
            f"--libri_dataset_split={split}",
        ]
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"  Done: {split}")

print("=== All LibriSpeech alignment manifests created ===")
PYEOF
