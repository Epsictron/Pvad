#!/bin/bash
# Script: 02b_run_nfa.sh
# PATH B — Step 2: Run NeMo Forced Aligner on custom datasets to produce word-level CTM.
# Requires GPU.
#
# Usage: bash scripts/02b_run_nfa.sh --config configs/datasets.yaml

set -euo pipefail

CONFIG="${1:-configs/datasets.yaml}"

python3 - "${CONFIG}" <<'PYEOF'
import sys
import yaml
import subprocess
import os

config_path = sys.argv[1]
with open(config_path) as f:
    config = yaml.safe_load(f)

alignment_cfg = config.get("alignment", {})
batch_size = alignment_cfg.get("nfa_batch_size", 16)

for ds_key, ds in config["datasets"].items():
    if ds.get("has_alignments", False):
        print(f"Skipping {ds_key} (has_alignments=true)")
        continue

    ctc_model = ds.get("ctc_model")
    if not ctc_model:
        print(f"ERROR: No ctc_model specified for {ds_key}, skipping NFA")
        continue

    manifest = ds["manifest"]
    ds_path = ds["path"]
    output_dir = f"/data/nfa_output/{ds_path}"

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running NFA for: {ds['name']} ({ds_key})")
    print(f"  Model: {ctc_model}")
    print(f"  Manifest: {manifest}")
    print(f"  Output: {output_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"{'='*60}")

    cmd = [
        "python", "NeMo/tools/nemo_forced_aligner/align.py",
        f"pretrained_name={ctc_model}",
        f"manifest_filepath={manifest}",
        f"output_dir={output_dir}",
        f"batch_size={batch_size}",
    ]
    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"  Done: {ds_key}")

print("\n=== All NFA runs complete ===")
PYEOF
