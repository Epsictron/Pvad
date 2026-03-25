#!/bin/bash
# run_pipeline.sh
# One-command pipeline: edit configs/datasets.yaml → run this → get WAV + PVAD labels.
#
# Usage:
#   bash run_pipeline.sh                          # train + val + test
#   bash run_pipeline.sh --split train            # train only
#   bash run_pipeline.sh --config my_config.yaml  # custom config path
#   bash run_pipeline.sh --skip-download          # skip LibriSpeech download (if already done)

set -euo pipefail

# ─── Parse args ───
CONFIG="configs/datasets.yaml"
SPLITS="train val test"
SKIP_DOWNLOAD=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)       CONFIG="$2"; shift 2 ;;
        --split)        SPLITS="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        -h|--help)
            echo "Usage: bash run_pipeline.sh [--config PATH] [--split SPLIT] [--skip-download]"
            echo ""
            echo "Options:"
            echo "  --config PATH      Path to datasets.yaml (default: configs/datasets.yaml)"
            echo "  --split SPLIT      Space-separated splits to generate (default: 'train val test')"
            echo "  --skip-download    Skip LibriSpeech download step"
            echo ""
            echo "Edit configs/datasets.yaml first, then run this script."
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "============================================================"
echo "  PVAD Data Generation Pipeline"
echo "============================================================"
echo "  Config:  ${CONFIG}"
echo "  Splits:  ${SPLITS}"
echo "  Time:    $(date)"
echo "============================================================"

# ─── Detect which paths to run ───
HAS_PATH_A=false
HAS_PATH_B=false

PATH_A_DATASETS=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
for k, v in c['datasets'].items():
    if v.get('has_alignments', False):
        print(k)
" 2>/dev/null || true)

PATH_B_DATASETS=$(python3 -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
for k, v in c['datasets'].items():
    if not v.get('has_alignments', False):
        print(k)
" 2>/dev/null || true)

if [ -n "${PATH_A_DATASETS}" ]; then HAS_PATH_A=true; fi
if [ -n "${PATH_B_DATASETS}" ]; then HAS_PATH_B=true; fi

echo ""
echo "  PATH A datasets (pre-aligned): ${PATH_A_DATASETS:-none}"
echo "  PATH B datasets (need NFA):    ${PATH_B_DATASETS:-none}"
echo ""

# ─── PATH A: LibriSpeech ───
if [ "${HAS_PATH_A}" = true ]; then
    echo "============================================================"
    echo "  PATH A: LibriSpeech (pre-aligned)"
    echo "============================================================"

    if [ "${SKIP_DOWNLOAD}" = false ]; then
        echo ""
        echo ">>> Step A1: Downloading LibriSpeech..."
        bash scripts/01a_download_librispeech.sh
    else
        echo ""
        echo ">>> Step A1: Skipped (--skip-download)"
    fi

    echo ""
    echo ">>> Step A2: Creating LibriSpeech alignment manifests..."
    bash scripts/02a_create_librispeech_manifest.sh "${CONFIG}"
fi

# ─── PATH B: Custom Datasets ───
if [ "${HAS_PATH_B}" = true ]; then
    echo ""
    echo "============================================================"
    echo "  PATH B: Custom Datasets (NFA + Silero VAD)"
    echo "============================================================"

    echo ""
    echo ">>> Step B1: Preparing NeMo manifests..."
    python3 scripts/01b_prepare_custom_manifest.py --config "${CONFIG}"

    echo ""
    echo ">>> Step B2: Running NeMo Forced Aligner (GPU)..."
    bash scripts/02b_run_nfa.sh "${CONFIG}"

    echo ""
    echo ">>> Step B3: Running Silero VAD..."
    python3 scripts/03b_run_silero_vad.py --config "${CONFIG}"

    echo ""
    echo ">>> Step B4: Refining CTM with VAD..."
    python3 scripts/04b_refine_ctm_with_vad.py --config "${CONFIG}"

    echo ""
    echo ">>> Step B5: Creating alignment manifests..."
    python3 scripts/05b_create_alignment_manifest.py --config "${CONFIG}"

    echo ""
    echo ">>> Step B6: Validating alignments..."
    python3 scripts/06b_validate_alignments.py --config "${CONFIG}" --num_samples 20
fi

# ─── Shared: Simulate + Generate Labels ───
echo ""
echo "============================================================"
echo "  Shared: Session Simulation + PVAD Label Generation"
echo "============================================================"

for SPLIT in ${SPLITS}; do
    echo ""
    echo ">>> Step 7: Generating multi-speaker sessions (split=${SPLIT})..."
    python3 scripts/07_generate_sessions.py --config "${CONFIG}" --split "${SPLIT}"

    echo ""
    echo ">>> Step 8: Converting RTTM → PVAD manifests (split=${SPLIT})..."
    python3 scripts/08_rttm_to_pvad_manifest.py --config "${CONFIG}" --split "${SPLIT}"
done

# ─── Validate + Stats ───
echo ""
echo "============================================================"
echo "  Validation & Statistics"
echo "============================================================"

echo ""
echo ">>> Step 9: Validating all sessions..."
python3 scripts/09_validate_all.py --config "${CONFIG}"

echo ""
echo ">>> Step 10: Computing statistics..."
python3 scripts/10_compute_stats.py --config "${CONFIG}"

# ─── Done ───
echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "============================================================"
echo ""
echo "  Outputs:"
echo "    Simulated sessions:  /data/simulated/<dataset>/<split>_<N>spk/"
echo "    PVAD manifests:      /data/pvad_manifests/"
echo "    Merged manifest:     /data/pvad_manifests/<split>_merged.json"
echo ""
echo "  To visualize a session:"
echo "    python scripts/11_visualize_session.py \\"
echo "      --wav /data/simulated/<dataset>/train_2spk/multispeaker_session_0.wav \\"
echo "      --rttm /data/simulated/<dataset>/train_2spk/multispeaker_session_0.rttm \\"
echo "      --output_png viz.png"
echo ""
