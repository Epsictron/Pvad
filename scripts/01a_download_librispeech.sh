#!/bin/bash
# Script: 01a_download_librispeech.sh
# Downloads LibriSpeech audio data and pre-computed word alignments.
# PATH A — Step 1
#
# Usage: bash scripts/01a_download_librispeech.sh [--data_root /data/LibriSpeech]

set -euo pipefail

DATA_ROOT="${1:-/data/LibriSpeech}"
ALIGNMENT_DIR="${2:-/data/LibriSpeech_Alignments}"

echo "=== Downloading LibriSpeech ==="
echo "Data root: ${DATA_ROOT}"

python NeMo/scripts/dataset_processing/get_librispeech_data.py \
  --data_root "${DATA_ROOT}" \
  --data_sets "train_clean_100" "train_clean_360" "dev_clean"

echo "=== Downloading LibriSpeech Alignments ==="
if [ -d "${ALIGNMENT_DIR}" ]; then
  echo "Alignments directory already exists at ${ALIGNMENT_DIR}, skipping clone."
else
  git clone https://github.com/CorentinJ/librispeech-alignments.git \
    "${ALIGNMENT_DIR}"
fi

echo "=== Done ==="
echo "Audio at: ${DATA_ROOT}"
echo "Alignments at: ${ALIGNMENT_DIR}"
