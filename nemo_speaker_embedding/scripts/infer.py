"""Extract speaker embeddings and verify speakers using a trained NeMo model.

Usage:
    # Single file — print embedding:
    python scripts/infer.py --nemo_model experiments/speaker_model.nemo --audio file1.wav

    # Two files — speaker verification (cosine similarity):
    python scripts/infer.py --nemo_model experiments/speaker_model.nemo --audio file1.wav file2.wav

    # Batch — extract embeddings for a manifest:
    python scripts/infer.py --nemo_model experiments/speaker_model.nemo --manifest eval.json --output embeddings.pt
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo.collections.asr.models import EncDecSpeakerLabelModel


def extract_single(model, audio_path: str) -> torch.Tensor:
    """Extract L2-normalized speaker embedding from a single audio file."""
    emb = model.get_embedding(audio_path)
    return F.normalize(emb, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_model", required=True, help="Path to .nemo checkpoint")
    parser.add_argument("--audio", nargs="*", help="WAV file(s) for verification")
    parser.add_argument("--manifest", help="NeMo manifest for batch extraction")
    parser.add_argument("--output", default="embeddings.pt", help="Output file for batch mode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Verification threshold")
    args = parser.parse_args()

    model = EncDecSpeakerLabelModel.restore_from(args.nemo_model)
    model.eval()
    model.freeze()

    # ── Single / pair mode ───────────────────────────────────────────
    if args.audio:
        embeddings = []
        for path in args.audio:
            emb = extract_single(model, path)
            print(f"{path}: shape={emb.shape}, norm={emb.norm().item():.4f}")
            embeddings.append(emb)

        if len(embeddings) == 2:
            sim = F.cosine_similarity(embeddings[0], embeddings[1]).item()
            print(f"\nCosine similarity: {sim:.4f}")
            print(f"Same speaker (threshold={args.threshold}): {sim > args.threshold}")

    # ── Batch manifest mode ──────────────────────────────────────────
    elif args.manifest:
        results = {}
        with open(args.manifest) as f:
            entries = [json.loads(line) for line in f]

        for entry in entries:
            emb = extract_single(model, entry["audio_filepath"])
            results[entry["audio_filepath"]] = {
                "embedding": emb.cpu(),
                "label": entry.get("label", "unknown"),
            }

        torch.save(results, args.output)
        print(f"Saved {len(results)} embeddings → {args.output}")

    else:
        parser.error("Provide --audio or --manifest")


if __name__ == "__main__":
    main()
