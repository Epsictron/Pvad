"""Extract speaker embeddings and compute similarity using a trained NeMo model.

Usage:
    python scripts/infer.py --nemo_model path/to/model.nemo --audio file1.wav file2.wav

If two files are given, prints cosine similarity (speaker verification).
"""

import argparse

import torch
from nemo.collections.asr.models import EncDecSpeakerLabelModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_model", required=True, help=".nemo checkpoint path")
    parser.add_argument("--audio", nargs="+", required=True, help="WAV file(s)")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = EncDecSpeakerLabelModel.restore_from(args.nemo_model)
    model.eval()
    model.freeze()

    embeddings = []
    for path in args.audio:
        emb = model.get_embedding(path)
        emb = torch.nn.functional.normalize(emb, dim=1)
        embeddings.append(emb)
        print(f"{path}: embedding shape {emb.shape}")

    if len(embeddings) == 2:
        sim = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1])
        same = sim.item() > args.threshold
        print(f"\nCosine similarity: {sim.item():.4f}")
        print(f"Same speaker (threshold={args.threshold}): {same}")


if __name__ == "__main__":
    main()
