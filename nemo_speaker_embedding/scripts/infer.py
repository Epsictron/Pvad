"""Inference — extract embeddings / verify speakers."""

import argparse
import torch
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecSpeakerLabelModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nemo_model", required=True)
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model = EncDecSpeakerLabelModel.restore_from(args.nemo_model)
    model.eval()
    model.freeze()

    embs = [F.normalize(model.get_embedding(f), dim=1) for f in args.audio]
    for f, e in zip(args.audio, embs):
        print(f"{f}: {e.shape}")

    if len(embs) == 2:
        sim = F.cosine_similarity(embs[0], embs[1]).item()
        print(f"Similarity: {sim:.4f} | Same speaker: {sim > args.threshold}")


if __name__ == "__main__":
    main()
