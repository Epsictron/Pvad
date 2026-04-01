"""Extract embeddings / verify speakers."""

import argparse
import torch.nn.functional as F
from nemo.collections.asr.models import EncDecSpeakerLabelModel

parser = argparse.ArgumentParser()
parser.add_argument("--nemo_model", required=True)
parser.add_argument("--audio", nargs="+", required=True)
args = parser.parse_args()

model = EncDecSpeakerLabelModel.restore_from(args.nemo_model)
model.eval()
model.freeze()

embs = [F.normalize(model.get_embedding(f), dim=1) for f in args.audio]
for f, e in zip(args.audio, embs):
    print(f"{f}: {e.shape}")

if len(embs) == 2:
    print(f"Cosine similarity: {F.cosine_similarity(embs[0], embs[1]).item():.4f}")
