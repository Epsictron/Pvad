"""Extract speaker embeddings from the dummy model."""

import yaml
import torch
from pathlib import Path

from model import DummySpeakerEmbeddingModel


def infer(
    cfg_path: str = "conf/config.yaml",
    ckpt_path: str = "dummy_speaker_model.pt",
) -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())["model"]

    model = DummySpeakerEmbeddingModel(
        n_mels=cfg["n_mels"],
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        num_speakers=cfg["num_speakers"],
    )
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Dummy input: 2-second utterance at ~100 frames/sec
    dummy_input = torch.randn(1, cfg["n_mels"], 200)
    embedding = model.get_embedding(dummy_input)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding (first 8 dims): {embedding[0, :8].tolist()}")


if __name__ == "__main__":
    infer()
