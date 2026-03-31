"""Minimal training loop for the dummy speaker embedding model."""

import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from model import DummySpeakerEmbeddingModel
from data.dataset import DummySpeakerDataset, collate_fn


def train(cfg_path: str = "conf/config.yaml") -> None:
    cfg = yaml.safe_load(Path(cfg_path).read_text())

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DummySpeakerEmbeddingModel(
        n_mels=model_cfg["n_mels"],
        hidden_dim=model_cfg["hidden_dim"],
        embedding_dim=model_cfg["embedding_dim"],
        num_speakers=model_cfg["num_speakers"],
    ).to(device)

    dataset = DummySpeakerDataset(
        num_samples=data_cfg["num_samples"],
        num_speakers=model_cfg["num_speakers"],
        n_mels=model_cfg["n_mels"],
        max_frames=data_cfg["max_frames"],
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"])

    model.train()
    for epoch in range(1, train_cfg["epochs"] + 1):
        total_loss = 0.0
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)

            out = model(features, labels)
            loss = out["loss"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{train_cfg['epochs']}  loss={avg_loss:.4f}")

    ckpt_path = "dummy_speaker_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    train()
