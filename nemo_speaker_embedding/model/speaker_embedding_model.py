"""Dummy NeMo Speaker Embedding Model."""

import torch
import torch.nn as nn


class DummyEncoder(nn.Module):
    """Simple 1D-CNN encoder that maps mel-spectrogram frames to frame-level features."""

    def __init__(self, n_mels: int = 80, hidden_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        return self.conv(x)  # (B, hidden_dim, T)


class StatisticsPooling(nn.Module):
    """Aggregate frame-level features into a single utterance-level vector."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        mean = x.mean(dim=2)
        std = x.std(dim=2)
        return torch.cat([mean, std], dim=1)  # (B, 2*C)


class DummySpeakerEmbeddingModel(nn.Module):
    """Minimal speaker-embedding model: encoder -> pooling -> linear projection.

    Produces fixed-size speaker embeddings from variable-length mel-spectrograms.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_speakers: int = 10,
    ):
        super().__init__()
        self.encoder = DummyEncoder(n_mels=n_mels, hidden_dim=hidden_dim)
        self.pool = StatisticsPooling()
        self.projector = nn.Linear(hidden_dim * 2, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            features: mel-spectrogram (B, n_mels, T)
            labels:   speaker ids     (B,) — optional, needed for training loss

        Returns:
            dict with 'embedding' and optionally 'loss'.
        """
        encoded = self.encoder(features)
        pooled = self.pool(encoded)
        embedding = self.projector(pooled)

        out: dict[str, torch.Tensor] = {"embedding": embedding}
        if labels is not None:
            logits = self.classifier(embedding)
            out["logits"] = logits
            out["loss"] = nn.functional.cross_entropy(logits, labels)
        return out

    @torch.no_grad()
    def get_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Extract normalised speaker embedding for inference."""
        emb = self.forward(features)["embedding"]
        return nn.functional.normalize(emb, dim=1)
