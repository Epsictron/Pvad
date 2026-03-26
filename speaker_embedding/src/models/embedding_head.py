"""Embedding projection head with optional batch normalization and L2 normalization."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EmbeddingHead(nn.Module):
    """Linear projection from pooled features to a fixed-size speaker embedding.

    The head applies a linear layer, optional batch normalization, and
    L2 normalization to produce unit-length embeddings suitable for
    cosine-similarity scoring.

    Args:
        input_dim: Dimensionality of the pooled input (typically 2*D from
            a statistics pooling layer).
        embedding_dim: Desired embedding dimensionality.
        batch_norm: Whether to apply ``BatchNorm1d`` after the linear
            projection.  Defaults to ``True``.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, embedding_dim)
        self.bn: nn.Module | None = nn.BatchNorm1d(embedding_dim) if batch_norm else None
        logger.info(
            "EmbeddingHead: input_dim=%d, embedding_dim=%d, batch_norm=%s",
            input_dim,
            embedding_dim,
            batch_norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize pooled features to a speaker embedding.

        Args:
            x: Pooled representation of shape (B, input_dim).

        Returns:
            L2-normalized embedding of shape (B, embedding_dim).
        """
        out = self.linear(x)  # (B, embedding_dim)
        if self.bn is not None:
            out = self.bn(out)  # (B, embedding_dim)
        out = F.normalize(out, p=2, dim=1)  # L2 normalization
        logger.debug("EmbeddingHead output shape: %s", out.shape)
        return out
