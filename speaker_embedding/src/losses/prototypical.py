"""Prototypical network loss for few-shot speaker verification."""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class PrototypicalLoss(nn.Module):
    """Prototypical network loss for metric learning.

    Computes prototypes from support embeddings and classifies query
    embeddings based on their distance to each prototype.

    Attributes:
        distance: Distance metric to use ('cosine' or 'euclidean').
        n_support: Number of support examples per speaker.
        temperature: Temperature scaling factor for distance logits.
    """

    def __init__(
        self,
        distance: str = "cosine",
        n_support: int = 1,
        temperature: float = 0.05,
    ) -> None:
        """Initializes PrototypicalLoss.

        Args:
            distance: Distance metric to use. One of 'cosine' or 'euclidean'.
            n_support: Number of support examples per speaker for computing
                prototypes.
            temperature: Temperature scaling factor applied to distance logits
                before softmax.

        Raises:
            ValueError: If distance is not 'cosine' or 'euclidean'.
        """
        super().__init__()
        if distance not in ("cosine", "euclidean"):
            raise ValueError(
                f"Unsupported distance metric: {distance}. "
                "Must be 'cosine' or 'euclidean'."
            )
        self.distance = distance
        self.n_support = n_support
        self.temperature = temperature

        logger.info(
            "Initialized PrototypicalLoss: distance=%s, n_support=%d, "
            "temperature=%.4f",
            distance,
            n_support,
            temperature,
        )

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Computes the prototypical loss.

        Groups embeddings by speaker label, splits each group into support
        and query sets, computes prototypes from support embeddings, and
        returns cross-entropy loss over distance-based logits.

        Args:
            embeddings: Input embeddings of shape (batch_size, embedding_dim).
            labels: Speaker labels of shape (batch_size,).

        Returns:
            Scalar cross-entropy loss computed over query-to-prototype
            distance logits.
        """
        unique_labels = torch.unique(labels)
        prototypes = []
        query_embeddings = []
        query_labels = []
        valid_class_idx = 0

        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]

            # Skip speakers with too few utterances for support + at least 1 query
            if class_embeddings.size(0) <= self.n_support:
                logger.debug(
                    "Skipping speaker %s: only %d utterances "
                    "(need > %d for support + query).",
                    label.item(),
                    class_embeddings.size(0),
                    self.n_support,
                )
                continue

            # Split into support and query
            support = class_embeddings[: self.n_support]
            query = class_embeddings[self.n_support :]

            # Compute prototype as mean of support embeddings
            prototype = support.mean(dim=0)
            prototypes.append(prototype)

            query_embeddings.append(query)
            query_labels.append(
                torch.full(
                    (query.size(0),),
                    valid_class_idx,
                    dtype=torch.long,
                    device=embeddings.device,
                )
            )
            valid_class_idx += 1

        if len(prototypes) == 0:
            logger.warning(
                "No valid speakers found for prototypical loss computation. "
                "Returning zero loss."
            )
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Stack prototypes: (num_valid_classes, embedding_dim)
        prototypes = torch.stack(prototypes)
        # Concatenate queries: (total_queries, embedding_dim)
        query_embeddings = torch.cat(query_embeddings, dim=0)
        # Concatenate query labels: (total_queries,)
        query_labels = torch.cat(query_labels, dim=0)

        # Compute distance logits
        logits = self._compute_logits(query_embeddings, prototypes)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, query_labels)

        return loss

    def _compute_logits(
        self, query_embeddings: Tensor, prototypes: Tensor
    ) -> Tensor:
        """Computes distance-based logits from queries to prototypes.

        Args:
            query_embeddings: Query embeddings of shape (num_queries, embedding_dim).
            prototypes: Prototype embeddings of shape (num_classes, embedding_dim).

        Returns:
            Logits of shape (num_queries, num_classes).
        """
        if self.distance == "cosine":
            # Cosine similarity / temperature
            query_norm = F.normalize(query_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            logits = torch.mm(query_norm, proto_norm.t()) / self.temperature
        elif self.distance == "euclidean":
            # Negative squared euclidean distance / temperature
            # ||q - p||^2 = ||q||^2 + ||p||^2 - 2*q.p
            dists = torch.cdist(query_embeddings, prototypes, p=2).pow(2)
            logits = -dists / self.temperature

        return logits
