"""Additive Angular Margin Softmax (AAM-Softmax) loss for speaker verification."""

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class AAMSoftmax(nn.Module):
    """Additive Angular Margin Softmax loss.

    Implements the AAM-Softmax loss function which adds an angular margin
    penalty to the target logit, encouraging larger angular separations
    between speaker embeddings of different classes.

    The learnable weight matrix W is part of the module state and MUST be
    saved in checkpoints.

    Attributes:
        num_classes: Number of speaker classes.
        embedding_dim: Dimensionality of the input embeddings.
        margin: Angular margin penalty in radians.
        scale: Scaling factor applied after margin addition.
        easy_margin: Whether to use easy margin mode.
        weight: Learnable weight matrix of shape (num_classes, embedding_dim).
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.2,
        scale: float = 30.0,
        easy_margin: bool = False,
    ) -> None:
        """Initializes AAMSoftmax.

        Args:
            num_classes: Number of speaker classes.
            embedding_dim: Dimensionality of the input embeddings.
            margin: Angular margin penalty in radians.
            scale: Scaling factor applied after margin addition.
            easy_margin: Whether to use easy margin mode. If True, when
                cos(theta) > 0 use cos(theta + m), else use cos(theta).
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.scale = scale
        self.easy_margin = easy_margin

        # Learnable weight matrix (class centers)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute margin constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Threshold for numerical stability: cos(pi - m)
        self.th = math.cos(math.pi - margin)
        # mm = sin(pi - m) * m
        self.mm = math.sin(math.pi - margin) * margin

        logger.info(
            "Initialized AAMSoftmax: num_classes=%d, embedding_dim=%d, "
            "margin=%.3f, scale=%.1f, easy_margin=%s",
            num_classes,
            embedding_dim,
            margin,
            scale,
            easy_margin,
        )

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        """Computes the AAM-Softmax loss.

        Args:
            embeddings: Input embeddings of shape (batch_size, embedding_dim).
            labels: Ground truth class labels of shape (batch_size,).

        Returns:
            Scalar cross-entropy loss with angular margin applied.
        """
        # L2-normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity: (batch_size, num_classes)
        cos_theta = F.linear(embeddings_norm, weight_norm)
        # Clamp for numerical stability
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sin(theta) from cos(theta)
        sin_theta = torch.sqrt(1.0 - cos_theta.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        if self.easy_margin:
            # Easy margin: use cos(theta+m) when cos(theta) > 0, else cos(theta)
            cos_theta_m = torch.where(
                cos_theta > 0, cos_theta_m, cos_theta
            )
        else:
            # Standard margin: use cos(theta+m) when cos(theta) > th,
            # else cos(theta) - mm for monotonicity
            cos_theta_m = torch.where(
                cos_theta > self.th, cos_theta_m, cos_theta - self.mm
            )

        # Create one-hot mask for target classes
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.unsqueeze(1).long(), 1.0)

        # Apply margin only to the target class
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # Scale logits
        logits = logits * self.scale

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss
