"""Normalization modules for acoustic features.

Provides per-utterance cepstral mean and variance normalization (CMVN)
and instance normalization wrappers suitable for GPU-based feature
pipelines.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class CMVN(nn.Module):
    """Cepstral Mean and Variance Normalization.

    Performs per-utterance mean subtraction and variance normalization
    along the time axis.  Handles the edge case where an utterance has
    zero variance (constant features) by leaving those dimensions
    unchanged rather than producing NaN.

    The module expects features shaped ``(B, C, T)`` where *C* is the
    number of feature channels and *T* is the time dimension.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialise CMVN.

        Args:
            eps: Small constant added to the standard deviation to
                avoid division by zero.
        """
        super().__init__()
        self.eps = eps

    def forward(self, features: Tensor) -> Tensor:
        """Apply per-utterance CMVN.

        Args:
            features: Input tensor of shape ``(B, C, T)``.

        Returns:
            Normalised tensor of the same shape.
        """
        # Compute statistics along the time axis (dim=-1), keep dims
        # for broadcasting.
        mean = features.mean(dim=-1, keepdim=True)
        std = features.std(dim=-1, keepdim=True)

        # Guard against zero-variance channels.
        std = std.clamp(min=self.eps)

        normalized = (features - mean) / std
        return normalized


class InstanceNorm(nn.Module):
    """Instance normalization wrapper for acoustic features.

    Thin wrapper around :class:`torch.nn.InstanceNorm1d` that accepts
    tensors shaped ``(B, C, T)`` and normalises each channel
    independently per instance.

    Args:
        num_features: Number of feature channels (*C*).
        eps: Epsilon for numerical stability.
        affine: If ``True``, learnable affine parameters are added.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
    ) -> None:
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(
            num_features=num_features,
            eps=eps,
            affine=affine,
        )
        logger.debug(
            "InstanceNorm created with num_features=%d, affine=%s",
            num_features,
            affine,
        )

    def forward(self, features: Tensor) -> Tensor:
        """Apply instance normalization.

        Args:
            features: Input tensor of shape ``(B, C, T)``.

        Returns:
            Normalised tensor of the same shape.
        """
        return self.instance_norm(features)
