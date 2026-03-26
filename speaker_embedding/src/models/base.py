"""Abstract base class for speaker encoder models."""

import abc
import logging
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseEncoder(nn.Module, abc.ABC):
    """Abstract base class for speaker encoder models.

    All encoder models must inherit from this class and implement
    the forward method that maps input features to frame-level
    representations.
    """

    @abc.abstractmethod
    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input features to frame-level representations.

        Args:
            features: Input features of shape (B, C, T).
            lengths: Valid lengths for each sample in the batch (B,).

        Returns:
            Tuple of (encoded, lengths) where encoded has shape (B, D, T')
            and lengths has shape (B,).
        """
        ...

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Return the output channel dimension D."""
        ...
