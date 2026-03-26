"""Dummy encoder model for development and testing.

# PLACEHOLDER: Replace with your production model
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .base import BaseEncoder

logger = logging.getLogger(__name__)

# PLACEHOLDER: Replace with your production model
_FEATURE_DIM_MAP = {
    "fbank": "n_mels",
    "mfcc": "n_mfcc",
    "spectrogram": "n_fft",
    "raw": None,
}


def _resolve_input_channels(config: DictConfig) -> int:
    """Resolve the number of input channels from the feature config.

    Args:
        config: Full Hydra/OmegaConf config containing a ``features`` group.

    Returns:
        Number of input channels for the first convolutional layer.
    """
    # PLACEHOLDER: Replace with your production model
    feat_type: str = config.features.type
    if feat_type == "raw":
        return 1
    if feat_type == "spectrogram":
        return config.features.n_fft // 2 + 1
    key = _FEATURE_DIM_MAP.get(feat_type)
    if key is None:
        raise ValueError(f"Unknown feature type: {feat_type}")
    return int(getattr(config.features, key))


class DummyEncoder(BaseEncoder):
    """Simple 3-layer 1D-CNN + GRU encoder.

    # PLACEHOLDER: Replace with your production model

    This is a lightweight encoder intended for quick iteration and
    integration testing.  It should **not** be used for production
    speaker verification.

    Args:
        config: Hydra/OmegaConf ``DictConfig`` with at least a
            ``features`` group so that input channels can be inferred.
    """

    _HIDDEN: int = 256  # PLACEHOLDER: Replace with your production model

    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        # PLACEHOLDER: Replace with your production model
        input_channels: int = _resolve_input_channels(config)
        logger.info(
            "DummyEncoder: input_channels=%d, hidden=%d (PLACEHOLDER)",
            input_channels,
            self._HIDDEN,
        )

        # --- 3 x (Conv1d + BatchNorm + ReLU) ---
        # PLACEHOLDER: Replace with your production model
        self.conv1 = nn.Conv1d(input_channels, self._HIDDEN, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(self._HIDDEN)

        self.conv2 = nn.Conv1d(self._HIDDEN, self._HIDDEN, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(self._HIDDEN)

        self.conv3 = nn.Conv1d(self._HIDDEN, self._HIDDEN, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(self._HIDDEN)

        self.relu = nn.ReLU(inplace=True)

        # --- GRU ---
        # PLACEHOLDER: Replace with your production model
        self.gru = nn.GRU(
            input_size=self._HIDDEN,
            hidden_size=self._HIDDEN,
            num_layers=1,
            batch_first=True,
        )

    # ------------------------------------------------------------------ #
    # BaseEncoder interface
    # ------------------------------------------------------------------ #

    @property
    def output_dim(self) -> int:
        """Return the output channel dimension D.

        Returns:
            256 (the hidden size used throughout this placeholder model).
        """
        # PLACEHOLDER: Replace with your production model
        return self._HIDDEN

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input features through CNN layers followed by a GRU.

        # PLACEHOLDER: Replace with your production model

        Args:
            features: Input features of shape (B, C, T).
            lengths: Valid lengths for each sample in the batch (B,).

        Returns:
            Tuple of (encoded, lengths) where encoded has shape
            (B, 256, T) and lengths is unchanged.
        """
        # PLACEHOLDER: Replace with your production model
        x = self.relu(self.bn1(self.conv1(features)))  # (B, H, T)
        x = self.relu(self.bn2(self.conv2(x)))          # (B, H, T)
        x = self.relu(self.bn3(self.conv3(x)))          # (B, H, T)

        # GRU expects (B, T, H)
        x = x.transpose(1, 2)                           # (B, T, H)
        x, _ = self.gru(x)                              # (B, T, H)
        x = x.transpose(1, 2)                           # (B, H, T)

        return x, lengths
