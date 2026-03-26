"""Temporal pooling layers for speaker embedding extraction."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _make_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Create a boolean mask from sequence lengths.

    Args:
        lengths: Integer tensor of shape (B,) with valid lengths.
        max_len: Maximum sequence length (T).

    Returns:
        Boolean tensor of shape (B, 1, T) where ``True`` indicates a
        valid (non-padded) time step.
    """
    # (B, T)
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask.unsqueeze(1)  # (B, 1, T)


class StatisticsPooling(nn.Module):
    """Mean + standard-deviation pooling over the time dimension.

    Given frame-level representations of shape (B, D, T) and corresponding
    ``lengths``, this module computes the mean and standard deviation
    across time for each sample and returns their concatenation of shape
    (B, 2*D).
    """

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute statistics pooling.

        Args:
            x: Frame-level features of shape (B, D, T).
            lengths: Valid lengths for each sample (B,).

        Returns:
            Pooled representation of shape (B, 2*D).
        """
        mask = _make_length_mask(lengths, x.size(2))  # (B, 1, T)
        # Mask padded positions to zero so they don't contribute.
        x_masked = x * mask.float()

        # Number of valid frames per sample, clamped to avoid division by zero.
        count = lengths.float().clamp(min=1.0).unsqueeze(1)  # (B, 1)

        mean = x_masked.sum(dim=2) / count  # (B, D)

        # Variance = E[x^2] - E[x]^2, then take sqrt for std.
        mean_sq = (x_masked ** 2).sum(dim=2) / count  # (B, D)
        var = (mean_sq - mean ** 2).clamp(min=1e-8)
        std = torch.sqrt(var)  # (B, D)

        pooled = torch.cat([mean, std], dim=1)  # (B, 2*D)
        logger.debug("StatisticsPooling output shape: %s", pooled.shape)
        return pooled


class AttentiveStatisticsPooling(nn.Module):
    """Attention-weighted mean + standard-deviation pooling.

    An attention mechanism assigns a scalar weight to each time step
    before computing weighted statistics.  This allows the model to
    focus on the most speaker-discriminative frames.

    Args:
        input_dim: Number of channels D in the input (B, D, T).
        attention_channels: Hidden dimension of the attention network.
    """

    def __init__(self, input_dim: int, attention_channels: int = 128) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(attention_channels, 1, kernel_size=1),
        )
        logger.info(
            "AttentiveStatisticsPooling: input_dim=%d, attention_channels=%d",
            input_dim,
            attention_channels,
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Compute attentive statistics pooling.

        Args:
            x: Frame-level features of shape (B, D, T).
            lengths: Valid lengths for each sample (B,).

        Returns:
            Pooled representation of shape (B, 2*D).
        """
        # Attention scores (B, 1, T).
        scores = self.attention(x)  # (B, 1, T)

        # Mask padded positions with -inf before softmax.
        mask = _make_length_mask(lengths, x.size(2))  # (B, 1, T)
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=2)  # (B, 1, T)

        # Weighted mean.
        weighted = x * weights  # (B, D, T)
        mean = weighted.sum(dim=2)  # (B, D)

        # Weighted standard deviation.
        mean_sq = (x ** 2 * weights).sum(dim=2)  # (B, D)
        var = (mean_sq - mean ** 2).clamp(min=1e-8)
        std = torch.sqrt(var)  # (B, D)

        pooled = torch.cat([mean, std], dim=1)  # (B, 2*D)
        logger.debug("AttentiveStatisticsPooling output shape: %s", pooled.shape)
        return pooled
