"""Custom speaker encoder that plugs into NeMo's EncDecSpeakerLabelModel.

NeMo expects the encoder to:
  - Accept input of shape (B, n_mels, T)
  - Return output of shape (B, feat_out, T')
  - Expose `_feat_out` so the decoder knows the channel dimension

Register in YAML via:
    encoder:
      _target_: custom_model.encoder.CustomSpeakerEncoder
      feat_in: 80
      feat_out: 1536
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w


class TDNNBlock(nn.Module):
    """Time-Delay Neural Network block with optional SE and residual."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        use_se: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, dilation=dilation, padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn(self.conv(x)))
        out = self.se(out)
        return out + self.residual(x)


class CustomSpeakerEncoder(nn.Module):
    """Lightweight TDNN-SE encoder for speaker embeddings.

    Architecture:
        TDNN(80→512, k=5, d=1)
      → TDNN(512→512, k=3, d=2) × 3  (with SE + residual)
      → TDNN(512→1536, k=1)           (channel expansion)

    Produces (B, 1536, T) for NeMo's SpeakerDecoder to pool + classify.
    """

    def __init__(self, feat_in: int = 80, feat_out: int = 1536):
        super().__init__()
        self._feat_out = feat_out

        self.layers = nn.Sequential(
            # Stem
            TDNNBlock(feat_in, 512, kernel_size=5, dilation=1, use_se=False),
            # Core blocks with increasing dilation
            TDNNBlock(512, 512, kernel_size=3, dilation=2, use_se=True),
            TDNNBlock(512, 512, kernel_size=3, dilation=3, use_se=True),
            TDNNBlock(512, 512, kernel_size=3, dilation=4, use_se=True),
            # Channel expansion
            nn.Conv1d(512, feat_out, kernel_size=1),
            nn.BatchNorm1d(feat_out),
            nn.ReLU(),
        )

    def forward(self, audio_signal, length=None):
        """
        Args:
            audio_signal: (B, feat_in, T) mel-spectrogram
            length: (B,) optional lengths — passed through for NeMo compat

        Returns:
            outputs: (B, feat_out, T)
            length:  (B,) unchanged
        """
        outputs = self.layers(audio_signal)
        if length is not None:
            return outputs, length
        return outputs
