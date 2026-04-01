"""Custom speaker encoder that plugs into NeMo's EncDecSpeakerLabelModel.

NeMo's EncDecSpeakerLabelModel calls:
    audio_signal, audio_signal_len = self.preprocessor(...)
    encoded, encoded_len = self.encoder(processed_signal=audio_signal, length=audio_signal_len)
    logits, embs = self.decoder(encoder_output=encoded, length=encoded_len)

So the encoder contract is:
    Input:  processed_signal (B, n_mels, T), length (B,)
    Output: encoded (B, feat_out, T'), length (B,)

The decoder reads `self.encoder._feat_out` to know the channel dimension.

Register in YAML:
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
        dropout: float = 0.1,
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
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.act(self.bn(self.conv(x))))
        out = self.se(out)
        return out + self.residual(x)


class CustomSpeakerEncoder(nn.Module):
    """Lightweight TDNN-SE encoder for speaker embeddings.

    Architecture (mirrors TitaNet structure at smaller scale):
        TDNN(80→512, k=5, d=1)             — stem
        TDNN(512→512, k=3, d=2) × 3        — core blocks with SE + residual
        TDNN(512→512, k=3, d=3) × 3        — deeper context
        TDNN(512→512, k=3, d=4) × 3        — widest context
        Conv1d(512→1536, k=1)              — channel expansion

    Output: (B, 1536, T) → fed to NeMo's SpeakerDecoder for pooling + AngularSoftmax.
    """

    def __init__(self, feat_in: int = 80, feat_out: int = 1536):
        super().__init__()
        self._feat_out = feat_out

        blocks = [
            # Stem — no SE, no residual
            TDNNBlock(feat_in, 512, kernel_size=5, dilation=1, dropout=0.0, use_se=False),
        ]
        # 3 groups of repeated TDNN-SE blocks with increasing dilation
        for dilation, repeats in [(2, 3), (3, 3), (4, 3)]:
            for _ in range(repeats):
                blocks.append(
                    TDNNBlock(512, 512, kernel_size=3, dilation=dilation, dropout=0.1, use_se=True)
                )
        # Channel expansion (like TitaNet's final 1×1 conv)
        blocks.extend([
            nn.Conv1d(512, feat_out, kernel_size=1),
            nn.BatchNorm1d(feat_out),
            nn.ReLU(),
        ])
        self.layers = nn.Sequential(*blocks)

    def forward(self, audio_signal=None, length=None, processed_signal=None, processed_signal_length=None):
        """NeMo-compatible forward.

        EncDecSpeakerLabelModel may call with either:
          encoder(processed_signal=x, length=l)
        or:
          encoder(audio_signal=x, length=l)

        Args:
            audio_signal / processed_signal: (B, feat_in, T)
            length / processed_signal_length: (B,)

        Returns:
            encoded: (B, feat_out, T)
            length:  (B,) unchanged (no stride > 1)
        """
        x = processed_signal if processed_signal is not None else audio_signal
        ln = processed_signal_length if processed_signal_length is not None else length

        encoded = self.layers(x)

        if ln is not None:
            return encoded, ln
        return encoded
