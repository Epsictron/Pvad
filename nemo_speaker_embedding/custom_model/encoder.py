"""Custom encoder backbone — the only from-scratch piece.
Everything else (preprocessor, decoder, loss, training) is NeMo.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * w


class TDNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.1, se=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation,
                              padding=(kernel - 1) // 2 * dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.se(self.drop(self.act(self.bn(self.conv(x))))) + self.res(x)


class CustomSpeakerEncoder(nn.Module):
    """TDNN-SE encoder. Plug into NeMo via _target_ in YAML."""

    def __init__(self, feat_in=80, feat_out=1536):
        super().__init__()
        self._feat_out = feat_out
        layers = [TDNNBlock(feat_in, 512, kernel=5, dilation=1, dropout=0, se=False)]
        for d in [2, 3, 4]:
            for _ in range(3):
                layers.append(TDNNBlock(512, 512, kernel=3, dilation=d))
        layers += [nn.Conv1d(512, feat_out, 1), nn.BatchNorm1d(feat_out), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, audio_signal=None, length=None,
                processed_signal=None, processed_signal_length=None):
        x = processed_signal if processed_signal is not None else audio_signal
        ln = processed_signal_length if processed_signal_length is not None else length
        out = self.net(x)
        return (out, ln) if ln is not None else out
