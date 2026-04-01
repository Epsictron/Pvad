"""Custom encoder — drop-in replacement for NeMo's ConvASREncoder."""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(ch, ch // r, bias=False), nn.ReLU(),
                                nn.Linear(ch // r, ch, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1)


class TDNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, d=1, drop=0.1, se=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=d, padding=(k-1)//2*d)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.se = SEBlock(out_ch) if se else nn.Identity()
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.se(self.drop(self.act(self.bn(self.conv(x))))) + self.res(x)


class CustomSpeakerEncoder(nn.Module):
    """TDNN-SE encoder. Same I/O contract as NeMo's ConvASREncoder."""

    def __init__(self, feat_in=80, feat_out=3072):
        super().__init__()
        self._feat_out = feat_out
        layers = [TDNNBlock(feat_in, 512, k=5, d=1, drop=0, se=False)]
        for d in [2, 3, 4]:
            for _ in range(3):
                layers.append(TDNNBlock(512, 512, k=3, d=d))
        layers += [nn.Conv1d(512, feat_out, 1), nn.BatchNorm1d(feat_out), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, audio_signal=None, length=None,
                processed_signal=None, processed_signal_length=None):
        x = processed_signal if processed_signal is not None else audio_signal
        ln = processed_signal_length if processed_signal_length is not None else length
        return (self.net(x), ln) if ln is not None else self.net(x)
