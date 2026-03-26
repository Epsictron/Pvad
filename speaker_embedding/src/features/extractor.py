"""GPU-based acoustic feature extraction using torchaudio transforms.

Supports filter-bank (fbank), MFCC, spectrogram, and raw waveform
pass-through.  Optionally applies per-utterance normalization and
SpecAugment (frequency / time masking).
"""

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio.transforms as T
from omegaconf import DictConfig
from torch import Tensor

from .normalization import CMVN, InstanceNorm

logger = logging.getLogger(__name__)

# Small constant added before log to avoid log(0).
_LOG_EPS = 1e-8


class FeatureExtractor(nn.Module):
    """Configurable acoustic feature extractor.

    Wraps :mod:`torchaudio.transforms` so that feature computation
    happens entirely on GPU with no disk I/O.

    Args:
        config: Hydra / OmegaConf configuration object.  Expected keys
            live under ``config.features`` and include:

            - **type** (``str``): One of ``"fbank"``, ``"mfcc"``,
              ``"spectrogram"``, or ``"raw"``.
            - **sample_rate** (``int``): Audio sample rate in Hz.
            - **n_fft** (``int``): FFT size.
            - **win_length** (``int``): Window length in samples.
            - **hop_length** (``int``): Hop length in samples.
            - **n_mels** (``int``): Number of mel filter-bank channels.
            - **n_mfcc** (``int``): Number of MFCC coefficients.
            - **f_min** (``float``): Minimum frequency for mel banks.
            - **f_max** (``Optional[float]``): Maximum frequency for mel
              banks (``None`` defaults to Nyquist).
            - **normalize** (``str``): Normalization strategy — one of
              ``"cmvn"``, ``"instance_norm"``, or ``"none"``.
            - **spec_augment** (``DictConfig | None``): Optional
              SpecAugment settings with ``freq_mask_param`` and
              ``time_mask_param``.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        feat_cfg = config.features
        self.feat_type: str = feat_cfg.type
        self.hop_length: int = feat_cfg.hop_length

        # ----- Feature transform ------------------------------------------
        if self.feat_type == "fbank":
            self.transform: nn.Module = T.MelSpectrogram(
                sample_rate=feat_cfg.sample_rate,
                n_fft=feat_cfg.n_fft,
                win_length=feat_cfg.win_length,
                hop_length=feat_cfg.hop_length,
                n_mels=feat_cfg.n_mels,
                f_min=feat_cfg.f_min,
                f_max=feat_cfg.f_max,
            )
            self._num_features = feat_cfg.n_mels
            logger.info("Feature extractor: fbank (%d mels)", feat_cfg.n_mels)

        elif self.feat_type == "mfcc":
            melkwargs = {
                "n_fft": feat_cfg.n_fft,
                "win_length": feat_cfg.win_length,
                "hop_length": feat_cfg.hop_length,
                "n_mels": feat_cfg.n_mels,
                "f_min": feat_cfg.f_min,
                "f_max": feat_cfg.f_max,
            }
            self.transform = T.MFCC(
                sample_rate=feat_cfg.sample_rate,
                n_mfcc=feat_cfg.n_mfcc,
                melkwargs=melkwargs,
            )
            self._num_features = feat_cfg.n_mfcc
            logger.info("Feature extractor: mfcc (%d coeffs)", feat_cfg.n_mfcc)

        elif self.feat_type == "spectrogram":
            self.transform = T.Spectrogram(
                n_fft=feat_cfg.n_fft,
                win_length=feat_cfg.win_length,
                hop_length=feat_cfg.hop_length,
            )
            self._num_features = feat_cfg.n_fft // 2 + 1
            logger.info("Feature extractor: spectrogram")

        elif self.feat_type == "raw":
            self.transform = nn.Identity()
            self._num_features = 1
            logger.info("Feature extractor: raw pass-through")

        else:
            raise ValueError(
                f"Unknown feature type '{self.feat_type}'. "
                "Expected one of: fbank, mfcc, spectrogram, raw."
            )

        # ----- Normalization ----------------------------------------------
        norm_type: str = getattr(feat_cfg, "normalize", "none")
        if norm_type == "cmvn":
            self.normalizer: Optional[nn.Module] = CMVN()
            logger.info("Normalization: CMVN")
        elif norm_type == "instance_norm":
            self.normalizer = InstanceNorm(num_features=self._num_features)
            logger.info("Normalization: InstanceNorm")
        elif norm_type == "none":
            self.normalizer = None
            logger.info("Normalization: none")
        else:
            raise ValueError(
                f"Unknown normalize type '{norm_type}'. "
                "Expected one of: cmvn, instance_norm, none."
            )

        # ----- SpecAugment ------------------------------------------------
        spec_aug_cfg = getattr(feat_cfg, "spec_augment", None)
        if spec_aug_cfg is not None:
            freq_mask_param: int = getattr(spec_aug_cfg, "freq_mask_param", 0)
            time_mask_param: int = getattr(spec_aug_cfg, "time_mask_param", 0)

            self.freq_masking: Optional[nn.Module] = (
                T.FrequencyMasking(freq_mask_param=freq_mask_param)
                if freq_mask_param > 0
                else None
            )
            self.time_masking: Optional[nn.Module] = (
                T.TimeMasking(time_mask_param=time_mask_param)
                if time_mask_param > 0
                else None
            )
            logger.info(
                "SpecAugment enabled: freq_mask_param=%d, time_mask_param=%d",
                freq_mask_param,
                time_mask_param,
            )
        else:
            self.freq_masking = None
            self.time_masking = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def num_features(self) -> int:
        """Return the number of output feature channels."""
        return self._num_features

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        waveform: Tensor,
        lengths: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Extract features from a batch of waveforms.

        Args:
            waveform: Raw audio tensor of shape ``(B, 1, S)`` or
                ``(B, S)`` where *S* is the number of samples.
            lengths: Integer tensor of shape ``(B,)`` giving the true
                number of samples per utterance (before padding).

        Returns:
            A tuple ``(features, feat_lengths)`` where *features* has
            shape ``(B, C, T)`` and *feat_lengths* holds the valid
            frame count per utterance.
        """
        # Ensure shape is (B, S) — squeeze channel dim if present.
        if waveform.dim() == 3 and waveform.size(1) == 1:
            waveform = waveform.squeeze(1)

        # --- Feature extraction -------------------------------------------
        features = self.transform(waveform)  # (B, C, T) or (B, S) for raw

        # For raw pass-through, add a channel dim so output is always (B, C, T).
        if self.feat_type == "raw":
            if features.dim() == 2:
                features = features.unsqueeze(1)

        # Log mel energy for fbank (add epsilon to avoid log(0)).
        if self.feat_type == "fbank":
            features = torch.log(features + _LOG_EPS)

        # --- Compute output lengths ---------------------------------------
        if self.feat_type == "raw":
            feat_lengths = lengths
        else:
            # Number of STFT frames produced for a given number of samples.
            feat_lengths = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1

        # --- Normalization ------------------------------------------------
        if self.normalizer is not None:
            features = self.normalizer(features)

        # --- SpecAugment (only during training) ---------------------------
        if self.training:
            if self.freq_masking is not None:
                features = self.freq_masking(features)
            if self.time_masking is not None:
                features = self.time_masking(features)

        return features, feat_lengths
