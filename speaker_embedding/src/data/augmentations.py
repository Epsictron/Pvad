"""Audio augmentation pipeline for speaker embedding training."""

import logging
import random
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as F
from omegaconf import DictConfig
from torch import Tensor

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Configurable audio augmentation pipeline.

    Supports additive noise, reverberation (RIR convolution), speed
    perturbation, and volume perturbation. Each augmentation is applied
    independently with its own probability.

    Attributes:
        config: Augmentation configuration from Hydra/OmegaConf.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the audio augmentor.

        Args:
            config: Configuration object with augmentation parameters.
                Expected structure under ``config.augmentations``:
                    noise:
                        enabled: bool
                        prob: float
                        snr_min: float (dB)
                        snr_max: float (dB)
                    reverb:
                        enabled: bool
                        prob: float
                        rir_path: str  (path to RIR audio file or directory)
                    speed:
                        enabled: bool
                        prob: float
                        speed_min: float (e.g., 0.9)
                        speed_max: float (e.g., 1.1)
                    volume:
                        enabled: bool
                        prob: float
                        gain_min: float (dB)
                        gain_max: float (dB)
        """
        self.config = config
        aug_cfg = config.augmentations

        self.noise_enabled = aug_cfg.get("noise", {}).get("enabled", False)
        self.reverb_enabled = aug_cfg.get("reverb", {}).get("enabled", False)
        self.speed_enabled = aug_cfg.get("speed", {}).get("enabled", False)
        self.volume_enabled = aug_cfg.get("volume", {}).get("enabled", False)

        logger.info(
            "AudioAugmentor initialized: noise=%s, reverb=%s, speed=%s, volume=%s.",
            self.noise_enabled,
            self.reverb_enabled,
            self.speed_enabled,
            self.volume_enabled,
        )

    def apply(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Apply the augmentation pipeline to a waveform.

        Each enabled augmentation is applied independently with its
        configured probability. The order is: speed perturbation, volume
        perturbation, additive noise, reverberation.

        Args:
            waveform: Input audio tensor of shape [C, T] or [1, T].
            sample_rate: Sample rate of the input waveform in Hz.

        Returns:
            Augmented waveform tensor with the same number of channels.
            Note that speed perturbation may change the time dimension.
        """
        if self.speed_enabled:
            waveform = self._apply_speed(waveform, sample_rate)

        if self.volume_enabled:
            waveform = self._apply_volume(waveform)

        if self.noise_enabled:
            waveform = self._apply_noise(waveform)

        if self.reverb_enabled:
            waveform = self._apply_reverb(waveform, sample_rate)

        return waveform

    def _apply_noise(self, waveform: Tensor) -> Tensor:
        """Add Gaussian noise at a random SNR level.

        Args:
            waveform: Input tensor of shape [C, T].

        Returns:
            Waveform with additive noise, or unchanged if probability check fails.
        """
        noise_cfg = self.config.augmentations.noise
        prob = float(noise_cfg.get("prob", 0.5))

        if random.random() > prob:
            return waveform

        snr_min = float(noise_cfg.get("snr_min", 5.0))
        snr_max = float(noise_cfg.get("snr_max", 20.0))
        snr_db = random.uniform(snr_min, snr_max)

        noise = torch.randn_like(waveform)

        # Compute signal and noise power
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()

        if noise_power == 0 or signal_power == 0:
            return waveform

        # Scale noise to achieve desired SNR
        snr_linear = 10.0 ** (snr_db / 10.0)
        scale = torch.sqrt(signal_power / (snr_linear * noise_power))
        waveform = waveform + scale * noise

        return waveform

    def _apply_reverb(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Apply reverberation via convolution with a room impulse response.

        Args:
            waveform: Input tensor of shape [C, T].
            sample_rate: Sample rate of the waveform in Hz.

        Returns:
            Reverberant waveform, or unchanged if probability check fails
            or RIR cannot be loaded.
        """
        reverb_cfg = self.config.augmentations.reverb
        prob = float(reverb_cfg.get("prob", 0.5))

        if random.random() > prob:
            return waveform

        rir_path = reverb_cfg.get("rir_path", None)
        if rir_path is None:
            logger.warning("Reverb enabled but no rir_path configured. Skipping.")
            return waveform

        try:
            rir, rir_sr = torchaudio.load(str(rir_path))
        except Exception as exc:
            logger.warning("Failed to load RIR from '%s': %s. Skipping.", rir_path, exc)
            return waveform

        # Resample RIR if needed
        if rir_sr != sample_rate:
            rir = torchaudio.transforms.Resample(rir_sr, sample_rate)(rir)

        # Use only the first channel of the RIR
        if rir.shape[0] > 1:
            rir = rir[0:1, :]

        # Normalize RIR
        rir = rir / (rir.abs().max() + 1e-8)

        # Convolve using torchaudio functional fftconvolve
        # Fall back to torch conv1d if fftconvolve is not available
        try:
            convolved = F.fftconvolve(waveform, rir)
        except AttributeError:
            # Manual convolution via F.convolve or torch.nn.functional.conv1d
            rir_flip = rir.flip(-1)
            padding = rir_flip.shape[-1] - 1
            convolved = torch.nn.functional.conv1d(
                waveform.unsqueeze(0),
                rir_flip.unsqueeze(0),
                padding=padding,
            ).squeeze(0)

        # Trim to original length
        convolved = convolved[:, : waveform.shape[-1]]

        # Normalize to match original RMS
        orig_rms = waveform.pow(2).mean().sqrt()
        conv_rms = convolved.pow(2).mean().sqrt()
        if conv_rms > 0:
            convolved = convolved * (orig_rms / conv_rms)

        return convolved

    def _apply_speed(self, waveform: Tensor, sample_rate: int) -> Tensor:
        """Apply speed perturbation by resampling.

        Changing the speed effectively alters the pitch and duration.
        The waveform is resampled from ``sample_rate * speed_factor`` to
        ``sample_rate``, which changes the playback speed.

        Args:
            waveform: Input tensor of shape [C, T].
            sample_rate: Original sample rate in Hz.

        Returns:
            Speed-perturbed waveform, or unchanged if probability check fails.
        """
        speed_cfg = self.config.augmentations.speed
        prob = float(speed_cfg.get("prob", 0.5))

        if random.random() > prob:
            return waveform

        speed_min = float(speed_cfg.get("speed_min", 0.9))
        speed_max = float(speed_cfg.get("speed_max", 1.1))
        speed_factor = random.uniform(speed_min, speed_max)

        if abs(speed_factor - 1.0) < 1e-4:
            return waveform

        # Resample: treat original as if it were at sample_rate * speed_factor,
        # then resample to sample_rate
        orig_freq = int(sample_rate * speed_factor)
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_freq, new_freq=sample_rate
        )
        waveform = resampler(waveform)

        return waveform

    def _apply_volume(self, waveform: Tensor) -> Tensor:
        """Apply random volume (gain) perturbation.

        Args:
            waveform: Input tensor of shape [C, T].

        Returns:
            Gain-adjusted waveform, or unchanged if probability check fails.
        """
        volume_cfg = self.config.augmentations.volume
        prob = float(volume_cfg.get("prob", 0.5))

        if random.random() > prob:
            return waveform

        gain_min = float(volume_cfg.get("gain_min", -6.0))
        gain_max = float(volume_cfg.get("gain_max", 6.0))
        gain_db = random.uniform(gain_min, gain_max)

        # Convert dB gain to linear scale
        gain_linear = 10.0 ** (gain_db / 20.0)
        waveform = waveform * gain_linear

        return waveform
