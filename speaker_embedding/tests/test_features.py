"""Tests for FeatureExtractor and normalization modules."""

import torch
import pytest
from omegaconf import OmegaConf

from speaker_embedding.src.features.extractor import FeatureExtractor
from speaker_embedding.src.features.normalization import CMVN, InstanceNorm


def _feature_config(feat_type="fbank", normalize="none"):
    """Build a minimal config for FeatureExtractor."""
    return OmegaConf.create({
        "features": {
            "type": feat_type,
            "sample_rate": 16000,
            "n_fft": 512,
            "win_length": 400,
            "hop_length": 160,
            "n_mels": 80,
            "n_mfcc": 40,
            "f_min": 20,
            "f_max": 7600,
            "normalize": normalize,
        },
    })


class TestFeatureExtractor:

    @pytest.mark.parametrize("feat_type", ["fbank", "mfcc", "spectrogram", "raw"])
    def test_forward_runs(self, feat_type):
        config = _feature_config(feat_type=feat_type)
        extractor = FeatureExtractor(config)
        extractor.eval()

        B, S = 2, 16000  # 1 second of audio at 16 kHz
        waveform = torch.randn(B, 1, S)
        lengths = torch.tensor([S, S])

        features, feat_lengths = extractor(waveform, lengths)
        assert features.dim() == 3  # (B, C, T)
        assert features.shape[0] == B
        assert feat_lengths.shape == (B,)

    def test_fbank_output_shape(self):
        config = _feature_config(feat_type="fbank")
        extractor = FeatureExtractor(config)
        extractor.eval()

        B, S = 2, 16000
        waveform = torch.randn(B, 1, S)
        lengths = torch.tensor([S, S])
        features, _ = extractor(waveform, lengths)
        # fbank should have n_mels channels
        assert features.shape[1] == 80

    def test_mfcc_output_shape(self):
        config = _feature_config(feat_type="mfcc")
        extractor = FeatureExtractor(config)
        extractor.eval()

        B, S = 2, 16000
        waveform = torch.randn(B, 1, S)
        lengths = torch.tensor([S, S])
        features, _ = extractor(waveform, lengths)
        assert features.shape[1] == 40  # n_mfcc

    def test_spectrogram_output_shape(self):
        config = _feature_config(feat_type="spectrogram")
        extractor = FeatureExtractor(config)
        extractor.eval()

        B, S = 2, 16000
        waveform = torch.randn(B, 1, S)
        lengths = torch.tensor([S, S])
        features, _ = extractor(waveform, lengths)
        assert features.shape[1] == 512 // 2 + 1  # n_fft // 2 + 1

    def test_raw_output_shape(self):
        config = _feature_config(feat_type="raw")
        extractor = FeatureExtractor(config)
        extractor.eval()

        B, S = 2, 16000
        waveform = torch.randn(B, 1, S)
        lengths = torch.tensor([S, S])
        features, feat_lengths = extractor(waveform, lengths)
        assert features.shape[1] == 1
        assert features.shape[2] == S

    def test_invalid_type_raises(self):
        config = _feature_config(feat_type="unknown")
        with pytest.raises(ValueError):
            FeatureExtractor(config)


class TestCMVN:

    def test_zero_mean(self):
        cmvn = CMVN()
        # (B, C, T) with non-zero mean
        features = torch.randn(4, 80, 100) + 5.0
        normalized = cmvn(features)
        # Mean along time axis should be approximately zero
        means = normalized.mean(dim=-1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_output_shape(self):
        cmvn = CMVN()
        features = torch.randn(2, 40, 50)
        normalized = cmvn(features)
        assert normalized.shape == features.shape


class TestInstanceNorm:

    def test_forward(self):
        norm = InstanceNorm(num_features=80)
        features = torch.randn(4, 80, 100)
        out = norm(features)
        assert out.shape == features.shape

    def test_output_shape(self):
        norm = InstanceNorm(num_features=40)
        features = torch.randn(2, 40, 50)
        out = norm(features)
        assert out.shape == features.shape
