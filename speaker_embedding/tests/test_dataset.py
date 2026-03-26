"""Tests for manifest loading and SpeakerDataset."""

import json

import numpy as np
import soundfile as sf
import torch
import pytest
from omegaconf import OmegaConf

from speaker_embedding.src.data.manifest import ManifestEntry, load_manifest
from speaker_embedding.src.data.dataset import SpeakerDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(path, entries):
    """Write a list of dicts as JSONL to *path*."""
    with open(path, "w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def _make_wav(path, num_samples=16000, sample_rate=16000):
    """Create a mono WAV file with random noise."""
    data = np.random.randn(num_samples).astype(np.float32)
    sf.write(str(path), data, sample_rate)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

class TestManifestLoading:

    def test_load_basic(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            {"audio_filepath": "/a.wav", "speaker": "spk0", "duration": 2.0, "gender": "m"},
            {"audio_filepath": "/b.wav", "speaker": "spk1", "duration": 3.0, "gender": "f"},
        ])
        entries = load_manifest(str(manifest))
        assert len(entries) == 2
        assert entries[0].audio_filepath == "/a.wav"
        assert entries[1].speaker == "spk1"

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_manifest("/nonexistent/manifest.jsonl")

    def test_filter_min_duration(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            {"audio_filepath": "/a.wav", "speaker": "spk0", "duration": 0.5},
            {"audio_filepath": "/b.wav", "speaker": "spk0", "duration": 2.0},
            {"audio_filepath": "/c.wav", "speaker": "spk0", "duration": 5.0},
        ])
        entries = load_manifest(str(manifest), min_duration=1.0)
        assert len(entries) == 2
        assert all(e.duration >= 1.0 for e in entries)

    def test_filter_max_duration(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            {"audio_filepath": "/a.wav", "speaker": "spk0", "duration": 1.0},
            {"audio_filepath": "/b.wav", "speaker": "spk0", "duration": 10.0},
            {"audio_filepath": "/c.wav", "speaker": "spk0", "duration": 50.0},
        ])
        entries = load_manifest(str(manifest), max_duration=15.0)
        assert len(entries) == 2
        assert all(e.duration <= 15.0 for e in entries)

    def test_filter_min_and_max_duration(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        _write_manifest(manifest, [
            {"audio_filepath": "/a.wav", "speaker": "spk0", "duration": 0.5},
            {"audio_filepath": "/b.wav", "speaker": "spk0", "duration": 3.0},
            {"audio_filepath": "/c.wav", "speaker": "spk0", "duration": 100.0},
        ])
        entries = load_manifest(str(manifest), min_duration=1.0, max_duration=10.0)
        assert len(entries) == 1
        assert entries[0].duration == 3.0


# ---------------------------------------------------------------------------
# ManifestEntry creation and validation
# ---------------------------------------------------------------------------

class TestManifestEntry:

    def test_creation(self):
        entry = ManifestEntry(audio_filepath="/a.wav", speaker="spk0", duration=2.5, gender="m")
        assert entry.audio_filepath == "/a.wav"
        assert entry.speaker == "spk0"
        assert entry.duration == 2.5
        assert entry.gender == "m"

    def test_default_gender(self):
        entry = ManifestEntry(audio_filepath="/a.wav", speaker="spk0", duration=1.0)
        assert entry.gender == "u"

    def test_invalid_gender_defaults_to_unknown(self):
        entry = ManifestEntry(audio_filepath="/a.wav", speaker="spk0", duration=1.0, gender="x")
        assert entry.gender == "u"


# ---------------------------------------------------------------------------
# SpeakerDataset
# ---------------------------------------------------------------------------

class TestSpeakerDataset:

    @pytest.fixture
    def dataset_with_audio(self, tmp_path):
        """Create a dataset backed by real WAV files."""
        sample_rate = 16000
        segment_length = 1.0  # 1 second
        entries = []
        for i in range(4):
            wav_path = tmp_path / f"utt{i}.wav"
            dur = 2.0  # 2 seconds of audio
            _make_wav(wav_path, num_samples=int(dur * sample_rate), sample_rate=sample_rate)
            entries.append(ManifestEntry(
                audio_filepath=str(wav_path),
                speaker=f"spk{i % 2}",
                duration=dur,
                gender="m" if i % 2 == 0 else "f",
            ))
        config = OmegaConf.create({
            "data": {"segment_length": segment_length},
            "features": {"sample_rate": sample_rate},
        })
        return SpeakerDataset(entries, config)

    def test_len(self, dataset_with_audio):
        assert len(dataset_with_audio) == 4

    def test_getitem_keys(self, dataset_with_audio):
        sample = dataset_with_audio[0]
        assert "audio" in sample
        assert "speaker_id" in sample
        assert "gender" in sample
        assert "index" in sample

    def test_getitem_audio_shape(self, dataset_with_audio):
        sample = dataset_with_audio[0]
        expected_samples = int(1.0 * 16000)  # segment_length * sample_rate
        assert sample["audio"].shape == (1, expected_samples)

    def test_speaker_id_is_int(self, dataset_with_audio):
        sample = dataset_with_audio[0]
        assert isinstance(sample["speaker_id"], int)
