"""Speaker embedding dataset with on-the-fly audio loading."""

import logging
import random
from typing import Dict, List

import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset

from .manifest import ManifestEntry

logger = logging.getLogger(__name__)


class SpeakerDataset(Dataset):
    """PyTorch dataset for speaker embedding training.

    Loads audio on-the-fly using torchaudio, crops or pads to a fixed
    segment length, and returns a dictionary with audio tensor, speaker ID,
    gender, and sample index.

    Attributes:
        entries: List of manifest entries backing this dataset.
        config: Hydra/OmegaConf configuration object.
        speaker_to_idx: Mapping from speaker string label to integer ID.
        segment_samples: Number of audio samples per segment.
        sample_rate: Target sample rate in Hz.
    """

    def __init__(self, manifest_entries: List[ManifestEntry], config: DictConfig) -> None:
        """Initialize the speaker dataset.

        Args:
            manifest_entries: List of ManifestEntry objects describing the data.
            config: Configuration object. Expected keys:
                - config.data.segment_length: Segment length in seconds.
                - config.features.sample_rate: Target sample rate in Hz.
        """
        super().__init__()
        self.entries = manifest_entries
        self.config = config
        self.sample_rate: int = config.features.sample_rate
        self.segment_samples: int = int(config.data.segment_length * self.sample_rate)

        # Build speaker-to-index mapping (sorted for determinism)
        unique_speakers = sorted(set(entry.speaker for entry in self.entries))
        self.speaker_to_idx: Dict[str, int] = {
            spk: idx for idx, spk in enumerate(unique_speakers)
        }

        logger.info(
            "SpeakerDataset created: %d utterances, %d speakers, "
            "segment_length=%.2fs (%d samples) at %d Hz.",
            len(self.entries),
            len(self.speaker_to_idx),
            config.data.segment_length,
            self.segment_samples,
            self.sample_rate,
        )

    def __len__(self) -> int:
        """Return the number of utterances in the dataset."""
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict:
        """Load and return a single sample.

        Args:
            index: Index of the sample to retrieve.

        Returns:
            Dictionary with keys:
                - 'audio': Tensor of shape [1, T] where T = segment_samples.
                - 'speaker_id': Integer speaker label.
                - 'gender': Gender string ('m', 'f', or 'u').
                - 'index': The original dataset index.
        """
        entry = self.entries[index]

        # Load audio
        waveform, sr = torchaudio.load(entry.audio_filepath)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        num_samples = waveform.shape[1]

        if num_samples > self.segment_samples:
            # Random offset for cropping
            max_offset = num_samples - self.segment_samples
            offset = random.randint(0, max_offset)
            waveform = waveform[:, offset : offset + self.segment_samples]
        elif num_samples < self.segment_samples:
            # Pad with zeros on the right
            pad_length = self.segment_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        return {
            "audio": waveform,  # [1, T]
            "speaker_id": self.speaker_to_idx[entry.speaker],
            "gender": entry.gender,
            "index": index,
        }
