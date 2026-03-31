"""Synthetic dataset for quick prototyping and smoke tests."""

import torch
from torch.utils.data import Dataset


class DummySpeakerDataset(Dataset):
    """Generates random mel-spectrogram tensors with speaker labels."""

    def __init__(
        self,
        num_samples: int = 200,
        num_speakers: int = 10,
        n_mels: int = 80,
        max_frames: int = 300,
    ):
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.n_mels = n_mels
        self.max_frames = max_frames

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        T = torch.randint(100, self.max_frames + 1, (1,)).item()
        features = torch.randn(self.n_mels, T)
        label = torch.tensor(idx % self.num_speakers, dtype=torch.long)
        return {"features": features, "label": label}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length spectrograms to the longest in the batch."""
    max_t = max(item["features"].shape[1] for item in batch)
    padded = []
    for item in batch:
        f = item["features"]
        pad_len = max_t - f.shape[1]
        if pad_len > 0:
            f = torch.nn.functional.pad(f, (0, pad_len))
        padded.append(f)
    features = torch.stack(padded)
    labels = torch.stack([item["label"] for item in batch])
    return {"features": features, "labels": labels}
