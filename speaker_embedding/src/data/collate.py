"""Collation function for speaker embedding data batches."""

import logging
from typing import Dict, List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict]) -> Dict:
    """Collate a list of dataset samples into a batched dictionary.

    Pads all audio tensors to the maximum length in the batch with zeros
    and stacks them into a single tensor.

    Args:
        batch: List of sample dictionaries, each with keys:
            - 'audio': Tensor of shape [1, T_i] (variable length).
            - 'speaker_id': Integer speaker label.
            - 'gender': Gender string ('m', 'f', or 'u').
            - 'index': Original dataset index.

    Returns:
        Dictionary with keys:
            - 'audio': Tensor of shape [B, 1, T_max] where T_max is the
              longest audio in the batch.
            - 'lengths': Tensor of shape [B] with the original (unpadded)
              number of samples for each utterance.
            - 'speaker_ids': Tensor of shape [B] with integer speaker labels.
            - 'genders': List of gender strings of length B.
    """
    # Extract individual fields
    audios: List[Tensor] = [sample["audio"] for sample in batch]
    speaker_ids: List[int] = [sample["speaker_id"] for sample in batch]
    genders: List[str] = [sample["gender"] for sample in batch]

    # Compute original lengths (number of samples per utterance)
    lengths = torch.tensor(
        [audio.shape[-1] for audio in audios], dtype=torch.long
    )

    # Find maximum length for padding
    max_length = int(lengths.max().item())

    # Pad each audio tensor to max_length and stack
    padded_audios: List[Tensor] = []
    for audio in audios:
        num_samples = audio.shape[-1]
        if num_samples < max_length:
            pad_amount = max_length - num_samples
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
        padded_audios.append(audio)

    audio_batch = torch.stack(padded_audios, dim=0)  # [B, 1, T_max]
    speaker_id_batch = torch.tensor(speaker_ids, dtype=torch.long)

    return {
        "audio": audio_batch,
        "lengths": lengths,
        "speaker_ids": speaker_id_batch,
        "genders": genders,
    }
