#!/usr/bin/env python
"""Batch embedding extraction."""

import argparse
import json
import logging

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from speaker_embedding.src.data.manifest import load_manifest
from speaker_embedding.src.features.extractor import FeatureExtractor
from speaker_embedding.src.models.dummy_model import DummyEncoder
from speaker_embedding.src.models.pooling import StatisticsPooling
from speaker_embedding.src.models.embedding_head import EmbeddingHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch embedding extraction.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to JSONL manifest.")
    parser.add_argument("--output_path", type=str, required=True, help="Output .npz file path.")
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for extraction.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for extraction.")
    return parser.parse_args()


def build_model(config, device):
    """Build encoder, pooling, and embedding head from config."""
    encoder = DummyEncoder(config).to(device)
    pooling = StatisticsPooling().to(device)
    embedding_head = EmbeddingHead(
        input_dim=encoder.output_dim * 2,
        embedding_dim=config.model.embedding_dim,
    ).to(device)
    return encoder, pooling, embedding_head


def load_checkpoint_and_config(checkpoint_path, config_path, device):
    """Load checkpoint and resolve configuration."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if config_path is not None:
        config = OmegaConf.load(config_path)
    elif "config" in checkpoint:
        config = OmegaConf.create(checkpoint["config"])
    else:
        raise RuntimeError(
            "No config found in checkpoint and --config_path not provided."
        )

    return checkpoint, config


@torch.no_grad()
def extract_batch(waveforms, lengths, feature_extractor, encoder, pooling, embedding_head, device):
    """Extract embeddings for a batch of waveforms.

    Args:
        waveforms: Tensor of shape (B, 1, S) or (B, S).
        lengths: Tensor of shape (B,) with sample counts.
        feature_extractor: FeatureExtractor module.
        encoder: Encoder module.
        pooling: Pooling module.
        embedding_head: EmbeddingHead module.
        device: Torch device.

    Returns:
        Numpy array of shape (B, embedding_dim).
    """
    waveforms = waveforms.to(device)
    lengths = lengths.to(device)

    features, feat_lengths = feature_extractor(waveforms, lengths)
    encoded, enc_lengths = encoder(features, feat_lengths)
    pooled = pooling(encoded, enc_lengths)
    embeddings = embedding_head(pooled)

    return embeddings.cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Loading checkpoint from %s", args.checkpoint_path)
    checkpoint, config = load_checkpoint_and_config(args.checkpoint_path, args.config_path, device)

    encoder, pooling, embedding_head = build_model(config, device)

    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if "pooling_state_dict" in checkpoint:
        pooling.load_state_dict(checkpoint["pooling_state_dict"])
    if "embedding_head_state_dict" in checkpoint:
        embedding_head.load_state_dict(checkpoint["embedding_head_state_dict"])

    encoder.eval()
    pooling.eval()
    embedding_head.eval()

    feature_extractor = FeatureExtractor(config)
    feature_extractor.eval()
    feature_extractor.to(device)

    # Load manifest
    logger.info("Loading manifest from %s", args.manifest_path)
    entries = load_manifest(args.manifest_path)
    logger.info("Loaded %d entries.", len(entries))

    sample_rate = config.features.sample_rate
    segment_length = config.data.segment_length
    segment_samples = int(segment_length * sample_rate)

    all_ids = []
    all_embeddings = []

    # Process in batches
    for batch_start in range(0, len(entries), args.batch_size):
        batch_entries = entries[batch_start : batch_start + args.batch_size]
        waveforms = []
        lengths = []
        ids = []

        for entry in batch_entries:
            try:
                waveform, sr = torchaudio.load(entry.audio_filepath)
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                num_samples = waveform.shape[1]
                actual_length = min(num_samples, segment_samples)

                # Pad or crop
                if num_samples > segment_samples:
                    waveform = waveform[:, :segment_samples]
                elif num_samples < segment_samples:
                    pad_len = segment_samples - num_samples
                    waveform = torch.nn.functional.pad(waveform, (0, pad_len))

                waveforms.append(waveform)
                lengths.append(actual_length)
                ids.append(entry.audio_filepath)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", entry.audio_filepath, exc)

        if not waveforms:
            continue

        waveform_batch = torch.cat(waveforms, dim=0).unsqueeze(1)  # (B, 1, S)
        length_batch = torch.tensor(lengths, dtype=torch.long)

        embeddings = extract_batch(
            waveform_batch, length_batch,
            feature_extractor, encoder, pooling, embedding_head, device,
        )

        all_ids.extend(ids)
        all_embeddings.append(embeddings)

        if (batch_start // args.batch_size + 1) % 10 == 0:
            logger.info(
                "Processed %d / %d entries.",
                min(batch_start + args.batch_size, len(entries)),
                len(entries),
            )

    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([])

    # Save as npz: keys are utterance IDs, plus a stacked array
    logger.info("Saving %d embeddings to %s", len(all_ids), args.output_path)
    np.savez(
        args.output_path,
        ids=np.array(all_ids),
        embeddings=all_embeddings,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
