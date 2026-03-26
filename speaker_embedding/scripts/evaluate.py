#!/usr/bin/env python
"""Evaluate speaker embeddings on trial lists."""

import argparse
import json
import logging
import sys

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from speaker_embedding.src.evaluation.metrics import compute_eer, compute_min_dcf
from speaker_embedding.src.evaluation.trials import TrialLoader, score_trials
from speaker_embedding.src.features.extractor import FeatureExtractor
from speaker_embedding.src.models.dummy_model import DummyEncoder
from speaker_embedding.src.models.pooling import StatisticsPooling
from speaker_embedding.src.models.embedding_head import EmbeddingHead

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate speaker embeddings on trial lists.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--trial_list", type=str, required=True, help="Path to trial list file.")
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML config (overrides checkpoint config).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run evaluation on.")
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
def extract_embedding(waveform, sample_rate, config, encoder, pooling, embedding_head, feature_extractor, device):
    """Extract a single embedding from a waveform tensor."""
    target_sr = config.features.sample_rate
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, S)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # (1, 1, S)

    waveform = waveform.to(device)
    lengths = torch.tensor([waveform.shape[-1]], device=device)

    features, feat_lengths = feature_extractor(waveform, lengths)
    encoded, enc_lengths = encoder(features, feat_lengths)
    pooled = pooling(encoded, enc_lengths)
    embedding = embedding_head(pooled)

    return embedding.squeeze(0).cpu().numpy()


def main():
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Loading checkpoint from %s", args.checkpoint_path)
    checkpoint, config = load_checkpoint_and_config(args.checkpoint_path, args.config_path, device)

    encoder, pooling, embedding_head = build_model(config, device)

    # Load state dicts if available
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

    # Load trial list
    logger.info("Loading trial list from %s", args.trial_list)
    trial_loader = TrialLoader(args.trial_list)
    trials = trial_loader.load()
    logger.info("Loaded %d trials.", len(trials))

    # Collect all unique utterance IDs from trials
    utterance_ids = set()
    for enroll_id, test_id, _ in trials:
        utterance_ids.add(enroll_id)
        utterance_ids.add(test_id)

    logger.info("Extracting embeddings for %d unique utterances.", len(utterance_ids))

    # Extract embeddings for each utterance
    embeddings = {}
    for utt_id in utterance_ids:
        try:
            waveform, sr = torchaudio.load(utt_id)
            emb = extract_embedding(
                waveform, sr, config, encoder, pooling, embedding_head,
                feature_extractor, device,
            )
            embeddings[utt_id] = emb
        except Exception as exc:
            logger.warning("Failed to process %s: %s", utt_id, exc)

    logger.info("Extracted embeddings for %d / %d utterances.", len(embeddings), len(utterance_ids))

    # Score trials
    scores, labels = score_trials(embeddings, trials)
    logger.info("Scored %d trials.", len(scores))

    if len(scores) == 0:
        logger.error("No trials could be scored. Check utterance paths.")
        sys.exit(1)

    # Compute metrics
    eer, eer_threshold = compute_eer(scores, labels)
    min_dcf = compute_min_dcf(
        scores, labels,
        p_target=config.get("evaluation", {}).get("mindcf_p_target", 0.01),
        c_fa=config.get("evaluation", {}).get("mindcf_c_fa", 1.0),
        c_miss=config.get("evaluation", {}).get("mindcf_c_miss", 1.0),
    )

    print(f"EER:    {eer * 100:.2f}% (threshold={eer_threshold:.4f})")
    print(f"minDCF: {min_dcf:.4f}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
