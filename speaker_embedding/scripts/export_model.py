#!/usr/bin/env python
"""Export model to ONNX or TorchScript."""

import argparse
import logging

import torch
from omegaconf import OmegaConf

from speaker_embedding.src.features.extractor import FeatureExtractor
from speaker_embedding.src.models.dummy_model import DummyEncoder
from speaker_embedding.src.models.pooling import StatisticsPooling
from speaker_embedding.src.models.embedding_head import EmbeddingHead
from speaker_embedding.src.utils.export import export_onnx, export_torchscript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export model to ONNX or TorchScript.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="onnx", help="Export format.")
    parser.add_argument("--output_path", type=str, required=True, help="Output file path.")
    parser.add_argument("--config_path", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--device", type=str, default="cpu", help="Device for export.")
    return parser.parse_args()


class ExportableModel(torch.nn.Module):
    """Wrapper that chains feature extraction, encoder, pooling, and head.

    This module accepts raw waveforms and produces speaker embeddings,
    suitable for tracing or scripting for deployment.
    """

    def __init__(self, feature_extractor, encoder, pooling, embedding_head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.pooling = pooling
        self.embedding_head = embedding_head

    def forward(self, waveform, lengths):
        """Forward pass from raw waveform to embedding.

        Args:
            waveform: (B, 1, S) raw audio.
            lengths: (B,) sample counts.

        Returns:
            (B, embedding_dim) speaker embeddings.
        """
        features, feat_lengths = self.feature_extractor(waveform, lengths)
        encoded, enc_lengths = self.encoder(features, feat_lengths)
        pooled = self.pooling(encoded, enc_lengths)
        embedding = self.embedding_head(pooled)
        return embedding


def main():
    args = parse_args()
    device = torch.device(args.device)

    logger.info("Loading checkpoint from %s", args.checkpoint_path)
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)

    if args.config_path is not None:
        config = OmegaConf.load(args.config_path)
    elif "config" in checkpoint:
        config = OmegaConf.create(checkpoint["config"])
    else:
        raise RuntimeError(
            "No config found in checkpoint and --config_path not provided."
        )

    # Build model components
    feature_extractor = FeatureExtractor(config).to(device)
    encoder = DummyEncoder(config).to(device)
    pooling = StatisticsPooling().to(device)
    embedding_head = EmbeddingHead(
        input_dim=encoder.output_dim * 2,
        embedding_dim=config.model.embedding_dim,
    ).to(device)

    # Load state dicts
    if "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    if "pooling_state_dict" in checkpoint:
        pooling.load_state_dict(checkpoint["pooling_state_dict"])
    if "embedding_head_state_dict" in checkpoint:
        embedding_head.load_state_dict(checkpoint["embedding_head_state_dict"])

    feature_extractor.eval()
    encoder.eval()
    pooling.eval()
    embedding_head.eval()

    model = ExportableModel(feature_extractor, encoder, pooling, embedding_head)
    model.eval()
    model.to(device)

    # Create dummy input for tracing
    sample_rate = config.features.sample_rate
    segment_samples = int(config.data.segment_length * sample_rate)
    dummy_waveform = torch.randn(1, 1, segment_samples, device=device)
    dummy_lengths = torch.tensor([segment_samples], device=device)

    if args.format == "onnx":
        logger.info("Exporting to ONNX: %s", args.output_path)
        export_onnx(
            model=model,
            dummy_input=(dummy_waveform, dummy_lengths),
            output_path=args.output_path,
            opset_version=config.get("export", {}).get("opset_version", 17),
        )
    elif args.format == "torchscript":
        logger.info("Exporting to TorchScript: %s", args.output_path)
        export_torchscript(
            model=model,
            dummy_input=(dummy_waveform, dummy_lengths),
            output_path=args.output_path,
        )

    logger.info("Export complete: %s", args.output_path)


if __name__ == "__main__":
    main()
