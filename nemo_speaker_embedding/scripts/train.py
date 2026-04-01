"""Train a speaker embedding model using NeMo's EncDecSpeakerLabelModel.

Based on: NVIDIA/NeMo/examples/speaker_tasks/recognition/speaker_reco.py

Pipeline: Audio → MelSpec → SpecAugment → Encoder → Decoder(pool+classify) → AngularSoftmaxLoss

Usage:
    cd nemo_speaker_embedding

    # With Hydra (recommended — same as NeMo's recipe):
    python scripts/train.py --config-path=../conf --config-name=speaker_model \
        model.train_ds.manifest_filepath=dummy_data/train_manifest.json \
        model.validation_ds.manifest_filepath=dummy_data/val_manifest.json \
        model.decoder.num_classes=10 \
        trainer.max_epochs=50

    # Or override everything on command line:
    python scripts/train.py --config-path=../conf --config-name=speaker_model \
        model.train_ds.manifest_filepath=/data/train.json \
        model.validation_ds.manifest_filepath=/data/val.json \
        model.decoder.num_classes=500 \
        trainer.devices=2 trainer.max_epochs=200
"""

import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf

# Ensure project root is on PYTHONPATH so custom_model.encoder resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager

seed_everything(42)


@hydra_runner(config_path="../conf", config_name="speaker_model")
def main(cfg):
    logging.info(f"Hydra config:\n{OmegaConf.to_yaml(cfg)}")

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = pl.Trainer(**cfg.trainer)

    # ── Experiment manager (logging, checkpoints, resume) ────────────
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    # ── Model ────────────────────────────────────────────────────────
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)

    logging.info(f"Model parameters: {sum(p.numel() for p in speaker_model.parameters()):,}")
    logging.info(f"Encoder: {type(speaker_model.encoder).__name__}")
    logging.info(f"Decoder: {type(speaker_model.decoder).__name__}")
    logging.info(f"Loss:    {type(speaker_model.loss).__name__}")

    # Save speaker labels to experiment dir
    if log_dir is not None:
        with open(os.path.join(log_dir, "labels.txt"), "w") as f:
            if speaker_model.labels is not None:
                for label in speaker_model.labels:
                    f.write(f"{label}\n")

    # ── Train ────────────────────────────────────────────────────────
    trainer.fit(speaker_model)

    # Save final .nemo checkpoint
    if not trainer.fast_dev_run and log_dir is not None:
        model_path = os.path.join(log_dir, "..", "speaker_model.nemo")
        speaker_model.save_to(model_path)
        logging.info(f"Model saved to {model_path}")

    # ── Optional: test evaluation ────────────────────────────────────
    if hasattr(cfg.model, "test_ds") and cfg.model.test_ds.manifest_filepath is not None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            test_trainer = pl.Trainer(devices=1, accelerator=cfg.trainer.accelerator)
            if speaker_model.prepare_test(test_trainer):
                test_trainer.test(speaker_model)


if __name__ == "__main__":
    main()
