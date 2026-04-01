"""Train a speaker embedding model using NeMo's EncDecSpeakerLabelModel.

This script uses the full NeMo pipeline:
  Preprocessor (MelSpec) → Encoder (custom TDNN-SE) → Decoder (SpeakerDecoder) → Loss

Usage:
    cd nemo_speaker_embedding
    python scripts/train.py                          # default config
    python scripts/train.py --config conf/speaker_model.yaml  # custom config
"""

import sys
import argparse
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf

# Ensure the project root is importable so custom_model.encoder resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils.exp_manager import exp_manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="conf/speaker_model.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = pl.Trainer(
        devices=cfg.trainer.get("devices", 1),
        accelerator=cfg.trainer.get("accelerator", "auto"),
        max_epochs=cfg.trainer.get("max_epochs", 50),
        accumulate_grad_batches=cfg.trainer.get("accumulate_grad_batches", 1),
        precision=cfg.trainer.get("precision", "32-true"),
        log_every_n_steps=cfg.trainer.get("log_every_n_steps", 10),
        val_check_interval=cfg.trainer.get("val_check_interval", 1.0),
        enable_checkpointing=cfg.trainer.get("enable_checkpointing", True),
    )

    # ── Experiment manager (logging + checkpoints) ───────────────────
    exp_manager(
        trainer,
        cfg={
            "exp_dir": cfg.get("exp_dir", "./experiments"),
            "name": cfg.get("name", "SpeakerModel"),
            "checkpoint_callback_params": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "always_save_nemo": True,
            },
            "resume_if_exists": True,
            "resume_ignore_no_checkpoint": True,
            "create_tensorboard_logger": True,
        },
    )

    # ── Model ────────────────────────────────────────────────────────
    model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Encoder: {type(model.encoder).__name__}")
    print(f"Decoder: {type(model.decoder).__name__}\n")

    # ── Train ────────────────────────────────────────────────────────
    trainer.fit(model)

    print(f"\nTraining complete. Checkpoints saved in: {cfg.get('exp_dir', './experiments')}")


if __name__ == "__main__":
    main()
