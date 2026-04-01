"""Finetune a pretrained NeMo speaker model (TitaNet/ECAPA) on custom data.

Keeps the pretrained encoder frozen initially, replaces the decoder head
for your number of speakers, and trains with AngularSoftmaxLoss.

Usage:
    cd nemo_speaker_embedding
    python scripts/finetune_pretrained.py \
        --pretrained titanet_large \
        --train_manifest dummy_data/train_manifest.json \
        --val_manifest dummy_data/val_manifest.json \
        --num_speakers 10 \
        --epochs 20

    # Unfreeze encoder after N epochs for full finetuning:
    python scripts/finetune_pretrained.py \
        --pretrained titanet_large \
        --train_manifest /data/train.json \
        --val_manifest /data/val.json \
        --num_speakers 500 \
        --epochs 50 \
        --unfreeze_encoder_epoch 5
"""

import argparse
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.modules import SpeakerDecoder
from nemo.utils.exp_manager import exp_manager


class UnfreezeEncoderCallback(pl.Callback):
    """Unfreeze the encoder after a given epoch for full finetuning."""

    def __init__(self, unfreeze_epoch: int):
        self.unfreeze_epoch = unfreeze_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.unfreeze_epoch:
            for param in pl_module.encoder.parameters():
                param.requires_grad = True
            print(f"[Epoch {self.unfreeze_epoch}] Encoder unfrozen — full finetuning starts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="titanet_large",
                        help="NeMo pretrained model name (titanet_large, ecapa_tdnn, etc.)")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--num_speakers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--unfreeze_encoder_epoch", type=int, default=-1,
                        help="Epoch to unfreeze encoder (-1 = always frozen)")
    args = parser.parse_args()

    # ── Load pretrained ──────────────────────────────────────────────
    model = EncDecSpeakerLabelModel.from_pretrained(args.pretrained)
    print(f"Loaded pretrained: {args.pretrained}")
    print(f"Original decoder num_classes: {model.decoder._num_classes}")

    # ── Freeze encoder ───────────────────────────────────────────────
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen.")

    # ── Replace decoder for new speaker count ────────────────────────
    model.decoder = SpeakerDecoder(
        feat_in=model.encoder._feat_out,
        num_classes=args.num_speakers,
        pool_mode="attention",
        emb_sizes=192,
        angular=True,
    )
    print(f"New decoder: num_classes={args.num_speakers}, angular=True")

    # ── Update data configs ──────────────────────────────────────────
    train_ds = OmegaConf.create({
        "manifest_filepath": args.train_manifest,
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": 2,
        "labels": None,
    })
    val_ds = OmegaConf.create({
        "manifest_filepath": args.val_manifest,
        "sample_rate": 16000,
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": 2,
    })
    model.setup_training_data(train_ds)
    model.setup_validation_data(val_ds)

    # ── Update optimizer (lower LR for finetuning) ───────────────────
    model.cfg.optim = OmegaConf.create({
        "name": "adam",
        "lr": args.lr,
        "weight_decay": 0.0001,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_ratio": 0.1,
            "min_lr": 0.0,
        },
    })

    # ── Trainer ──────────────────────────────────────────────────────
    callbacks = []
    if args.unfreeze_encoder_epoch >= 0:
        callbacks.append(UnfreezeEncoderCallback(args.unfreeze_encoder_epoch))

    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        max_epochs=args.epochs,
        precision="32-true",
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        callbacks=callbacks,
    )

    log_dir = exp_manager(
        trainer,
        cfg=OmegaConf.create({
            "exp_dir": "./experiments",
            "name": f"finetune_{args.pretrained}",
            "create_tensorboard_logger": True,
            "create_checkpoint_callback": True,
            "checkpoint_callback_params": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 2,
                "always_save_nemo": True,
            },
            "resume_if_exists": True,
            "resume_ignore_no_checkpoint": True,
        }),
    )

    # ── Train ────────────────────────────────────────────────────────
    trainer.fit(model)

    # Save final model
    if log_dir is not None:
        model_path = os.path.join(log_dir, "..", f"finetuned_{args.pretrained}.nemo")
        model.save_to(model_path)
        print(f"Saved to {model_path}")


if __name__ == "__main__":
    main()
