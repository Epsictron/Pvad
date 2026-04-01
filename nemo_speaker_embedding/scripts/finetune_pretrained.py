"""Finetune a pretrained NeMo speaker model (TitaNet/ECAPA) on custom data.

Usage:
    cd nemo_speaker_embedding
    python scripts/finetune_pretrained.py \
        --pretrained titanet_large \
        --train_manifest dummy_data/train_manifest.json \
        --val_manifest dummy_data/val_manifest.json \
        --num_speakers 10 \
        --epochs 20
"""

import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.utils.exp_manager import exp_manager


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="titanet_large",
                        help="NeMo pretrained model name")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--num_speakers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    # ── Load pretrained ──────────────────────────────────────────────
    model = EncDecSpeakerLabelModel.from_pretrained(args.pretrained)

    # ── Update decoder for new speaker count ─────────────────────────
    model.cfg.decoder.num_classes = args.num_speakers
    from nemo.collections.asr.modules import SpeakerDecoder
    model.decoder = SpeakerDecoder(
        feat_in=model.decoder._feat_in,
        num_classes=args.num_speakers,
        pool_mode="xvector",
        emb_sizes=list(model.cfg.decoder.emb_sizes),
    )

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

    # ── Update optimizer ─────────────────────────────────────────────
    model.cfg.optim.lr = args.lr

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = pl.Trainer(
        devices=1,
        accelerator="auto",
        max_epochs=args.epochs,
        precision="32-true",
        log_every_n_steps=10,
    )

    exp_manager(
        trainer,
        cfg={
            "exp_dir": "./experiments",
            "name": f"finetune_{args.pretrained}",
            "checkpoint_callback_params": {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 2,
                "always_save_nemo": True,
            },
            "resume_if_exists": True,
            "resume_ignore_no_checkpoint": True,
            "create_tensorboard_logger": True,
        },
    )

    trainer.fit(model)
    print("Finetuning complete.")


if __name__ == "__main__":
    main()
