"""Finetune pretrained TitaNet/ECAPA on custom data."""

import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.collections.asr.modules import SpeakerDecoder
from nemo.utils.exp_manager import exp_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="titanet_large")
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--num_speakers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    args = parser.parse_args()

    model = EncDecSpeakerLabelModel.from_pretrained(args.pretrained)

    # Replace decoder head for your speakers
    model.decoder = SpeakerDecoder(
        feat_in=model.encoder._feat_out,
        num_classes=args.num_speakers,
        pool_mode="attention",
        emb_sizes=192,
        angular=True,
    )

    model.setup_training_data(OmegaConf.create({
        "manifest_filepath": args.train_manifest, "sample_rate": 16000,
        "batch_size": args.batch_size, "shuffle": True, "labels": None,
    }))
    model.setup_validation_data(OmegaConf.create({
        "manifest_filepath": args.val_manifest, "sample_rate": 16000,
        "batch_size": args.batch_size, "shuffle": False,
    }))

    trainer = pl.Trainer(devices=1, accelerator="auto", max_epochs=args.epochs)
    exp_manager(trainer, cfg=OmegaConf.create({
        "exp_dir": "./experiments", "name": f"finetune_{args.pretrained}",
        "create_tensorboard_logger": True, "create_checkpoint_callback": True,
    }))
    trainer.fit(model)
    model.save_to(f"finetuned_{args.pretrained}.nemo")


if __name__ == "__main__":
    main()
