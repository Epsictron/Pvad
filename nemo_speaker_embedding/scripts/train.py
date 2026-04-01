"""Train — thin wrapper around NeMo's speaker_reco.py recipe."""

import sys
from pathlib import Path
from pytorch_lightning import seed_everything

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytorch_lightning as pl
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

seed_everything(42)


@hydra_runner(config_path="../conf", config_name="speaker_model")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)
    model.save_to("speaker_model.nemo")


if __name__ == "__main__":
    main()
