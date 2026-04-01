# Copied from NeMo speaker_reco.py.
# Only addition: swap encoder after model init (NeMo blocks custom _target_).

import os, sys, importlib.util
from pathlib import Path

# Load custom encoder via absolute path (Hydra changes cwd, so sys.path won't work)
_encoder_path = Path(__file__).resolve().parent.parent / "custom_model" / "encoder.py"
_spec = importlib.util.spec_from_file_location("custom_model.encoder", _encoder_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CustomSpeakerEncoder = _mod.CustomSpeakerEncoder

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from nemo.collections.asr.models import EncDecSpeakerLabelModel
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

seed_everything(42)


@hydra_runner(config_path="../conf", config_name="speaker_model")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    log_dir = exp_manager(trainer, cfg.get("exp_manager", None))
    speaker_model = EncDecSpeakerLabelModel(cfg=cfg.model, trainer=trainer)

    # Swap encoder to custom model (NeMo _target_ only allows approved namespaces)
    speaker_model.encoder = CustomSpeakerEncoder(
        feat_in=cfg.model.preprocessor.features,
        feat_out=cfg.model.decoder.feat_in,
    )

    if log_dir is not None:
        with open(os.path.join(log_dir, 'labels.txt'), 'w') as f:
            if speaker_model.labels is not None:
                for label in speaker_model.labels:
                    f.write(f'{label}\n')

    trainer.fit(speaker_model)

    if not trainer.fast_dev_run:
        speaker_model.save_to(os.path.join(log_dir, '..', 'spkr.nemo'))


if __name__ == '__main__':
    main()
