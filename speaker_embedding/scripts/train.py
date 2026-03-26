#!/usr/bin/env python
"""Training entry point for speaker embedding models."""
import hydra
from omegaconf import DictConfig
from speaker_embedding.src.training.trainer import Trainer
from speaker_embedding.src.utils.seed import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="train_normal")
def main(config: DictConfig) -> None:
    set_seed(config.seed)
    trainer = Trainer(config)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
