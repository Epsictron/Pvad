"""Curriculum learning scheduler for progressive speaker embedding training.

Controls stage transitions that adjust segment length, augmentation
probability, and speakers-per-batch as training progresses.
"""

import logging
from typing import Dict, List

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class CurriculumScheduler:
    """Manages curriculum learning stages during training.

    Reads stage definitions from ``config.training.curriculum.stages``
    and provides accessors that return the appropriate parameter values
    for a given epoch.  Each stage specifies a name, a number of epochs,
    a segment length, an augmentation probability, and the number of
    speakers per batch.

    Args:
        config: Hydra/OmegaConf configuration.  Expected structure::

            training:
              curriculum:
                stages:
                  - name: "easy"
                    epochs: 50
                    segment_length: 4.0
                    augmentation_prob: 0.1
                    speakers_per_batch: 8
                  - name: "medium"
                    epochs: 50
                    ...
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the curriculum scheduler from configuration.

        Args:
            config: Full Hydra/OmegaConf configuration object.
        """
        stages_cfg = config.training.curriculum.stages
        self.stages: List[Dict] = []
        cumulative_epoch = 0

        for stage in stages_cfg:
            start_epoch = cumulative_epoch
            end_epoch = cumulative_epoch + stage.epochs
            self.stages.append(
                {
                    "name": stage.name,
                    "start_epoch": start_epoch,
                    "end_epoch": end_epoch,
                    "segment_length": stage.segment_length,
                    "augmentation_prob": stage.augmentation_prob,
                    "speakers_per_batch": stage.speakers_per_batch,
                }
            )
            cumulative_epoch = end_epoch

        self._previous_stage_name: str = ""

        logger.info(
            "CurriculumScheduler initialized with %d stages: %s",
            len(self.stages),
            [s["name"] for s in self.stages],
        )

    def get_stage(self, epoch: int) -> Dict:
        """Return the stage configuration for a given epoch.

        Args:
            epoch: Current training epoch (0-based).

        Returns:
            Dictionary with keys ``name``, ``start_epoch``, ``end_epoch``,
            ``segment_length``, ``augmentation_prob``, and
            ``speakers_per_batch``.
        """
        for stage in self.stages:
            if stage["start_epoch"] <= epoch < stage["end_epoch"]:
                return stage
        # If epoch exceeds all stages, return the last stage.
        return self.stages[-1]

    def get_segment_length(self, epoch: int) -> float:
        """Return the segment length in seconds for the given epoch.

        Args:
            epoch: Current training epoch (0-based).

        Returns:
            Segment length in seconds.
        """
        return self.get_stage(epoch)["segment_length"]

    def get_augmentation_prob(self, epoch: int) -> float:
        """Return the augmentation probability for the given epoch.

        Args:
            epoch: Current training epoch (0-based).

        Returns:
            Probability of applying augmentation (0.0 to 1.0).
        """
        return self.get_stage(epoch)["augmentation_prob"]

    def get_speakers_per_batch(self, epoch: int) -> int:
        """Return the number of speakers per batch for the given epoch.

        Args:
            epoch: Current training epoch (0-based).

        Returns:
            Number of speakers to include in each batch.
        """
        return self.get_stage(epoch)["speakers_per_batch"]

    def should_update(self, epoch: int) -> bool:
        """Check whether a stage boundary has been crossed.

        Returns ``True`` on the first call for each new stage, which
        allows the caller to rebuild data loaders or adjust settings.

        Args:
            epoch: Current training epoch (0-based).

        Returns:
            ``True`` if the current stage differs from the previous one.
        """
        current_stage = self.get_stage(epoch)
        stage_name = current_stage["name"]

        if stage_name != self._previous_stage_name:
            logger.info(
                "Curriculum stage transition at epoch %d: '%s' -> '%s' "
                "(segment_length=%.2f, augmentation_prob=%.2f, "
                "speakers_per_batch=%d).",
                epoch,
                self._previous_stage_name or "(none)",
                stage_name,
                current_stage["segment_length"],
                current_stage["augmentation_prob"],
                current_stage["speakers_per_batch"],
            )
            self._previous_stage_name = stage_name
            return True
        return False
