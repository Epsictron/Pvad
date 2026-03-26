"""Combined loss module for joint AAM-Softmax and prototypical training."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

from .aam_softmax import AAMSoftmax
from .prototypical import PrototypicalLoss

logger = logging.getLogger(__name__)


class CombinedLoss(nn.Module):
    """Combined AAM-Softmax and Prototypical loss.

    Computes a weighted combination of AAM-Softmax and Prototypical losses
    with an optional alpha warmup schedule for transitioning between the
    two losses over training.

    Attributes:
        aam_softmax: The AAM-Softmax loss module.
        prototypical: The Prototypical loss module.
        alpha: Current weighting factor (1.0 = pure AAM-Softmax,
            0.0 = pure Prototypical).
        alpha_start: Initial alpha value for warmup schedule.
        alpha_end: Final alpha value for warmup schedule.
        warmup_epochs: Number of epochs over which to interpolate alpha.
        use_schedule: Whether alpha scheduling is enabled.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initializes CombinedLoss from a configuration object.

        Args:
            config: Configuration containing loss parameters. Expected keys:
                - aam_softmax.num_classes: Number of speaker classes.
                - aam_softmax.embedding_dim: Embedding dimensionality.
                - aam_softmax.margin: Angular margin (default 0.2).
                - aam_softmax.scale: Logit scale (default 30.0).
                - aam_softmax.easy_margin: Easy margin mode (default False).
                - prototypical.distance: Distance metric (default 'cosine').
                - prototypical.n_support: Support set size (default 1).
                - prototypical.temperature: Temperature (default 0.05).
                - alpha: Initial/fixed weighting factor (default 0.5).
                - schedule.enabled: Whether to use alpha schedule (default False).
                - schedule.alpha_start: Starting alpha (default 1.0).
                - schedule.alpha_end: Ending alpha (default 0.5).
                - schedule.warmup_epochs: Epochs for warmup (default 10).
        """
        super().__init__()

        # Build AAM-Softmax loss
        aam_cfg = config.get("aam_softmax", {})
        self.aam_softmax = AAMSoftmax(
            num_classes=aam_cfg.get("num_classes", 1000),
            embedding_dim=aam_cfg.get("embedding_dim", 192),
            margin=aam_cfg.get("margin", 0.2),
            scale=aam_cfg.get("scale", 30.0),
            easy_margin=aam_cfg.get("easy_margin", False),
        )

        # Build Prototypical loss
        proto_cfg = config.get("prototypical", {})
        self.prototypical = PrototypicalLoss(
            distance=proto_cfg.get("distance", "cosine"),
            n_support=proto_cfg.get("n_support", 1),
            temperature=proto_cfg.get("temperature", 0.05),
        )

        # Alpha weighting
        self.alpha = float(config.get("alpha", 0.5))

        # Schedule configuration
        schedule_cfg = config.get("schedule", {})
        self.use_schedule = bool(schedule_cfg.get("enabled", False))
        self.alpha_start = float(schedule_cfg.get("alpha_start", 1.0))
        self.alpha_end = float(schedule_cfg.get("alpha_end", 0.5))
        self.warmup_epochs = int(schedule_cfg.get("warmup_epochs", 10))

        if self.use_schedule:
            self.alpha = self.alpha_start

        logger.info(
            "Initialized CombinedLoss: alpha=%.3f, schedule_enabled=%s",
            self.alpha,
            self.use_schedule,
        )
        if self.use_schedule:
            logger.info(
                "Alpha schedule: start=%.3f, end=%.3f, warmup_epochs=%d",
                self.alpha_start,
                self.alpha_end,
                self.warmup_epochs,
            )

    def forward(
        self, embeddings: Tensor, labels: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Computes the combined loss.

        Args:
            embeddings: Input embeddings of shape (batch_size, embedding_dim).
            labels: Speaker labels of shape (batch_size,).

        Returns:
            A tuple of:
                - total_loss: Weighted combination of AAM-Softmax and
                  Prototypical losses.
                - loss_dict: Dictionary with individual loss values and alpha:
                  {"aam_loss", "proto_loss", "total_loss", "alpha"}.
        """
        aam_loss = self.aam_softmax(embeddings, labels)
        proto_loss = self.prototypical(embeddings, labels)

        total_loss = self.alpha * aam_loss + (1.0 - self.alpha) * proto_loss

        loss_dict = {
            "aam_loss": aam_loss.item(),
            "proto_loss": proto_loss.item(),
            "total_loss": total_loss.item(),
            "alpha": self.alpha,
        }

        return total_loss, loss_dict

    def update_alpha(self, epoch: int) -> None:
        """Updates alpha based on the warmup schedule.

        Performs linear interpolation of alpha from alpha_start to alpha_end
        over warmup_epochs. After warmup_epochs, alpha remains at alpha_end.
        Has no effect if scheduling is disabled.

        Args:
            epoch: Current training epoch (0-indexed).
        """
        if not self.use_schedule:
            return

        if epoch >= self.warmup_epochs:
            self.alpha = self.alpha_end
        else:
            # Linear interpolation from alpha_start to alpha_end
            progress = epoch / max(self.warmup_epochs, 1)
            self.alpha = self.alpha_start + progress * (
                self.alpha_end - self.alpha_start
            )

        logger.debug("Updated alpha to %.4f at epoch %d", self.alpha, epoch)
