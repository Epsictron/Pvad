"""Learning rate scheduler factory.

Builds a PyTorch learning rate scheduler from the Hydra/OmegaConf
configuration, supporting warmup, cosine annealing, cyclic, one-cycle,
and step-based schedules.
"""

import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CyclicLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
    StepLR,
    _LRScheduler,
)
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def build_lr_scheduler(optimizer: Optimizer, config: DictConfig) -> _LRScheduler:
    """Build a learning rate scheduler from configuration.

    Dispatches on ``config.training.lr_scheduler.name`` to create one of
    several supported schedules.  The ``"warmup"`` and ``"cosine"``
    variants automatically prepend a linear warmup phase using
    :class:`~torch.optim.lr_scheduler.SequentialLR`.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        config: Full Hydra/OmegaConf configuration.  Expected keys under
            ``config.training.lr_scheduler``:

            - **name** (``str``): One of ``"warmup"``, ``"cosine"``,
              ``"cyclic"``, ``"one_cycle"``, ``"step"``.
            - **warmup_epochs** (``int``): Number of warmup epochs
              (used by ``"warmup"`` and ``"cosine"``).
            - **min_lr** (``float``): Minimum LR for cosine annealing.
            - **step_size** (``int``): Period for ``StepLR``.
            - **gamma** (``float``): Multiplicative factor for ``StepLR``.
            - **max_lr** (``float``): Peak LR for ``"cyclic"`` and
              ``"one_cycle"`` schedules.

    Returns:
        A PyTorch learning rate scheduler instance.

    Raises:
        ValueError: If the scheduler name is not recognized.
    """
    sched_cfg = config.training.lr_scheduler
    name: str = sched_cfg.name
    warmup_epochs: int = sched_cfg.warmup_epochs
    total_epochs: int = config.training.epochs

    if name == "warmup":
        scheduler = _build_warmup(optimizer, warmup_epochs)
        logger.info(
            "LR scheduler: linear warmup for %d epochs then constant.",
            warmup_epochs,
        )

    elif name == "cosine":
        scheduler = _build_cosine_with_warmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=sched_cfg.min_lr,
        )
        logger.info(
            "LR scheduler: linear warmup (%d epochs) + cosine annealing "
            "(T_max=%d, min_lr=%g).",
            warmup_epochs,
            total_epochs - warmup_epochs,
            sched_cfg.min_lr,
        )

    elif name == "cyclic":
        scheduler = CyclicLR(
            optimizer,
            base_lr=config.training.optimizer.lr,
            max_lr=sched_cfg.max_lr,
            step_size_up=sched_cfg.step_size,
            mode="triangular2",
            cycle_momentum=False,
        )
        logger.info(
            "LR scheduler: CyclicLR (base_lr=%g, max_lr=%g, step_size=%d).",
            config.training.optimizer.lr,
            sched_cfg.max_lr,
            sched_cfg.step_size,
        )

    elif name == "one_cycle":
        scheduler = OneCycleLR(
            optimizer,
            max_lr=sched_cfg.max_lr,
            total_steps=total_epochs,
            pct_start=warmup_epochs / total_epochs if total_epochs > 0 else 0.3,
            anneal_strategy="cos",
            final_div_factor=sched_cfg.max_lr / sched_cfg.min_lr
            if sched_cfg.min_lr > 0
            else 1e4,
        )
        logger.info(
            "LR scheduler: OneCycleLR (max_lr=%g, total_steps=%d).",
            sched_cfg.max_lr,
            total_epochs,
        )

    elif name == "step":
        scheduler = StepLR(
            optimizer,
            step_size=sched_cfg.step_size,
            gamma=sched_cfg.gamma,
        )
        logger.info(
            "LR scheduler: StepLR (step_size=%d, gamma=%g).",
            sched_cfg.step_size,
            sched_cfg.gamma,
        )

    else:
        raise ValueError(
            f"Unknown LR scheduler name '{name}'. "
            "Expected one of: warmup, cosine, cyclic, one_cycle, step."
        )

    return scheduler


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _build_warmup(optimizer: Optimizer, warmup_epochs: int) -> _LRScheduler:
    """Build a linear warmup scheduler that holds constant after warmup.

    Args:
        optimizer: Target optimizer.
        warmup_epochs: Number of epochs for the linear ramp.

    Returns:
        A :class:`~torch.optim.lr_scheduler.LinearLR` that ramps from
        a fraction of the base LR to the full base LR.
    """
    return LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )


def _build_cosine_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr: float,
) -> _LRScheduler:
    """Build a warmup + cosine annealing scheduler using SequentialLR.

    Args:
        optimizer: Target optimizer.
        warmup_epochs: Number of epochs for the linear warmup phase.
        total_epochs: Total number of training epochs.
        min_lr: Minimum learning rate at the end of cosine annealing.

    Returns:
        A :class:`~torch.optim.lr_scheduler.SequentialLR` combining
        ``LinearLR`` (warmup) and ``CosineAnnealingLR`` (decay).
    """
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / max(warmup_epochs, 1),
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return scheduler
