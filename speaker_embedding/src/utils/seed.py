"""Reproducibility utilities for seeding all random number generators."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set the random seed for all relevant RNGs and enable determinism flags.

    Seeds Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices).
    Additionally sets ``torch.backends.cudnn.deterministic``,
    ``torch.backends.cudnn.benchmark``, and the ``CUBLAS_WORKSPACE_CONFIG``
    environment variable for fully deterministic cuBLAS behaviour.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic algorithm selection.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    # cuBLAS workspace configuration for deterministic reductions.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    logger.info(
        "Random seed set to %d (deterministic mode enabled).", seed
    )


def worker_init_fn(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` that seeds each worker uniquely.

    Each worker receives a seed offset of ``base_seed + worker_id`` where
    ``base_seed`` is derived from PyTorch's initial seed for that worker.
    This avoids duplicated augmentation patterns across workers.

    Args:
        worker_id: Worker index provided by the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
