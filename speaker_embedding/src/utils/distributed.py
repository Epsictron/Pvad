"""Distributed Data Parallel (DDP) setup utilities.

Provides helpers for initialising and tearing down PyTorch distributed
process groups, as well as rank / world-size queries that gracefully
fall back to single-GPU defaults.
"""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed(backend: str = "nccl") -> int:
    """Initialise the distributed process group and set the CUDA device.

    Args:
        backend: Communication backend to use.  ``"nccl"`` is recommended
            for GPU training; ``"gloo"`` can be used for CPU-only runs.

    Returns:
        The local rank of the current process.

    Raises:
        RuntimeError: If required environment variables (``RANK``,
            ``WORLD_SIZE``, ``LOCAL_RANK``) are not set.
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )
        logger.info(
            "Distributed process group initialised: rank=%d, world_size=%d, "
            "local_rank=%d, backend=%s",
            rank,
            world_size,
            local_rank,
            backend,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info("CUDA device set to local_rank=%d", local_rank)

    return local_rank


def cleanup_distributed() -> None:
    """Destroy the distributed process group if it is initialised."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


def is_main_process() -> bool:
    """Return ``True`` if the current process is rank 0 or not distributed."""
    return get_rank() == 0


def get_rank() -> int:
    """Return the global rank of the current process.

    Returns:
        ``0`` when running without a distributed process group.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Return the world size of the current distributed group.

    Returns:
        ``1`` when running without a distributed process group.
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1
