"""Checkpoint save / load utilities.

Persists full training state -- model weights, loss-module parameters,
optimiser, scheduler, gradient scaler, RNG states, and the run
configuration -- so that training can be resumed exactly.
"""

import logging
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _collect_rng_states() -> Dict[str, Any]:
    """Snapshot Python, NumPy, Torch CPU, and Torch CUDA RNG states."""
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def _restore_rng_states(rng_states: Dict[str, Any]) -> None:
    """Restore RNG states previously collected by ``_collect_rng_states``."""
    if "python" in rng_states:
        random.setstate(rng_states["python"])
    if "numpy" in rng_states:
        np.random.set_state(rng_states["numpy"])
    if "torch_cpu" in rng_states:
        torch.random.set_rng_state(rng_states["torch_cpu"])
    if "torch_cuda" in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states["torch_cuda"])


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    loss_module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    best_metric: float,
    config: Any,
    rng_states: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a full training checkpoint to disk.

    Args:
        path: Destination file path (e.g. ``checkpoints/epoch_5.pt``).
        model: The model (or ``DistributedDataParallel`` wrapper).
        loss_module: Loss / criterion module with learnable parameters
            (e.g. AAM-Softmax).
        optimizer: Optimiser instance.
        scheduler: Learning-rate scheduler instance.
        scaler: ``GradScaler`` used for mixed-precision training, or
            ``None`` if unused.
        epoch: Current epoch number (0-based).
        step: Global training step counter.
        best_metric: Best validation metric observed so far.
        config: Run configuration (OmegaConf ``DictConfig`` or plain dict).
            Stored as a snapshot for reproducibility.
        rng_states: Pre-collected RNG states.  When ``None`` (the default)
            the function snapshots them automatically.
    """
    # Unwrap DDP if necessary.
    model_state = (
        model.module.state_dict()
        if hasattr(model, "module")
        else model.state_dict()
    )

    loss_state = (
        loss_module.module.state_dict()
        if hasattr(loss_module, "module")
        else loss_module.state_dict()
    )

    checkpoint: Dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "best_metric": best_metric,
        "model_state_dict": model_state,
        "loss_module_state_dict": loss_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "config": config,
        "rng_states": rng_states if rng_states is not None else _collect_rng_states(),
    }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(
        "Checkpoint saved: %s  (epoch=%d, step=%d, best_metric=%.6f)",
        path,
        epoch,
        step,
        best_metric,
    )


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    loss_module: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """Load a checkpoint and restore training state.

    Only the *model* weights are mandatory; the remaining components are
    restored only when the corresponding argument is not ``None``.

    Args:
        path: Path to the checkpoint file.
        model: Model instance to load weights into.
        loss_module: Optional loss module to restore.
        optimizer: Optional optimiser to restore.
        scheduler: Optional scheduler to restore.
        scaler: Optional ``GradScaler`` to restore.

    Returns:
        A metadata dictionary with keys ``epoch``, ``step``,
        ``best_metric``, and ``config``.
    """
    checkpoint: Dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)

    # Restore model (handle DDP wrapper).
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(checkpoint["model_state_dict"])

    if loss_module is not None and "loss_module_state_dict" in checkpoint:
        target_loss = (
            loss_module.module
            if hasattr(loss_module, "module")
            else loss_module
        )
        target_loss.load_state_dict(checkpoint["loss_module_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    # Restore RNG states when available.
    if "rng_states" in checkpoint and checkpoint["rng_states"] is not None:
        _restore_rng_states(checkpoint["rng_states"])
        logger.info("RNG states restored from checkpoint.")

    metadata: Dict[str, Any] = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "config": checkpoint.get("config"),
    }

    logger.info(
        "Checkpoint loaded: %s  (epoch=%d, step=%d, best_metric=%.6f)",
        path,
        metadata["epoch"],
        metadata["step"],
        metadata["best_metric"],
    )
    return metadata


def find_best_checkpoint(
    checkpoint_dir: str,
    monitor: str = "eer",
    mode: str = "min",
) -> Optional[str]:
    """Scan a directory for the best checkpoint file based on a metric.

    Checkpoint filenames are expected to contain the monitored metric as
    ``<monitor>=<value>`` (e.g. ``epoch_5_eer=0.0321.pt``).

    Args:
        checkpoint_dir: Directory to scan.
        monitor: Metric name to look for in filenames.
        mode: ``"min"`` to select the lowest value, ``"max"`` for the
            highest.

    Returns:
        Absolute path to the best checkpoint, or ``None`` if no matching
        files are found.
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.is_dir():
        logger.warning("Checkpoint directory does not exist: %s", checkpoint_dir)
        return None

    pattern = re.compile(rf"{re.escape(monitor)}=([\d.eE+\-]+)")
    best_path: Optional[str] = None
    best_value: Optional[float] = None

    for fpath in ckpt_dir.iterdir():
        if not fpath.is_file():
            continue
        match = pattern.search(fpath.name)
        if match is None:
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue

        if best_value is None:
            best_value = value
            best_path = str(fpath)
        elif mode == "min" and value < best_value:
            best_value = value
            best_path = str(fpath)
        elif mode == "max" and value > best_value:
            best_value = value
            best_path = str(fpath)

    if best_path is not None:
        logger.info(
            "Best checkpoint found: %s (%s=%.6f)",
            best_path,
            monitor,
            best_value,
        )
    else:
        logger.warning(
            "No checkpoint matching monitor='%s' found in %s",
            monitor,
            checkpoint_dir,
        )

    return best_path
