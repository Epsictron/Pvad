"""Gradient utilities for mixed-precision training and gradient management."""

import logging

import torch
import torch.nn as nn
from torch.amp import GradScaler

logger = logging.getLogger(__name__)


def clip_gradients(model: nn.Module, max_norm: float) -> float:
    """Clip gradients by global norm.

    Applies :func:`torch.nn.utils.clip_grad_norm_` to all parameters of
    *model* and returns the total gradient norm before clipping.

    Args:
        model: The model whose gradients should be clipped.
        max_norm: Maximum allowed global norm of the gradients.

    Returns:
        The total gradient norm (before clipping) as a float.
    """
    total_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_norm
    )
    total_norm_value = total_norm.item()
    logger.debug("Gradient norm: %.4f (max_norm=%.4f)", total_norm_value, max_norm)
    return total_norm_value


def setup_gradient_scaler(enabled: bool) -> GradScaler:
    """Create a GradScaler for mixed-precision training.

    Args:
        enabled: Whether the scaler should actually scale gradients.
            When ``False`` the scaler is created in a no-op state so that
            the training loop can use it unconditionally.

    Returns:
        A :class:`torch.amp.GradScaler` instance.
    """
    scaler = GradScaler("cuda", enabled=enabled)
    logger.info("GradScaler created (enabled=%s).", enabled)
    return scaler


def accumulate_gradients(
    loss: torch.Tensor,
    scaler: GradScaler,
    accumulation_steps: int,
    step: int,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Handle gradient accumulation with mixed-precision support.

    Scales the loss by ``1 / accumulation_steps`` before the backward pass
    so that the effective gradient is the mean over the accumulation window.
    The optimizer is stepped only every *accumulation_steps* iterations.

    Args:
        loss: Scalar loss tensor for the current micro-batch.
        scaler: A :class:`torch.amp.GradScaler` (may be in no-op mode).
        accumulation_steps: Number of micro-batches to accumulate before
            an optimizer step.
        step: The current micro-batch index within the epoch (0-based).
        optimizer: The optimizer to step when accumulation is complete.
    """
    scaled_loss = loss / accumulation_steps
    scaler.scale(scaled_loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        logger.debug(
            "Optimizer stepped at micro-batch %d (accumulation_steps=%d).",
            step,
            accumulation_steps,
        )
