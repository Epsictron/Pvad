"""Logging utilities for training runs.

Provides structured logging to console and file, TensorBoard integration,
CSV metric logging, and optional Weights & Biases (W&B) support.
"""

import csv
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard Python logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, rank: int = 0) -> logging.Logger:
    """Configure file and console logging.

    Only rank 0 emits messages to the console and log file; other ranks
    are set to ``WARNING`` to avoid duplicated output.

    Args:
        log_dir: Directory where ``train.log`` will be written.
        rank: Distributed rank of the current process.

    Returns:
        The root ``logging.Logger`` instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)

    # Remove any pre-existing handlers to avoid duplicate output.
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        # Console handler.
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler.
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "train.log"), mode="a"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger


# ---------------------------------------------------------------------------
# TensorBoard logger wrapper
# ---------------------------------------------------------------------------

class TBLogger:
    """Thin wrapper around TensorBoard ``SummaryWriter``.

    Provides a concise API for the most common logging operations and
    silently no-ops when ``log_dir`` is ``None`` (e.g. non-main ranks).

    Args:
        log_dir: Directory for TensorBoard event files.  When ``None``
            all write operations become no-ops.
    """

    def __init__(self, log_dir: Optional[str] = None) -> None:
        self._writer = None
        if log_dir is not None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(log_dir, exist_ok=True)
                self._writer = SummaryWriter(log_dir=log_dir)
                logger.info("TensorBoard logging to %s", log_dir)
            except ImportError:
                logger.warning(
                    "tensorboard is not installed; TBLogger will be a no-op."
                )

    # -- public API ---------------------------------------------------------

    def log_scalar(
        self, tag: str, value: float, step: int
    ) -> None:
        """Log a single scalar value.

        Args:
            tag: Identifier for the scalar (e.g. ``"train/loss"``).
            value: Scalar value.
            step: Global step number.
        """
        if self._writer is not None:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalars under a common group.

        Args:
            main_tag: Group name (e.g. ``"learning_rate"``).
            tag_scalar_dict: Mapping of sub-tag to scalar value.
            step: Global step number.
        """
        if self._writer is not None:
            self._writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(
        self, tag: str, values: Any, step: int, bins: str = "tensorflow"
    ) -> None:
        """Log a histogram of values.

        Args:
            tag: Identifier for the histogram.
            values: Values to build the histogram from (tensor or array).
            step: Global step number.
            bins: Binning strategy (default ``"tensorflow"``).
        """
        if self._writer is not None:
            self._writer.add_histogram(tag, values, step, bins=bins)

    def close(self) -> None:
        """Flush and close the underlying ``SummaryWriter``."""
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            logger.info("TBLogger closed.")


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

class CSVLogger:
    """Append-only CSV logger for per-epoch metrics.

    Args:
        log_path: Path to the CSV file.  Parent directories are created
            automatically.
    """

    def __init__(self, log_path: str) -> None:
        self._path = log_path
        self._file = None
        self._writer: Optional[csv.DictWriter] = None
        self._fieldnames: Optional[list] = None

        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        logger.info("CSVLogger will write to %s", log_path)

    # -- public API ---------------------------------------------------------

    def log(self, epoch: int, metrics: Dict[str, Union[int, float]]) -> None:
        """Append a row of metrics for the given epoch.

        On the first call the CSV header is written based on the keys of
        *metrics*.  Subsequent calls must provide the same set of keys.

        Args:
            epoch: Epoch number.
            metrics: Mapping of metric names to their values.
        """
        row = {"epoch": epoch, **metrics}

        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._file = open(self._path, mode="a", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            # Write header only if the file is empty / new.
            if os.path.getsize(self._path) == 0:
                self._writer.writeheader()

        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            logger.info("CSVLogger closed.")


# ---------------------------------------------------------------------------
# Optional Weights & Biases integration
# ---------------------------------------------------------------------------

def setup_wandb(config: Any) -> None:
    """Initialise a Weights & Biases run if enabled in the config.

    The function expects ``config.wandb.enable`` to be a boolean flag.
    When ``True``, it attempts to import ``wandb`` and initialise a run
    using fields from ``config.wandb`` (project, entity, name, tags, etc.).

    Args:
        config: Run configuration (OmegaConf ``DictConfig`` or plain
            object with attribute access).  Must have a ``wandb``
            sub-config with at least an ``enable`` field.
    """
    try:
        wandb_cfg = config.wandb
    except AttributeError:
        logger.debug("No wandb configuration found; skipping W&B setup.")
        return

    if not getattr(wandb_cfg, "enable", False):
        logger.debug("W&B disabled in config; skipping.")
        return

    try:
        import wandb  # type: ignore[import-untyped]
    except ImportError:
        logger.warning(
            "wandb.enable is True but the 'wandb' package is not installed."
        )
        return

    init_kwargs: Dict[str, Any] = {
        "project": getattr(wandb_cfg, "project", "speaker-embedding"),
        "config": (
            dict(config) if isinstance(config, dict) else
            config  # OmegaConf will be handled by wandb
        ),
    }
    if getattr(wandb_cfg, "entity", None):
        init_kwargs["entity"] = wandb_cfg.entity
    if getattr(wandb_cfg, "name", None):
        init_kwargs["name"] = wandb_cfg.name
    if getattr(wandb_cfg, "tags", None):
        init_kwargs["tags"] = list(wandb_cfg.tags)

    wandb.init(**init_kwargs)
    logger.info(
        "W&B run initialised: project=%s, name=%s",
        init_kwargs.get("project"),
        init_kwargs.get("name", "(auto)"),
    )
