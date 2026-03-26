"""Main training loop for speaker embedding models with DDP support."""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..data import SpeakerDataset, collate_fn, load_manifest
from ..features import FeatureExtractor
from ..losses import AAMSoftmax, CombinedLoss, PrototypicalLoss
from ..models import (
    AttentiveStatisticsPooling,
    BaseEncoder,
    DummyEncoder,
    EmbeddingHead,
    StatisticsPooling,
)
from .curriculum import CurriculumScheduler
from .gradient_utils import clip_gradients, setup_gradient_scaler
from .lr_scheduler import build_lr_scheduler

logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates speaker embedding model training.

    Handles model construction, loss selection, optimizer and scheduler
    setup, data loading, mixed-precision training, gradient clipping,
    checkpointing, and optional distributed data-parallel training.

    Args:
        config: Hydra/OmegaConf configuration object containing all
            training hyperparameters.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the trainer.

        Args:
            config: Full Hydra/OmegaConf configuration object.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rank: int = int(os.environ.get("LOCAL_RANK", 0))
        self.is_distributed: bool = config.training.distributed.enable
        self.is_main_process: bool = self.rank == 0

        self.model: Optional[nn.Module] = None
        self.loss_fn: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.curriculum: Optional[CurriculumScheduler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.num_classes: int = 0

    def setup(self) -> None:
        """Build all components required for training.

        Constructs the model (encoder + pooling + embedding head), loss
        function, optimizer (including loss parameters), learning rate
        scheduler, gradient scaler, and data loaders.  Wraps the model
        in :class:`~torch.nn.parallel.DistributedDataParallel` when
        distributed training is enabled.
        """
        if self.is_distributed:
            dist.init_process_group(backend=self.config.training.distributed.backend)
            self.rank = dist.get_rank()
            self.is_main_process = self.rank == 0
            self.device = torch.device("cuda", self.rank)
            torch.cuda.set_device(self.device)
            if self.is_main_process:
                logger.info(
                    "Distributed training initialized: backend=%s, world_size=%d.",
                    self.config.training.distributed.backend,
                    dist.get_world_size(),
                )

        # Build data loaders first so we know num_classes
        self._build_dataloaders()

        # Feature extractor (runs on GPU, no DDP wrapping needed)
        self.feature_extractor = FeatureExtractor(self.config).to(self.device)

        # Model
        self.model = self._build_model().to(self.device)

        # Loss
        self.loss_fn = self._build_loss().to(self.device)

        # DDP wrapping
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
            )
            if self.is_main_process:
                logger.info("Model wrapped with DistributedDataParallel.")

        # Optimizer (includes both model AND loss parameters)
        self.optimizer = self._build_optimizer()

        # LR scheduler
        self.scheduler = build_lr_scheduler(self.optimizer, self.config)

        # Gradient scaler for mixed precision
        self.scaler = setup_gradient_scaler(
            enabled=self.config.training.mixed_precision
        )

        # Curriculum scheduler (if applicable)
        if self.config.training.mode == "curriculum":
            self.curriculum = CurriculumScheduler(self.config)
            if self.is_main_process:
                logger.info("Curriculum training mode enabled.")

        # Optional: torch.compile
        if self.config.training.compile_model:
            self.model = torch.compile(self.model)
            if self.is_main_process:
                logger.info("Model compiled with torch.compile.")

        if self.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            loss_params = sum(p.numel() for p in self.loss_fn.parameters())
            logger.info(
                "Setup complete: model_params=%d, loss_params=%d, device=%s.",
                total_params,
                loss_params,
                self.device,
            )

    def train(self) -> None:
        """Run the full training loop over all epochs.

        For each epoch: runs one training epoch, validates, saves
        checkpoints, and logs metrics.  When curriculum mode is active
        the data loaders are rebuilt on stage boundaries.
        """
        total_epochs: int = self.config.training.epochs
        checkpoint_dir = Path(self.config.checkpoint.dir)
        if self.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.is_main_process:
            logger.info("Starting training for %d epochs.", total_epochs)

        for epoch in range(total_epochs):
            # Curriculum stage transitions
            if self.curriculum is not None and self.curriculum.should_update(epoch):
                self._build_dataloaders()
                if self.is_main_process:
                    logger.info("Data loaders rebuilt for new curriculum stage.")

            # Distributed sampler epoch
            if self.is_distributed and hasattr(self.train_loader, "sampler"):
                sampler = self.train_loader.sampler
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Validate
            val_metrics: Dict[str, float] = {}
            eval_interval: int = self.config.evaluation.interval
            if (epoch + 1) % eval_interval == 0:
                val_metrics = self.validate(epoch)

            # LR scheduler step
            self.scheduler.step()

            # Logging (rank 0 only)
            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch %d/%d - lr=%.6g - train: %s - val: %s",
                    epoch + 1,
                    total_epochs,
                    current_lr,
                    _format_metrics(train_metrics),
                    _format_metrics(val_metrics) if val_metrics else "N/A",
                )

            # Checkpointing (rank 0 only)
            save_every: int = self.config.checkpoint.save_every
            if self.is_main_process and (epoch + 1) % save_every == 0:
                self._save_checkpoint(epoch, train_metrics, val_metrics)

        if self.is_distributed:
            dist.destroy_process_group()
        if self.is_main_process:
            logger.info("Training complete.")

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            Dictionary of training metrics (at minimum ``loss`` and
            ``grad_norm``).
        """
        self.model.train()
        self.loss_fn.train()
        self.feature_extractor.train()

        total_loss = 0.0
        total_grad_norm = 0.0
        num_batches = 0
        accumulation_steps: int = self.config.training.gradient.accumulation_steps
        clip_norm: float = self.config.training.gradient.clip_norm
        log_every: int = self.config.logging.log_every_n_steps

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            audio = batch["audio"].to(self.device, non_blocking=True)
            labels = batch["speaker_id"].to(self.device, non_blocking=True)
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(self.device, non_blocking=True)
            else:
                lengths = torch.full(
                    (audio.size(0),), audio.size(-1), device=self.device, dtype=torch.long
                )

            # Feature extraction on GPU
            with torch.amp.autocast("cuda", enabled=self.config.training.mixed_precision):
                features, feat_lengths = self.feature_extractor(audio, lengths)
                embeddings = self.model(features, feat_lengths)
                loss = self.loss_fn(embeddings, labels)
                loss = loss / accumulation_steps

            # Backward
            self.scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = clip_gradients(self.model, clip_norm)
                # Also clip loss parameters
                if sum(p.numel() for p in self.loss_fn.parameters()) > 0:
                    clip_gradients(self.loss_fn, clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                total_grad_norm += grad_norm
            else:
                grad_norm = 0.0

            total_loss += loss.item() * accumulation_steps
            num_batches += 1

            # Periodic logging
            if self.is_main_process and (step + 1) % log_every == 0:
                logger.info(
                    "Epoch %d - step %d/%d - batch_loss=%.4f - grad_norm=%.4f",
                    epoch + 1,
                    step + 1,
                    len(self.train_loader),
                    loss.item() * accumulation_steps,
                    grad_norm,
                )

        avg_loss = total_loss / max(num_batches, 1)
        optimizer_steps = max(num_batches // accumulation_steps, 1)
        avg_grad_norm = total_grad_norm / optimizer_steps

        return {"loss": avg_loss, "grad_norm": avg_grad_norm}

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Evaluate the model on the validation set.

        Args:
            epoch: Current epoch number (0-based).

        Returns:
            Dictionary of validation metrics (at minimum ``loss``).
            Returns an empty dict when no validation data is configured.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.loss_fn.eval()
        self.feature_extractor.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            audio = batch["audio"].to(self.device, non_blocking=True)
            labels = batch["speaker_id"].to(self.device, non_blocking=True)
            lengths = batch.get("lengths")
            if lengths is not None:
                lengths = lengths.to(self.device, non_blocking=True)
            else:
                lengths = torch.full(
                    (audio.size(0),), audio.size(-1), device=self.device, dtype=torch.long
                )

            with torch.amp.autocast("cuda", enabled=self.config.training.mixed_precision):
                features, feat_lengths = self.feature_extractor(audio, lengths)
                embeddings = self.model(features, feat_lengths)
                loss = self.loss_fn(embeddings, labels)

            total_loss += loss.item()
            num_batches += 1

        metrics: Dict[str, float] = {}
        if num_batches > 0:
            metrics["loss"] = total_loss / num_batches

        if self.is_main_process:
            logger.info(
                "Validation at epoch %d: %s",
                epoch + 1,
                _format_metrics(metrics),
            )

        return metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> nn.Module:
        """Construct the full model pipeline (encoder + pooling + head).

        Returns:
            An :class:`nn.Module` that accepts ``(features, lengths)``
            and returns embeddings of shape ``(B, embedding_dim)``.
        """
        model_name: str = self.config.model.name
        embedding_dim: int = self.config.model.embedding_dim

        # Encoder
        if model_name == "dummy":
            encoder = DummyEncoder(self.config)
        else:
            # For production models (ecapa_tdnn, resnet34, custom) the
            # encoder registry would be consulted here.
            raise ValueError(
                f"Unknown model name '{model_name}'. "
                "Expected one of: dummy. Register additional encoders as needed."
            )

        encoder_out_dim: int = encoder.output_dim

        # Pooling
        pooling = AttentiveStatisticsPooling(input_dim=encoder_out_dim)
        pooled_dim = encoder_out_dim * 2  # statistics pooling doubles dim

        # Embedding head
        head = EmbeddingHead(
            input_dim=pooled_dim,
            embedding_dim=embedding_dim,
        )

        model = _SpeakerModel(encoder=encoder, pooling=pooling, head=head)

        if self.is_main_process:
            logger.info(
                "Model built: encoder=%s (out_dim=%d), pooling=%s, "
                "embedding_dim=%d.",
                model_name,
                encoder_out_dim,
                pooling.__class__.__name__,
                embedding_dim,
            )

        return model

    def _build_loss(self) -> nn.Module:
        """Build the loss function based on configuration.

        Returns:
            A loss module that accepts ``(embeddings, labels)`` and
            returns a scalar loss.
        """
        loss_type: str = self.config.loss.type

        if loss_type == "aam":
            loss_fn = AAMSoftmax(
                num_classes=self.num_classes,
                embedding_dim=self.config.model.embedding_dim,
                margin=self.config.loss.margin,
                scale=self.config.loss.scale,
                easy_margin=self.config.loss.easy_margin,
            )
        elif loss_type == "proto":
            loss_fn = PrototypicalLoss(
                distance=self.config.loss.distance,
                n_support=self.config.loss.n_support,
                temperature=self.config.loss.temperature,
            )
        elif loss_type == "combined":
            loss_fn = CombinedLoss(
                alpha=self.config.loss.alpha,
                aam_config=self.config.loss.aam,
                proto_config=self.config.loss.proto,
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(
                f"Unknown loss type '{loss_type}'. "
                "Expected one of: aam, proto, combined."
            )

        if self.is_main_process:
            logger.info("Loss function built: %s.", loss_type)

        return loss_fn

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build the optimizer for both model and loss parameters.

        Returns:
            A PyTorch optimizer.
        """
        opt_cfg = self.config.training.optimizer
        name: str = opt_cfg.name

        # Combine model and loss parameters (AAM weights are learnable)
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())

        if name == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
            )
        elif name == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                betas=tuple(opt_cfg.betas),
                weight_decay=opt_cfg.weight_decay,
            )
        elif name == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer name '{name}'. "
                "Expected one of: adam, adamw, sgd."
            )

        if self.is_main_process:
            logger.info(
                "Optimizer built: %s (lr=%g, weight_decay=%g).",
                name,
                opt_cfg.lr,
                opt_cfg.weight_decay,
            )

        return optimizer

    def _build_dataloaders(self) -> None:
        """Create training and validation data loaders.

        Loads manifest files, builds :class:`SpeakerDataset` instances,
        and wraps them in :class:`DataLoader`.  When curriculum mode is
        active, the segment length from the current stage is applied.
        """
        data_cfg = self.config.data

        # Training manifest
        train_entries = load_manifest(
            path=data_cfg.manifest_path,
            min_duration=data_cfg.min_duration,
            max_duration=data_cfg.max_duration,
        )
        train_dataset = SpeakerDataset(train_entries, self.config)
        self.num_classes = len(train_dataset.speaker_to_idx)

        # Sampler
        train_sampler: Optional[DistributedSampler] = None
        shuffle = True
        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_dataset, shuffle=True
            )
            shuffle = False

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=data_cfg.num_workers,
            pin_memory=data_cfg.pin_memory,
            collate_fn=collate_fn,
            drop_last=True,
        )

        if self.is_main_process:
            logger.info(
                "Train dataloader: %d samples, %d batches, %d classes.",
                len(train_dataset),
                len(self.train_loader),
                self.num_classes,
            )

        # Validation manifest (optional)
        val_manifest_path = data_cfg.val_manifest_path
        if val_manifest_path is not None:
            val_entries = load_manifest(
                path=val_manifest_path,
                min_duration=data_cfg.min_duration,
                max_duration=data_cfg.max_duration,
            )
            val_dataset = SpeakerDataset(val_entries, self.config)

            val_sampler: Optional[DistributedSampler] = None
            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_dataset, shuffle=False
                )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.training.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                collate_fn=collate_fn,
                drop_last=False,
            )

            if self.is_main_process:
                logger.info(
                    "Val dataloader: %d samples, %d batches.",
                    len(val_dataset),
                    len(self.val_loader),
                )
        else:
            self.val_loader = None

    def _save_checkpoint(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Save a training checkpoint to disk.

        Args:
            epoch: Current epoch number (0-based).
            train_metrics: Training metrics for this epoch.
            val_metrics: Validation metrics for this epoch.
        """
        checkpoint_dir = Path(self.config.checkpoint.dir)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"

        # Unwrap DDP if necessary
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, DDP)
            else self.model.state_dict()
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model_state,
            "loss_state_dict": self.loss_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info("Checkpoint saved: %s", checkpoint_path)


class _SpeakerModel(nn.Module):
    """Wrapper combining encoder, pooling, and embedding head.

    This internal module sequences the three components so that DDP
    wrapping and :func:`torch.compile` work on a single ``nn.Module``.

    Args:
        encoder: A :class:`BaseEncoder` producing frame-level features.
        pooling: A pooling module reducing frames to a single vector.
        head: An :class:`EmbeddingHead` projecting to the embedding space.
    """

    def __init__(
        self,
        encoder: BaseEncoder,
        pooling: nn.Module,
        head: EmbeddingHead,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooling = pooling
        self.head = head

    def forward(
        self, features: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: encode, pool, and project.

        Args:
            features: Input features of shape ``(B, C, T)``.
            lengths: Valid lengths for each sample ``(B,)``.

        Returns:
            Speaker embeddings of shape ``(B, embedding_dim)``.
        """
        encoded, enc_lengths = self.encoder(features, lengths)
        pooled = self.pooling(encoded, enc_lengths)
        embeddings = self.head(pooled)
        return embeddings


def _format_metrics(metrics: Dict[str, float]) -> str:
    """Format a metrics dictionary as a human-readable string.

    Args:
        metrics: Mapping from metric name to value.

    Returns:
        Comma-separated key=value string.
    """
    if not metrics:
        return "{}"
    return ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
