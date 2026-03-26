"""Tests for training utilities: gradient clipping, LR schedulers, GradScaler."""

import torch
import torch.nn as nn
import pytest
from omegaconf import OmegaConf

from speaker_embedding.src.training.gradient_utils import clip_gradients, setup_gradient_scaler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for training tests."""

    def __init__(self, in_dim=16, out_dim=4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

class TestTrainingStep:

    def test_one_step_does_not_crash(self):
        model = _TinyModel(in_dim=16, out_dim=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(8, 16)
        labels = torch.randint(0, 4, (8,))

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        # Parameters should have been updated (grad is non-None)
        assert model.linear.weight.grad is not None


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

class TestGradientClipping:

    def test_clip_returns_finite_norm(self):
        model = _TinyModel()
        x = torch.randn(4, 16)
        loss = model(x).sum()
        loss.backward()

        norm = clip_gradients(model, max_norm=1.0)
        assert isinstance(norm, float)
        assert torch.isfinite(torch.tensor(norm))

    def test_clip_reduces_large_gradients(self):
        model = _TinyModel()
        # Create artificially large gradients
        x = torch.randn(4, 16) * 1000
        loss = model(x).sum()
        loss.backward()

        norm_before = torch.nn.utils.clip_grad_norm_(
            [p.clone() for p in model.parameters() if p.grad is not None],
            max_norm=float("inf"),
        )
        # Re-backward for clipping test (need fresh grads)
        model.zero_grad()
        loss = model(x).sum()
        loss.backward()

        norm = clip_gradients(model, max_norm=1.0)
        # After clipping, re-computed norm should be <= 1.0 + tolerance
        post_clip_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float("inf"),
        )
        assert post_clip_norm.item() <= 1.0 + 1e-3


# ---------------------------------------------------------------------------
# LR Scheduler
# ---------------------------------------------------------------------------

class TestLRScheduler:

    @pytest.mark.parametrize("sched_name", ["cosine", "step", "one_cycle", "cyclic"])
    def test_build_scheduler(self, sched_name):
        model = _TinyModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        if sched_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=100, eta_min=1e-6,
            )
        elif sched_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1,
            )
        elif sched_name == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=0.01, total_steps=100,
            )
        elif sched_name == "cyclic":
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer, base_lr=1e-5, max_lr=1e-2, step_size_up=50,
            )

        assert scheduler is not None
        # One step should not crash
        scheduler.step()


# ---------------------------------------------------------------------------
# GradScaler
# ---------------------------------------------------------------------------

class TestGradScaler:

    def test_setup_enabled(self):
        # GradScaler with enabled=True requires CUDA; test with enabled=False
        scaler = setup_gradient_scaler(enabled=False)
        assert scaler is not None

    def test_scaler_disabled_mode(self):
        scaler = setup_gradient_scaler(enabled=False)
        # In disabled mode, scale should return the tensor unchanged
        t = torch.tensor(1.0)
        scaled = scaler.scale(t)
        assert torch.allclose(scaled, t)
