"""Tests for AAMSoftmax, PrototypicalLoss, and CombinedLoss."""

import torch
import pytest
from omegaconf import OmegaConf

from speaker_embedding.src.losses.aam_softmax import AAMSoftmax
from speaker_embedding.src.losses.prototypical import PrototypicalLoss
from speaker_embedding.src.losses.combined import CombinedLoss


class TestAAMSoftmax:

    def test_forward_returns_scalar(self):
        num_classes, emb_dim = 10, 64
        loss_fn = AAMSoftmax(num_classes=num_classes, embedding_dim=emb_dim)

        embeddings = torch.randn(8, emb_dim)
        labels = torch.randint(0, num_classes, (8,))
        loss = loss_fn(embeddings, labels)

        assert loss.dim() == 0  # scalar
        assert loss.item() > 0

    def test_loss_positive(self):
        loss_fn = AAMSoftmax(num_classes=5, embedding_dim=32)
        embeddings = torch.randn(4, 32)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(embeddings, labels)
        assert loss.item() > 0

    def test_weight_matrix_shape(self):
        num_classes, emb_dim = 100, 192
        loss_fn = AAMSoftmax(num_classes=num_classes, embedding_dim=emb_dim)
        assert loss_fn.weight.shape == (num_classes, emb_dim)

    def test_backward(self):
        loss_fn = AAMSoftmax(num_classes=5, embedding_dim=32)
        embeddings = torch.randn(4, 32, requires_grad=True)
        labels = torch.randint(0, 5, (4,))
        loss = loss_fn(embeddings, labels)
        loss.backward()
        assert embeddings.grad is not None


class TestPrototypicalLoss:

    def test_forward_with_grouped_embeddings(self):
        """Create embeddings where each speaker has n_support+1 utterances."""
        emb_dim = 64
        n_support = 1
        loss_fn = PrototypicalLoss(distance="cosine", n_support=n_support, temperature=0.05)

        # 3 speakers, each with 3 utterances (1 support + 2 query)
        embeddings = torch.randn(9, emb_dim)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

        loss = loss_fn(embeddings, labels)
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_euclidean_distance(self):
        loss_fn = PrototypicalLoss(distance="euclidean", n_support=1, temperature=0.05)
        embeddings = torch.randn(6, 32)
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = loss_fn(embeddings, labels)
        assert loss.dim() == 0

    def test_insufficient_utterances_returns_zero(self):
        """When no speaker has enough utterances, loss should be zero."""
        loss_fn = PrototypicalLoss(distance="cosine", n_support=5)
        # Each speaker has only 1 utterance (need > 5)
        embeddings = torch.randn(3, 32)
        labels = torch.tensor([0, 1, 2])
        loss = loss_fn(embeddings, labels)
        assert loss.item() == 0.0


class TestCombinedLoss:

    def _make_config(self, num_classes=10, emb_dim=64):
        return OmegaConf.create({
            "aam_softmax": {
                "num_classes": num_classes,
                "embedding_dim": emb_dim,
                "margin": 0.2,
                "scale": 30.0,
                "easy_margin": False,
            },
            "prototypical": {
                "distance": "cosine",
                "n_support": 1,
                "temperature": 0.05,
            },
            "alpha": 0.5,
            "schedule": {"enabled": False},
        })

    def test_forward_returns_correct_keys(self):
        config = self._make_config()
        loss_fn = CombinedLoss(config)

        # 4 speakers, 3 utterances each -> prototypical can work
        embeddings = torch.randn(12, 64)
        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

        total_loss, loss_dict = loss_fn(embeddings, labels)
        assert total_loss.dim() == 0
        assert "aam_loss" in loss_dict
        assert "proto_loss" in loss_dict
        assert "total_loss" in loss_dict
        assert "alpha" in loss_dict

    def test_alpha_weighting(self):
        config = self._make_config()
        loss_fn = CombinedLoss(config)
        assert loss_fn.alpha == 0.5

    def test_alpha_schedule_update(self):
        config = OmegaConf.create({
            "aam_softmax": {
                "num_classes": 10,
                "embedding_dim": 64,
                "margin": 0.2,
                "scale": 30.0,
            },
            "prototypical": {
                "distance": "cosine",
                "n_support": 1,
                "temperature": 0.05,
            },
            "alpha": 0.5,
            "schedule": {
                "enabled": True,
                "alpha_start": 1.0,
                "alpha_end": 0.5,
                "warmup_epochs": 10,
            },
        })
        loss_fn = CombinedLoss(config)
        assert loss_fn.alpha == 1.0  # starts at alpha_start

        loss_fn.update_alpha(5)
        assert 0.5 < loss_fn.alpha < 1.0  # somewhere between start and end

        loss_fn.update_alpha(10)
        assert loss_fn.alpha == 0.5  # at alpha_end after warmup
