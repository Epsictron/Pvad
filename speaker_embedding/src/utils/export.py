"""Model export utilities for ONNX and TorchScript formats."""

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_onnx(
    model: nn.Module,
    config: Any,
    output_path: str,
) -> None:
    """Export the model to ONNX format with dynamic batch and time axes.

    The exported graph uses the opset version specified in
    ``config.export.opset_version`` (defaults to 14 if absent).  When
    ``onnxruntime`` is available the exported model is verified with a
    dummy forward pass.

    Args:
        model: The model to export (should already be on CPU and in eval
            mode).
        config: Run configuration providing ``config.export.opset_version``
            and ``config.features.n_mels`` (number of input features).
        output_path: Destination file path for the ``.onnx`` file.
    """
    model.eval()
    model.cpu()

    opset_version: int = getattr(
        getattr(config, "export", None), "opset_version", 14
    )
    n_mels: int = getattr(
        getattr(config, "features", None), "n_mels", 80
    )

    # Create a dummy input: (batch, n_mels, time).
    dummy_input = torch.randn(1, n_mels, 200)

    dynamic_axes = {
        "input": {0: "batch_size", 2: "time"},
        "output": {0: "batch_size"},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    logger.info(
        "ONNX model exported to %s (opset=%d).", output_path, opset_version
    )

    # Optional verification with ONNX Runtime.
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]

        sess = ort.InferenceSession(output_path)
        ort_inputs = {sess.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = sess.run(None, ort_inputs)
        logger.info(
            "ONNX Runtime verification passed. Output shape: %s",
            ort_outputs[0].shape,
        )
    except ImportError:
        logger.warning(
            "onnxruntime is not installed; skipping ONNX verification."
        )
    except Exception as exc:
        logger.error("ONNX Runtime verification failed: %s", exc)


def export_torchscript(
    model: nn.Module,
    config: Any,
    output_path: str,
) -> None:
    """Export the model to TorchScript via ``torch.jit.trace``.

    Args:
        model: The model to export (should already be on CPU and in eval
            mode).
        config: Run configuration providing ``config.features.n_mels``
            (number of input features).
        output_path: Destination file path for the ``.pt`` TorchScript
            archive.
    """
    model.eval()
    model.cpu()

    n_mels: int = getattr(
        getattr(config, "features", None), "n_mels", 80
    )

    dummy_input = torch.randn(1, n_mels, 200)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)

    traced.save(output_path)
    logger.info("TorchScript model exported to %s.", output_path)
