# Speaker Embedding

A modular PyTorch pipeline for training, evaluating, and exporting speaker embedding models. Supports configurable feature extraction, multiple loss functions (AAM-Softmax, Prototypical, Combined), gender-balanced sampling, curriculum learning, and mixed-precision training.

## Project Structure

```
speaker_embedding/
├── configs/              # Hydra/OmegaConf YAML configurations
│   ├── base.yaml         # Base config with all defaults
│   ├── train_normal.yaml # Standard training config
│   ├── train_curriculum.yaml
│   ├── model/            # Model-specific configs
│   └── loss/             # Loss-specific configs
├── scripts/              # Entry-point scripts
│   ├── train.py          # Hydra-based training entry point
│   ├── evaluate.py       # Evaluate on trial lists (EER, minDCF)
│   ├── extract_embeddings.py  # Batch embedding extraction to .npz
│   ├── validate_manifest.py   # Manifest validation and statistics
│   └── export_model.py   # Export to ONNX or TorchScript
├── src/                  # Library source code
│   ├── data/             # Dataset, manifest loading, samplers
│   ├── features/         # Feature extraction and normalization
│   ├── losses/           # AAM-Softmax, Prototypical, Combined
│   ├── models/           # Encoder architectures, pooling, embedding head
│   ├── training/         # Trainer, LR schedulers, gradient utilities
│   ├── evaluation/       # Metrics (EER, minDCF), trial scoring
│   └── utils/            # Checkpointing, logging, distributed, export, seed
├── tests/                # Pytest test suite
├── setup.py
├── requirements.txt
└── README.md
```

## Quick Start

### Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

### Train

```bash
python scripts/train.py data.manifest_path=/path/to/train.jsonl
```

Override any config value via the command line (Hydra syntax):

```bash
python scripts/train.py \
    data.manifest_path=/path/to/train.jsonl \
    training.epochs=50 \
    training.batch_size=128 \
    loss=combined
```

### Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint_path checkpoints/best.pt \
    --trial_list /path/to/trials.txt
```

### Extract Embeddings

```bash
python scripts/extract_embeddings.py \
    --checkpoint_path checkpoints/best.pt \
    --manifest_path /path/to/manifest.jsonl \
    --output_path embeddings.npz
```

### Export

```bash
python scripts/export_model.py \
    --checkpoint_path checkpoints/best.pt \
    --format onnx \
    --output_path model.onnx
```

## Config System

Configuration is managed by [Hydra](https://hydra.cc/) with [OmegaConf](https://omegaconf.readthedocs.io/). The base config (`configs/base.yaml`) defines all default values. Training configs (`train_normal.yaml`, `train_curriculum.yaml`) compose the base with model and loss sub-configs. Any value can be overridden from the command line.

## License

TBD
