# NeMo Speaker Embedding — Custom Model

Trains a speaker embedding model using **NVIDIA NeMo's `EncDecSpeakerLabelModel` pipeline**
with a custom encoder backbone.

## Structure

```
nemo_speaker_embedding/
├── conf/
│   └── speaker_model.yaml        # full NeMo config
├── custom_model/
│   └── encoder.py                # custom encoder plugged into NeMo
├── data_prep/
│   ├── create_manifest.py        # build NeMo manifests from audio dirs
│   └── generate_dummy_data.py    # generate synthetic wav files for testing
├── scripts/
│   ├── train.py                  # NeMo + PyTorch Lightning training
│   ├── infer.py                  # extract embeddings & verify speakers
│   └── finetune_pretrained.py    # finetune TitaNet/ECAPA on custom data
└── requirements.txt
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Generate dummy data (10 speakers, 20 utterances each)
python data_prep/generate_dummy_data.py

# 3. Train
python scripts/train.py

# 4. Extract embeddings
python scripts/infer.py --nemo_model experiments/SpeakerModel/checkpoints/*.nemo --audio test.wav
```

## Using Your Own Data

Organise audio as:
```
my_data/
  speaker_001/
    utt_001.wav
    utt_002.wav
  speaker_002/
    ...
```

Then generate a NeMo manifest:
```bash
python data_prep/create_manifest.py --data_dir my_data --output train_manifest.json
```

Update `conf/speaker_model.yaml` paths and `num_classes` accordingly.
