"""Generate synthetic WAV files for smoke-testing the NeMo speaker pipeline.

Creates a directory tree with N speakers × M utterances of random audio,
then writes train and val manifests.

Usage:
    python generate_dummy_data.py [--output_dir dummy_data --num_speakers 10 --utts_per_speaker 20]
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf


def generate(output_dir: str, num_speakers: int, utts_per_speaker: int, sr: int = 16000) -> None:
    os.makedirs(output_dir, exist_ok=True)
    train_entries, val_entries = [], []

    for spk_idx in range(num_speakers):
        spk_id = f"spk_{spk_idx:04d}"
        spk_dir = os.path.join(output_dir, spk_id)
        os.makedirs(spk_dir, exist_ok=True)

        for utt_idx in range(utts_per_speaker):
            duration = np.random.uniform(1.5, 4.0)
            samples = int(sr * duration)
            audio = np.random.randn(samples).astype(np.float32) * 0.01

            fname = f"utt_{utt_idx:04d}.wav"
            fpath = os.path.abspath(os.path.join(spk_dir, fname))
            sf.write(fpath, audio, sr)

            entry = {
                "audio_filepath": fpath,
                "duration": round(duration, 3),
                "label": spk_id,
                "offset": 0.0,
            }
            if utt_idx < utts_per_speaker - 2:
                train_entries.append(entry)
            else:
                val_entries.append(entry)

    for name, entries in [("train_manifest.json", train_entries), ("val_manifest.json", val_entries)]:
        path = os.path.join(output_dir, name)
        with open(path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"Wrote {len(entries)} entries → {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="dummy_data")
    parser.add_argument("--num_speakers", type=int, default=10)
    parser.add_argument("--utts_per_speaker", type=int, default=20)
    args = parser.parse_args()
    generate(args.output_dir, args.num_speakers, args.utts_per_speaker)
