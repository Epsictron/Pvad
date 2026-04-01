"""Build NeMo-format JSON-lines manifest from a directory of speaker audio.

Expected directory layout:
    data_dir/
        speaker_001/
            utt_001.wav
            utt_002.wav
        speaker_002/
            ...

Usage:
    python create_manifest.py --data_dir /path/to/audio --output train_manifest.json
"""

import argparse
import json
import os

import soundfile as sf


def create_manifest(data_dir: str, output: str, min_dur: float = 0.5) -> None:
    entries = []
    for speaker_id in sorted(os.listdir(data_dir)):
        speaker_dir = os.path.join(data_dir, speaker_id)
        if not os.path.isdir(speaker_dir):
            continue
        for fname in sorted(os.listdir(speaker_dir)):
            if not fname.endswith((".wav", ".flac", ".ogg")):
                continue
            fpath = os.path.abspath(os.path.join(speaker_dir, fname))
            info = sf.info(fpath)
            if info.duration < min_dur:
                continue
            entries.append({
                "audio_filepath": fpath,
                "duration": round(info.duration, 3),
                "label": speaker_id,
                "offset": 0.0,
            })

    with open(output, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    labels = {e["label"] for e in entries}
    print(f"Wrote {len(entries)} utterances, {len(labels)} speakers → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="train_manifest.json")
    parser.add_argument("--min_dur", type=float, default=0.5)
    args = parser.parse_args()
    create_manifest(args.data_dir, args.output, args.min_dur)
