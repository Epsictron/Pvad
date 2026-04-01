"""Generate synthetic WAV files for smoke-testing."""

import argparse, json, os
import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="dummy_data")
    parser.add_argument("--num_speakers", type=int, default=10)
    parser.add_argument("--utts_per_speaker", type=int, default=20)
    args = parser.parse_args()

    train, val = [], []
    for s in range(args.num_speakers):
        spk = f"spk_{s:04d}"
        d = os.path.join(args.output_dir, spk)
        os.makedirs(d, exist_ok=True)
        for u in range(args.utts_per_speaker):
            dur = np.random.uniform(1.5, 4.0)
            audio = np.random.randn(int(16000 * dur)).astype(np.float32) * 0.01
            path = os.path.abspath(os.path.join(d, f"utt_{u:04d}.wav"))
            sf.write(path, audio, 16000)
            entry = {"audio_filepath": path, "duration": round(dur, 3), "label": spk, "offset": 0.0}
            (val if u >= args.utts_per_speaker - 2 else train).append(entry)

    for name, entries in [("train_manifest.json", train), ("val_manifest.json", val)]:
        p = os.path.join(args.output_dir, name)
        with open(p, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        print(f"{len(entries)} entries → {p}")


if __name__ == "__main__":
    main()
