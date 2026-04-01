"""Build NeMo manifest from: data_dir/speaker_id/utt.wav"""

import argparse, json, os
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output", default="manifest.json")
    args = parser.parse_args()

    with open(args.output, "w") as f:
        for spk in sorted(os.listdir(args.data_dir)):
            spk_dir = os.path.join(args.data_dir, spk)
            if not os.path.isdir(spk_dir):
                continue
            for wav in sorted(os.listdir(spk_dir)):
                if not wav.endswith((".wav", ".flac")):
                    continue
                path = os.path.abspath(os.path.join(spk_dir, wav))
                dur = sf.info(path).duration
                f.write(json.dumps({"audio_filepath": path, "duration": round(dur, 3),
                                    "label": spk, "offset": 0.0}) + "\n")
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
