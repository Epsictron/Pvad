"""data_dir/speaker_id/utt.wav → NeMo manifest."""

import argparse, json, os
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--output", default="manifest.json")
args = parser.parse_args()

with open(args.output, "w") as f:
    for spk in sorted(os.listdir(args.data_dir)):
        d = os.path.join(args.data_dir, spk)
        if not os.path.isdir(d):
            continue
        for wav in sorted(os.listdir(d)):
            if not wav.endswith((".wav", ".flac")):
                continue
            p = os.path.abspath(os.path.join(d, wav))
            f.write(json.dumps({"audio_filepath": p, "duration": round(sf.info(p).duration, 3),
                                "label": spk, "offset": 0.0}) + "\n")
