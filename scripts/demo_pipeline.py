#!/usr/bin/env python3
"""
Script: demo_pipeline.py
End-to-end demo: supply sample audio → generate sessions → plot everything.

Takes a folder of sample WAV files, simulates multi-speaker sessions,
and produces a 5-panel visualization:
  1. Time-domain waveform
  2. Spectrogram
  3. RTTM speaker labels
  4. VAD activity
  5. Final PVAD label (target speaker highlighted)

Usage:
    # Prepare a folder with WAV files named like: spk01_001.wav, spk02_001.wav, ...
    # Speaker ID is extracted from the filename prefix before the first underscore.

    python scripts/demo_pipeline.py \
        --audio_dir /path/to/sample_wavs \
        --num_speakers 2 \
        --num_sessions 3 \
        --output_dir /data/demo_output

    # Or use the built-in synthetic test data (no audio files needed):
    python scripts/demo_pipeline.py --synthetic --output_dir /data/demo_output
"""

import argparse
import json
import os
import random
import struct
import wave
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Audio I/O helpers (pure-Python fallback when soundfile is unavailable)
# ---------------------------------------------------------------------------

def read_wav(path: str) -> tuple:
    """Read a WAV file → (samples as float32 numpy array, sample_rate)."""
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # mono
        return data, sr
    except ImportError:
        pass
    # Fallback: stdlib wave module
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
    if sw == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if ch > 1:
        samples = samples[::ch]
    return samples, sr


def write_wav(path: str, data: np.ndarray, sr: int = 16000):
    """Write a float32 numpy array to a 16-bit WAV file."""
    try:
        import soundfile as sf
        sf.write(path, data, sr, subtype="PCM_16")
        return
    except ImportError:
        pass
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Synthetic test-tone generator (when user has no audio files)
# ---------------------------------------------------------------------------

def generate_synthetic_utterance(speaker_freq: float, duration: float,
                                 sr: int = 16000) -> np.ndarray:
    """Generate a simple tone burst that represents a speaker utterance."""
    t = np.arange(int(sr * duration)) / sr
    # Tone + slight AM modulation to mimic speech envelope
    envelope = 0.3 + 0.7 * np.abs(np.sin(2 * np.pi * 3.0 * t))  # ~3 Hz modulation
    signal = 0.5 * envelope * np.sin(2 * np.pi * speaker_freq * t)
    # Fade in/out to avoid clicks
    fade = int(0.01 * sr)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    return signal.astype(np.float32)


def create_synthetic_dataset(output_dir: str, num_speakers: int = 3,
                             utts_per_speaker: int = 5,
                             sr: int = 16000) -> str:
    """Create a folder of synthetic WAV files for testing."""
    os.makedirs(output_dir, exist_ok=True)
    freqs = [200 + i * 120 for i in range(num_speakers)]  # Different pitch per speaker

    for spk_idx in range(num_speakers):
        spk_id = f"spk{spk_idx + 1:02d}"
        for utt_idx in range(utts_per_speaker):
            duration = random.uniform(1.5, 4.0)
            audio = generate_synthetic_utterance(freqs[spk_idx], duration, sr)
            fname = f"{spk_id}_{utt_idx + 1:03d}.wav"
            write_wav(os.path.join(output_dir, fname), audio, sr)

    print(f"  Created {num_speakers * utts_per_speaker} synthetic utterances in {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Session simulator (lightweight, standalone)
# ---------------------------------------------------------------------------

def discover_speakers(audio_dir: str) -> dict:
    """Scan audio_dir for WAVs. Returns {speaker_id: [list of wav paths]}."""
    speakers = {}
    for f in sorted(os.listdir(audio_dir)):
        if not f.lower().endswith(".wav"):
            continue
        # Speaker ID = prefix before first underscore
        spk_id = f.split("_")[0]
        speakers.setdefault(spk_id, []).append(os.path.join(audio_dir, f))
    return speakers


def simulate_session(
    speaker_utts: dict,
    num_speakers: int,
    target_duration: float = 30.0,
    turn_gap_mean: float = 0.6,
    turn_gap_min: float = 0.2,
    turn_gap_max: float = 1.5,
    sr: int = 16000,
) -> tuple:
    """
    Simulate a multi-speaker session by concatenating utterances with gaps.

    Returns: (session_audio, rttm_segments, vad_segments)
      rttm_segments: [(speaker, start_sec, duration_sec), ...]
      vad_segments:  [(start_sec, end_sec), ...]  — all speech regions
    """
    available = list(speaker_utts.keys())
    if len(available) < num_speakers:
        raise ValueError(f"Need {num_speakers} speakers, only {len(available)} available")

    chosen = random.sample(available, num_speakers)
    # Preload utterances
    utt_pool = {}
    for spk in chosen:
        utt_pool[spk] = [read_wav(p) for p in speaker_utts[spk]]

    session = np.zeros(0, dtype=np.float32)
    rttm = []
    vad = []
    current_time = 0.0

    while current_time < target_duration:
        # Pick next speaker (round-robin with randomness)
        spk = chosen[len(rttm) % num_speakers]
        if num_speakers > 1 and random.random() < 0.3:
            spk = random.choice(chosen)

        # Pick a random utterance
        audio, utt_sr = random.choice(utt_pool[spk])
        if utt_sr != sr:
            # Simple resample by linear interpolation
            ratio = sr / utt_sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(np.linspace(0, len(audio) - 1, new_len),
                              np.arange(len(audio)), audio).astype(np.float32)

        utt_duration = len(audio) / sr

        # Would this exceed target?
        if current_time + utt_duration > target_duration + 2.0:
            break

        # Add to session
        session = np.concatenate([session, audio])
        rttm.append((spk, current_time, utt_duration))
        vad.append((current_time, current_time + utt_duration))
        current_time += utt_duration

        # Add silence gap
        gap = np.clip(random.gauss(turn_gap_mean, 0.15), turn_gap_min, turn_gap_max)
        if current_time + gap < target_duration:
            silence = np.zeros(int(gap * sr), dtype=np.float32)
            session = np.concatenate([session, silence])
            current_time += gap

    return session, rttm, vad, chosen


# ---------------------------------------------------------------------------
# RTTM / Label I/O
# ---------------------------------------------------------------------------

def write_rttm(rttm_segments: list, output_path: str, session_id: str = "session"):
    """Write RTTM file."""
    with open(output_path, "w") as f:
        for spk, start, dur in rttm_segments:
            f.write(f"SPEAKER {session_id} 1 {start:.4f} {dur:.4f} <NA> <NA> {spk} <NA> <NA>\n")


def compute_frame_labels(segments: list, total_duration: float,
                         frame_hop_s: float = 0.01) -> np.ndarray:
    """Convert (start, end) segments → binary frame-level labels."""
    num_frames = int(total_duration / frame_hop_s) + 1
    labels = np.zeros(num_frames, dtype=np.float32)
    for start, end in segments:
        i0 = int(start / frame_hop_s)
        i1 = int(end / frame_hop_s)
        labels[i0:min(i1 + 1, num_frames)] = 1.0
    return labels


def compute_pvad_label(rttm_segments: list, target_speaker: str,
                       total_duration: float,
                       frame_hop_s: float = 0.01) -> np.ndarray:
    """Compute PVAD label: 1 when target_speaker is active, 0 otherwise."""
    target_segs = [(start, start + dur) for spk, start, dur in rttm_segments
                   if spk == target_speaker]
    return compute_frame_labels(target_segs, total_duration, frame_hop_s)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_session(
    session_audio: np.ndarray,
    rttm_segments: list,
    vad_segments: list,
    speakers: list,
    target_speaker: str,
    sr: int,
    output_path: str,
    session_id: str = "session_0",
):
    """
    5-panel plot:
      1. Time-domain waveform
      2. Spectrogram
      3. RTTM speaker labels
      4. VAD label
      5. Final PVAD label (target speaker)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    total_duration = len(session_audio) / sr
    frame_hop = 0.01
    time_audio = np.arange(len(session_audio)) / sr

    # Compute labels
    vad_label = compute_frame_labels(vad_segments, total_duration, frame_hop)
    pvad_label = compute_pvad_label(rttm_segments, target_speaker, total_duration, frame_hop)
    frame_times = np.arange(len(vad_label)) * frame_hop

    # Per-speaker labels for RTTM panel
    spk_labels = {}
    for spk in speakers:
        segs = [(start, start + dur) for s, start, dur in rttm_segments if s == spk]
        spk_labels[spk] = compute_frame_labels(segs, total_duration, frame_hop)

    COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    fig, axes = plt.subplots(5, 1, figsize=(18, 14), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.5, 1, 1]})

    # ── Panel 1: Waveform ──
    ax = axes[0]
    ax.plot(time_audio, session_audio, linewidth=0.3, color="#555555")
    # Color-code speech regions by speaker
    for i, (spk, start, dur) in enumerate(rttm_segments):
        idx = speakers.index(spk)
        color = COLORS[idx % len(COLORS)]
        s_i = int(start * sr)
        e_i = min(int((start + dur) * sr), len(session_audio))
        ax.fill_between(time_audio[s_i:e_i], session_audio[s_i:e_i],
                        alpha=0.3, color=color)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{session_id} | {len(speakers)} speakers | target: {target_speaker}",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, total_duration)

    # ── Panel 2: Spectrogram ──
    ax = axes[1]
    n_fft = 512
    hop_length = 160  # 10ms at 16kHz
    # Compute spectrogram manually
    num_frames_spec = (len(session_audio) - n_fft) // hop_length + 1
    if num_frames_spec > 0:
        spec = np.zeros((n_fft // 2 + 1, num_frames_spec))
        window = np.hanning(n_fft)
        for i in range(num_frames_spec):
            frame = session_audio[i * hop_length : i * hop_length + n_fft] * window
            spec[:, i] = np.abs(np.fft.rfft(frame))
        spec_db = 20 * np.log10(spec + 1e-10)
        spec_db = np.clip(spec_db, spec_db.max() - 80, spec_db.max())
        extent = [0, total_duration, 0, sr / 2 / 1000]
        ax.imshow(spec_db, aspect="auto", origin="lower", extent=extent,
                  cmap="magma", interpolation="bilinear")
    ax.set_ylabel("Freq (kHz)")
    ax.set_title("Spectrogram", fontsize=10)

    # ── Panel 3: RTTM Speaker Labels ──
    ax = axes[2]
    for i, spk in enumerate(speakers):
        color = COLORS[i % len(COLORS)]
        label = spk_labels[spk]
        ax.fill_between(frame_times, i, i + label * 0.8,
                        color=color, alpha=0.8, label=spk)
    ax.set_yticks([x + 0.4 for x in range(len(speakers))])
    ax.set_yticklabels(speakers)
    ax.set_ylim(-0.1, len(speakers))
    ax.set_title("RTTM Speaker Labels", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    # ── Panel 4: VAD Label ──
    ax = axes[3]
    ax.fill_between(frame_times, 0, vad_label, color="#FF9800", alpha=0.7)
    ax.set_ylabel("VAD")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("VAD Activity (all speakers)", fontsize=10)

    # ── Panel 5: Final PVAD Label ──
    ax = axes[4]
    target_idx = speakers.index(target_speaker)
    target_color = COLORS[target_idx % len(COLORS)]
    ax.fill_between(frame_times, 0, pvad_label, color=target_color, alpha=0.8)
    ax.set_ylabel("PVAD")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"PVAD Label — Target: {target_speaker}", fontsize=10,
                 fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demo pipeline: sample audio → sessions → visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With your own audio files (named spk01_001.wav, spk02_001.wav, ...):
  python scripts/demo_pipeline.py --audio_dir ./my_samples --output_dir ./demo_out

  # With built-in synthetic test data (no audio files needed):
  python scripts/demo_pipeline.py --synthetic --output_dir ./demo_out
        """,
    )
    parser.add_argument("--audio_dir", type=str, default=None,
                        help="Directory with sample WAV files (spkID_xxx.wav)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic test tones instead of real audio")
    parser.add_argument("--output_dir", type=str, default="demo_output",
                        help="Output directory for sessions + plots")
    parser.add_argument("--num_speakers", type=int, default=2,
                        help="Number of speakers per session (1, 2, or 3)")
    parser.add_argument("--num_sessions", type=int, default=3,
                        help="Number of sessions to generate")
    parser.add_argument("--session_duration", type=float, default=30.0,
                        help="Target session duration in seconds")
    parser.add_argument("--sr", type=int, default=16000,
                        help="Sample rate")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to datasets.yaml — reads turn_gap and session_length from it")
    parser.add_argument("--turn_gap_mean", type=float, default=None)
    parser.add_argument("--turn_gap_min", type=float, default=None)
    parser.add_argument("--turn_gap_max", type=float, default=None)
    args = parser.parse_args()

    # ── Load defaults from config if provided ──
    if args.config and os.path.isfile(args.config):
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        sim = cfg.get("simulation", {})
        tg = sim.get("turn_gap", {})
        if args.turn_gap_mean is None:
            args.turn_gap_mean = tg.get("mean", 0.6)
        if args.turn_gap_min is None:
            args.turn_gap_min = tg.get("min", 0.2)
        if args.turn_gap_max is None:
            args.turn_gap_max = tg.get("max", 1.5)
        if args.session_duration == 30.0:  # only override if user didn't set it
            args.session_duration = sim.get("session_length", 30.0)
        print(f"  Loaded config from {args.config}")
    else:
        if args.turn_gap_mean is None:
            args.turn_gap_mean = 0.6
        if args.turn_gap_min is None:
            args.turn_gap_min = 0.2
        if args.turn_gap_max is None:
            args.turn_gap_max = 1.5

    print(f"  Turn gap: mean={args.turn_gap_mean}, min={args.turn_gap_min}, max={args.turn_gap_max}")
    print(f"  Session duration: {args.session_duration}s")

    random.seed(42)
    np.random.seed(42)

    # ── Step 1: Get audio ──
    if args.synthetic:
        synth_dir = os.path.join(args.output_dir, "synthetic_wavs")
        print("Step 1: Generating synthetic test audio...")
        create_synthetic_dataset(synth_dir, num_speakers=max(args.num_speakers, 3),
                                 utts_per_speaker=5, sr=args.sr)
        audio_dir = synth_dir
    elif args.audio_dir:
        audio_dir = args.audio_dir
        if not os.path.isdir(audio_dir):
            print(f"ERROR: Audio directory not found: {audio_dir}")
            return
        print(f"Step 1: Using audio from {audio_dir}")
    else:
        print("ERROR: Provide --audio_dir or --synthetic")
        return

    # ── Step 2: Discover speakers ──
    speaker_utts = discover_speakers(audio_dir)
    print(f"\nStep 2: Found {len(speaker_utts)} speakers:")
    for spk, files in speaker_utts.items():
        print(f"  {spk}: {len(files)} utterances")

    if len(speaker_utts) < args.num_speakers:
        print(f"ERROR: Need {args.num_speakers} speakers, only found {len(speaker_utts)}")
        return

    # ── Step 3: Generate sessions ──
    sessions_dir = os.path.join(args.output_dir, "sessions")
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(sessions_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    manifest_entries = []

    for sess_idx in range(args.num_sessions):
        session_id = f"session_{sess_idx:03d}"
        print(f"\nStep 3.{sess_idx + 1}: Generating {session_id} "
              f"({args.num_speakers} speakers, ~{args.session_duration}s)...")

        session_audio, rttm_segments, vad_segments, chosen_speakers = simulate_session(
            speaker_utts=speaker_utts,
            num_speakers=args.num_speakers,
            target_duration=args.session_duration,
            turn_gap_mean=args.turn_gap_mean,
            turn_gap_min=args.turn_gap_min,
            turn_gap_max=args.turn_gap_max,
            sr=args.sr,
        )

        # Write WAV
        wav_path = os.path.join(sessions_dir, f"{session_id}.wav")
        write_wav(wav_path, session_audio, args.sr)
        print(f"  WAV: {wav_path} ({len(session_audio) / args.sr:.1f}s)")

        # Write RTTM
        rttm_path = os.path.join(sessions_dir, f"{session_id}.rttm")
        write_rttm(rttm_segments, rttm_path, session_id)
        print(f"  RTTM: {rttm_path} ({len(rttm_segments)} segments)")

        # ── Step 4: Plot for each target speaker ──
        for target_spk in chosen_speakers:
            plot_path = os.path.join(plots_dir, f"{session_id}_target_{target_spk}.png")
            print(f"\nStep 4: Plotting {session_id} → target={target_spk}")
            plot_session(
                session_audio=session_audio,
                rttm_segments=rttm_segments,
                vad_segments=vad_segments,
                speakers=chosen_speakers,
                target_speaker=target_spk,
                sr=args.sr,
                output_path=plot_path,
                session_id=session_id,
            )

            # Manifest entry
            manifest_entries.append({
                "audio_filepath": os.path.abspath(wav_path),
                "duration": round(len(session_audio) / args.sr, 3),
                "rttm_filepath": os.path.abspath(rttm_path),
                "target_speaker": target_spk,
                "num_speakers": args.num_speakers,
                "plot": os.path.abspath(plot_path),
            })

    # ── Step 5: Write manifest ──
    manifest_path = os.path.join(args.output_dir, "demo_manifest.json")
    with open(manifest_path, "w") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"{'=' * 60}")
    print(f"  Sessions:  {sessions_dir}/ ({args.num_sessions} sessions)")
    print(f"  Plots:     {plots_dir}/ ({len(manifest_entries)} plots)")
    print(f"  Manifest:  {manifest_path}")
    print(f"  PVAD samples: {len(manifest_entries)}")


if __name__ == "__main__":
    main()
