"""Manifest loading, validation, and statistics for speaker embedding training."""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """A single entry from a JSONL manifest file.

    Attributes:
        audio_filepath: Path to the audio file.
        speaker: Speaker identifier string.
        duration: Duration of the audio in seconds. Must be positive.
        gender: Gender label, one of 'm', 'f', or 'u' (unknown). Defaults to 'u'.
    """

    audio_filepath: str
    speaker: str
    duration: float
    gender: str = "u"

    def __post_init__(self) -> None:
        """Validate gender field after initialization."""
        if self.gender not in ("m", "f", "u"):
            logger.warning(
                "Invalid gender '%s' for entry '%s', defaulting to 'u'.",
                self.gender,
                self.audio_filepath,
            )
            self.gender = "u"


def load_manifest(
    path: str,
    min_duration: float = 0.0,
    max_duration: float = float("inf"),
) -> List[ManifestEntry]:
    """Load a JSONL manifest file and filter entries by duration.

    Each line of the manifest is a JSON object with fields: audio_filepath,
    speaker, duration, and optionally gender.

    Args:
        path: Path to the JSONL manifest file.
        min_duration: Minimum duration in seconds (inclusive). Entries shorter
            than this are discarded.
        max_duration: Maximum duration in seconds (inclusive). Entries longer
            than this are discarded.

    Returns:
        List of ManifestEntry objects that pass the duration filter.

    Raises:
        FileNotFoundError: If the manifest file does not exist.
        json.JSONDecodeError: If a line is not valid JSON.
        KeyError: If a required field is missing from a manifest line.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest file not found: {path}")

    entries: List[ManifestEntry] = []
    total_lines = 0
    skipped_duration = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total_lines += 1

            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Invalid JSON at line %d in %s: %s", line_num, path, exc)
                raise

            # Validate required fields
            for required_field in ("audio_filepath", "speaker", "duration"):
                if required_field not in data:
                    raise KeyError(
                        f"Missing required field '{required_field}' at line {line_num} "
                        f"in {path}"
                    )

            duration = float(data["duration"])
            if duration < min_duration or duration > max_duration:
                skipped_duration += 1
                continue

            entry = ManifestEntry(
                audio_filepath=str(data["audio_filepath"]),
                speaker=str(data["speaker"]),
                duration=duration,
                gender=data.get("gender", "u"),
            )
            entries.append(entry)

    # Log statistics after loading
    stats = get_manifest_stats(entries)
    logger.info(
        "Loaded manifest '%s': %d/%d entries kept (skipped %d by duration filter). "
        "%d speakers, duration range [%.2f, %.2f]s, total %.2f hours.",
        path,
        len(entries),
        total_lines,
        skipped_duration,
        stats["num_speakers"],
        stats["duration_stats"]["min"],
        stats["duration_stats"]["max"],
        stats["duration_stats"]["total"] / 3600.0,
    )

    return entries


def validate_manifest(entries: List[ManifestEntry]) -> bool:
    """Validate a list of manifest entries.

    Checks that all entries have required fields populated, durations are
    positive, and audio files exist on disk. Missing audio files produce
    warnings but do not cause validation failure.

    Args:
        entries: List of ManifestEntry objects to validate.

    Returns:
        True if all entries pass validation, False otherwise.
    """
    is_valid = True

    for idx, entry in enumerate(entries):
        # Check required string fields are non-empty
        if not entry.audio_filepath:
            logger.error("Entry %d: audio_filepath is empty.", idx)
            is_valid = False

        if not entry.speaker:
            logger.error("Entry %d: speaker is empty.", idx)
            is_valid = False

        # Check duration is positive
        if entry.duration <= 0:
            logger.error(
                "Entry %d (%s): duration must be positive, got %.4f.",
                idx,
                entry.audio_filepath,
                entry.duration,
            )
            is_valid = False

        # Check gender is valid
        if entry.gender not in ("m", "f", "u"):
            logger.error(
                "Entry %d (%s): invalid gender '%s'.",
                idx,
                entry.audio_filepath,
                entry.gender,
            )
            is_valid = False

        # Warn if audio file does not exist (not a hard failure)
        if not os.path.isfile(entry.audio_filepath):
            logger.warning(
                "Entry %d: audio file not found: %s",
                idx,
                entry.audio_filepath,
            )

    if is_valid:
        logger.info("Manifest validation passed for %d entries.", len(entries))
    else:
        logger.error("Manifest validation failed.")

    return is_valid


def get_manifest_stats(entries: List[ManifestEntry]) -> Dict:
    """Compute summary statistics for a list of manifest entries.

    Args:
        entries: List of ManifestEntry objects.

    Returns:
        Dictionary with keys:
            - num_utterances: Total number of entries.
            - num_speakers: Number of unique speakers.
            - gender_distribution: Dict mapping gender label to count.
            - duration_stats: Dict with min, max, mean, and total duration.
    """
    if not entries:
        return {
            "num_utterances": 0,
            "num_speakers": 0,
            "gender_distribution": {},
            "duration_stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "total": 0.0},
        }

    speakers = set()
    gender_counts: Dict[str, int] = {}
    durations: List[float] = []

    for entry in entries:
        speakers.add(entry.speaker)
        gender_counts[entry.gender] = gender_counts.get(entry.gender, 0) + 1
        durations.append(entry.duration)

    total_duration = sum(durations)
    return {
        "num_utterances": len(entries),
        "num_speakers": len(speakers),
        "gender_distribution": gender_counts,
        "duration_stats": {
            "min": min(durations),
            "max": max(durations),
            "mean": total_duration / len(durations),
            "total": total_duration,
        },
    }
