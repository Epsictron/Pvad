"""Gender-balanced and curriculum samplers for speaker embedding training."""

import logging
import math
import random
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler

from .manifest import ManifestEntry

logger = logging.getLogger(__name__)


class GenderBalancedSampler(Sampler):
    """Sampler that produces gender-balanced batches.

    Each batch of size B contains B/2 male utterances and B/2 female
    utterances. Within each gender group, speakers are sampled uniformly,
    and for each selected speaker, ``utterances_per_speaker`` utterances
    are drawn.

    Unknown-gender entries ('u') are handled according to the chosen
    strategy: 'proportional', 'male', 'female', or 'exclude'.

    Attributes:
        batch_size: Number of samples per batch (must be even).
        utterances_per_speaker: Number of utterances drawn per speaker
            within a batch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        manifest_entries: List[ManifestEntry],
        batch_size: int,
        utterances_per_speaker: int = 2,
        unknown_gender_strategy: str = "proportional",
        seed: int = 42,
    ) -> None:
        """Initialize the gender-balanced sampler.

        Args:
            manifest_entries: List of ManifestEntry objects.
            batch_size: Total batch size. Must be even.
            utterances_per_speaker: Number of utterances to draw per speaker
                in each batch.
            unknown_gender_strategy: How to handle entries with gender='u'.
                One of 'proportional', 'male', 'female', 'exclude'.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If batch_size is odd or strategy is invalid.
        """
        super().__init__()
        if batch_size % 2 != 0:
            raise ValueError(f"batch_size must be even, got {batch_size}")

        valid_strategies = ("proportional", "male", "female", "exclude")
        if unknown_gender_strategy not in valid_strategies:
            raise ValueError(
                f"unknown_gender_strategy must be one of {valid_strategies}, "
                f"got '{unknown_gender_strategy}'"
            )

        self.manifest_entries = manifest_entries
        self.batch_size = batch_size
        self.utterances_per_speaker = utterances_per_speaker
        self.unknown_gender_strategy = unknown_gender_strategy
        self.seed = seed

        # Build gender-indexed pools: speaker -> list of entry indices
        self.male_speakers: Dict[str, List[int]] = defaultdict(list)
        self.female_speakers: Dict[str, List[int]] = defaultdict(list)

        self._build_pools()

        logger.info(
            "GenderBalancedSampler: %d male speakers, %d female speakers, "
            "batch_size=%d, utterances_per_speaker=%d, strategy='%s'.",
            len(self.male_speakers),
            len(self.female_speakers),
            self.batch_size,
            self.utterances_per_speaker,
            self.unknown_gender_strategy,
        )

    def _build_pools(self) -> None:
        """Build speaker pools partitioned by gender."""
        self.male_speakers = defaultdict(list)
        self.female_speakers = defaultdict(list)
        unknown_entries: List[tuple] = []  # (index, speaker)

        for idx, entry in enumerate(self.manifest_entries):
            if entry.gender == "m":
                self.male_speakers[entry.speaker].append(idx)
            elif entry.gender == "f":
                self.female_speakers[entry.speaker].append(idx)
            else:
                unknown_entries.append((idx, entry.speaker))

        # Handle unknown gender entries
        if unknown_entries:
            if self.unknown_gender_strategy == "exclude":
                logger.info(
                    "Excluding %d unknown-gender entries.", len(unknown_entries)
                )
            elif self.unknown_gender_strategy == "male":
                for idx, speaker in unknown_entries:
                    self.male_speakers[speaker].append(idx)
            elif self.unknown_gender_strategy == "female":
                for idx, speaker in unknown_entries:
                    self.female_speakers[speaker].append(idx)
            elif self.unknown_gender_strategy == "proportional":
                # Distribute unknown entries proportionally based on
                # existing male/female counts
                n_male = sum(len(v) for v in self.male_speakers.values())
                n_female = sum(len(v) for v in self.female_speakers.values())
                total = n_male + n_female
                if total == 0:
                    # No gendered data at all; split 50/50
                    male_ratio = 0.5
                else:
                    male_ratio = n_male / total

                rng = random.Random(self.seed)
                for idx, speaker in unknown_entries:
                    if rng.random() < male_ratio:
                        self.male_speakers[speaker].append(idx)
                    else:
                        self.female_speakers[speaker].append(idx)

    def _speakers_per_half_batch(self) -> int:
        """Return number of speakers to sample per gender group in a batch."""
        half_batch = self.batch_size // 2
        return max(1, half_batch // self.utterances_per_speaker)

    def __iter__(self) -> Iterator[List[int]]:
        """Yield batches of indices with gender balance.

        Yields:
            List of integer indices forming one batch.
        """
        rng = random.Random(self.seed)
        speakers_per_group = self._speakers_per_half_batch()
        half_batch = self.batch_size // 2

        male_speaker_list = list(self.male_speakers.keys())
        female_speaker_list = list(self.female_speakers.keys())

        if not male_speaker_list or not female_speaker_list:
            logger.warning(
                "One or both gender pools are empty (male=%d, female=%d). "
                "Cannot produce gender-balanced batches.",
                len(male_speaker_list),
                len(female_speaker_list),
            )
            return

        num_batches = len(self)

        for _ in range(num_batches):
            batch_indices: List[int] = []

            # Sample male half
            batch_indices.extend(
                self._sample_gender_group(
                    self.male_speakers, male_speaker_list, speakers_per_group,
                    half_batch, rng,
                )
            )

            # Sample female half
            batch_indices.extend(
                self._sample_gender_group(
                    self.female_speakers, female_speaker_list, speakers_per_group,
                    half_batch, rng,
                )
            )

            rng.shuffle(batch_indices)
            yield batch_indices

    def _sample_gender_group(
        self,
        speaker_pool: Dict[str, List[int]],
        speaker_list: List[str],
        speakers_per_group: int,
        target_count: int,
        rng: random.Random,
    ) -> List[int]:
        """Sample indices from a single gender group.

        Args:
            speaker_pool: Mapping from speaker to list of entry indices.
            speaker_list: List of speaker keys for random selection.
            speakers_per_group: Number of speakers to sample.
            target_count: Exact number of utterance indices to return.
            rng: Random number generator instance.

        Returns:
            List of entry indices of length target_count.
        """
        indices: List[int] = []

        # Sample speakers uniformly (with replacement if needed)
        selected_speakers = rng.choices(speaker_list, k=speakers_per_group)

        for speaker in selected_speakers:
            utts = speaker_pool[speaker]
            sampled = rng.choices(utts, k=self.utterances_per_speaker)
            indices.extend(sampled)

        # Trim or pad to exactly target_count
        if len(indices) > target_count:
            indices = indices[:target_count]
        elif len(indices) < target_count:
            # Fill remaining slots by sampling more from the pool
            all_indices = [
                idx for utt_list in speaker_pool.values() for idx in utt_list
            ]
            extra = rng.choices(all_indices, k=target_count - len(indices))
            indices.extend(extra)

        return indices

    def __len__(self) -> int:
        """Return the number of batches per epoch.

        Computed as total utterances in both pools divided by batch size.
        """
        total_utterances = sum(
            len(v) for v in self.male_speakers.values()
        ) + sum(len(v) for v in self.female_speakers.values())
        if total_utterances == 0 or self.batch_size == 0:
            return 0
        return max(1, total_utterances // self.batch_size)


class CurriculumSampler(GenderBalancedSampler):
    """Curriculum-aware extension of GenderBalancedSampler.

    Allows dynamic adjustment of the sampling density (speakers per batch)
    based on the current curriculum stage. Early stages may use fewer
    speakers per batch (easier task), while later stages increase diversity.

    Attributes:
        speakers_per_batch: Current number of speakers to sample per
            gender group per batch. Overrides the default calculation
            when set via ``set_stage``.
    """

    def __init__(
        self,
        manifest_entries: List[ManifestEntry],
        batch_size: int,
        utterances_per_speaker: int = 2,
        unknown_gender_strategy: str = "proportional",
        seed: int = 42,
    ) -> None:
        """Initialize the curriculum sampler.

        Args:
            manifest_entries: List of ManifestEntry objects.
            batch_size: Total batch size. Must be even.
            utterances_per_speaker: Number of utterances to draw per speaker.
            unknown_gender_strategy: Strategy for unknown-gender entries.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            manifest_entries=manifest_entries,
            batch_size=batch_size,
            utterances_per_speaker=utterances_per_speaker,
            unknown_gender_strategy=unknown_gender_strategy,
            seed=seed,
        )
        self._custom_speakers_per_batch: Optional[int] = None

    def set_stage(self, stage_config: Dict) -> None:
        """Update sampling parameters based on curriculum stage.

        Args:
            stage_config: Dictionary with curriculum stage parameters.
                Expected keys:
                - 'speakers_per_batch' (int, optional): Override number of
                  speakers per gender group per batch.
                - 'batch_size' (int, optional): Override batch size.
                - 'utterances_per_speaker' (int, optional): Override
                  utterances per speaker.
        """
        if "speakers_per_batch" in stage_config:
            self._custom_speakers_per_batch = int(stage_config["speakers_per_batch"])
            logger.info(
                "CurriculumSampler stage update: speakers_per_batch=%d.",
                self._custom_speakers_per_batch,
            )

        if "batch_size" in stage_config:
            new_batch_size = int(stage_config["batch_size"])
            if new_batch_size % 2 != 0:
                raise ValueError(
                    f"batch_size must be even, got {new_batch_size}"
                )
            self.batch_size = new_batch_size
            logger.info(
                "CurriculumSampler stage update: batch_size=%d.",
                self.batch_size,
            )

        if "utterances_per_speaker" in stage_config:
            self.utterances_per_speaker = int(stage_config["utterances_per_speaker"])
            logger.info(
                "CurriculumSampler stage update: utterances_per_speaker=%d.",
                self.utterances_per_speaker,
            )

    def _speakers_per_half_batch(self) -> int:
        """Return number of speakers per gender group, respecting curriculum override."""
        if self._custom_speakers_per_batch is not None:
            return self._custom_speakers_per_batch
        return super()._speakers_per_half_batch()
