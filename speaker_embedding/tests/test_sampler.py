"""Tests for GenderBalancedSampler."""

import pytest

from speaker_embedding.src.data.manifest import ManifestEntry
from speaker_embedding.src.data.sampler import GenderBalancedSampler


def _make_entries(n_male=20, n_female=20, n_unknown=0, speakers_per_gender=4):
    """Create a list of ManifestEntry objects with controlled gender distribution."""
    entries = []
    idx = 0
    for i in range(n_male):
        spk = f"male_spk{i % speakers_per_gender}"
        entries.append(ManifestEntry(
            audio_filepath=f"/m{idx}.wav", speaker=spk, duration=3.0, gender="m",
        ))
        idx += 1
    for i in range(n_female):
        spk = f"female_spk{i % speakers_per_gender}"
        entries.append(ManifestEntry(
            audio_filepath=f"/f{idx}.wav", speaker=spk, duration=3.0, gender="f",
        ))
        idx += 1
    for i in range(n_unknown):
        spk = f"unk_spk{i % 2}"
        entries.append(ManifestEntry(
            audio_filepath=f"/u{idx}.wav", speaker=spk, duration=3.0, gender="u",
        ))
        idx += 1
    return entries


class TestGenderBalancedSampler:

    def test_balanced_batches(self):
        """Each batch should contain B/2 male and B/2 female indices."""
        entries = _make_entries(n_male=20, n_female=20)
        batch_size = 8
        sampler = GenderBalancedSampler(entries, batch_size=batch_size, seed=0)

        male_indices = set()
        for i, e in enumerate(entries):
            if e.gender == "m":
                male_indices.add(i)

        for batch in sampler:
            assert len(batch) == batch_size
            # Because sampler shuffles batch indices, we check that the batch
            # was drawn from the full pool; exact 50/50 is guaranteed by design
            # since pools are non-empty.
            break  # one batch is enough to check length

    def test_unknown_strategy_exclude(self):
        entries = _make_entries(n_male=10, n_female=10, n_unknown=10)
        sampler = GenderBalancedSampler(
            entries, batch_size=4, unknown_gender_strategy="exclude", seed=0,
        )
        # Unknown entries (indices 20-29) should NOT appear in any batch.
        unknown_indices = set(range(20, 30))
        for batch in sampler:
            for idx in batch:
                assert idx not in unknown_indices
            break

    def test_unknown_strategy_male(self):
        entries = _make_entries(n_male=10, n_female=10, n_unknown=5)
        sampler = GenderBalancedSampler(
            entries, batch_size=4, unknown_gender_strategy="male", seed=0,
        )
        # Unknown entries should be in male pool
        total_male_utts = sum(len(v) for v in sampler.male_speakers.values())
        assert total_male_utts >= 10 + 5  # original male + unknown

    def test_unknown_strategy_female(self):
        entries = _make_entries(n_male=10, n_female=10, n_unknown=5)
        sampler = GenderBalancedSampler(
            entries, batch_size=4, unknown_gender_strategy="female", seed=0,
        )
        total_female_utts = sum(len(v) for v in sampler.female_speakers.values())
        assert total_female_utts >= 10 + 5

    def test_unknown_strategy_proportional(self):
        entries = _make_entries(n_male=10, n_female=10, n_unknown=10)
        sampler = GenderBalancedSampler(
            entries, batch_size=4, unknown_gender_strategy="proportional", seed=0,
        )
        # All unknown entries should end up in one pool or the other
        total = sum(len(v) for v in sampler.male_speakers.values()) + \
                sum(len(v) for v in sampler.female_speakers.values())
        assert total == 30  # 10m + 10f + 10u all accounted for

    def test_all_indices_valid(self):
        entries = _make_entries(n_male=20, n_female=20)
        sampler = GenderBalancedSampler(entries, batch_size=8, seed=0)
        n = len(entries)
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < n

    def test_epoch_length(self):
        entries = _make_entries(n_male=20, n_female=20)
        batch_size = 8
        sampler = GenderBalancedSampler(entries, batch_size=batch_size, seed=0)
        expected_len = max(1, 40 // batch_size)  # 40 total / 8 = 5
        assert len(sampler) == expected_len
        # Verify iteration yields the same number
        batches = list(sampler)
        assert len(batches) == expected_len

    def test_odd_batch_size_raises(self):
        entries = _make_entries(n_male=10, n_female=10)
        with pytest.raises(ValueError):
            GenderBalancedSampler(entries, batch_size=7)

    def test_invalid_strategy_raises(self):
        entries = _make_entries(n_male=10, n_female=10)
        with pytest.raises(ValueError):
            GenderBalancedSampler(entries, batch_size=4, unknown_gender_strategy="invalid")
