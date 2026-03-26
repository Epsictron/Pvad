"""Trial loading and scoring utilities for speaker verification."""

from typing import Dict, List, Tuple

import numpy as np
import torch


class TrialLoader:
    """Loader for speaker verification trial list files.

    A trial list file contains one trial per line in the format::

        label enroll_id test_id

    where ``label`` is 1 (same speaker) or 0 (different speaker).

    Args:
        trial_path: Path to the trial list file.
    """

    def __init__(self, trial_path: str) -> None:
        self.trial_path = trial_path

    def load(self) -> List[Tuple[str, str, int]]:
        """Load the trial list from disk.

        Returns:
            A list of tuples ``(enroll_id, test_id, label)`` where
            *enroll_id* and *test_id* are utterance identifiers and
            *label* is 1 for target trials or 0 for non-target trials.
        """
        trials: List[Tuple[str, str, int]] = []
        with open(self.trial_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                label = int(parts[0])
                enroll_id = parts[1]
                test_id = parts[2]
                trials.append((enroll_id, test_id, label))
        return trials


@torch.no_grad()
def score_trials(
    embeddings: Dict[str, np.ndarray],
    trials: List[Tuple[str, str, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Score trial pairs using cosine similarity.

    For each trial ``(enroll_id, test_id, label)`` the cosine similarity
    between the corresponding embeddings is computed.  Trials whose
    utterance identifiers are missing from *embeddings* are silently
    skipped.

    Args:
        embeddings: Mapping from utterance identifier to its embedding
            vector (1-D numpy array).
        trials: List of ``(enroll_id, test_id, label)`` tuples as
            returned by :meth:`TrialLoader.load`.

    Returns:
        A tuple ``(scores, labels)`` where *scores* is a 1-D float array
        of cosine similarities and *labels* is a 1-D int array of the
        corresponding ground-truth labels.
    """
    scores_list: List[float] = []
    labels_list: List[int] = []

    for enroll_id, test_id, label in trials:
        if enroll_id not in embeddings or test_id not in embeddings:
            continue

        enroll_emb = embeddings[enroll_id]
        test_emb = embeddings[test_id]

        # Convert to torch tensors for cosine similarity computation.
        enroll_tensor = torch.from_numpy(np.asarray(enroll_emb, dtype=np.float32)).unsqueeze(0)
        test_tensor = torch.from_numpy(np.asarray(test_emb, dtype=np.float32)).unsqueeze(0)

        similarity = torch.nn.functional.cosine_similarity(
            enroll_tensor, test_tensor
        ).item()

        scores_list.append(similarity)
        labels_list.append(label)

    scores = np.array(scores_list, dtype=np.float64)
    labels = np.array(labels_list, dtype=np.int32)

    return scores, labels
