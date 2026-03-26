"""Speaker verification evaluation metrics."""

from typing import Tuple

import numpy as np


def compute_eer(
    scores: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """Compute the Equal Error Rate (EER).

    The EER is the point where the False Acceptance Rate (FAR) equals the
    False Rejection Rate (FRR). The threshold at this operating point is
    also returned.

    Args:
        scores: Similarity scores where higher values indicate greater
            likelihood of the same speaker.
        labels: Binary labels where 1 indicates same speaker (target)
            and 0 indicates different speaker (non-target).

    Returns:
        A tuple of (eer, threshold_at_eer). The EER is expressed as a
        fraction in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    # Sort thresholds in descending order so that FAR starts low and FRR
    # starts high as we sweep from strict to lenient.
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]

    num_target = np.sum(labels == 1)
    num_nontarget = np.sum(labels == 0)

    # Sweep thresholds from lowest to highest score.  At threshold t every
    # score >= t is accepted.
    # FAR  = #(non-target accepted) / #(non-target)
    # FRR  = #(target rejected)     / #(target)
    #
    # Walking from the lowest threshold upward, each score that crosses below
    # the threshold turns from accepted to rejected.
    far = np.zeros(len(sorted_scores) + 1, dtype=np.float64)
    frr = np.zeros(len(sorted_scores) + 1, dtype=np.float64)

    # Index 0: threshold = -inf  -> everything accepted
    far[0] = 1.0 if num_nontarget > 0 else 0.0
    frr[0] = 0.0

    cum_nontarget = 0
    cum_target = 0
    for i in range(len(sorted_scores)):
        if sorted_labels[i] == 0:
            cum_nontarget += 1
        else:
            cum_target += 1
        # Threshold just above sorted_scores[i]: scores[i] is now rejected.
        far[i + 1] = (num_nontarget - cum_nontarget) / max(num_nontarget, 1)
        frr[i + 1] = cum_target / max(num_target, 1)

    # Find the crossover point where FAR and FRR intersect.
    diffs = far - frr
    # Find the index just before the sign change.
    idx = np.where(diffs[:-1] * diffs[1:] <= 0)[0]

    if len(idx) == 0:
        # Fallback: pick the point with the smallest absolute difference.
        abs_diffs = np.abs(diffs)
        best = np.argmin(abs_diffs)
        eer = float((far[best] + frr[best]) / 2.0)
        threshold = float(sorted_scores[min(best, len(sorted_scores) - 1)])
        return eer, threshold

    # Linear interpolation at the first crossover.
    i = idx[0]
    if diffs[i] == diffs[i + 1]:
        alpha = 0.5
    else:
        alpha = float(diffs[i] / (diffs[i] - diffs[i + 1]))

    eer = float(far[i] + alpha * (far[i + 1] - far[i]))

    # Interpolate threshold.  Index 0 corresponds to -inf, indices 1..N map
    # to sorted_scores[0..N-1].
    if i == 0:
        threshold = float(sorted_scores[0])
    elif i >= len(sorted_scores):
        threshold = float(sorted_scores[-1])
    else:
        t_low = sorted_scores[i - 1]
        t_high = sorted_scores[min(i, len(sorted_scores) - 1)]
        threshold = float(t_low + alpha * (t_high - t_low))

    return eer, threshold


def compute_min_dcf(
    scores: np.ndarray,
    labels: np.ndarray,
    p_target: float = 0.01,
    c_fa: float = 1.0,
    c_miss: float = 1.0,
) -> float:
    """Compute the minimum Detection Cost Function (minDCF).

    The DCF at a given threshold is defined as::

        DCF = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)

    The result is normalized by the best cost achievable without any
    system (i.e., always accept or always reject)::

        DCF_norm = DCF / min(c_miss * p_target, c_fa * (1 - p_target))

    The minimum over all thresholds is returned.

    Args:
        scores: Similarity scores where higher values indicate greater
            likelihood of the same speaker.
        labels: Binary labels where 1 indicates same speaker (target)
            and 0 indicates different speaker (non-target).
        p_target: Prior probability of a target trial.
        c_fa: Cost of a false acceptance.
        c_miss: Cost of a miss (false rejection).

    Returns:
        The minimum normalized DCF value.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    num_target = np.sum(labels == 1)
    num_nontarget = np.sum(labels == 0)

    if num_target == 0 or num_nontarget == 0:
        return 0.0

    # Use every unique score as a candidate threshold, plus +/- inf.
    thresholds = np.sort(scores)
    thresholds = np.concatenate(
        [thresholds, [thresholds[-1] + 1.0]]
    )

    min_dcf = float("inf")
    default_cost = min(c_miss * p_target, c_fa * (1 - p_target))

    for thresh in thresholds:
        # Scores >= threshold are accepted as target.
        decisions = scores >= thresh
        p_miss = np.sum((labels == 1) & ~decisions) / num_target
        p_fa = np.sum((labels == 0) & decisions) / num_nontarget

        dcf = c_miss * p_miss * p_target + c_fa * p_fa * (1 - p_target)
        dcf_norm = dcf / default_cost
        if dcf_norm < min_dcf:
            min_dcf = dcf_norm

    return float(min_dcf)
