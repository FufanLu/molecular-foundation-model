"""Threshold selection, class weighting, and retrieval probes for evaluation.

These helpers keep the evaluation protocol honest:

- Per-label decision thresholds are selected on a validation split and only
  then applied, once, to a held-out test split.
- ``pos_weight`` is computed from a training split only, so rare labels keep
  gradient signal without peeking at validation/test label frequencies.
- Retrieval probes measure whether the shared projection space groups
  molecules with identical sensory label sets, which is the observable
  behaviour the alignment objectives claim to encourage.
"""

from __future__ import annotations

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def _f1(expected: np.ndarray, predicted: np.ndarray) -> float:
    true_positive = float(((predicted == 1) & (expected == 1)).sum())
    false_positive = float(((predicted == 1) & (expected == 0)).sum())
    false_negative = float(((predicted == 0) & (expected == 1)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def compute_pos_weight(targets: np.ndarray, cap: float = 10.0) -> np.ndarray:
    """Per-label negative/positive ratio over observed entries, clamped to [1, cap].

    Unknown entries (-1) never count as negatives. Labels without observed
    positives get weight 1.0 so their BCE is unchanged rather than amplified.
    """
    targets = np.asarray(targets, dtype=np.float64)
    if cap < 1.0:
        raise ValueError("pos_weight cap must be at least 1.0")
    positives = (targets == 1).sum(axis=0)
    negatives = (targets == 0).sum(axis=0)
    weight = np.clip(negatives / np.maximum(positives, 1), 1.0, cap)
    # No positive evidence (including fully unknown labels): pos_weight would
    # never apply to a real positive, so keep the BCE unchanged instead of
    # recording a misleading ratio.
    weight[positives == 0] = 1.0
    return weight.astype(np.float32)


def select_thresholds(
    logits: np.ndarray,
    targets: np.ndarray,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Pick the F1-maximising decision threshold per label on the given split.

    Labels with no observed positives keep the 0.5 default, because an
    F1-optimal threshold is undefined without positive evidence.
    """
    probabilities = _sigmoid(logits)
    if grid is None:
        grid = np.round(np.arange(0.02, 0.99, 0.02), 2)
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float64)
    for column in range(targets.shape[1]):
        observed = targets[:, column] >= 0
        expected = targets[observed, column]
        if expected.size == 0 or not (expected == 1).any():
            continue
        column_probabilities = probabilities[observed, column]
        best_threshold, best_f1 = 0.5, -1.0
        for threshold in grid:
            score = _f1(expected, (column_probabilities >= threshold).astype(np.float64))
            if score > best_f1 + 1e-12:
                best_f1, best_threshold = score, float(threshold)
        thresholds[column] = best_threshold
    return thresholds


def macro_f1(
    logits: np.ndarray,
    targets: np.ndarray,
    labels: tuple[str, ...],
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """Per-label F1 over observed rows only; unknown entries (-1) are excluded."""
    probabilities = _sigmoid(np.asarray(logits, dtype=np.float64))
    if thresholds is None:
        thresholds = np.full(targets.shape[1], 0.5, dtype=np.float64)
    scores: dict[str, float] = {}
    for column, label in enumerate(labels):
        observed = targets[:, column] >= 0
        scores[label] = (
            _f1(
                targets[observed, column],
                (probabilities[observed, column] >= thresholds[column]).astype(np.float64),
            )
            if observed.any()
            else float("nan")
        )
    scores["macro"] = float(np.nanmean(list(scores.values())))
    return scores


def profile_retrieval(
    projections: np.ndarray,
    targets: np.ndarray,
    query_mask: np.ndarray,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, float | int]:
    """Rank molecules by projection similarity; relevant = identical label set.

    Every molecule with observed labels forms the candidate pool. Each query
    is excluded from its own ranking so the probe cannot score by identity.
    Queries whose label set is unique in the pool are dropped from the
    average and are not counted in ``queries``.
    """
    result: dict[str, float | int] = {f"recall@{k}": float("nan") for k in ks}
    result["mrr"] = float("nan")
    result["queries"] = 0
    observed = (targets >= 0).any(axis=1)
    queries = np.flatnonzero(np.asarray(query_mask, dtype=bool) & observed)
    candidates = np.flatnonzero(observed)
    if queries.size == 0 or candidates.size < 2:
        return result
    positives = targets == 1
    similarity = np.asarray(projections, dtype=np.float64)[queries] @ projections[candidates].T
    position_of = {int(index): position for position, index in enumerate(candidates)}
    for row, query in enumerate(queries):
        similarity[row, position_of[int(query)]] = -np.inf
    relevant = (positives[queries][:, None, :] == positives[candidates][None, :, :]).all(axis=-1)
    for row, query in enumerate(queries):
        relevant[row, position_of[int(query)]] = False  # self is never a valid hit
    has_relevant = relevant.any(axis=1)
    if not has_relevant.any():
        return result
    order = np.argsort(-similarity[has_relevant], axis=1)
    sorted_relevance = np.take_along_axis(relevant[has_relevant], order, axis=1)
    for k in ks:
        result[f"recall@{k}"] = float(sorted_relevance[:, :k].any(axis=1).mean())
    result["mrr"] = float((1.0 / (sorted_relevance.argmax(axis=1) + 1)).mean())
    result["queries"] = int(has_relevant.sum())
    return result
