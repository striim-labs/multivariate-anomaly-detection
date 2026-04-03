"""
TranAD Anomaly Scoring and Threshold Calibration

Computes anomaly scores from TranAD reconstruction errors and calibrates
thresholds using POT, percentile, or F1-maximizing methods.

Reference files:
  - tranad/pot.py (pot_eval, bf_search, adjust_predicts, calc_point2point)
  - tranad/diagnosis.py (hit_att, ndcg)
  - tranad/main.py lines 274-345 (inference + scoring flow)
"""

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score, roc_auc_score

from src.spot import SPOT
from src.model import TranADNet
from src.utils import convert_to_windows

logger = logging.getLogger(__name__)


@dataclass
class POTParams:
    """Parameters for POT threshold calibration.

    Per-machine values from reference (SMD dataset):
      machine-1-1: q=1e-5, level=0.99995, scale=1.06
      machine-2-1: q=1e-5, level=0.95,    scale=0.9
      machine-3-2: q=1e-5, level=0.99,    scale=1.0
      machine-3-7: q=1e-5, level=0.99995, scale=1.06
    """

    q: float = 1e-5
    level: float = 0.99995
    scale: float = 1.06


# Per-machine POT parameters from reference (src/constants.py)
DEFAULT_POT_PARAMS = {
    "machine-1-1": POTParams(q=1e-5, level=0.99995, scale=1.06),
    "machine-2-1": POTParams(q=1e-5, level=0.999, scale=0.9),
    "machine-3-2": POTParams(q=1e-5, level=0.99, scale=1.0),
    "machine-3-7": POTParams(q=1e-5, level=0.99995, scale=1.06),
}


# -- Inference --


def score_batch(
    model: TranADNet,
    data: np.ndarray,
    window_size: int = 10,
    device: str | torch.device = "cpu",
    scoring_mode: str = "phase2_only",
) -> np.ndarray:
    """Run TranAD inference and return per-dimension MSE scores.

    Args:
        model: Trained TranADNet (should be in eval mode).
        data: Raw normalized time series, shape (N, n_features).
        window_size: Sliding window size (must match model training).
        device: Torch device.
        scoring_mode: "phase2_only" (reference code) uses z[1] only.
            "averaged" (paper Eq. 13) uses 0.5*MSE(x1) + 0.5*MSE(x2).

    Returns:
        Anomaly scores, shape (N, n_features). Per-dimension MSE.
    """
    model.eval()
    device = torch.device(device) if isinstance(device, str) else device
    n_features = data.shape[1]

    # Match model dtype (float32 or float64)
    model_dtype = next(model.parameters()).dtype
    data_tensor = torch.from_numpy(data).to(model_dtype).to(device)
    windows = convert_to_windows(data_tensor, window_size)  # (N, W, F)

    loss_fn = nn.MSELoss(reduction="none")

    with torch.no_grad():
        # (N, W, F) -> (W, N, F)
        window = windows.permute(1, 0, 2)
        N = window.shape[1]
        elem = window[-1, :, :].view(1, N, n_features)

        x1, x2 = model(window, elem)

        if scoring_mode == "averaged":
            # Paper Eq. 13: s = 0.5*||O1 - W|| + 0.5*||O_hat_2 - W||
            loss = 0.5 * loss_fn(x1, elem)[0] + 0.5 * loss_fn(x2, elem)[0]
        else:
            # Reference code: Phase 2 output only (z[1])
            loss = loss_fn(x2, elem)[0]

    # Always return float32 for consistent downstream processing
    return loss.cpu().float().numpy()


# -- Threshold Calibration --


def calibrate_threshold(
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    method: str = "pot",
    pot_params: POTParams | None = None,
    percentile: float = 99.0,
    f1_search_steps: int = 100,
) -> dict:
    """Calibrate anomaly threshold using the specified method.

    Operates on aggregated scores (mean across features), matching the
    reference pipeline's final evaluation step (tranad/main.py:338-340).

    Args:
        train_scores: Training set scores, shape (N_train, n_features).
        test_scores: Test set scores, shape (N_test, n_features).
        labels: Ground truth labels, shape (N_test, n_features) or (N_test,).
        method: One of "pot", "percentile", "f1_max".
        pot_params: POT-specific parameters (used when method="pot").
        percentile: Percentile value (used when method="percentile").
        f1_search_steps: Search steps (used when method="f1_max").

    Returns:
        dict with 'threshold', 'method', and 'details'.
    """
    # Aggregate to 1-D
    train_1d = np.mean(train_scores, axis=1)
    test_1d = np.mean(test_scores, axis=1)
    if labels.ndim == 2:
        labels_1d = (np.sum(labels, axis=1) >= 1).astype(int)
    else:
        labels_1d = labels

    if method == "pot":
        params = pot_params or POTParams()
        threshold = _pot_threshold(train_1d, test_1d, params)
        return {
            "threshold": float(threshold),
            "method": "pot",
            "details": {"q": params.q, "level": params.level, "scale": params.scale},
        }
    elif method == "percentile":
        threshold = _percentile_threshold(train_1d, percentile)
        return {
            "threshold": float(threshold),
            "method": "percentile",
            "details": {"percentile": percentile},
        }
    elif method == "f1_max":
        threshold, metrics = _f1_max_threshold(
            test_1d, labels_1d, f1_search_steps
        )
        return {
            "threshold": float(threshold),
            "method": "f1_max",
            "details": metrics,
        }
    else:
        raise ValueError(f"Unknown method: {method}. Use pot, percentile, or f1_max.")


def _pot_threshold(
    train_scores_1d: np.ndarray,
    test_scores_1d: np.ndarray,
    pot_params: POTParams,
) -> float:
    """Compute POT threshold on 1-D score arrays.

    Implements the retry loop from tranad/pot.py:135-141:
    while SPOT.initialize fails, reduce level by *0.999.

    After computing the POT threshold, validates it by checking the
    predicted anomaly rate. If >20% of test data would be flagged (which
    indicates the threshold is unreasonably low relative to the test
    score distribution), falls back to the 99.9th percentile of test
    scores as a conservative threshold.
    """
    lms = pot_params.level
    max_retries = 1000
    for _ in range(max_retries):
        try:
            s = SPOT(pot_params.q)
            s.fit(train_scores_1d, test_scores_1d)
            s.initialize(level=lms, min_extrema=False, verbose=False)
        except Exception:
            lms = lms * 0.999
        else:
            break

    ret = s.run(dynamic=False)
    pot_th = np.mean(ret["thresholds"]) * pot_params.scale

    # Validate: if threshold flags >20% of test data, it's unreliable.
    anomaly_rate = float(np.mean(test_scores_1d > pot_th))
    if anomaly_rate > 0.20:
        pot_th = float(np.percentile(test_scores_1d, 99.9))

    return pot_th


def _percentile_threshold(
    train_scores_1d: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """Compute threshold as a percentile of training scores."""
    return float(np.percentile(train_scores_1d, percentile))


def _f1_max_threshold(
    test_scores_1d: np.ndarray,
    labels_1d: np.ndarray,
    step_num: int = 100,
) -> tuple[float, dict]:
    """Find threshold that maximizes F1 via brute-force search.

    Reference: tranad/pot.py, bf_search().
    """
    start = float(test_scores_1d.min())
    end = float(test_scores_1d.max())
    search_range = end - start

    best_f1 = -1.0
    best_threshold = start
    best_metrics = {}

    for i in range(step_num):
        threshold = start + search_range * (i + 1) / step_num
        predict = adjust_predicts(
            test_scores_1d, labels_1d, threshold
        )
        actual = (labels_1d > 0.1).astype(float)
        metrics = _calc_point2point(predict, actual)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
            best_metrics = metrics

    return best_threshold, best_metrics


# -- Evaluation --


def evaluate(
    test_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    """Evaluate anomaly detection with point-adjustment protocol.

    Pipeline:
    1. Aggregate scores: mean across features -> (N_test,)
    2. Aggregate labels: (sum >= 1) -> binary (N_test,)
    3. Apply threshold to get predictions
    4. Apply point-adjustment (adjust_predicts)
    5. Compute precision, recall, F1, AUC

    Returns:
        dict with 'f1', 'precision', 'recall', 'roc_auc',
        'TP', 'TN', 'FP', 'FN', 'threshold'.
    """
    score_1d = np.mean(test_scores, axis=1)
    if labels.ndim == 2:
        labels_1d = (np.sum(labels, axis=1) >= 1).astype(float)
    else:
        labels_1d = labels.astype(float)

    predict = adjust_predicts(score_1d, labels_1d, threshold)
    actual = (labels_1d > 0.1).astype(float)
    metrics = _calc_point2point(predict, actual)
    metrics["threshold"] = threshold
    return metrics


def adjust_predicts(
    score: np.ndarray,
    label: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Point-adjustment protocol for time series anomaly detection.

    If ANY point within a contiguous anomaly segment is correctly
    predicted, ALL points in that segment are marked as detected.

    This is the standard evaluation protocol used by OmniAnomaly,
    TranAD, and other TSAD benchmarks.

    Reference: tranad/pot.py, adjust_predicts() (lines 29-75).
    """
    score = np.asarray(score)
    label = np.asarray(label)
    predict = (score > threshold).astype(float)
    actual = (label > 0.1)
    anomaly_state = False

    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    if not predict[j]:
                        predict[j] = 1.0
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = 1.0

    return predict


def _calc_point2point(
    predict: np.ndarray,
    actual: np.ndarray,
) -> dict:
    """Compute precision/recall/F1/AUC from binary predictions.

    Reference: tranad/pot.py, calc_point2point().
    """
    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 1e-5)
    recall = TP / (TP + FN + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    try:
        roc_auc = roc_auc_score(actual, predict)
    except Exception:
        roc_auc = 0.0
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc_auc),
        "TP": int(TP),
        "TN": int(TN),
        "FP": int(FP),
        "FN": int(FN),
    }


# -- Root Cause Diagnosis --


def diagnose(
    test_scores: np.ndarray,
    interp_labels: np.ndarray,
    ps: list[int] | None = None,
) -> dict:
    """Compute root cause attribution metrics.

    Reference: tranad/diagnosis.py, hit_att() and ndcg().
    """
    if ps is None:
        ps = [100, 150]
    result = {}
    result.update(_hit_att(test_scores, interp_labels, ps))
    result.update(_ndcg(test_scores, interp_labels, ps))
    return result


def _hit_att(
    ascore: np.ndarray,
    labels: np.ndarray,
    ps: list[int],
) -> dict:
    """HitRate@k%: fraction of true anomalous dims in top-k ranked dims.

    Reference: tranad/diagnosis.py, hit_att().
    """
    res = {}
    for p in ps:
        hit_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            a = np.argsort(a).tolist()[::-1]  # dims ranked by score (descending)
            l = set(np.where(l == 1)[0])  # true anomalous dims
            if l:
                size = round(p * len(l) / 100)
                a_p = set(a[:size])
                hit = len(a_p.intersection(l)) / len(l)
                hit_scores.append(hit)
        res[f"Hit@{p}%"] = float(np.mean(hit_scores)) if hit_scores else 0.0
    return res


def _ndcg(
    ascore: np.ndarray,
    labels: np.ndarray,
    ps: list[int],
) -> dict:
    """NDCG@k% ranking quality of anomalous dimension identification.

    Reference: tranad/diagnosis.py, ndcg().
    """
    res = {}
    for p in ps:
        ndcg_scores = []
        for i in range(ascore.shape[0]):
            a, l = ascore[i], labels[i]
            labs = list(np.where(l == 1)[0])
            if labs:
                k_p = round(p * len(labs) / 100)
                try:
                    hit = ndcg_score(
                        l.reshape(1, -1), a.reshape(1, -1), k=k_p
                    )
                except Exception:
                    continue
                ndcg_scores.append(hit)
        res[f"NDCG@{p}%"] = float(np.mean(ndcg_scores)) if ndcg_scores else 0.0
    return res


# -- Feature Attribution --


def compute_feature_baselines(
    train_scores: np.ndarray,
    percentile: float = 95.0,
    floor: float = 1e-8,
) -> np.ndarray:
    """Compute per-feature baseline scores from training data.

    The baseline represents "normal" reconstruction error for each feature.
    Features that are inherently harder to reconstruct have higher baselines.
    Elevation-ratio normalization (score / baseline) removes the bias toward
    always-high-error features.

    Returns:
        Baseline scores, shape (n_features,). Each value >= floor.
    """
    baselines = np.percentile(train_scores, percentile, axis=0)  # (F,)
    baselines = np.maximum(baselines, floor)
    return baselines


def attribute_dimensions(
    segment_scores: np.ndarray,
    baselines: np.ndarray,
    min_elevation: float = 2.0,
    contribution_threshold: float = 0.80,
    max_features: int = 10,
    feature_labels: list[str] | None = None,
    batch_data: np.ndarray | None = None,
    history_data: np.ndarray | None = None,
) -> list[dict]:
    """Attribute an anomaly segment to specific feature dimensions.

    Ranks features by their mean elevation ratio (score / baseline) across
    the segment, and returns the top contributors that explain at least
    ``contribution_threshold`` of the total excess score.
    """
    n_features = segment_scores.shape[1]
    if feature_labels is None:
        feature_labels = [f"dim_{i}" for i in range(n_features)]

    # Mean elevation ratio per feature across the segment
    elevation_ratios = segment_scores / baselines  # (T, F)
    mean_elevation = np.mean(elevation_ratios, axis=0)  # (F,)

    # Contribution: feature's share of total excess score across segment
    excess = np.maximum(segment_scores - baselines, 0)  # (T, F)
    feature_excess = np.sum(excess, axis=0)  # (F,)
    total_excess = np.sum(feature_excess)

    if total_excess < 1e-12:
        return []

    contributions = feature_excess / total_excess  # (F,)

    # Rank by mean elevation descending
    ranked_indices = np.argsort(mean_elevation)[::-1]

    attributed = []
    cumulative_contribution = 0.0

    for idx in ranked_indices:
        idx = int(idx)
        if mean_elevation[idx] < min_elevation:
            break
        if len(attributed) >= max_features:
            break
        entry = {
            "dim": idx,
            "label": feature_labels[idx],
            "mean_elevation": round(float(mean_elevation[idx]), 4),
            "contribution": round(float(contributions[idx]), 4),
        }
        if batch_data is not None:
            mean_src = history_data if history_data is not None else batch_data
            mean_val = float(np.mean(mean_src[:, idx]))
            batch_dim = batch_data[:, idx]
            extreme_val = float(
                batch_dim[np.argmax(np.abs(batch_dim - mean_val))]
            )
            entry["mean_value"] = round(mean_val, 6)
            entry["extreme_value"] = round(extreme_val, 6)
        attributed.append(entry)
        cumulative_contribution += contributions[idx]
        if cumulative_contribution >= contribution_threshold:
            break

    return attributed


def find_anomaly_segments(
    predictions: np.ndarray,
) -> list[tuple[int, int]]:
    """Find contiguous runs of predicted anomalies.

    Returns:
        List of (start, end) tuples (inclusive on both ends).
    """
    binary = (np.asarray(predictions) > 0.5).astype(int)
    padded = np.concatenate([[0], binary, [0]])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def build_segment_summaries(
    test_scores: np.ndarray,
    predictions: np.ndarray,
    baselines: np.ndarray,
    feature_labels: list[str] | None = None,
    min_elevation: float = 2.0,
    contribution_threshold: float = 0.80,
    normalized_data: np.ndarray | None = None,
    history_data: np.ndarray | None = None,
) -> list[dict]:
    """Build structured attribution summaries for all anomaly segments."""
    segments = find_anomaly_segments(predictions)
    score_1d = np.mean(test_scores, axis=1)

    summaries = []
    for start, end in segments:
        seg_scores = test_scores[start : end + 1]  # (T, F)
        seg_1d = score_1d[start : end + 1]

        peak_offset = int(np.argmax(seg_1d))

        attributed = attribute_dimensions(
            seg_scores,
            baselines,
            min_elevation=min_elevation,
            contribution_threshold=contribution_threshold,
            feature_labels=feature_labels,
            batch_data=normalized_data,
            history_data=history_data,
        )

        summaries.append({
            "segment_start": int(start),
            "segment_end": int(end),
            "segment_length": int(end - start + 1),
            "peak_score": round(float(seg_1d[peak_offset]), 6),
            "peak_timestamp": int(start + peak_offset),
            "mean_score": round(float(np.mean(seg_1d)), 6),
            "attributed_dimensions": attributed,
        })

    return summaries


def diagnose_with_elevation(
    test_scores: np.ndarray,
    interp_labels: np.ndarray,
    baselines: np.ndarray,
    ps: list[int] | None = None,
) -> dict:
    """Compute root cause attribution metrics using elevation-ratio ranking.

    Same as diagnose(), but ranks features by elevation ratio
    (score / baseline) instead of raw score.
    """
    if ps is None:
        ps = [100, 150]

    elevation_scores = test_scores / baselines  # (N, F)
    raw_result = diagnose(elevation_scores, interp_labels, ps)
    return {f"{k}_elev": v for k, v in raw_result.items()}
