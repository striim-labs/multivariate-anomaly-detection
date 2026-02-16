"""
TranAD Anomaly Scoring and Threshold Calibration

Computes anomaly scores from TranAD reconstruction errors and calibrates
thresholds using POT, percentile, or F1-maximizing methods.

Reference files:
  - tranad/pot.py (pot_eval, bf_search, adjust_predicts, calc_point2point)
  - tranad/diagnosis.py (hit_att, ndcg)
  - tranad/main.py lines 274-345 (inference + scoring flow)
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score, roc_auc_score

from spot import SPOT
from tranad_model import TranADNet
from tranad_utils import convert_to_windows


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


class TranADScorer:
    """Anomaly scoring with threshold calibration (POT, percentile, F1-max).

    Follows the reference scoring pipeline from tranad/main.py:
    1. Run inference to get per-dimension reconstruction errors
    2. Calibrate threshold on train errors + test errors
    3. Apply point-adjustment protocol for evaluation
    4. Compute F1/precision/recall/AUC metrics
    """

    # ── Inference ──────────────────────────────────────────────────────

    @staticmethod
    def score_batch(
        model: TranADNet,
        data: np.ndarray,
        window_size: int = 10,
        device: str | torch.device = "cpu",
    ) -> np.ndarray:
        """Run TranAD inference and return per-dimension MSE scores.

        Replicates the reference test-time backprop (tranad/main.py:274-281):
        - Uses Phase 2 output (z[1]) only
        - MSE with reduction='none' gives per-dimension scores
        - Processes all data in a single forward pass

        Args:
            model: Trained TranADNet (should be in eval mode).
            data: Raw normalized time series, shape (N, n_features), float32.
            window_size: Sliding window size (must match model training).
            device: Torch device.

        Returns:
            Anomaly scores, shape (N, n_features). Per-dimension MSE
            between the Phase 2 reconstruction and the ground truth.
        """
        model.eval()
        device = torch.device(device) if isinstance(device, str) else device
        n_features = data.shape[1]

        data_tensor = torch.from_numpy(data).float().to(device)
        windows = convert_to_windows(data_tensor, window_size)  # (N, W, F)

        loss_fn = nn.MSELoss(reduction="none")

        with torch.no_grad():
            # (N, W, F) -> (W, N, F)
            window = windows.permute(1, 0, 2)
            N = window.shape[1]
            elem = window[-1, :, :].view(1, N, n_features)

            x1, x2 = model(window, elem)

            # Use Phase 2 output only (z[1] in reference)
            # loss shape: (1, N, F) -> [0] -> (N, F)
            loss = loss_fn(x2, elem)[0]

        return loss.cpu().numpy()

    # ── Threshold Calibration ──────────────────────────────────────────

    def calibrate_threshold(
        self,
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
            threshold = self._pot_threshold(train_1d, test_1d, params)
            return {
                "threshold": float(threshold),
                "method": "pot",
                "details": {"q": params.q, "level": params.level, "scale": params.scale},
            }
        elif method == "percentile":
            threshold = self._percentile_threshold(train_1d, percentile)
            return {
                "threshold": float(threshold),
                "method": "percentile",
                "details": {"percentile": percentile},
            }
        elif method == "f1_max":
            threshold, metrics = self._f1_max_threshold(
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
        self,
        train_scores_1d: np.ndarray,
        test_scores_1d: np.ndarray,
        pot_params: POTParams,
    ) -> float:
        """Compute POT threshold on 1-D score arrays.

        Implements the retry loop from tranad/pot.py:135-141:
        while SPOT.initialize fails, reduce level by *0.999.

        Args:
            train_scores_1d: Training scores, shape (N_train,).
            test_scores_1d: Test scores, shape (N_test,).
            pot_params: POT parameters.

        Returns:
            Calibrated threshold (float).
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
        return pot_th

    @staticmethod
    def _percentile_threshold(
        train_scores_1d: np.ndarray,
        percentile: float = 99.0,
    ) -> float:
        """Compute threshold as a percentile of training scores."""
        return float(np.percentile(train_scores_1d, percentile))

    @staticmethod
    def _f1_max_threshold(
        test_scores_1d: np.ndarray,
        labels_1d: np.ndarray,
        step_num: int = 100,
    ) -> tuple[float, dict]:
        """Find threshold that maximizes F1 via brute-force search.

        Reference: tranad/pot.py, bf_search().

        Args:
            test_scores_1d: Test scores, shape (N_test,).
            labels_1d: Binary labels, shape (N_test,).
            step_num: Number of candidate thresholds.

        Returns:
            (best_threshold, best_metrics_dict)
        """
        start = float(test_scores_1d.min())
        end = float(test_scores_1d.max())
        search_range = end - start

        best_f1 = -1.0
        best_threshold = start
        best_metrics = {}

        for i in range(step_num):
            threshold = start + search_range * (i + 1) / step_num
            predict = TranADScorer._adjust_predicts(
                test_scores_1d, labels_1d, threshold
            )
            actual = (labels_1d > 0.1).astype(float)
            metrics = TranADScorer._calc_point2point(predict, actual)
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = threshold
                best_metrics = metrics

        return best_threshold, best_metrics

    # ── Evaluation ─────────────────────────────────────────────────────

    @staticmethod
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

        Args:
            test_scores: Shape (N_test, n_features).
            labels: Shape (N_test, n_features) or (N_test,).
            threshold: Anomaly threshold.

        Returns:
            dict with 'f1', 'precision', 'recall', 'roc_auc',
            'TP', 'TN', 'FP', 'FN', 'threshold'.
        """
        score_1d = np.mean(test_scores, axis=1)
        if labels.ndim == 2:
            labels_1d = (np.sum(labels, axis=1) >= 1).astype(float)
        else:
            labels_1d = labels.astype(float)

        predict = TranADScorer._adjust_predicts(score_1d, labels_1d, threshold)
        actual = (labels_1d > 0.1).astype(float)
        metrics = TranADScorer._calc_point2point(predict, actual)
        metrics["threshold"] = threshold
        return metrics

    @staticmethod
    def _adjust_predicts(
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

        Args:
            score: Anomaly scores, shape (N,).
            label: Ground truth binary labels, shape (N,).
            threshold: Detection threshold.

        Returns:
            Adjusted binary predictions, shape (N,), dtype float.
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

    @staticmethod
    def _calc_point2point(
        predict: np.ndarray,
        actual: np.ndarray,
    ) -> dict:
        """Compute precision/recall/F1/AUC from binary predictions.

        Reference: tranad/pot.py, calc_point2point().

        Args:
            predict: Binary predictions, shape (N,).
            actual: Binary ground truth, shape (N,).

        Returns:
            dict with 'f1', 'precision', 'recall', 'roc_auc',
            'TP', 'TN', 'FP', 'FN'.
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

    # ── Root Cause Diagnosis ───────────────────────────────────────────

    @staticmethod
    def diagnose(
        test_scores: np.ndarray,
        interp_labels: np.ndarray,
        ps: list[int] | None = None,
    ) -> dict:
        """Compute root cause attribution metrics.

        Reference: tranad/diagnosis.py, hit_att() and ndcg().

        Args:
            test_scores: Per-dimension scores, shape (N_test, n_features).
            interp_labels: Per-dimension binary labels, shape (N_test, n_features).
            ps: Percentile levels for HitRate/NDCG (default: [100, 150]).

        Returns:
            dict with 'Hit@100%', 'Hit@150%', 'NDCG@100%', 'NDCG@150%'.
        """
        if ps is None:
            ps = [100, 150]
        result = {}
        result.update(TranADScorer._hit_att(test_scores, interp_labels, ps))
        result.update(TranADScorer._ndcg(test_scores, interp_labels, ps))
        return result

    @staticmethod
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

    @staticmethod
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
