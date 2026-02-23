"""
TranAD Streaming Detector

Maintains per-device sliding window buffers and scores incoming telemetry.
Implements BaseDetector for use in both REST and streaming paths.
"""

import logging

import numpy as np

from base_detector import BaseDetector
from tranad_registry import TranADRegistry
from tranad_scorer import TranADScorer

logger = logging.getLogger(__name__)


class TranADStreamingDetector(BaseDetector):
    """Streaming anomaly detector wrapping TranAD for real-time use.

    Manages a sliding window buffer of recent timesteps. When enough
    data has accumulated (>= window_size), calls score_batch and
    returns detection results.

    Args:
        store_id: Store identifier.
        device_id: Device identifier within the store.
        registry: TranADRegistry instance.
        norm_params: Normalization params array, shape (2, n_features).
        scorer_state: Dict with 'threshold' and 'feature_baselines'.
        device: Torch device string.
        window_size: Sliding window size (default 10).
        buffer_size: Maximum buffer length before truncation (default 1000).
    """

    def __init__(
        self,
        store_id: int,
        device_id: int,
        registry: TranADRegistry,
        norm_params: np.ndarray,
        scorer_state: dict,
        device: str = "cpu",
        window_size: int = 10,
        buffer_size: int = 1000,
    ):
        self._store_id = store_id
        self._device_id = device_id
        self._machine_key = f"machine-{store_id}-{device_id}"
        self._device = device
        self._window_size = window_size
        self._buffer_size = buffer_size

        self._threshold = scorer_state["threshold"]
        self._baselines = np.array(scorer_state.get("feature_baselines", []))
        self._min_vals = norm_params[0]
        self._max_vals = norm_params[1]

        # Load model
        self._model, self._config = registry.get_model(self._machine_key, device)
        self._n_features = self._config.n_features

        # Buffer for incoming raw timesteps
        self._buffer: list[np.ndarray] = []
        self._total_received = 0
        self._ready = True

        logger.info(
            "TranADStreamingDetector initialized for store=%d device=%d "
            "(features=%d, window=%d, threshold=%.6f)",
            store_id,
            device_id,
            self._n_features,
            self._window_size,
            self._threshold,
        )

    def detect(self, data: np.ndarray) -> dict:
        """Run anomaly detection on incoming data.

        Args:
            data: Raw (unnormalized) timestep(s), shape (n_features,) for
                  single or (N, n_features) for batch.

        Returns:
            Dict with 'has_result', and if True: 'scores_1d', 'predictions',
            'segments', 'n_anomalies', 'buffer_size'.
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)

        if data.shape[1] != self._n_features:
            raise ValueError(
                f"Expected {self._n_features} features, got {data.shape[1]}"
            )

        for row in data:
            self._buffer.append(row)
            self._total_received += 1

        # Truncate if buffer exceeds max size
        if len(self._buffer) > self._buffer_size:
            self._buffer = self._buffer[-self._buffer_size :]

        if len(self._buffer) < self._window_size:
            return {
                "has_result": False,
                "buffer_size": len(self._buffer),
                "min_required": self._window_size,
            }

        # Normalize the full buffer
        raw = np.array(self._buffer, dtype=np.float64)
        normalized = (raw - self._min_vals) / (
            self._max_vals - self._min_vals + 1e-4
        )

        # Score
        scores = TranADScorer.score_batch(
            self._model,
            normalized,
            window_size=self._window_size,
            device=self._device,
        )
        scores_1d = np.mean(scores, axis=1)
        predictions = (scores_1d > self._threshold).astype(int)

        # Build segments with attribution
        # History = buffer rows before the current batch; current batch for extreme_value
        segments = []
        if self._baselines.size > 0 and int(predictions.sum()) > 0:
            n_current = len(data)
            history = raw[:-n_current] if len(raw) > n_current else None
            segments = TranADScorer.build_segment_summaries(
                scores, predictions, self._baselines,
                normalized_data=raw, history_data=history,
            )

        return {
            "has_result": True,
            "scores_1d": scores_1d.tolist(),
            "predictions": predictions.tolist(),
            "segments": segments,
            "n_anomalies": int(predictions.sum()),
            "buffer_size": len(self._buffer),
            "total_received": self._total_received,
        }

    def get_stats(self) -> dict:
        return {
            "detector": "tranad",
            "store_id": self._store_id,
            "device_id": self._device_id,
            "device": self._device,
            "window_size": self._window_size,
            "n_features": self._n_features,
            "threshold": self._threshold,
            "buffer_size": len(self._buffer),
            "total_received": self._total_received,
            "ready": self._ready,
        }

    def get_name(self) -> str:
        return f"TranAD Streaming Detector (store={self._store_id}, device={self._device_id})"

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def min_samples_required(self) -> int:
        return self._window_size
