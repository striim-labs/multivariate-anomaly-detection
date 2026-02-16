"""
TranAD Streaming Detector for Kafka/Spark Pipeline

Maintains per-device sliding window buffers and scores incoming telemetry.

"""

import numpy as np

from base_detector import BaseDetector


class TranADStreamingDetector(BaseDetector):
    """Streaming anomaly detector wrapping TranAD for real-time use."""

    def detect(self, data: np.ndarray) -> dict:
        raise NotImplementedError("Streaming detector not yet implemented")

    def get_stats(self) -> dict:
        return {"detector": "tranad", "status": "not_implemented"}

    def get_name(self) -> str:
        return "TranAD Streaming Detector"

    @property
    def is_ready(self) -> bool:
        return False

    @property
    def min_samples_required(self) -> int:
        return 10
