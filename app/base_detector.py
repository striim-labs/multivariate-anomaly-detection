"""
Abstract base class for streaming anomaly detectors.

All detector implementations (TranAD, etc.) should inherit from BaseDetector
and implement the required abstract methods.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Abstract interface for streaming anomaly detectors."""

    @abstractmethod
    def detect(self, data: np.ndarray) -> dict:
        """Run anomaly detection on incoming data."""
        ...

    @abstractmethod
    def get_stats(self) -> dict:
        """Return detector configuration and status."""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return the detector display name."""
        ...

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the detector model is loaded and ready."""
        ...

    @property
    @abstractmethod
    def min_samples_required(self) -> int:
        """Minimum number of samples needed before detection can begin."""
        ...
