"""
TranAD Model Architecture

Transformer-based anomaly detection with two-phase self-conditioning
and adversarial training. Adapted from imperial-qore/TranAD (BSD-3-Clause).

"""

from dataclasses import dataclass


@dataclass
class TranADConfig:
    """TranAD hyperparameters."""

    window_size: int = 10
    n_features: int = 38
    n_heads: int = 8
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    d_model: int = 0  # Set to n_features * 2 if 0
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 5
    batch_size: int = 128


class TranADNet:
    """Core TranAD neural network. TODO: Implement in Phase 2."""

    pass


class TranAD:
    """TranAD training/inference wrapper. TODO: Implement in Phase 2."""

    pass
