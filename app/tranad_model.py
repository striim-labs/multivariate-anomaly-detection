"""
TranAD Model Architecture

Transformer-based anomaly detection with two-phase self-conditioning
and adversarial training. Adapted from imperial-qore/TranAD (BSD-3-Clause).

Reference files:
  - tranad/models.py (class TranAD, lines 491-524)
  - tranad/dlutils.py (PositionalEncoding, TransformerEncoderLayer, TransformerDecoderLayer)
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class TranADConfig:
    """TranAD hyperparameters.

    Default values match the reference implementation for SMD (38 features).
    d_model is always 2 * n_features (set automatically if 0).
    n_heads must evenly divide d_model.
    """

    window_size: int = 10
    n_features: int = 38
    n_heads: int = 38
    n_encoder_layers: int = 1
    n_decoder_layers: int = 1
    d_model: int = 0  # Auto-set to n_features * 2 if 0
    d_feedforward: int = 16
    dropout: float = 0.1
    lr: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 5
    batch_size: int = 128
    scheduler_step: int = 5
    scheduler_gamma: float = 0.9

    def __post_init__(self):
        if self.d_model == 0:
            self.d_model = self.n_features * 2
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Sums sin and cos into the same dimensions (reference quirk, not interleaved).
    Reference: tranad/dlutils.py, class PositionalEncoding.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor, pos: int = 0) -> torch.Tensor:
        x = x + self.pe[pos : pos + x.size(0), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Custom encoder layer without LayerNorm, using LeakyReLU.

    This intentionally differs from PyTorch's built-in nn.TransformerEncoderLayer:
    no LayerNorm is applied, and activation is LeakyReLU instead of ReLU.
    Reference: tranad/dlutils.py, class TransformerEncoderLayer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    """Custom decoder layer without LayerNorm, using LeakyReLU.

    Has self-attention, cross-attention, and FFN — each with residual + dropout.
    No masking is applied (matching reference behavior).
    Reference: tranad/dlutils.py, class TransformerDecoderLayer.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class TranADNet(nn.Module):
    """Full TranAD architecture with two-phase self-conditioning.

    Architecture:
        - Shared encoder: TransformerEncoder with custom layer (no LayerNorm)
        - Decoder 1 & 2: Separate TransformerDecoders with custom layers
        - Shared projection: Linear(2*feats, feats) + Sigmoid

    Forward pass (two phases):
        Phase 1: focus_score = zeros -> encode -> decode1 -> x1
        Phase 2: focus_score = (x1 - src)^2 -> encode -> decode2 -> x2

    Reference: tranad/models.py, class TranAD (lines 491-524).
    """

    def __init__(self, config: TranADConfig):
        super().__init__()
        self.n_feats = config.n_features
        self.n_window = config.window_size

        d_model = config.d_model
        nhead = config.n_heads
        d_ff = config.d_feedforward
        dropout = config.dropout

        self.pos_encoder = PositionalEncoding(d_model, dropout, config.window_size)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            enable_nested_tensor=False,
        )

        decoder_layer1 = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout
        )
        self.transformer_decoder1 = nn.TransformerDecoder(
            decoder_layer1, num_layers=config.n_decoder_layers
        )

        decoder_layer2 = TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff, dropout=dropout
        )
        self.transformer_decoder2 = nn.TransformerDecoder(
            decoder_layer2, num_layers=config.n_decoder_layers
        )

        self.fcn = nn.Sequential(nn.Linear(d_model, config.n_features), nn.Sigmoid())

    def encode(
        self, src: torch.Tensor, c: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode window + focus score, prepare decoder inputs.

        Args:
            src: window, shape (window_size, batch, n_features)
            c: focus score, shape (window_size, batch, n_features)
            tgt: last element, shape (1, batch, n_features)

        Returns:
            (tgt_doubled, memory): shapes (1, batch, d_model), (window_size, batch, d_model)
        """
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-phase forward pass with self-conditioning.

        Args:
            src: window, shape (window_size, batch, n_features)
            tgt: last element of window, shape (1, batch, n_features)

        Returns:
            (x1, x2): Phase 1 and Phase 2 reconstructions,
                       each shape (1, batch, n_features)
        """
        # Phase 1: focus score is zeros
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))

        # Phase 2: focus score is squared error from Phase 1
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))

        return x1, x2
