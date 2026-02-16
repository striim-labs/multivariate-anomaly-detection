"""Shared utilities for TranAD training and inference."""

import torch


def convert_to_windows(data: torch.Tensor, window_size: int) -> torch.Tensor:
    """Convert time series to sliding windows.

    For position i, window contains data[i-W:i] (padded with data[0] at front).
    Vectorized using torch.Tensor.unfold for efficiency.

    Reference: tranad/main.py, convert_to_windows().

    Args:
        data: (N, features) tensor
        window_size: window length (default 10)

    Returns:
        (N, window_size, features) tensor
    """
    n, f = data.shape
    pad = data[0:1].expand(window_size, f)
    padded = torch.cat([pad, data], dim=0)  # (N + W, F)
    windows = padded.unfold(0, window_size, 1)  # (N + 1, F, W)
    windows = windows[:n].permute(0, 2, 1)  # (N, W, F)
    return windows


def auto_device(preference: str = "auto") -> torch.device:
    """Select best available torch device.

    Args:
        preference: "auto", "cpu", "cuda", or "mps".

    Returns:
        torch.device
    """
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
