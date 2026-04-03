"""
TranAD Training Utilities

Reusable training components: epoch runners, loss weighting, early stopping.
Used by code/3_train_model.py and code/6_optimize.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import TranADConfig, TranADNet


def compute_loss_weight(epoch: int, config: TranADConfig) -> float:
    """Compute the evolving weight for Phase 1 loss term.

    Returns w such that:
      non-adversarial: loss = w * MSE(x1) + (1-w) * MSE(x2)
      adversarial:     L1 = w * MSE(x1) + (1-w) * MSE(x2)
                       L2 = w * MSE(x1) - (1-w) * MSE(x2)

    epoch_inverse: w = 1/(epoch+1)          -- reference code
    exponential_decay: w = epsilon^{-(epoch+1)}  -- paper Eq. 9
    """
    n = epoch + 1
    if config.loss_weighting == "epoch_inverse":
        return 1.0 / n
    elif config.loss_weighting == "exponential_decay":
        return config.epsilon ** (-n)
    else:
        raise ValueError(f"Unknown loss_weighting: {config.loss_weighting}")


class EarlyStopping:
    """Early stopping monitor for validation loss."""

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop. Saves best model state.

        Returns True if patience exhausted.
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Restore model to the best weights seen."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(
    model: TranADNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
    config: TranADConfig,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Supports two loss modes:
      - Non-adversarial (default): single combined loss matching reference code
      - Adversarial (paper Eq. 8-9): separate L1/L2 with sign flip
    """
    model.train()
    w = compute_loss_weight(epoch, config)
    losses = []

    for (d,) in dataloader:
        d = d.to(device)
        local_bs = d.shape[0]

        # (batch, window_size, features) -> (window_size, batch, features)
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, config.n_features)

        x1, x2 = model(window, elem)

        if config.adversarial_loss:
            recon_p1 = torch.mean(loss_fn(x1, elem))
            recon_p2 = torch.mean(loss_fn(x2, elem))

            l1 = w * recon_p1 + (1 - w) * recon_p2
            l2 = w * recon_p1 - (1 - w) * recon_p2

            optimizer.zero_grad()
            l1.backward(retain_graph=True)
            l2.backward()

            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )

            optimizer.step()
            losses.append((l1.item() + l2.item()) / 2)
        else:
            l1 = w * loss_fn(x1, elem) + (1 - w) * loss_fn(x2, elem)
            loss = torch.mean(l1)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)

            if config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip_norm
                )

            optimizer.step()
            losses.append(loss.item())

    return sum(losses) / len(losses)


@torch.no_grad()
def validate_epoch(
    model: TranADNet,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    epoch: int,
    config: TranADConfig,
    device: torch.device,
) -> float:
    """Run one validation epoch (no gradient computation)."""
    model.eval()
    w = compute_loss_weight(epoch, config)
    losses = []

    for (d,) in dataloader:
        d = d.to(device)
        local_bs = d.shape[0]

        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, config.n_features)

        x1, x2 = model(window, elem)

        l = w * loss_fn(x1, elem) + (1 - w) * loss_fn(x2, elem)
        losses.append(torch.mean(l).item())

    return sum(losses) / len(losses)
