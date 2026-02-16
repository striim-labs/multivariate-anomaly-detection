"""
Train TranAD on preprocessed SMD data.

Loads normalized .npy arrays, builds sliding windows, trains the model,
and saves a checkpoint. Supports adversarial loss (paper Eq. 8-9),
configurable loss weighting, early stopping, and float64 dtype.

Usage:
    uv run python scripts/train_smd.py --machine machine-1-1
    uv run python scripts/train_smd.py --machine machine-1-1 --adversarial-loss --early-stopping-patience 3
    uv run python scripts/train_smd.py --machine machine-1-1 --dtype float64 --loss-weighting exponential_decay

Reference: tranad/main.py, TranAD branch of backprop() (lines 251-273).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add app/ to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from tranad_model import TranADConfig, TranADNet
from tranad_utils import auto_device, convert_to_windows


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

    Reference: tranad/main.py, backprop() TranAD training branch.
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
            # Paper Eq. 8-9: separate losses with opposite signs on adversarial term
            recon_p1 = torch.mean(loss_fn(x1, elem))
            recon_p2 = torch.mean(loss_fn(x2, elem))

            l1 = w * recon_p1 + (1 - w) * recon_p2  # D1 minimizes adversarial
            l2 = w * recon_p1 - (1 - w) * recon_p2  # D2 maximizes adversarial

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
            # Reference code: single combined loss
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


def main():
    parser = argparse.ArgumentParser(description="Train TranAD on SMD")
    parser.add_argument(
        "--machine", type=str, default="machine-1-1", help="Machine name"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/smd/processed"),
        help="Path to preprocessed data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tranad"),
        help="Path to save checkpoints",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cpu, cuda, mps, or auto",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--d-feedforward", type=int, default=None, help="FFN hidden dimension"
    )

    # Phase 3: new config options
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default=None,
        help="Model and data dtype",
    )
    parser.add_argument(
        "--loss-weighting",
        type=str,
        choices=["epoch_inverse", "exponential_decay"],
        default=None,
        help="Loss weight schedule",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon for exponential_decay weighting",
    )
    parser.add_argument(
        "--adversarial-loss",
        action="store_true",
        default=False,
        help="Use separate L1/L2 adversarial losses (paper Eq. 8-9)",
    )
    parser.add_argument(
        "--use-layer-norm",
        action="store_true",
        default=False,
        help="Add LayerNorm to transformer layers (paper Eq. 4)",
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
        help="Max gradient norm (0 to disable)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Early stopping patience (0 to disable)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=None,
        help="Fraction of training data for validation",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs with early stopping",
    )
    args = parser.parse_args()

    # Build config with overrides
    config = TranADConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.d_feedforward is not None:
        config.d_feedforward = args.d_feedforward
    if args.dtype is not None:
        config.dtype = args.dtype
    if args.loss_weighting is not None:
        config.loss_weighting = args.loss_weighting
    if args.epsilon is not None:
        config.epsilon = args.epsilon
    if args.adversarial_loss:
        config.adversarial_loss = True
    if args.use_layer_norm:
        config.use_layer_norm = True
    if args.gradient_clip_norm is not None:
        config.gradient_clip_norm = args.gradient_clip_norm
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience
    if args.val_split is not None:
        config.val_split = args.val_split
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs

    # Device selection (float64 + MPS may not work, fallback to CPU)
    device = auto_device(args.device)
    if config.dtype == "float64" and device.type == "mps":
        print("Warning: float64 may not be fully supported on MPS, falling back to CPU")
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load preprocessed data
    train_path = args.data_dir / f"{args.machine}_train.npy"
    if not train_path.exists():
        print(f"Error: {train_path} not found. Run preprocess_smd.py first.")
        sys.exit(1)

    train_data = np.load(train_path)
    n_features = train_data.shape[1]
    config.n_features = n_features
    config.d_model = 0  # Trigger auto-set in __post_init__
    config.__post_init__()

    print(f"Training {args.machine}: {train_data.shape[0]} samples, {n_features} features")
    print(
        f"Config: epochs={config.epochs}, batch_size={config.batch_size}, "
        f"lr={config.lr}, dtype={config.dtype}"
    )
    print(
        f"  loss_weighting={config.loss_weighting}, adversarial={config.adversarial_loss}, "
        f"layer_norm={config.use_layer_norm}"
    )

    # Convert to tensor with correct dtype and build windows
    torch_dtype = torch.float64 if config.dtype == "float64" else torch.float32
    train_tensor = torch.from_numpy(train_data).to(torch_dtype)
    windows = convert_to_windows(train_tensor, config.window_size)
    print(f"Windows: {windows.shape}")

    # Set up dataloaders (with optional train/val split for early stopping)
    use_early_stopping = config.early_stopping_patience > 0
    if use_early_stopping:
        n_total = windows.shape[0]
        n_val = int(n_total * config.val_split)
        n_train = n_total - n_val
        train_windows = windows[:n_train]
        val_windows = windows[n_train:]

        train_loader = DataLoader(
            TensorDataset(train_windows), batch_size=config.batch_size
        )
        val_loader = DataLoader(
            TensorDataset(val_windows), batch_size=config.batch_size
        )
        stopper = EarlyStopping(patience=config.early_stopping_patience)
        total_epochs = config.max_epochs
        print(
            f"Early stopping: patience={config.early_stopping_patience}, "
            f"train={n_train}, val={n_val}"
        )
    else:
        train_loader = DataLoader(TensorDataset(windows), batch_size=config.batch_size)
        val_loader = None
        stopper = None
        total_epochs = config.epochs

    # Create model
    model = TranADNet(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = nn.MSELoss(reduction="none")

    # Training loop
    final_epoch = 0
    epoch_loss = 0.0
    for epoch in range(total_epochs):
        epoch_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, epoch, config, device
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        final_epoch = epoch + 1

        if use_early_stopping and val_loader is not None and stopper is not None:
            val_loss = validate_epoch(
                model, val_loader, loss_fn, epoch, config, device
            )
            print(
                f"  Epoch {final_epoch}/{total_epochs}  "
                f"Train: {epoch_loss:.6f}  Val: {val_loss:.6f}  LR: {lr:.6f}"
            )
            if stopper.step(val_loss, model):
                print(
                    f"  Early stopping at epoch {final_epoch} "
                    f"(patience={config.early_stopping_patience})"
                )
                stopper.restore_best(model)
                break
        else:
            print(
                f"  Epoch {final_epoch}/{total_epochs}  "
                f"Loss: {epoch_loss:.6f}  LR: {lr:.6f}"
            )

    # Save checkpoint
    ckpt_dir = args.output_dir / args.machine
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save(
        {
            "epoch": final_epoch - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "final_loss": epoch_loss,
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
