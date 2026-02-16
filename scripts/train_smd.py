"""
Train TranAD on preprocessed SMD data.

Loads normalized .npy arrays, builds sliding windows, trains the model,
and saves a checkpoint. Expects data from preprocess_smd.py.

Usage:
    uv run python scripts/train_smd.py --machine machine-1-1
    uv run python scripts/train_smd.py --machine machine-1-1 --epochs 10

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


def train_epoch(
    model: TranADNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
    n_features: int,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Loss formula: l1 = (1/n)*MSE(x1, elem) + (1 - 1/n)*MSE(x2, elem)
    where n = epoch + 1. This gradually shifts weight from decoder 1 to decoder 2.

    Reference: tranad/main.py, backprop() TranAD training branch.

    Args:
        model: TranADNet instance
        dataloader: yields (windows, windows) batches
        optimizer: AdamW optimizer
        loss_fn: MSELoss(reduction='none')
        epoch: current epoch (0-indexed)
        n_features: number of features
        device: torch device

    Returns:
        Mean loss over all batches
    """
    model.train()
    n = epoch + 1
    losses = []

    for (d,) in dataloader:
        d = d.to(device)
        local_bs = d.shape[0]

        # (batch, window_size, features) -> (window_size, batch, features)
        window = d.permute(1, 0, 2)
        elem = window[-1, :, :].view(1, local_bs, n_features)

        x1, x2 = model(window, elem)

        l1 = (1 / n) * loss_fn(x1, elem) + (1 - 1 / n) * loss_fn(x2, elem)
        loss = torch.mean(l1)

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        losses.append(loss.item())

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
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cpu, cuda, mps, or auto",
    )
    args = parser.parse_args()

    device = auto_device(args.device)
    print(f"Device: {device}")

    # Build config with overrides
    config = TranADConfig()
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr

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
    print(f"Config: epochs={config.epochs}, batch_size={config.batch_size}, lr={config.lr}")

    # Convert to tensor and build windows
    train_tensor = torch.from_numpy(train_data).float()
    windows = convert_to_windows(train_tensor, config.window_size)
    print(f"Windows: {windows.shape}")

    # DataLoader (no shuffle to match reference for reproducibility)
    dataset = TensorDataset(windows)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

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
    for epoch in range(config.epochs):
        epoch_loss = train_epoch(
            model, dataloader, optimizer, loss_fn, epoch, n_features, device
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch + 1}/{config.epochs}  Loss: {epoch_loss:.6f}  LR: {lr:.6f}")

    # Save checkpoint
    ckpt_dir = args.output_dir / args.machine
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save(
        {
            "epoch": config.epochs - 1,
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
