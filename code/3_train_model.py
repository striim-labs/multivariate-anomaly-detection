"""
Train TranAD on preprocessed SMD data.

Loads normalized .npy arrays, builds sliding windows, trains the model,
and saves a checkpoint. Supports adversarial loss (paper Eq. 8-9),
configurable loss weighting, early stopping, and float64 dtype.

Usage:
    uv run python code/3_train_model.py --machine machine-1-1
    uv run python code/3_train_model.py --machine machine-1-1 --epochs 20
    uv run python code/3_train_model.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TranADConfig, TranADNet
from src.scorer import DEFAULT_POT_PARAMS, POTParams
from src.train import EarlyStopping, train_epoch, validate_epoch
from src.utils import auto_device, convert_to_windows

MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]

# Paper-reported results for comparison (Table 2)
PAPER_RESULTS = {
    "avg_precision": 0.9262,
    "avg_recall": 0.9974,
    "avg_f1": 0.9605,
}


def train_single_machine(args, machine: str) -> float | None:
    """Train a single machine and return final loss, or None on failure."""
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

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

    # Device selection
    device = auto_device(args.device)
    if config.dtype == "float64" and device.type == "mps":
        print("Warning: float64 may not be fully supported on MPS, falling back to CPU")
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load preprocessed data
    train_path = data_dir / f"{machine}_train.npy"
    if not train_path.exists():
        print(f"Error: {train_path} not found. Run code/0_verify_setup.py first.")
        return None

    train_data = np.load(train_path)
    n_features = train_data.shape[1]
    config.n_features = n_features
    config.d_model = 0
    config.__post_init__()

    print(f"Training {machine}: {train_data.shape[0]} samples, {n_features} features")
    print(f"Config: epochs={config.epochs}, batch_size={config.batch_size}, "
          f"lr={config.lr}, dtype={config.dtype}")
    print(f"  loss_weighting={config.loss_weighting}, adversarial={config.adversarial_loss}, "
          f"layer_norm={config.use_layer_norm}")

    # Convert to tensor and build windows
    torch_dtype = torch.float64 if config.dtype == "float64" else torch.float32
    train_tensor = torch.from_numpy(train_data).to(torch_dtype)
    windows = convert_to_windows(train_tensor, config.window_size)
    print(f"Windows: {windows.shape}")

    # Set up dataloaders
    use_early_stopping = config.early_stopping_patience > 0
    if use_early_stopping:
        n_total = windows.shape[0]
        n_val = int(n_total * config.val_split)
        n_train = n_total - n_val
        train_windows = windows[:n_train]
        val_windows = windows[n_train:]

        train_loader = DataLoader(TensorDataset(train_windows), batch_size=config.batch_size)
        val_loader = DataLoader(TensorDataset(val_windows), batch_size=config.batch_size)
        stopper = EarlyStopping(patience=config.early_stopping_patience)
        total_epochs = config.max_epochs
        print(f"Early stopping: patience={config.early_stopping_patience}, "
              f"train={n_train}, val={n_val}")
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
            val_loss = validate_epoch(model, val_loader, loss_fn, epoch, config, device)
            print(f"  Epoch {final_epoch}/{total_epochs}  "
                  f"Train: {epoch_loss:.6f}  Val: {val_loss:.6f}  LR: {lr:.6f}")
            if stopper.step(val_loss, model):
                print(f"  Early stopping at epoch {final_epoch} "
                      f"(patience={config.early_stopping_patience})")
                stopper.restore_best(model)
                break
        else:
            print(f"  Epoch {final_epoch}/{total_epochs}  "
                  f"Loss: {epoch_loss:.6f}  LR: {lr:.6f}")

    # Save checkpoint
    ckpt_dir = output_dir / machine
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
    return epoch_loss


def train_all_machines(args):
    """Train and evaluate all reference machines."""
    results = {}

    for machine in (args.machines or MACHINES):
        print(f"\n{'='*60}")
        print(f"Machine: {machine}")
        print(f"{'='*60}")

        loss = train_single_machine(args, machine)
        if loss is not None:
            # Evaluate after training
            print(f"\n  Evaluating {machine}...")
            eval_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "code" / "4_evaluate_model.py"),
                "--machine", machine,
                "--device", args.device,
            ]
            pot = DEFAULT_POT_PARAMS.get(machine, POTParams())
            eval_cmd.extend(["--pot-level", str(pot.level), "--pot-scale", str(pot.scale)])

            result = subprocess.run(eval_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Evaluation failed: {result.stderr[-500:]}")
                continue

            # Load eval results
            eval_path = PROJECT_ROOT / args.output_dir / machine / "eval_results.json"
            if eval_path.exists():
                with open(eval_path) as f:
                    metrics = json.load(f)
                results[machine] = metrics
                print(f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                      f"R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}")

    # Summary table
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Machine':<16} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8}")
        print("-" * 70)

        f1_values, prec_values, rec_values = [], [], []
        for machine in (args.machines or MACHINES):
            if machine in results:
                m = results[machine]
                print(f"{machine:<16} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                      f"{m['recall']:>8.4f} {m['roc_auc']:>8.4f}")
                f1_values.append(m["f1"])
                prec_values.append(m["precision"])
                rec_values.append(m["recall"])

        if f1_values:
            print("-" * 70)
            avg_f1 = np.mean(f1_values)
            avg_prec = np.mean(prec_values)
            avg_rec = np.mean(rec_values)
            print(f"{'Average':<16} {avg_f1:>8.4f} {avg_prec:>10.4f} {avg_rec:>8.4f}")
            print(f"{'Paper (Table 2)':<16} {PAPER_RESULTS['avg_f1']:>8.4f} "
                  f"{PAPER_RESULTS['avg_precision']:>10.4f} "
                  f"{PAPER_RESULTS['avg_recall']:>8.4f}")
            print(f"\nGap from paper: F1={avg_f1 - PAPER_RESULTS['avg_f1']:+.4f}")

        # Save summary
        summary_path = PROJECT_ROOT / args.output_dir / "summary.json"
        summary = {
            "machines": results,
            "avg_f1": float(np.mean(f1_values)) if f1_values else 0,
            "avg_precision": float(np.mean(prec_values)) if prec_values else 0,
            "avg_recall": float(np.mean(rec_values)) if rec_values else 0,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Train TranAD on SMD")
    parser.add_argument("--machine", type=str, default="machine-1-1", help="Machine name")
    parser.add_argument("--all", action="store_true", help="Train all 4 reference machines")
    parser.add_argument("--machines", nargs="*", default=None,
                        help="Override machine list for --all")
    parser.add_argument("--data-dir", type=str, default="data/smd/processed")
    parser.add_argument("--output-dir", type=str, default="models/tranad")
    parser.add_argument("--device", type=str, default="auto")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-feedforward", type=int, default=None)
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default=None)
    parser.add_argument("--loss-weighting", type=str,
                        choices=["epoch_inverse", "exponential_decay"], default=None)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--adversarial-loss", action="store_true", default=False)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)

    args = parser.parse_args()

    if args.all:
        train_all_machines(args)
    else:
        train_single_machine(args, args.machine)


if __name__ == "__main__":
    main()
