"""
Train TranAD on preprocessed SMD data.

Trains a baseline model with deliberately conservative defaults so you can
see the pipeline working before optimizing. The grid sweep (code/4_grid_sweep.py)
finds the best configuration and retrains a production-quality model.

By default, checkpoints are saved to models/tranad/initial/ so the pre-trained
reference artifacts in models/tranad/machine-*/ are never overwritten.

Usage:
    uv run python code/1_train_model.py --machine machine-1-1
    uv run python code/1_train_model.py --machine machine-1-1 --epochs 10
    uv run python code/1_train_model.py --all
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TranADConfig
from src.scorer import DEFAULT_POT_PARAMS, POTParams
from src.train import train_full
from src.utils import auto_device

MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]

# Paper-reported results for comparison (Table 2)
PAPER_RESULTS = {
    "avg_precision": 0.9262,
    "avg_recall": 0.9974,
    "avg_f1": 0.9605,
}

# Baseline defaults: an honest first attempt with a smaller feedforward
# network and a moderate learning rate. Analogous to the LSTM repo's
# hidden_dim=24 baseline -- a reasonable starting point that leaves room
# for the grid sweep to find a better architecture and training config.
BASELINE_DEFAULTS = {
    "epochs": 5,
    "lr": 0.001,
    "d_feedforward": 8,
    "loss_weighting": "epoch_inverse",
    "scoring_mode": "phase2_only",
    "early_stopping_patience": 3,
    "val_split": 0.2,
    "max_epochs": 30,
}


def train_single_machine(args, machine: str) -> float | None:
    """Train a single machine and return final loss, or None on failure."""
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir

    # Build config: start from baseline defaults, apply CLI overrides
    config = TranADConfig(
        epochs=BASELINE_DEFAULTS["epochs"],
        lr=BASELINE_DEFAULTS["lr"],
        d_feedforward=BASELINE_DEFAULTS["d_feedforward"],
        loss_weighting=BASELINE_DEFAULTS["loss_weighting"],
        scoring_mode=BASELINE_DEFAULTS["scoring_mode"],
        early_stopping_patience=BASELINE_DEFAULTS["early_stopping_patience"],
        val_split=BASELINE_DEFAULTS["val_split"],
        max_epochs=BASELINE_DEFAULTS["max_epochs"],
    )
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

    # Train using the shared helper
    model, final_epoch, epoch_loss = train_full(
        config, train_data, device, seed=args.seed
    )

    # Save checkpoint
    ckpt_dir = output_dir / machine
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.ckpt"
    torch.save(
        {
            "epoch": final_epoch - 1,
            "model_state_dict": model.state_dict(),
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
                str(PROJECT_ROOT / "code" / "2_evaluate_model.py"),
                "--machine", machine,
                "--model-dir", args.output_dir,
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
    parser = argparse.ArgumentParser(description="Train TranAD baseline on SMD")
    parser.add_argument("--machine", type=str, default="machine-1-1", help="Machine name")
    parser.add_argument("--all", action="store_true", help="Train all 4 reference machines")
    parser.add_argument("--machines", nargs="*", default=None,
                        help="Override machine list for --all")
    parser.add_argument("--data-dir", type=str, default="data/smd/processed")
    parser.add_argument("--output-dir", type=str, default="models/tranad/initial")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Training hyperparameters (override baseline defaults)
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
