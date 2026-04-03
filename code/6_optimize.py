"""
Hyperparameter sweep for TranAD on SMD.

Runs a grid search over parameter combinations, trains a model for each,
evaluates with POT, and saves results to a CSV.

Usage:
    uv run python code/6_optimize.py --machine machine-1-1 --quick
    uv run python code/6_optimize.py --machine machine-1-1
"""

import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import TranADConfig, TranADNet
from src.scorer import DEFAULT_POT_PARAMS, POTParams, calibrate_threshold, evaluate, score_batch
from src.train import EarlyStopping, train_epoch, validate_epoch
from src.utils import auto_device, convert_to_windows


QUICK_GRID = {
    "dtype": ["float32"],
    "loss_weighting": ["epoch_inverse", "exponential_decay"],
    "adversarial_loss": [False, True],
    "scoring_mode": ["phase2_only", "averaged"],
    "lr": [0.0001, 0.001, 0.01],
    "d_feedforward": [16, 64],
    "use_layer_norm": [False],
}

FULL_GRID = {
    "dtype": ["float32", "float64"],
    "loss_weighting": ["epoch_inverse", "exponential_decay"],
    "adversarial_loss": [False, True],
    "scoring_mode": ["phase2_only", "averaged"],
    "lr": [0.0001, 0.0005, 0.001, 0.005, 0.01],
    "d_feedforward": [16, 32, 64],
    "use_layer_norm": [False, True],
}

CSV_COLUMNS = [
    "trial", "dtype", "loss_weighting", "adversarial_loss", "scoring_mode",
    "lr", "d_feedforward", "use_layer_norm", "f1", "precision", "recall",
    "roc_auc", "threshold", "TP", "TN", "FP", "FN", "epochs_trained",
    "train_time_s", "final_train_loss", "status",
]


def build_config(n_features: int, params: dict, max_epochs: int) -> TranADConfig:
    return TranADConfig(
        n_features=n_features,
        n_heads=n_features,
        d_feedforward=params["d_feedforward"],
        use_layer_norm=params["use_layer_norm"],
        dtype=params["dtype"],
        lr=params["lr"],
        loss_weighting=params["loss_weighting"],
        adversarial_loss=params["adversarial_loss"],
        scoring_mode=params["scoring_mode"],
        early_stopping_patience=3,
        val_split=0.2,
        max_epochs=max_epochs,
    )


def run_trial(
    config: TranADConfig,
    train_data: np.ndarray,
    test_data: np.ndarray,
    labels: np.ndarray,
    pot_params: POTParams,
    device: torch.device,
) -> dict:
    torch_dtype = torch.float64 if config.dtype == "float64" else torch.float32
    train_tensor = torch.from_numpy(train_data).to(torch_dtype)
    windows = convert_to_windows(train_tensor, config.window_size)

    n_total = windows.shape[0]
    n_val = int(n_total * config.val_split)
    n_train = n_total - n_val
    train_windows = windows[:n_train]
    val_windows = windows[n_train:]

    train_loader = DataLoader(TensorDataset(train_windows), batch_size=config.batch_size)
    val_loader = DataLoader(TensorDataset(val_windows), batch_size=config.batch_size)

    model = TranADNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.scheduler_step, gamma=config.scheduler_gamma
    )
    loss_fn = nn.MSELoss(reduction="none")
    stopper = EarlyStopping(patience=config.early_stopping_patience)

    start_time = time.time()
    final_epoch = 0
    final_loss = 0.0
    for epoch in range(config.max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch, config, device)
        scheduler.step()
        val_loss = validate_epoch(model, val_loader, loss_fn, epoch, config, device)
        final_epoch = epoch + 1
        final_loss = train_loss

        if stopper.step(val_loss, model):
            stopper.restore_best(model)
            break
    train_time = time.time() - start_time

    train_scores = score_batch(model, train_data, config.window_size, device, config.scoring_mode)
    test_scores = score_batch(model, test_data, config.window_size, device, config.scoring_mode)

    cal = calibrate_threshold(
        train_scores, test_scores, labels, method="pot", pot_params=pot_params
    )

    metrics = evaluate(test_scores, labels, cal["threshold"])
    metrics["epochs_trained"] = final_epoch
    metrics["train_time_s"] = round(train_time, 1)
    metrics["final_train_loss"] = final_loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for TranAD")
    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument("--data-dir", type=str, default="data/smd/processed")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-sweep-epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    device = auto_device(args.device)
    print(f"Device: {device}")

    train_data = np.load(data_dir / f"{args.machine}_train.npy")
    test_data = np.load(data_dir / f"{args.machine}_test.npy")
    labels = np.load(data_dir / f"{args.machine}_interp_labels.npy")
    n_features = train_data.shape[1]
    print(f"Data loaded: train={train_data.shape}, test={test_data.shape}")

    grid = QUICK_GRID if args.quick else FULL_GRID
    param_names = sorted(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in param_names)))
    grid_type = "quick" if args.quick else "full"
    print(f"Grid: {grid_type}, {len(combos)} combinations")

    pot_params = DEFAULT_POT_PARAMS.get(args.machine, POTParams())

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"sweep_{args.machine}_{grid_type}.csv"

    completed_trials: set[int] = set()
    if args.resume and results_path.exists():
        with open(results_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") == "ok":
                    completed_trials.add(int(row["trial"]))
        print(f"Resuming: {len(completed_trials)} trials already completed")

    write_header = not results_path.exists() or not args.resume
    if write_header:
        with open(results_path, "w", newline="") as f:
            csv.writer(f).writerow(CSV_COLUMNS)

    best_f1 = -1.0
    best_params: dict = {}

    for i, values in enumerate(combos):
        params = dict(zip(param_names, values))
        trial_num = i + 1

        if trial_num in completed_trials:
            continue

        trial_device = device
        if params["dtype"] == "float64" and device.type == "mps":
            trial_device = torch.device("cpu")

        params_str = ", ".join(f"{k}={v}" for k, v in sorted(params.items()))
        print(f"\n[{trial_num}/{len(combos)}] {params_str}")

        try:
            config = build_config(n_features, params, args.max_sweep_epochs)
            metrics = run_trial(config, train_data, test_data, labels, pot_params, trial_device)

            f1 = metrics["f1"]
            print(f"  F1={f1:.4f}  P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  epochs={metrics['epochs_trained']}  "
                  f"time={metrics['train_time_s']}s")

            if f1 > best_f1:
                best_f1 = f1
                best_params = params.copy()
                print("  *** New best F1! ***")

            row = [
                trial_num, params["dtype"], params["loss_weighting"],
                params["adversarial_loss"], params["scoring_mode"],
                params["lr"], params["d_feedforward"], params["use_layer_norm"],
                f"{metrics['f1']:.6f}", f"{metrics['precision']:.6f}",
                f"{metrics['recall']:.6f}", f"{metrics['roc_auc']:.6f}",
                f"{metrics['threshold']:.6f}", metrics["TP"], metrics["TN"],
                metrics["FP"], metrics["FN"], metrics["epochs_trained"],
                metrics["train_time_s"], f"{metrics['final_train_loss']:.6f}", "ok",
            ]
        except Exception as e:
            print(f"  FAILED: {e}")
            row = [trial_num] + [params.get(k, "") for k in param_names]
            row += [""] * (len(CSV_COLUMNS) - len(param_names) - 1)
            row[-1] = f"error: {e}"

        with open(results_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    print("\n" + "=" * 70)
    print(f"SWEEP COMPLETE: {len(combos)} trials")
    print(f"Results saved to: {results_path}")
    print(f"\nBest F1: {best_f1:.4f}")
    if best_params:
        print("Best params:")
        for k, v in sorted(best_params.items()):
            print(f"  {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
