"""
Train and evaluate TranAD on all 4 reference SMD machines.

Trains each machine with the specified hyperparameters (or defaults),
evaluates with per-machine POT parameters from the reference, and
prints a summary table comparing to the paper's reported results.

Usage:
    uv run python scripts/train_all_machines.py
    uv run python scripts/train_all_machines.py --adversarial-loss --early-stopping-patience 3
    uv run python scripts/train_all_machines.py --lr 0.001 --d-feedforward 64 --dtype float64
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add app/ to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from tranad_scorer import POTParams

# The 4 non-trivial machines evaluated in the TranAD paper (Table 2)
MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]

# Per-machine POT parameters from reference (src/constants.py)
POT_PARAMS = {
    "machine-1-1": POTParams(q=1e-5, level=0.99995, scale=1.06),
    "machine-2-1": POTParams(q=1e-5, level=0.999, scale=0.9),
    "machine-3-2": POTParams(q=1e-5, level=0.99, scale=1.0),
    "machine-3-7": POTParams(q=1e-5, level=0.99995, scale=1.06),
}

# Paper-reported results for comparison (Table 2)
PAPER_RESULTS = {
    "avg_precision": 0.9262,
    "avg_recall": 0.9974,
    "avg_f1": 0.9605,
}


def ensure_preprocessed(machine: str, data_dir: Path, raw_dir: Path) -> bool:
    """Check if preprocessed data exists, run preprocessing if not."""
    train_path = data_dir / f"{machine}_train.npy"
    if train_path.exists():
        return True

    print(f"  Preprocessing {machine}...")
    scripts_dir = Path(__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "preprocess_smd.py"),
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(data_dir),
            "--machines",
            machine,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  Preprocessing failed: {result.stderr}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate TranAD on all reference SMD machines"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/smd/processed")
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/smd/raw")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/tranad")
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--machines",
        nargs="*",
        default=None,
        help="Override machine list (default: all 4 reference machines)",
    )

    # Training hyperparameters (passed through to train_smd.py)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--d-feedforward", type=int, default=None)
    parser.add_argument(
        "--dtype", type=str, choices=["float32", "float64"], default=None
    )
    parser.add_argument(
        "--loss-weighting",
        type=str,
        choices=["epoch_inverse", "exponential_decay"],
        default=None,
    )
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--adversarial-loss", action="store_true", default=False)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--val-split", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)

    # Scoring
    parser.add_argument(
        "--scoring-mode",
        type=str,
        choices=["phase2_only", "averaged"],
        default=None,
    )

    args = parser.parse_args()
    machines = args.machines or MACHINES

    # Build common train/evaluate arguments
    train_args = []
    eval_args = []
    for flag, val in [
        ("--epochs", args.epochs),
        ("--batch-size", args.batch_size),
        ("--lr", args.lr),
        ("--d-feedforward", args.d_feedforward),
        ("--dtype", args.dtype),
        ("--loss-weighting", args.loss_weighting),
        ("--epsilon", args.epsilon),
        ("--gradient-clip-norm", args.gradient_clip_norm),
        ("--early-stopping-patience", args.early_stopping_patience),
        ("--val-split", args.val_split),
        ("--max-epochs", args.max_epochs),
    ]:
        if val is not None:
            train_args.extend([flag, str(val)])

    if args.adversarial_loss:
        train_args.append("--adversarial-loss")
    if args.use_layer_norm:
        train_args.append("--use-layer-norm")
    if args.scoring_mode:
        eval_args.extend(["--scoring-mode", args.scoring_mode])

    scripts_dir = Path(__file__).resolve().parent
    results = {}

    for machine in machines:
        print(f"\n{'='*60}")
        print(f"Machine: {machine}")
        print(f"{'='*60}")

        # Ensure data is preprocessed
        if not ensure_preprocessed(machine, args.data_dir, args.raw_dir):
            print(f"  Skipping {machine} (preprocessing failed)")
            continue

        # Train
        print(f"  Training...")
        train_cmd = [
            sys.executable,
            str(scripts_dir / "train_smd.py"),
            "--machine",
            machine,
            "--data-dir",
            str(args.data_dir),
            "--output-dir",
            str(args.model_dir),
            "--device",
            args.device,
        ] + train_args

        result = subprocess.run(train_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Training failed:")
            print(result.stderr[-500:] if result.stderr else "No stderr")
            continue
        # Print last few lines of training output
        train_lines = result.stdout.strip().split("\n")
        for line in train_lines[-6:]:
            print(f"    {line}")

        # Evaluate with per-machine POT params
        pot = POT_PARAMS.get(machine, POTParams())
        print(f"  Evaluating (POT: level={pot.level}, scale={pot.scale})...")
        eval_cmd = [
            sys.executable,
            str(scripts_dir / "evaluate_smd.py"),
            "--machine",
            machine,
            "--data-dir",
            str(args.data_dir),
            "--model-dir",
            str(args.model_dir),
            "--device",
            args.device,
            "--pot-level",
            str(pot.level),
            "--pot-scale",
            str(pot.scale),
        ] + eval_args

        result = subprocess.run(eval_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Evaluation failed:")
            print(result.stderr[-500:] if result.stderr else "No stderr")
            continue

        # Load eval results
        eval_path = args.model_dir / machine / "eval_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                metrics = json.load(f)
            results[machine] = metrics
            print(
                f"  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
                f"R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}"
            )
            if "Hit@100%_elev" in metrics:
                raw_hit = metrics.get("Hit@100%", 0)
                elev_hit = metrics["Hit@100%_elev"]
                print(
                    f"  Hit@100%: raw={raw_hit:.3f} elev={elev_hit:.3f} "
                    f"delta={elev_hit - raw_hit:+.3f}"
                )
        else:
            print(f"  Warning: eval_results.json not found")

    # Summary table
    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"{'Machine':<16} {'F1':>8} {'Precision':>10} {'Recall':>8} {'AUC':>8}")
        print("-" * 70)

        f1_values = []
        prec_values = []
        rec_values = []
        for machine in machines:
            if machine in results:
                m = results[machine]
                print(
                    f"{machine:<16} {m['f1']:>8.4f} {m['precision']:>10.4f} "
                    f"{m['recall']:>8.4f} {m['roc_auc']:>8.4f}"
                )
                f1_values.append(m["f1"])
                prec_values.append(m["precision"])
                rec_values.append(m["recall"])
            else:
                print(f"{machine:<16} {'FAILED':>8}")

        if f1_values:
            print("-" * 70)
            avg_f1 = np.mean(f1_values)
            avg_prec = np.mean(prec_values)
            avg_rec = np.mean(rec_values)
            print(
                f"{'Average':<16} {avg_f1:>8.4f} {avg_prec:>10.4f} {avg_rec:>8.4f}"
            )
            print(
                f"{'Paper (Table 2)':<16} {PAPER_RESULTS['avg_f1']:>8.4f} "
                f"{PAPER_RESULTS['avg_precision']:>10.4f} "
                f"{PAPER_RESULTS['avg_recall']:>8.4f}"
            )
            print(f"\nGap from paper: F1={avg_f1 - PAPER_RESULTS['avg_f1']:+.4f}")

            # Elevation-ratio diagnosis summary
            hit_raw = [results[m].get("Hit@100%", 0) for m in machines if m in results and "Hit@100%" in results[m]]
            hit_elev = [results[m].get("Hit@100%_elev", 0) for m in machines if m in results and "Hit@100%_elev" in results[m]]
            if hit_raw and hit_elev:
                avg_hit_raw = np.mean(hit_raw)
                avg_hit_elev = np.mean(hit_elev)
                print(f"\nDiagnosis (avg Hit@100%): raw={avg_hit_raw:.4f}  "
                      f"elev={avg_hit_elev:.4f}  delta={avg_hit_elev - avg_hit_raw:+.4f}")
        print("=" * 70)

        # Save summary
        summary_path = args.model_dir / "summary.json"
        summary = {
            "machines": results,
            "avg_f1": float(np.mean(f1_values)) if f1_values else 0,
            "avg_precision": float(np.mean(prec_values)) if prec_values else 0,
            "avg_recall": float(np.mean(rec_values)) if rec_values else 0,
        }
        # Add elevation diagnosis averages if available
        hit_elev_vals = [results[m].get("Hit@100%_elev") for m in machines if m in results and "Hit@100%_elev" in results[m]]
        if hit_elev_vals:
            summary["avg_hit100_raw"] = float(np.mean([results[m].get("Hit@100%", 0) for m in machines if m in results and "Hit@100%" in results[m]]))
            summary["avg_hit100_elev"] = float(np.mean(hit_elev_vals))
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
