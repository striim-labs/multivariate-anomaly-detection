"""
Evaluate TranAD on preprocessed SMD data.

Loads a trained checkpoint, runs batch inference on train and test data,
calibrates anomaly thresholds, and prints/saves evaluation metrics.

Usage:
    uv run python scripts/evaluate_smd.py --machine machine-1-1
    uv run python scripts/evaluate_smd.py --machine machine-1-1 --method percentile
    uv run python scripts/evaluate_smd.py --machine machine-1-1 --method f1_max

Reference: tranad/main.py lines 318-345 (scoring section).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add app/ to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from tranad_registry import TranADRegistry
from tranad_scorer import POTParams, TranADScorer
from tranad_utils import auto_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate TranAD on SMD")
    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/smd/processed")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/tranad")
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pot",
        choices=["pot", "percentile", "f1_max"],
    )
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--pot-q", type=float, default=1e-5)
    parser.add_argument("--pot-level", type=float, default=0.99995)
    parser.add_argument("--pot-scale", type=float, default=1.06)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = auto_device(args.device)
    print(f"Device: {device}")

    # --- Load model via registry ---
    registry = TranADRegistry(base_dir=args.model_dir)
    model, config = registry.get_model(args.machine, device=device)
    print(
        f"Loaded model for {args.machine}: {config.n_features} features, "
        f"window_size={config.window_size}"
    )

    # --- Load preprocessed data ---
    train_path = args.data_dir / f"{args.machine}_train.npy"
    test_path = args.data_dir / f"{args.machine}_test.npy"
    labels_path = args.data_dir / f"{args.machine}_interp_labels.npy"

    for p in [train_path, test_path, labels_path]:
        if not p.exists():
            print(f"Error: {p} not found. Run preprocess_smd.py first.")
            sys.exit(1)

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    interp_labels = np.load(labels_path)
    print(
        f"Data: train={train_data.shape}, test={test_data.shape}, "
        f"labels={interp_labels.shape}"
    )

    # --- Score train and test data ---
    scorer = TranADScorer()

    print("Scoring training data...")
    train_scores = scorer.score_batch(
        model, train_data.astype(np.float32), config.window_size, device
    )
    print(
        f"  Train scores: shape={train_scores.shape}, "
        f"mean={train_scores.mean():.6f}, max={train_scores.max():.6f}"
    )

    print("Scoring test data...")
    test_scores = scorer.score_batch(
        model, test_data.astype(np.float32), config.window_size, device
    )
    print(
        f"  Test scores: shape={test_scores.shape}, "
        f"mean={test_scores.mean():.6f}, max={test_scores.max():.6f}"
    )

    # --- Calibrate threshold ---
    pot_params = POTParams(q=args.pot_q, level=args.pot_level, scale=args.pot_scale)

    print(f"Calibrating threshold (method={args.method})...")
    cal_result = scorer.calibrate_threshold(
        train_scores=train_scores,
        test_scores=test_scores,
        labels=interp_labels,
        method=args.method,
        pot_params=pot_params,
        percentile=args.percentile,
    )
    threshold = cal_result["threshold"]
    print(f"  Threshold: {threshold:.6f}")

    # --- Evaluate ---
    print("Evaluating with point-adjustment...")
    metrics = scorer.evaluate(test_scores, interp_labels, threshold)

    # --- Root cause diagnosis ---
    print("Computing diagnosis metrics...")
    diag = scorer.diagnose(test_scores, interp_labels)
    metrics.update(diag)

    # --- Print results ---
    print()
    print("=" * 60)
    print(f"Results for {args.machine} (method={args.method})")
    print("=" * 60)
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  ROC/AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Threshold:   {threshold:.6f}")
    print(
        f"  TP={metrics['TP']}, TN={metrics['TN']}, "
        f"FP={metrics['FP']}, FN={metrics['FN']}"
    )
    for key in ["Hit@100%", "Hit@150%", "NDCG@100%", "NDCG@150%"]:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.4f}")
    print("=" * 60)

    # --- Save results ---
    output_path = args.model_dir / args.machine / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, float)):
            save_metrics[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            save_metrics[k] = int(v)
        else:
            save_metrics[k] = v
    save_metrics["method"] = args.method
    save_metrics["machine"] = args.machine

    with open(output_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # --- Save scorer state to registry ---
    registry.save_scorer_state(args.machine, cal_result)
    print(f"Scorer state saved to registry")


if __name__ == "__main__":
    main()
