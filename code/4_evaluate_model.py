"""
Evaluate TranAD on preprocessed SMD data.

Loads a trained checkpoint, runs batch inference on train and test data,
calibrates anomaly thresholds, and prints/saves evaluation metrics.

Usage:
    uv run python code/4_evaluate_model.py --machine machine-1-1
    uv run python code/4_evaluate_model.py --machine machine-1-1 --method percentile
    uv run python code/4_evaluate_model.py --all --from-saved
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.registry import TranADRegistry
from src.scorer import (
    DEFAULT_POT_PARAMS,
    POTParams,
    adjust_predicts,
    build_segment_summaries,
    calibrate_threshold,
    compute_feature_baselines,
    diagnose,
    diagnose_with_elevation,
    evaluate,
    score_batch,
)
from src.utils import auto_device

MACHINES = ["machine-1-1", "machine-2-1", "machine-3-2", "machine-3-7"]

PAPER_RESULTS = {
    "avg_precision": 0.9262,
    "avg_recall": 0.9974,
    "avg_f1": 0.9605,
}


def evaluate_machine(args, machine: str) -> dict | None:
    """Evaluate a single machine. Returns metrics dict or None on failure."""
    data_dir = PROJECT_ROOT / args.data_dir
    model_dir = PROJECT_ROOT / args.model_dir

    device = auto_device(args.device)
    print(f"Device: {device}")

    # Load model via registry
    registry = TranADRegistry(base_dir=model_dir)
    model, config = registry.get_model(machine, device=device)
    print(f"Loaded model for {machine}: {config.n_features} features, "
          f"window_size={config.window_size}")

    # Load preprocessed data
    for suffix in ["_train.npy", "_test.npy", "_interp_labels.npy"]:
        if not (data_dir / f"{machine}{suffix}").exists():
            print(f"Error: {data_dir / f'{machine}{suffix}'} not found. "
                  f"Run code/0_verify_setup.py first.")
            return None

    train_data = np.load(data_dir / f"{machine}_train.npy")
    test_data = np.load(data_dir / f"{machine}_test.npy")
    interp_labels = np.load(data_dir / f"{machine}_interp_labels.npy")
    print(f"Data: train={train_data.shape}, test={test_data.shape}, "
          f"labels={interp_labels.shape}")

    # Score train and test data
    print(f"Scoring training data (mode={args.scoring_mode})...")
    train_scores = score_batch(model, train_data, config.window_size, device, args.scoring_mode)
    print(f"  Train scores: shape={train_scores.shape}, "
          f"mean={train_scores.mean():.6f}, max={train_scores.max():.6f}")

    # Compute per-feature baselines
    baselines = compute_feature_baselines(train_scores)
    print(f"  Feature baselines: shape={baselines.shape}, "
          f"min={baselines.min():.6f}, max={baselines.max():.6f}")

    print(f"Scoring test data (mode={args.scoring_mode})...")
    test_scores = score_batch(model, test_data, config.window_size, device, args.scoring_mode)
    print(f"  Test scores: shape={test_scores.shape}, "
          f"mean={test_scores.mean():.6f}, max={test_scores.max():.6f}")

    # Calibrate threshold
    pot_params = POTParams(q=args.pot_q, level=args.pot_level, scale=args.pot_scale)

    print(f"Calibrating threshold (method={args.method})...")
    cal_result = calibrate_threshold(
        train_scores=train_scores,
        test_scores=test_scores,
        labels=interp_labels,
        method=args.method,
        pot_params=pot_params,
        percentile=args.percentile,
    )
    threshold = cal_result["threshold"]
    print(f"  Threshold: {threshold:.6f}")

    # Evaluate
    print("Evaluating with point-adjustment...")
    metrics = evaluate(test_scores, interp_labels, threshold)

    # Root cause diagnosis
    print("Computing diagnosis metrics...")
    diag = diagnose(test_scores, interp_labels)
    metrics.update(diag)

    # Elevation-ratio diagnosis
    print("Computing elevation-ratio diagnosis...")
    diag_elev = diagnose_with_elevation(test_scores, interp_labels, baselines)
    metrics.update(diag_elev)

    # Segment-level feature attribution
    print("Computing feature attribution...")
    score_1d = np.mean(test_scores, axis=1)
    if interp_labels.ndim == 2:
        labels_1d = (np.sum(interp_labels, axis=1) >= 1).astype(float)
    else:
        labels_1d = interp_labels.astype(float)

    raw_predictions = (score_1d > threshold).astype(float)
    raw_segments = build_segment_summaries(test_scores, raw_predictions, baselines)

    adjusted_predictions = adjust_predicts(score_1d, labels_1d, threshold)
    adjusted_segments = build_segment_summaries(test_scores, adjusted_predictions, baselines)

    print(f"  Raw segments: {len(raw_segments)}, Adjusted segments: {len(adjusted_segments)}")
    for i, seg in enumerate(raw_segments[:5]):
        n_attr = len(seg["attributed_dimensions"])
        top_dims = [d["label"] for d in seg["attributed_dimensions"][:3]]
        print(f"  Segment {i + 1}: [{seg['segment_start']}-{seg['segment_end']}] "
              f"peak={seg['peak_score']:.4f}, {n_attr} attributed dims: {top_dims}")

    # Print results
    print()
    print("=" * 60)
    print(f"Results for {machine} (method={args.method})")
    print("=" * 60)
    print(f"  F1:          {metrics['f1']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  ROC/AUC:     {metrics['roc_auc']:.4f}")
    print(f"  Threshold:   {threshold:.6f}")
    print(f"  TP={metrics['TP']}, TN={metrics['TN']}, "
          f"FP={metrics['FP']}, FN={metrics['FN']}")
    for key in ["Hit@100%", "Hit@150%", "NDCG@100%", "NDCG@150%"]:
        if key in metrics:
            elev_key = f"{key}_elev"
            raw_val = metrics[key]
            elev_val = metrics.get(elev_key, 0.0)
            delta = elev_val - raw_val
            print(f"  {key:<12s} raw={raw_val:.4f}  elev={elev_val:.4f}  delta={delta:+.4f}")
    print(f"  Segments: {len(raw_segments)} raw, {len(adjusted_segments)} adjusted")
    print("=" * 60)

    # Save results
    output_path = model_dir / machine / "eval_results.json"
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
    save_metrics["machine"] = machine
    save_metrics["n_anomaly_segments"] = len(raw_segments)

    with open(output_path, "w") as f:
        json.dump(save_metrics, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save attribution results
    attribution_path = model_dir / machine / "attribution_results.json"
    with open(attribution_path, "w") as f:
        json.dump({"raw_segments": raw_segments, "adjusted_segments": adjusted_segments}, f, indent=2)
    print(f"Attribution saved to {attribution_path}")

    # Save scorer state to registry
    cal_result["feature_baselines"] = baselines
    registry.save_scorer_state(machine, cal_result)
    print("Scorer state saved to registry")

    return save_metrics


def show_from_saved(model_dir: Path) -> None:
    """Read existing eval_results.json files and print summary table."""
    results = {}
    for machine in MACHINES:
        eval_path = model_dir / machine / "eval_results.json"
        if eval_path.exists():
            with open(eval_path) as f:
                results[machine] = json.load(f)

    if not results:
        print("No saved results found.")
        return

    w = 86
    print(f"\n{'=' * w}")
    print("SAVED RESULTS (from eval_results.json)")
    print(f"{'=' * w}")
    print(f"{'Machine':<16} {'Method':<10} {'Thresh':>10} "
          f"{'P':>7} {'R':>7} {'F1':>7} {'AUC':>7}")
    print("-" * w)

    f1s, ps, rs = [], [], []
    for machine in MACHINES:
        if machine not in results:
            print(f"{machine:<16} {'MISSING':>10}")
            continue
        m = results[machine]
        method = m.get("method", "pot")
        print(f"{machine:<16} {method:<10} {m['threshold']:>10.4f} "
              f"{m['precision']:>7.3f} {m['recall']:>7.3f} "
              f"{m['f1']:>7.3f} {m['roc_auc']:>7.3f}")
        f1s.append(m["f1"])
        ps.append(m["precision"])
        rs.append(m["recall"])

    if f1s:
        print("-" * w)
        print(f"{'Average':<16} {'':10} {'':>10} "
              f"{np.mean(ps):>7.3f} {np.mean(rs):>7.3f} {np.mean(f1s):>7.3f}")
        print(f"{'Paper (Table 2)':<16} {'':10} {'':>10} "
              f"{PAPER_RESULTS['avg_precision']:>7.3f} "
              f"{PAPER_RESULTS['avg_recall']:>7.3f} "
              f"{PAPER_RESULTS['avg_f1']:>7.3f}")
        gap = np.mean(f1s) - PAPER_RESULTS["avg_f1"]
        print(f"\nGap from paper: F1={gap:+.4f}")
    print("=" * w)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TranAD on SMD")
    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument("--all", action="store_true", help="Evaluate all reference machines")
    parser.add_argument("--from-saved", action="store_true",
                        help="Show saved results without re-scoring")
    parser.add_argument("--data-dir", type=str, default="data/smd/processed")
    parser.add_argument("--model-dir", type=str, default="models/tranad")
    parser.add_argument("--method", type=str, default="pot",
                        choices=["pot", "percentile", "f1_max"])
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--pot-q", type=float, default=1e-5)
    parser.add_argument("--pot-level", type=float, default=0.99995)
    parser.add_argument("--pot-scale", type=float, default=1.06)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--scoring-mode", type=str, default="phase2_only",
                        choices=["phase2_only", "averaged"])
    args = parser.parse_args()

    if args.from_saved:
        show_from_saved(PROJECT_ROOT / args.model_dir)
    elif args.all:
        for machine in MACHINES:
            print(f"\n{'='*60}")
            print(f"Machine: {machine}")
            print(f"{'='*60}")
            pot = DEFAULT_POT_PARAMS.get(machine, POTParams())
            args.pot_level = pot.level
            args.pot_scale = pot.scale
            evaluate_machine(args, machine)
        show_from_saved(PROJECT_ROOT / args.model_dir)
    else:
        evaluate_machine(args, args.machine)


if __name__ == "__main__":
    main()
