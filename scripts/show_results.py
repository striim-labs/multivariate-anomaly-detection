"""
Display TranAD evaluation results across all SMD machines.

Two modes:
  --from-saved    Read existing eval_results.json files (instant, POT-only)
  (default)       Load models, score data, compare POT vs percentile sweep

Usage:
    uv run python scripts/show_results.py --from-saved
    uv run python scripts/show_results.py
    uv run python scripts/show_results.py --percentiles 99.5 99.9 99.99
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

DEFAULT_PERCENTILES = [99.0, 99.9, 99.99, 99.999]


def show_from_saved(model_dir: Path, machines: list[str]) -> None:
    """Read existing eval_results.json files and print a POT-only summary table."""
    results = {}
    for machine in machines:
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
    print(
        f"{'Machine':<16} {'Method':<10} {'Thresh':>10} "
        f"{'P':>7} {'R':>7} {'F1':>7} {'AUC':>7}"
    )
    print("-" * w)

    f1s, ps, rs = [], [], []
    for machine in machines:
        if machine not in results:
            print(f"{machine:<16} {'MISSING':>10}")
            continue
        m = results[machine]
        method = m.get("method", "pot")
        print(
            f"{machine:<16} {method:<10} {m['threshold']:>10.4f} "
            f"{m['precision']:>7.3f} {m['recall']:>7.3f} "
            f"{m['f1']:>7.3f} {m['roc_auc']:>7.3f}"
        )
        f1s.append(m["f1"])
        ps.append(m["precision"])
        rs.append(m["recall"])

    if f1s:
        print("-" * w)
        print(
            f"{'Average':<16} {'':10} {'':>10} "
            f"{np.mean(ps):>7.3f} {np.mean(rs):>7.3f} {np.mean(f1s):>7.3f}"
        )
        print(
            f"{'Paper (Table 2)':<16} {'':10} {'':>10} "
            f"{PAPER_RESULTS['avg_precision']:>7.3f} "
            f"{PAPER_RESULTS['avg_recall']:>7.3f} "
            f"{PAPER_RESULTS['avg_f1']:>7.3f}"
        )
        gap = np.mean(f1s) - PAPER_RESULTS["avg_f1"]
        print(f"\nGap from paper: F1={gap:+.4f}")
    print("=" * w)


def evaluate_all(
    registry: TranADRegistry,
    scorer: TranADScorer,
    data_dir: Path,
    machines: list[str],
    device: str,
    percentiles: list[float],
    scoring_mode: str,
) -> list[dict]:
    """Load models, score data, evaluate with POT and multiple percentiles.

    Returns a list of result dicts, one per machine.
    """
    results = []

    for machine in machines:
        print(f"\nEvaluating {machine}...")

        # Load model
        try:
            model, config = registry.get_model(machine, device=device)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        # Load data
        train_path = data_dir / f"{machine}_train.npy"
        test_path = data_dir / f"{machine}_test.npy"
        labels_path = data_dir / f"{machine}_interp_labels.npy"
        missing = False
        for p in [train_path, test_path, labels_path]:
            if not p.exists():
                print(f"  Skipping: {p} not found")
                missing = True
                break
        if missing:
            continue

        train_data = np.load(train_path)
        test_data = np.load(test_path)
        labels = np.load(labels_path)

        # Score (once per machine)
        print(f"  Scoring (mode={scoring_mode})...")
        train_scores = scorer.score_batch(
            model, train_data, config.window_size, device, scoring_mode
        )
        test_scores = scorer.score_batch(
            model, test_data, config.window_size, device, scoring_mode
        )

        # POT calibration + evaluation
        pot_params = POT_PARAMS.get(machine, POTParams())
        pot_cal = scorer.calibrate_threshold(
            train_scores, test_scores, labels,
            method="pot", pot_params=pot_params,
        )
        pot_th = pot_cal["threshold"]
        pot_metrics = scorer.evaluate(test_scores, labels, pot_th)

        # Detect POT fallback: compare to test p99.9
        test_1d = np.mean(test_scores, axis=1)
        test_p999 = float(np.percentile(test_1d, 99.9))
        pot_fell_back = abs(pot_th - test_p999) < 1e-6

        print(
            f"  POT: th={pot_th:.4f} F1={pot_metrics['f1']:.3f}"
            f"{'  [fallback to test p99.9]' if pot_fell_back else ''}"
        )

        # Percentile sweep (cheap — no re-scoring needed)
        pct_results = {}
        for pct in percentiles:
            pct_cal = scorer.calibrate_threshold(
                train_scores, test_scores, labels,
                method="percentile", percentile=pct,
            )
            pct_th = pct_cal["threshold"]
            pct_metrics = scorer.evaluate(test_scores, labels, pct_th)
            pct_results[pct] = {
                "threshold": pct_th,
                "f1": pct_metrics["f1"],
                "precision": pct_metrics["precision"],
                "recall": pct_metrics["recall"],
            }
            print(f"  p{pct}: th={pct_th:.6f} F1={pct_metrics['f1']:.3f}")

        results.append({
            "machine": machine,
            "pot_threshold": pot_th,
            "pot_f1": pot_metrics["f1"],
            "pot_precision": pot_metrics["precision"],
            "pot_recall": pot_metrics["recall"],
            "pot_fell_back": pot_fell_back,
            "pct": pct_results,
        })

    return results


def print_comparison_table(results: list[dict], percentiles: list[float]) -> None:
    """Print POT results + percentile F1 sweep table."""
    # Build dynamic column headers for percentile sweep
    pct_headers = [f"p{p}" for p in percentiles]
    pct_col_width = max(len(h) for h in pct_headers)
    pct_col_width = max(pct_col_width, 7)  # minimum width for F1 values

    # Calculate table width
    pot_section_width = 42  # "  Thresh      P      R     F1"
    pct_section_width = len(percentiles) * (pct_col_width + 1) + 1
    machine_col = 18
    w = machine_col + pot_section_width + pct_section_width
    w = max(w, 86)

    print(f"\n{'=' * w}")
    print("RESULTS: POT vs Percentile Sweep (F1)")
    print(f"{'=' * w}")

    # Header row 1: section labels
    pct_label = "Percentile F1"
    print(
        f"{'':>{machine_col}}"
        f"{'POT':^{pot_section_width}}|"
        f"{pct_label:^{pct_section_width - 1}}"
    )

    # Header row 2: column names
    pct_header_str = " ".join(f"{h:>{pct_col_width}}" for h in pct_headers)
    print(
        f"{'Machine':<{machine_col}}"
        f"{'Thresh':>10} {'P':>6} {'R':>6} {'F1':>6}   |"
        f" {pct_header_str}"
    )
    print("-" * w)

    pot_f1s, pot_ps, pot_rs = [], [], []
    pct_f1_by_col = {p: [] for p in percentiles}
    fallback_machines = []

    for r in results:
        flag = " *" if r["pot_fell_back"] else "  "
        if r["pot_fell_back"]:
            fallback_machines.append(r["machine"])

        pct_f1_str = " ".join(
            f"{r['pct'][p]['f1']:>{pct_col_width}.3f}" for p in percentiles
        )
        print(
            f"{r['machine']:<{machine_col - 2}}{flag}"
            f"{r['pot_threshold']:>10.4f} "
            f"{r['pot_precision']:>6.3f} {r['pot_recall']:>6.3f} "
            f"{r['pot_f1']:>6.3f}   |"
            f" {pct_f1_str}"
        )

        pot_f1s.append(r["pot_f1"])
        pot_ps.append(r["pot_precision"])
        pot_rs.append(r["pot_recall"])
        for p in percentiles:
            pct_f1_by_col[p].append(r["pct"][p]["f1"])

    if pot_f1s:
        print("-" * w)
        avg_pct_str = " ".join(
            f"{np.mean(pct_f1_by_col[p]):>{pct_col_width}.3f}" for p in percentiles
        )
        print(
            f"{'Average':<{machine_col}}"
            f"{'':>10} "
            f"{np.mean(pot_ps):>6.3f} {np.mean(pot_rs):>6.3f} "
            f"{np.mean(pot_f1s):>6.3f}   |"
            f" {avg_pct_str}"
        )
        print(
            f"{'Paper (Table 2)':<{machine_col}}"
            f"{'':>10} "
            f"{PAPER_RESULTS['avg_precision']:>6.3f} "
            f"{PAPER_RESULTS['avg_recall']:>6.3f} "
            f"{PAPER_RESULTS['avg_f1']:>6.3f}   |"
        )
        gap = np.mean(pot_f1s) - PAPER_RESULTS["avg_f1"]
        print(f"\nGap from paper: F1={gap:+.4f}")

    if fallback_machines:
        print(
            f"* POT fell back to test p99.9 "
            f"(anomaly rate > 20% with raw POT threshold)"
        )
    print("=" * w)


def main():
    parser = argparse.ArgumentParser(
        description="Display TranAD evaluation results across SMD machines"
    )
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="Read existing eval_results.json files (instant, POT-only)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/smd/processed"),
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/tranad"),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=DEFAULT_PERCENTILES,
        help="Percentile values to sweep (default: 99.0 99.9 99.99 99.999)",
    )
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default="phase2_only",
        choices=["phase2_only", "averaged"],
    )
    parser.add_argument(
        "--machines",
        nargs="*",
        default=None,
        help="Override machine list (default: all 4 reference machines)",
    )
    args = parser.parse_args()
    machines = args.machines or MACHINES

    if args.from_saved:
        show_from_saved(args.model_dir, machines)
    else:
        device = auto_device(args.device)
        print(f"Device: {device}")

        registry = TranADRegistry(base_dir=args.model_dir)
        scorer = TranADScorer()

        results = evaluate_all(
            registry, scorer, args.data_dir, machines,
            device, args.percentiles, args.scoring_mode,
        )

        if results:
            print_comparison_table(results, args.percentiles)


if __name__ == "__main__":
    main()
