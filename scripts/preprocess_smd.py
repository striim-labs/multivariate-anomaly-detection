"""
Preprocess raw SMD data: normalize, parse labels, save as .npy.

For each machine:
  1. Load train/test/test_label/interpretation_label .txt files
  2. Min-max normalize using training set statistics
  3. Parse interpretation labels into per-dimension binary matrix
  4. Save to data/smd/processed/

Usage:
    uv run python scripts/preprocess_smd.py
    uv run python scripts/preprocess_smd.py --machines machine-1-1 machine-2-1
"""

import argparse
from pathlib import Path

import numpy as np


def normalize(
    data: np.ndarray,
    min_vals: np.ndarray | None = None,
    max_vals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature min-max normalization to [0, 1).

    Reference: tranad/preprocess.py, normalize3().

    Args:
        data: shape (N, features)
        min_vals: per-feature minimums (computed from data if None)
        max_vals: per-feature maximums (computed from data if None)

    Returns:
        (normalized_data, min_vals, max_vals)
    """
    if min_vals is None:
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
    return (data - min_vals) / (max_vals - min_vals + 1e-4), min_vals, max_vals


def parse_interpretation_labels(filepath: Path, shape: tuple[int, int]) -> np.ndarray:
    """Parse interpretation label file into binary matrix.

    Format per line: "start-end:dim1,dim2,dim3"
    Positions are 1-indexed. Converts to 0-indexed with start-1:end-1 for rows
    and int(i)-1 for dimension indices.

    Reference: tranad/preprocess.py, load_and_save2().

    Args:
        filepath: path to interpretation label .txt file
        shape: (n_timesteps, n_features) for the output matrix

    Returns:
        Binary matrix of shape (n_timesteps, n_features)
    """
    labels = np.zeros(shape, dtype=np.float64)
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pos, values = line.split(":")[0], line.split(":")[1].split(",")
            start = int(pos.split("-")[0])
            end = int(pos.split("-")[1])
            dims = [int(i) - 1 for i in values]
            labels[start - 1 : end - 1, dims] = 1
    return labels


def preprocess_machine(
    machine: str, raw_dir: Path, output_dir: Path
) -> dict[str, tuple]:
    """Preprocess a single machine's data.

    Args:
        machine: machine name, e.g. "machine-1-1"
        raw_dir: path to data/smd/raw/
        output_dir: path to data/smd/processed/

    Returns:
        dict with shapes of saved arrays
    """
    # Load raw data
    train = np.genfromtxt(
        raw_dir / "train" / f"{machine}.txt", dtype=np.float64, delimiter=","
    )
    test = np.genfromtxt(
        raw_dir / "test" / f"{machine}.txt", dtype=np.float64, delimiter=","
    )
    test_label = np.genfromtxt(
        raw_dir / "test_label" / f"{machine}.txt", dtype=np.float64, delimiter=","
    )

    # Normalize using training set statistics
    train_norm, min_vals, max_vals = normalize(train)
    test_norm, _, _ = normalize(test, min_vals, max_vals)

    # Parse interpretation labels
    interp_path = raw_dir / "interpretation_label" / f"{machine}.txt"
    interp_labels = parse_interpretation_labels(interp_path, test.shape)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{machine}_train.npy", train_norm)
    np.save(output_dir / f"{machine}_test.npy", test_norm)
    np.save(output_dir / f"{machine}_test_labels.npy", test_label)
    np.save(output_dir / f"{machine}_interp_labels.npy", interp_labels)
    np.save(
        output_dir / f"{machine}_norm_params.npy",
        np.stack([min_vals, max_vals]),
    )

    shapes = {
        "train": train_norm.shape,
        "test": test_norm.shape,
        "test_labels": test_label.shape,
        "interp_labels": interp_labels.shape,
        "norm_params": (2, train.shape[1]),
    }
    return shapes


def discover_machines(raw_dir: Path) -> list[str]:
    """Find all machine names from the train directory."""
    train_dir = raw_dir / "train"
    machines = sorted(
        f.stem for f in train_dir.glob("*.txt")
    )
    return machines


def main():
    parser = argparse.ArgumentParser(description="Preprocess SMD data")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/smd/raw"),
        help="Path to raw SMD data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/smd/processed"),
        help="Path to save processed data",
    )
    parser.add_argument(
        "--machines",
        nargs="*",
        default=None,
        help="Specific machines to process (default: all)",
    )
    args = parser.parse_args()

    machines = args.machines or discover_machines(args.raw_dir)
    print(f"Processing {len(machines)} machines from {args.raw_dir}")

    for machine in machines:
        shapes = preprocess_machine(machine, args.raw_dir, args.output_dir)
        print(
            f"  {machine}: train={shapes['train']}, test={shapes['test']}, "
            f"labels={shapes['test_labels']}, interp={shapes['interp_labels']}"
        )

    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
