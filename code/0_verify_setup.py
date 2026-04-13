"""
Verify Setup: Download Data, Preprocess, and Check Artifacts

Downloads the SMD dataset if missing, preprocesses it, and verifies
that all required data files and model checkpoints exist.

Usage:
    uv run python code/0_verify_setup.py
    uv run python code/0_verify_setup.py --force-download
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    REFERENCE_MACHINES,
    discover_machines,
    download_smd_dataset,
    get_all_machine_names,
    preprocess_machine,
)


def verify_imports() -> bool:
    """Check that all required packages are importable."""
    required = [
        "torch", "numpy", "pandas", "fastapi", "uvicorn", "pydantic",
        "sklearn", "scipy", "requests", "yaml", "tqdm", "rich",
    ]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"  Missing packages: {', '.join(missing)}")
        print("  Run: uv sync")
        return False
    print(f"  All {len(required)} required packages importable")
    return True


def verify_data(raw_dir: Path, processed_dir: Path, force_download: bool = False) -> bool:
    """Download and preprocess data if needed, verify files exist."""
    ok = True

    # Download if raw data missing
    train_dir = raw_dir / "train"
    if not train_dir.exists() or len(list(train_dir.glob("*.txt"))) < 28 or force_download:
        print("  Downloading SMD dataset...")
        download_smd_dataset(raw_dir, force=force_download)
    else:
        n_files = len(list(train_dir.glob("*.txt")))
        print(f"  Raw data: {n_files} machines in {raw_dir}")

    # Preprocess if processed data missing
    machines_to_process = []
    all_machines = get_all_machine_names()
    for machine in all_machines:
        if not (processed_dir / f"{machine}_train.npy").exists():
            machines_to_process.append(machine)

    if machines_to_process:
        print(f"  Preprocessing {len(machines_to_process)} machines...")
        for machine in machines_to_process:
            try:
                shapes = preprocess_machine(machine, raw_dir, processed_dir)
                print(f"    {machine}: train={shapes['train']}, test={shapes['test']}")
            except Exception as e:
                print(f"    {machine}: FAILED - {e}")
                ok = False
    else:
        print(f"  Processed data: all {len(all_machines)} machines in {processed_dir}")

    return ok


def verify_models(model_dir: Path) -> bool:
    """Check that model checkpoints exist for reference machines."""
    ok = True
    for machine in REFERENCE_MACHINES:
        ckpt = model_dir / machine / "model.ckpt"
        state = model_dir / machine / "scorer_state.json"
        if ckpt.exists() and state.exists():
            print(f"  {machine}: model.ckpt + scorer_state.json")
        elif ckpt.exists():
            print(f"  {machine}: model.ckpt (no scorer_state.json - run code/2_evaluate_model.py)")
        else:
            print(f"  {machine}: MISSING - run code/1_train_model.py")
            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description="Verify setup for TranAD project")
    parser.add_argument("--force-download", action="store_true",
                        help="Re-download SMD data even if it exists")
    args = parser.parse_args()

    raw_dir = PROJECT_ROOT / "data" / "smd" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "smd" / "processed"
    model_dir = PROJECT_ROOT / "models" / "tranad"

    print("=" * 60)
    print("TranAD Multivariate Anomaly Detection - Setup Verification")
    print("=" * 60)

    print("\n1. Checking dependencies...")
    deps_ok = verify_imports()

    print("\n2. Checking data...")
    data_ok = verify_data(raw_dir, processed_dir, args.force_download)

    print("\n3. Checking model checkpoints...")
    models_ok = verify_models(model_dir)

    print("\n" + "=" * 60)
    if deps_ok and data_ok and models_ok:
        print("All checks passed! Ready to go.")
        print("\nNext steps:")
        print("  uv run python code/1_train_model.py --machine machine-1-1")
    else:
        print("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
