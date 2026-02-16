"""
SMD Dataset Download Script

Downloads the Server Machine Dataset (SMD) from the OmniAnomaly GitHub repository.
The SMD dataset contains 28 server machines with 38 features each, divided into
train/test splits with per-point anomaly labels and per-dimension attribution labels.

Usage:
    uv run python scripts/download_smd.py
    uv run python scripts/download_smd.py --output-dir data/smd/raw --force
    uv run python scripts/download_smd.py --validate-only
"""

import argparse
import logging
import sys
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

BASE_URL = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/"
    "master/ServerMachineDataset"
)

MACHINE_GROUPS: dict[int, list[int]] = {
    1: list(range(1, 9)),   # machine-1-1 through machine-1-8
    2: list(range(1, 10)),  # machine-2-1 through machine-2-9
    3: list(range(1, 12)),  # machine-3-1 through machine-3-11
}

SUBDIRECTORIES = ["train", "test", "test_label", "interpretation_label"]

EXPECTED_FEATURES = 38


def get_all_machine_names() -> list[str]:
    """Return all 28 machine names in order."""
    names = []
    for group, ids in MACHINE_GROUPS.items():
        for machine_id in ids:
            names.append(f"machine-{group}-{machine_id}")
    return names


def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(
    url: str, dest: Path, session: requests.Session, timeout: int = 30
) -> bool:
    """Download a single file. Returns True on success."""
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(response.text)
        return True
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            logger.warning(f"Not found (404): {url}")
        else:
            logger.error(f"HTTP error downloading {url}: {e}")
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_all(output_dir: Path, force: bool = False) -> None:
    """Download all SMD data files."""
    machines = get_all_machine_names()
    total_files = len(machines) * len(SUBDIRECTORIES)

    logger.info(
        f"Downloading SMD dataset: {len(machines)} machines x "
        f"{len(SUBDIRECTORIES)} subdirectories = {total_files} files"
    )
    logger.info(f"Output directory: {output_dir}")

    session = create_session()
    downloaded = 0
    skipped = 0
    failed = 0

    with tqdm(total=total_files, desc="Downloading SMD") as pbar:
        for subdir in SUBDIRECTORIES:
            for machine in machines:
                filename = f"{machine}.txt"
                url = f"{BASE_URL}/{subdir}/{filename}"
                dest = output_dir / subdir / filename

                if dest.exists() and not force:
                    skipped += 1
                    pbar.update(1)
                    continue

                if download_file(url, dest, session):
                    downloaded += 1
                else:
                    failed += 1

                pbar.update(1)

    logger.info(
        f"Download complete: {downloaded} downloaded, "
        f"{skipped} skipped (existing), {failed} failed"
    )

    if failed > 0:
        logger.warning(f"{failed} files failed to download. Re-run to retry.")


def count_lines(filepath: Path) -> int:
    """Count non-empty lines in a file."""
    with open(filepath) as f:
        return sum(1 for line in f if line.strip())


def count_columns(filepath: Path) -> int:
    """Count columns in the first line of a comma-delimited file."""
    with open(filepath) as f:
        first_line = f.readline().strip()
        if not first_line:
            return 0
        return len(first_line.split(","))


def validate_downloads(data_dir: Path) -> bool:
    """Validate downloaded SMD data for completeness and consistency."""
    machines = get_all_machine_names()
    success = True
    total_train_rows = 0
    total_test_rows = 0

    logger.info(f"Validating SMD data in {data_dir}...")

    for machine in machines:
        train_file = data_dir / "train" / f"{machine}.txt"
        test_file = data_dir / "test" / f"{machine}.txt"
        label_file = data_dir / "test_label" / f"{machine}.txt"
        interp_file = data_dir / "interpretation_label" / f"{machine}.txt"

        # Check all 4 files exist
        missing = []
        for f in [train_file, test_file, label_file, interp_file]:
            if not f.exists():
                missing.append(f.name)
                success = False

        if missing:
            logger.error(f"{machine}: missing files: {', '.join(missing)}")
            continue

        # Count rows
        train_rows = count_lines(train_file)
        test_rows = count_lines(test_file)
        label_rows = count_lines(label_file)
        interp_segments = count_lines(interp_file)

        # Validate test rows == test_label rows
        if test_rows != label_rows:
            logger.error(
                f"{machine}: test rows ({test_rows}) != "
                f"test_label rows ({label_rows})"
            )
            success = False

        # interpretation_label uses segment format (start-end:dims),
        # not one row per timestep — just check file is non-empty
        if interp_segments == 0:
            logger.warning(f"{machine}: interpretation_label is empty")

        # Validate column count
        train_cols = count_columns(train_file)
        test_cols = count_columns(test_file)

        if train_cols != EXPECTED_FEATURES:
            logger.warning(
                f"{machine}: train has {train_cols} columns "
                f"(expected {EXPECTED_FEATURES})"
            )
        if test_cols != EXPECTED_FEATURES:
            logger.warning(
                f"{machine}: test has {test_cols} columns "
                f"(expected {EXPECTED_FEATURES})"
            )

        total_train_rows += train_rows
        total_test_rows += test_rows

        logger.info(
            f"  {machine}: train={train_rows} rows, test={test_rows} rows, "
            f"labels={label_rows} rows, interp_segments={interp_segments}, "
            f"cols={train_cols}"
        )

    logger.info(
        f"Total: {total_train_rows} train rows, "
        f"{total_test_rows} test rows across {len(machines)} machines"
    )

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Download SMD dataset from OmniAnomaly repository"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/smd/raw"),
        help="Output directory for raw SMD files (default: data/smd/raw)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing downloads, don't download",
    )
    args = parser.parse_args()

    if args.validate_only:
        success = validate_downloads(args.output_dir)
    else:
        download_all(args.output_dir, force=args.force)
        success = validate_downloads(args.output_dir)

    if success:
        logger.info("All 28 machines downloaded and validated successfully")
    else:
        logger.error("Validation failed - see errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
