#!/usr/bin/env python3
"""
TranAD Demo Data Generator

Reads raw SMD test data and periodically writes JSON scoring request files
to the samples/demo/ directory. Each file contains a contiguous window of
10 timesteps, ready to be sent to the /score API endpoint via curl.

Usage:
    uv run python scripts/generate_demo_data.py
    uv run python scripts/generate_demo_data.py --machine machine-3-2 --interval 5
    uv run python scripts/generate_demo_data.py --interval 0 --max-files 50

End-to-end workflow:
    1. docker compose -f docker-compose.rest.yml up --build
    2. uv run python scripts/generate_demo_data.py
    3. curl -X POST http://localhost:8000/score -H "Content-Type: application/json" \\
       -d @samples/demo/score_request_0.json
"""

import argparse
import json
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rich.console import Console

console = Console()

WINDOW_SIZE = 10  # Rows per scoring request (matches model window_size)


def parse_machine_id(machine_id: str) -> tuple[int, int]:
    """Parse 'machine-X-Y' into (store_id, device_id)."""
    m = re.match(r"^machine-(\d+)-(\d+)$", machine_id)
    if not m:
        console.print(
            f"[red]Invalid machine ID: {machine_id!r} (expected 'machine-X-Y')[/red]"
        )
        sys.exit(1)
    return int(m.group(1)), int(m.group(2))


def load_raw_data(data_dir: str, machine_id: str) -> np.ndarray:
    """Load raw CSV test data, shape (N, 38)."""
    path = Path(data_dir) / f"{machine_id}.txt"
    if not path.exists():
        console.print(f"[red]Data file not found: {path}[/red]")
        console.print("Run: uv run python scripts/download_smd.py")
        sys.exit(1)
    return np.genfromtxt(path, dtype=np.float64, delimiter=",")


def build_request(
    store_id: int,
    device_id: int,
    window: np.ndarray,
    filename: str,
) -> dict:
    """Build a scoring request dict matching samples/score_request.json format."""
    return {
        "store_id": store_id,
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "filename": filename,
        "data": window.tolist(),
        "include_per_timestep": False,
        "include_attribution": True,
        "scoring_mode": "phase2_only",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate demo scoring request files from raw SMD test data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  uv run python scripts/generate_demo_data.py
  uv run python scripts/generate_demo_data.py --machine machine-3-2 --interval 5
  uv run python scripts/generate_demo_data.py --interval 0 --max-files 50
""",
    )
    parser.add_argument(
        "--machine",
        default="machine-1-1",
        help="Machine ID in format machine-X-Y (default: machine-1-1)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Seconds between file writes (default: 10.0, use 0 for no delay)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/smd/raw/test",
        help="Path to raw SMD test data (default: data/smd/raw/test)",
    )
    parser.add_argument(
        "--output-dir",
        default="samples/demo",
        help="Output directory for generated JSON files (default: samples/demo)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Stop after N files (default: 0 = generate until data exhausted)",
    )
    args = parser.parse_args()

    # Parse machine ID and load data
    store_id, device_id = parse_machine_id(args.machine)
    data = load_raw_data(args.data_dir, args.machine)

    # Calculate windows
    total_windows = len(data) // WINDOW_SIZE
    if total_windows == 0:
        console.print("[red]Not enough data rows for even one window[/red]")
        sys.exit(1)

    max_files = args.max_files if args.max_files > 0 else total_windows
    num_files = min(max_files, total_windows)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Startup banner
    console.print()
    console.print("[bold blue]TranAD Demo Data Generator[/bold blue]")
    console.print(
        f"  Machine:    [cyan]{args.machine}[/cyan]"
        f" (store_id={store_id}, device_id={device_id})"
    )
    console.print(f"  Data:       {len(data):,} rows x {data.shape[1]} features")
    console.print(f"  Windows:    {num_files:,} files of {WINDOW_SIZE} rows each")
    console.print(f"  Output:     {output_dir.resolve()}")
    console.print(f"  Interval:   {args.interval}s between writes")
    console.print()

    # Graceful Ctrl+C
    stop = False

    def signal_handler(sig, frame):
        nonlocal stop
        console.print("\n[yellow]Interrupted -- stopping generation...[/yellow]")
        stop = True

    signal.signal(signal.SIGINT, signal_handler)

    # Generate files
    files_written = 0

    for i in range(num_files):
        if stop:
            break

        row_start = i * WINDOW_SIZE
        row_end = row_start + WINDOW_SIZE
        window = data[row_start:row_end]

        filename = f"score_request_{i}.json"
        request = build_request(store_id, device_id, window, filename)
        filepath = output_dir / filename

        with open(filepath, "w") as f:
            json.dump(request, f, indent=2)

        files_written += 1
        remaining = num_files - files_written

        console.print(
            f"  [green]Wrote[/green] {filename}"
            f"  [dim](rows {row_start}-{row_end - 1},"
            f" {remaining:,} remaining)[/dim]"
        )

        # Sleep between writes (skip on last file or if interval is 0)
        if args.interval > 0 and i < num_files - 1 and not stop:
            time.sleep(args.interval)

    # Summary
    console.print()
    if stop:
        console.print(
            f"[yellow]Stopped early.[/yellow]"
            f" Wrote {files_written:,} of {num_files:,} files."
        )
    else:
        console.print(
            f"[green]Done.[/green] Wrote {files_written:,} files to {output_dir}/"
        )

    console.print()
    console.print("[dim]To score a file:[/dim]")
    console.print(
        f"  curl -X POST http://localhost:8000/score"
        f' -H "Content-Type: application/json"'
        f" -d @{output_dir}/score_request_0.json"
    )


if __name__ == "__main__":
    main()
