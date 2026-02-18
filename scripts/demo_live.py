#!/usr/bin/env python3
"""
TranAD Live Anomaly Detection Demo

Simulates real-time telemetry streaming by sending batches of raw sensor
data from SMD test sets to separate per-cluster /score REST API instances.
Runs multiple machines concurrently and displays detection results with
root-cause attribution in a rich terminal interface.

Usage:
    # Default (two clusters on ports 8000 and 8001)
    uv run python scripts/demo_live.py

    # Custom interval and batch size
    uv run python scripts/demo_live.py --interval 0.5 --batch-size 100

    # Skip ahead to anomaly zones for a quick demo
    uv run python scripts/demo_live.py --skip-to 6400 --max-ticks 20 --interval 0.3

    # Custom server mapping
    uv run python scripts/demo_live.py \\
        --machines machine-2-1@http://cluster-a:8000 machine-3-2@http://cluster-b:8000
"""

import argparse
import dataclasses
import re
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Color assignments per machine (cycles if more than 4)
MACHINE_STYLES = ["cyan", "magenta", "green", "yellow"]

console = Console()


# ── Data classes ──────────────────────────────────────────────────


@dataclasses.dataclass
class MachineSpec:
    machine_id: str
    store_id: int
    device_id: int
    server_url: str
    color: str


@dataclasses.dataclass
class MachineStats:
    machine_id: str
    server_url: str
    total_ticks: int = 0
    total_timesteps: int = 0
    total_anomalies: int = 0
    total_segments: int = 0
    latencies_ms: list[float] = dataclasses.field(default_factory=list)
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        return (
            sum(self.latencies_ms) / len(self.latencies_ms)
            if self.latencies_ms
            else 0
        )

    @property
    def max_latency_ms(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0

    @property
    def anomaly_ratio(self) -> float:
        return (
            self.total_anomalies / self.total_timesteps
            if self.total_timesteps
            else 0
        )


# ── Helpers ──────────────────────────────────────────────────────


def parse_machine_spec(spec: str, color: str) -> MachineSpec:
    """Parse 'machine-X-Y@http://host:port' into a MachineSpec."""
    if "@" in spec:
        machine_id, server_url = spec.split("@", 1)
    else:
        machine_id = spec
        server_url = "http://localhost:8000"

    m = re.match(r"^machine-(\d+)-(\d+)$", machine_id)
    if not m:
        console.print(f"[red]Invalid machine ID: {machine_id!r} (expected 'machine-X-Y')[/red]")
        sys.exit(1)

    return MachineSpec(
        machine_id=machine_id,
        store_id=int(m.group(1)),
        device_id=int(m.group(2)),
        server_url=server_url.rstrip("/"),
        color=color,
    )


def load_raw_data(data_dir: str, machine_id: str) -> np.ndarray:
    """Load raw CSV test data, shape (N, 38)."""
    path = Path(data_dir) / f"{machine_id}.txt"
    if not path.exists():
        console.print(f"[red]Data file not found: {path}[/red]")
        console.print("Run: uv run python scripts/download_smd.py")
        sys.exit(1)
    return np.genfromtxt(path, dtype=np.float64, delimiter=",")


# ── Server checks ────────────────────────────────────────────────


def check_server(spec: MachineSpec) -> str | None:
    """GET /health on a server. Returns status string or None on failure."""
    try:
        resp = requests.get(f"{spec.server_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json().get("status", "unknown")
    except requests.ConnectionError:
        return None
    except Exception as e:
        console.print(f"[red]Health check failed for {spec.server_url}: {e}[/red]")
        return None


# ── Display ──────────────────────────────────────────────────────


def display_tick(
    spec: MachineSpec,
    tick: int,
    row_start: int,
    row_end: int,
    result: dict,
    latency_ms: float,
    is_first_tick: bool,
):
    """Print a tick line, expanding into anomaly alerts if needed."""
    ts = datetime.now().strftime("%H:%M:%S")
    port = spec.server_url.rsplit(":", 1)[-1] if ":" in spec.server_url else ""
    n_anomalies = result.get("n_anomalies", 0)

    # First-tick annotation for model loading latency
    suffix = ""
    if is_first_tick and latency_ms > 500:
        suffix = "  [dim](model loaded)[/dim]"

    if n_anomalies == 0:
        console.print(
            f"[dim]{ts}[/dim]  [{spec.color} bold]{spec.machine_id}[/{spec.color} bold]"
            f" [dim]-> :{port}[/dim]"
            f"  tick {tick:>4d}"
            f"  rows {row_start}-{row_end}"
            f"   [green]0 anomalies[/green]"
            f"   [dim]{latency_ms:.0f}ms[/dim]"
            f"{suffix}"
        )
    else:
        console.print(
            f"[dim]{ts}[/dim]  [{spec.color} bold]{spec.machine_id}[/{spec.color} bold]"
            f" [dim]-> :{port}[/dim]"
            f"  tick {tick:>4d}"
            f"  rows {row_start}-{row_end}"
            f"   [bold red]{n_anomalies} anomalies[/bold red]"
            f"   [dim]{latency_ms:.0f}ms[/dim]"
            f"{suffix}"
        )

        for seg in result.get("anomaly_segments", []):
            seg_start = seg["segment_start"]
            seg_end = seg["segment_end"]
            seg_len = seg["segment_length"]
            peak = seg["peak_score"]
            # Offset segment indices to absolute row positions
            abs_start = row_start + seg_start
            abs_end = row_start + seg_end

            console.print(
                f"           [bold red]ANOMALY[/bold red]"
                f"  rows {abs_start}-{abs_end} ({seg_len} pts)"
                f"  peak={peak:.4f}"
            )

            for attr in seg.get("attributed_dimensions", [])[:5]:
                label = attr["label"]
                elev = attr["mean_elevation"]
                contrib = attr["contribution"]
                console.print(
                    f"             [yellow]->[/yellow] {label}:"
                    f" {elev:,.1f}x baseline ({contrib:.1%})"
                )


def print_summary(all_stats: list[MachineStats], elapsed: float):
    """Print a rich summary table."""
    console.print()
    table = Table(title="Demo Summary", show_lines=True, expand=False)
    table.add_column("Machine", style="bold", no_wrap=True)
    table.add_column("Server", style="dim", no_wrap=True)
    table.add_column("Ticks", justify="right")
    table.add_column("Rows", justify="right")
    table.add_column("Anom.", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("Segs", justify="right")
    table.add_column("Avg ms", justify="right")
    table.add_column("Max ms", justify="right")
    table.add_column("Err", justify="right")

    for stats in all_stats:
        # Shorten server URL to host:port
        short_server = stats.server_url.replace("http://", "").replace("https://", "")
        table.add_row(
            stats.machine_id,
            short_server,
            f"{stats.total_ticks:,}",
            f"{stats.total_timesteps:,}",
            f"{stats.total_anomalies:,}",
            f"{stats.anomaly_ratio:.2%}",
            f"{stats.total_segments:,}",
            f"{stats.avg_latency_ms:.1f}",
            f"{stats.max_latency_ms:.1f}",
            str(stats.errors) if stats.errors else "[green]0[/green]",
        )

    console.print(table)
    console.print(f"\n  Total elapsed: {elapsed:.1f}s")


# ── Per-machine worker ───────────────────────────────────────────


def run_machine(
    spec: MachineSpec,
    data: np.ndarray,
    batch_size: int,
    interval: float,
    skip_to: int,
    max_ticks: int,
    stats: MachineStats,
    stop_event: threading.Event,
):
    """Send batches from one machine's data to its dedicated server."""
    session = requests.Session()
    tick = 0

    for offset in range(skip_to, len(data), batch_size):
        if stop_event.is_set():
            break
        if max_ticks > 0 and tick >= max_ticks:
            break

        batch = data[offset : offset + batch_size]
        if len(batch) < 10:
            break

        payload = {
            "store_id": spec.store_id,
            "device_id": spec.device_id,
            "data": batch.tolist(),
            "include_attribution": True,
            "include_per_timestep": False,
        }

        t0 = time.monotonic()
        try:
            resp = session.post(
                f"{spec.server_url}/score",
                json=payload,
                timeout=30,
            )
            if resp.status_code != 200:
                console.print(
                    f"  [{spec.color}]{spec.machine_id}[/{spec.color}]"
                    f" [red]HTTP {resp.status_code}: {resp.text[:200]}[/red]"
                )
                stats.errors += 1
                tick += 1
                continue

            result = resp.json()
        except requests.ConnectionError:
            console.print(
                f"  [{spec.color}]{spec.machine_id}[/{spec.color}]"
                f" [red]Connection lost to {spec.server_url}[/red]"
            )
            stats.errors += 1
            break
        except Exception as e:
            console.print(
                f"  [{spec.color}]{spec.machine_id}[/{spec.color}]"
                f" [red]Error: {e}[/red]"
            )
            stats.errors += 1
            tick += 1
            continue

        latency_ms = (time.monotonic() - t0) * 1000

        # Update stats
        stats.total_ticks += 1
        stats.total_timesteps += result.get("n_timesteps", len(batch))
        stats.total_anomalies += result.get("n_anomalies", 0)
        stats.total_segments += len(result.get("anomaly_segments", []))
        stats.latencies_ms.append(latency_ms)

        # Display
        row_end = offset + len(batch) - 1
        display_tick(spec, tick, offset, row_end, result, latency_ms, tick == 0)

        tick += 1

        # Sleep (interruptible via stop_event)
        if not stop_event.wait(timeout=interval):
            pass  # timeout expired = normal, keep going

    session.close()


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="TranAD Live Anomaly Detection Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  uv run python scripts/demo_live.py
  uv run python scripts/demo_live.py --interval 0.5 --batch-size 100
  uv run python scripts/demo_live.py --skip-to 6400 --max-ticks 20 --interval 0.3
  uv run python scripts/demo_live.py \\
      --machines machine-2-1@http://host-a:8000 machine-3-2@http://host-b:8000
""",
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        default=[
            "machine-2-1@http://localhost:8000",
            "machine-3-2@http://localhost:8001",
        ],
        help="Machine specs as machine-X-Y@http://host:port (default: two local clusters)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Timesteps per scoring request (default: 50, min: 10)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Seconds between requests per machine (default: 1.0)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/smd/raw/test",
        help="Path to raw SMD test data (default: data/smd/raw/test)",
    )
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=0,
        help="Stop after N ticks per machine (default: 0 = unlimited)",
    )
    parser.add_argument(
        "--skip-to",
        type=int,
        default=0,
        help="Skip to timestep N before starting (default: 0)",
    )

    args = parser.parse_args()

    if args.batch_size < 10:
        console.print("[red]Error: --batch-size must be >= 10 (model window_size)[/red]")
        sys.exit(1)

    # Parse machine specs
    specs = [
        parse_machine_spec(s, MACHINE_STYLES[i % len(MACHINE_STYLES)])
        for i, s in enumerate(args.machines)
    ]

    # Load data for all machines
    machine_data: list[tuple[MachineSpec, np.ndarray]] = []
    for spec in specs:
        data = load_raw_data(args.data_dir, spec.machine_id)
        machine_data.append((spec, data))

    # Check server health and show startup panel
    panel_lines = []
    all_healthy = True
    for i, (spec, data) in enumerate(machine_data):
        status = check_server(spec)
        if status is None:
            panel_lines.append(
                f"  Cluster {i + 1}:  [{spec.color}]{spec.machine_id}[/{spec.color}]"
                f" -> {spec.server_url}  [bold red](unreachable)[/bold red]"
            )
            all_healthy = False
        else:
            panel_lines.append(
                f"  Cluster {i + 1}:  [{spec.color}]{spec.machine_id}[/{spec.color}]"
                f" -> {spec.server_url}  [green]({status})[/green]"
            )
            panel_lines.append(
                f"              {len(data):,} timesteps x {data.shape[1]} features loaded"
            )

    panel_lines.append("")
    panel_lines.append(
        f"  Batch size: {args.batch_size} timesteps  |  Interval: {args.interval}s"
    )
    if args.skip_to > 0:
        panel_lines.append(f"  Skipping to timestep: {args.skip_to}")
    if args.max_ticks > 0:
        panel_lines.append(f"  Max ticks: {args.max_ticks}")

    console.print()
    console.print(
        Panel(
            "\n".join(panel_lines),
            title="TranAD Live Anomaly Detection Demo",
            border_style="blue",
        )
    )
    console.print()

    if not all_healthy:
        console.print("[red]One or more servers are unreachable.[/red]")
        console.print("Start the servers first:")
        console.print("  docker compose -f docker-compose.demo.yml up --build")
        console.print("  # or locally:")
        for i, (spec, _) in enumerate(machine_data):
            port = spec.server_url.rsplit(":", 1)[-1] if ":" in spec.server_url else "8000"
            console.print(
                f"  PYTHONPATH=app PORT={port} PRELOAD_MACHINES={spec.machine_id}"
                f" uv run python app/main.py"
            )
        sys.exit(1)

    # Create stats trackers
    all_stats = [
        MachineStats(machine_id=spec.machine_id, server_url=spec.server_url)
        for spec, _ in machine_data
    ]

    # Launch threads
    stop_event = threading.Event()
    threads: list[threading.Thread] = []
    for (spec, data), stats in zip(machine_data, all_stats):
        t = threading.Thread(
            target=run_machine,
            args=(
                spec,
                data,
                args.batch_size,
                args.interval,
                args.skip_to,
                args.max_ticks,
                stats,
                stop_event,
            ),
            daemon=True,
        )
        threads.append(t)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        console.print("\n[yellow]Interrupted — stopping all machines...[/yellow]")
        stop_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    t_start = time.monotonic()
    for t in threads:
        t.start()

    # Wait for all threads (check periodically so signals are handled)
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.5)

    elapsed = time.monotonic() - t_start
    print_summary(all_stats, elapsed)


if __name__ == "__main__":
    main()
