"""
Plot feature attribution heatmaps for anomaly segments.

Loads saved test scores and baselines, computes elevation ratios,
and generates per-segment heatmaps showing which features drove
each anomaly.

Usage:
    uv run python scripts/plot_attribution.py --machine machine-1-1
    uv run python scripts/plot_attribution.py --machine machine-1-1 --segment 0
    uv run python scripts/plot_attribution.py --machine machine-1-1 --context 50
    uv run python scripts/plot_attribution.py --machine machine-1-1 --output-dir plots/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add app/ to import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from tranad_registry import TranADRegistry
from tranad_scorer import TranADScorer
from tranad_utils import auto_device


def plot_segment_heatmap(
    test_scores: np.ndarray,
    baselines: np.ndarray,
    threshold: float,
    segment: dict,
    context: int = 20,
    title: str = "",
) -> go.Figure:
    """Create a plotly heatmap of elevation ratios for one anomaly segment.

    Top subplot: 1D aggregated score with threshold line.
    Bottom subplot: elevation ratio heatmap (features x time).

    Args:
        test_scores: Full test scores, shape (N_test, n_features).
        baselines: Per-feature baselines, shape (n_features,).
        threshold: Detection threshold for the 1D score.
        segment: Segment summary dict from build_segment_summaries.
        context: Timestamps to show before/after segment.
        title: Plot title.

    Returns:
        plotly Figure.
    """
    seg_start = segment["segment_start"]
    seg_end = segment["segment_end"]
    n_test = test_scores.shape[0]
    n_features = test_scores.shape[1]

    # View window with context padding
    view_start = max(0, seg_start - context)
    view_end = min(n_test, seg_end + context + 1)

    view_scores = test_scores[view_start:view_end]  # (T_view, F)
    elevation = view_scores / baselines  # (T_view, F)
    score_1d = np.mean(view_scores, axis=1)  # (T_view,)
    timestamps = np.arange(view_start, view_end)

    # Attributed dimension indices for highlighting
    attr_dims = [d["dim"] for d in segment.get("attributed_dimensions", [])]
    feature_labels = [
        f"{'> ' if i in attr_dims else ''}dim_{i}"
        for i in range(n_features)
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.25, 0.75],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Aggregated Score (1D)", "Elevation Ratio by Feature"],
    )

    # Top: 1D score line
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=score_1d,
            mode="lines",
            name="Score (1D)",
            line=dict(color="steelblue", width=1.5),
        ),
        row=1,
        col=1,
    )
    # Threshold line
    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="threshold",
        row=1,
        col=1,
    )

    # Segment boundary shading (both subplots)
    for row in [1, 2]:
        fig.add_vrect(
            x0=seg_start - 0.5,
            x1=seg_end + 0.5,
            fillcolor="rgba(255, 0, 0, 0.08)",
            line=dict(color="red", width=1, dash="dot"),
            row=row,
            col=1,
        )

    # Bottom: elevation ratio heatmap
    # Cap elevation at a reasonable max for color scale readability
    elev_capped = np.clip(elevation.T, 0, 20)  # (F, T_view)

    fig.add_trace(
        go.Heatmap(
            z=elev_capped,
            x=timestamps,
            y=feature_labels,
            colorscale=[
                [0.0, "rgb(49, 54, 149)"],     # deep blue (well below baseline)
                [0.05, "rgb(165, 191, 221)"],   # light blue
                [0.1, "rgb(245, 245, 245)"],    # white (at baseline, ratio ~1-2)
                [0.25, "rgb(253, 174, 97)"],    # orange (ratio ~5)
                [0.5, "rgb(215, 48, 39)"],      # red (ratio ~10)
                [1.0, "rgb(103, 0, 31)"],       # dark red (ratio >= 20)
            ],
            zmin=0,
            zmax=20,
            colorbar=dict(title="Elevation<br>Ratio", len=0.7, y=0.3),
            hovertemplate=(
                "t=%{x}<br>%{y}<br>elevation=%{z:.2f}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # Layout
    if not title:
        n_attr = len(segment.get("attributed_dimensions", []))
        title = (
            f"Segment [{seg_start}-{seg_end}] "
            f"(length={segment['segment_length']}, "
            f"peak={segment['peak_score']:.4f}, "
            f"{n_attr} attributed dims)"
        )

    fig.update_layout(
        title=title,
        height=max(500, 200 + n_features * 12),
        width=max(800, (view_end - view_start) * 4 + 200),
        showlegend=False,
    )
    fig.update_xaxes(title_text="Timestamp", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=1, col=1)

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot feature attribution heatmaps for anomaly segments"
    )
    parser.add_argument("--machine", type=str, default="machine-1-1")
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/tranad")
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/smd/processed")
    )
    parser.add_argument(
        "--segment",
        type=int,
        default=None,
        help="Specific segment index to plot (default: all)",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=20,
        help="Timestamps of context before/after segment",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Save HTML files here (default: show interactively)",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--scoring-mode",
        type=str,
        default="phase2_only",
        choices=["phase2_only", "averaged"],
    )
    parser.add_argument(
        "--segment-type",
        type=str,
        default="raw",
        choices=["raw", "adjusted"],
        help="Which segment type to plot (default: raw)",
    )
    parser.add_argument(
        "--tp-only",
        action="store_true",
        default=False,
        help="Only plot segments that overlap with ground truth anomalies",
    )
    args = parser.parse_args()

    # Load scorer state (baselines + threshold)
    registry = TranADRegistry(base_dir=args.model_dir)
    scorer_state = registry.get_scorer_state(args.machine)
    if scorer_state is None:
        print(f"Error: No scorer state for {args.machine}. Run evaluate_smd.py first.")
        sys.exit(1)

    if "feature_baselines" not in scorer_state:
        print(f"Error: scorer_state.json missing feature_baselines. "
              f"Re-run evaluate_smd.py to generate them.")
        sys.exit(1)

    baselines = np.array(scorer_state["feature_baselines"])
    threshold = scorer_state["threshold"]

    # Load attribution results
    attr_path = args.model_dir / args.machine / "attribution_results.json"
    if not attr_path.exists():
        print(f"Error: {attr_path} not found. Run evaluate_smd.py first.")
        sys.exit(1)

    with open(attr_path) as f:
        attr_results = json.load(f)

    segment_key = f"{args.segment_type}_segments"
    segments = attr_results.get(segment_key, [])
    if not segments:
        print(f"No {args.segment_type} segments found for {args.machine}.")
        sys.exit(0)

    print(f"Found {len(segments)} {args.segment_type} segments for {args.machine}")

    # Filter to TP segments only (those overlapping ground truth anomalies)
    if args.tp_only:
        labels_path = args.data_dir / f"{args.machine}_test_labels.npy"
        if not labels_path.exists():
            print(f"Error: {labels_path} not found (needed for --tp-only).")
            sys.exit(1)
        test_labels = np.load(labels_path)  # (N_test,)
        tp_segments = []
        for seg in segments:
            seg_labels = test_labels[seg["segment_start"] : seg["segment_end"] + 1]
            if np.any(seg_labels > 0):
                tp_segments.append(seg)
        print(f"  Filtered to {len(tp_segments)} TP segments "
              f"(removed {len(segments) - len(tp_segments)} FP segments)")
        segments = tp_segments
        if not segments:
            print("No TP segments to plot.")
            sys.exit(0)

    # Score test data (needed for heatmap)
    device = auto_device(args.device)
    model, config = registry.get_model(args.machine, device=device)

    test_path = args.data_dir / f"{args.machine}_test.npy"
    if not test_path.exists():
        print(f"Error: {test_path} not found.")
        sys.exit(1)

    test_data = np.load(test_path)
    print(f"Scoring test data ({test_data.shape})...")
    test_scores = TranADScorer.score_batch(
        model, test_data, config.window_size, device, args.scoring_mode
    )

    # Determine which segments to plot
    if args.segment is not None:
        if args.segment >= len(segments):
            print(f"Error: segment {args.segment} out of range (0-{len(segments)-1})")
            sys.exit(1)
        segments_to_plot = [(args.segment, segments[args.segment])]
    else:
        segments_to_plot = list(enumerate(segments))

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, seg in segments_to_plot:
        title = (
            f"{args.machine} — {args.segment_type} segment {idx} "
            f"[{seg['segment_start']}-{seg['segment_end']}]"
        )
        fig = plot_segment_heatmap(
            test_scores,
            baselines,
            threshold,
            seg,
            context=args.context,
            title=title,
        )

        if args.output_dir:
            html_path = args.output_dir / f"{args.machine}_seg{idx}_{args.segment_type}.html"
            fig.write_html(str(html_path))
            print(f"  Saved: {html_path}")
        else:
            fig.show()
            if len(segments_to_plot) > 1:
                input(f"  Showing segment {idx}. Press Enter for next...")


if __name__ == "__main__":
    main()
