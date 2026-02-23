"""
TranAD Anomaly Detection Server

Multivariate time series anomaly detection using TranAD.
Exposes FastAPI REST endpoints for on-demand scoring with
per-feature root cause attribution.
"""

import logging
import os
import re
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException

from schemas import (
    AnomalySegment,
    AttributedDimension,
    ConfigResponse,
    DeviceIdentifier,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    ScoringRequest,
    ScoringResponse,
    TimestepResult,
)
from tranad_registry import TranADRegistry
from tranad_scorer import TranADScorer
from tranad_utils import auto_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global state ───────────────────────────────────────────────────

MODEL_DIR = os.getenv("MODEL_PATH", "models/tranad")
DATA_DIR = os.getenv("DATA_DIR", "data/smd/processed")
DEVICE = os.getenv("DEVICE", "cpu")
PRELOAD_MACHINES = os.getenv("PRELOAD_MACHINES", "")

registry = TranADRegistry(base_dir=MODEL_DIR)
scorer = TranADScorer()
device = auto_device(DEVICE)

# Per-machine caches for norm_params and scorer_state
_norm_params_cache: dict[str, np.ndarray] = {}
_scorer_state_cache: dict[str, dict] = {}

# Per-machine rolling buffer of raw input data for baseline computation.
# Each deque stores individual timestep rows (np arrays of shape (n_features,)).
ROLLING_BUFFER_BATCHES = 20  # number of previous batches (of 10 rows each)
_raw_data_buffer: dict[str, deque] = {}

# Pattern for parsing machine keys like "machine-3-7" → (3, 7)
_MACHINE_KEY_RE = re.compile(r"^machine-(\d+)-(\d+)$")


def _machine_key(store_id: int, device_id: int) -> str:
    """Construct internal filesystem key from store/device IDs."""
    return f"machine-{store_id}-{device_id}"


def _parse_machine_key(key: str) -> DeviceIdentifier | None:
    """Parse 'machine-X-Y' into a DeviceIdentifier, or None if invalid."""
    m = _MACHINE_KEY_RE.match(key)
    if m:
        return DeviceIdentifier(store_id=int(m.group(1)), device_id=int(m.group(2)))
    return None


def _list_devices() -> list[DeviceIdentifier]:
    """List all available devices from the registry."""
    devices = []
    for key in registry.list_machines():
        ident = _parse_machine_key(key)
        if ident:
            devices.append(ident)
    return devices


def _loaded_devices() -> list[DeviceIdentifier]:
    """List all currently loaded devices."""
    devices = []
    for key in _scorer_state_cache:
        ident = _parse_machine_key(key)
        if ident:
            devices.append(ident)
    return devices


def _load_machine_resources(machine_key: str) -> None:
    """Load and cache model, norm_params, and scorer_state for a machine.

    Idempotent — safe to call multiple times.
    """
    # Model (cached internally by registry)
    registry.get_model(machine_key, device)

    # Norm params
    if machine_key not in _norm_params_cache:
        norm_params = registry.get_norm_params(machine_key, data_dir=DATA_DIR)
        if norm_params is None:
            raise FileNotFoundError(
                f"No norm_params found for '{machine_key}' in {DATA_DIR}"
            )
        _norm_params_cache[machine_key] = norm_params

    # Scorer state (threshold + baselines)
    if machine_key not in _scorer_state_cache:
        scorer_state = registry.get_scorer_state(machine_key)
        if scorer_state is None:
            raise FileNotFoundError(
                f"No scorer_state.json found for '{machine_key}'"
            )
        _scorer_state_cache[machine_key] = scorer_state


# ── Lifespan ───────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting TranAD Anomaly Detection Server")
    logger.info("Device: %s | Model dir: %s | Data dir: %s", device, MODEL_DIR, DATA_DIR)
    logger.info("Available devices: %s", registry.list_machines())

    if PRELOAD_MACHINES:
        for mid in PRELOAD_MACHINES.split(","):
            mid = mid.strip()
            if mid:
                try:
                    _load_machine_resources(mid)
                    logger.info("Preloaded model for %s", mid)
                except Exception as e:
                    logger.error("Failed to preload %s: %s", mid, e)

    yield

    logger.info("Shutting down")
    registry.clear_cache()


app = FastAPI(
    title="TranAD Anomaly Detection",
    description="Multivariate time series anomaly detection using TranAD",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ──────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    loaded = _loaded_devices()
    return HealthResponse(
        status="ready" if loaded else "starting",
        detector="tranad",
        available_devices=_list_devices(),
        loaded_devices=loaded,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available device models."""
    return ModelsResponse(devices=_list_devices())


@app.get("/config", response_model=ConfigResponse)
async def config():
    """Return model configuration and status."""
    models_info = []
    for key in _scorer_state_cache:
        ident = _parse_machine_key(key)
        if not ident:
            continue
        state = _scorer_state_cache[key]
        _, cfg = registry.get_model(key, device)
        models_info.append(
            ModelInfo(
                store_id=ident.store_id,
                device_id=ident.device_id,
                threshold=state["threshold"],
                threshold_method=state.get("method", "unknown"),
                n_features=cfg.n_features,
                window_size=cfg.window_size,
            )
        )

    return ConfigResponse(
        available_devices=_list_devices(),
        loaded_devices=_loaded_devices(),
        device=str(device),
        default_scoring_mode="phase2_only",
        models=models_info,
    )


@app.post("/score", response_model=ScoringResponse)
async def score(request: ScoringRequest):
    """Score a batch of telemetry data for anomalies.

    Accepts raw (unnormalized) time series data. The API:
    1. Loads the device-specific model, norm_params, and threshold
    2. Normalizes data using training-time min/max
    3. Runs TranAD inference to get per-dimension scores
    4. Aggregates scores, applies threshold
    5. Finds anomaly segments and attributes root cause features
    """
    store_id = request.store_id
    device_id = request.device_id
    machine_key = _machine_key(store_id, device_id)
    t_start = time.monotonic()

    # 1. Load resources (lazy, cached after first call)
    try:
        _load_machine_resources(machine_key)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model, cfg = registry.get_model(machine_key, device)
    norm_params = _norm_params_cache[machine_key]
    scorer_state = _scorer_state_cache[machine_key]
    threshold = scorer_state["threshold"]
    baselines = np.array(scorer_state.get("feature_baselines", []))

    # 2. Convert and validate input
    data = np.array(request.data, dtype=np.float64)
    if data.shape[1] != cfg.n_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Feature count mismatch: model expects {cfg.n_features} features, "
                f"got {data.shape[1]}"
            ),
        )

    # 3. Normalize: (data - min) / (max - min + eps)
    min_vals = norm_params[0]
    max_vals = norm_params[1]
    normalized = (data - min_vals) / (max_vals - min_vals + 1e-4)

    # 4. Score
    scores = TranADScorer.score_batch(
        model,
        normalized,
        window_size=cfg.window_size,
        device=str(device),
        scoring_mode=request.scoring_mode,
    )

    # 5. Aggregate and threshold
    scores_1d = np.mean(scores, axis=1)
    predictions = (scores_1d > threshold).astype(int)
    n_anomalies = int(predictions.sum())

    # 6. Rolling buffer: snapshot history *before* appending current batch
    buf = _raw_data_buffer.setdefault(
        machine_key, deque(maxlen=ROLLING_BUFFER_BATCHES * cfg.window_size)
    )
    history = np.array(buf) if len(buf) > 0 else None
    buf.extend(data)  # append current batch rows for future requests

    # 7. Build segments with attribution
    segments: list[AnomalySegment] = []
    if request.include_attribution and baselines.size > 0 and n_anomalies > 0:
        raw_summaries = TranADScorer.build_segment_summaries(
            scores, predictions, baselines,
            normalized_data=data, history_data=history,
        )
        for s in raw_summaries:
            segments.append(
                AnomalySegment(
                    segment_start=s["segment_start"],
                    segment_end=s["segment_end"],
                    segment_length=s["segment_length"],
                    peak_score=s["peak_score"],
                    peak_timestamp=s["peak_timestamp"],
                    mean_score=s["mean_score"],
                    attributed_dimensions=[
                        AttributedDimension(**d)
                        for d in s["attributed_dimensions"]
                    ],
                )
            )
    elif n_anomalies > 0:
        seg_boundaries = TranADScorer.find_anomaly_segments(predictions)
        for start, end in seg_boundaries:
            seg_1d = scores_1d[start : end + 1]
            peak_offset = int(np.argmax(seg_1d))
            segments.append(
                AnomalySegment(
                    segment_start=start,
                    segment_end=end,
                    segment_length=end - start + 1,
                    peak_score=round(float(seg_1d[peak_offset]), 6),
                    peak_timestamp=start + peak_offset,
                    mean_score=round(float(np.mean(seg_1d)), 6),
                    attributed_dimensions=[],
                )
            )

    # 8. Per-dimension means (from history if available, else current batch)
    mean_src = history if history is not None else data
    dimension_means = np.mean(mean_src, axis=0).round(6).tolist()

    # 9. Per-timestep results (optional)
    per_timestep = None
    if request.include_per_timestep:
        per_timestep = [
            TimestepResult(
                index=i,
                score=round(float(scores_1d[i]), 6),
                is_anomaly=bool(predictions[i]),
            )
            for i in range(len(scores_1d))
        ]

    elapsed_ms = (time.monotonic() - t_start) * 1000
    logger.info(
        "Scored store=%d device=%d: %d timesteps, %d anomalies, %d segments, %.1fms",
        store_id,
        device_id,
        len(data),
        n_anomalies,
        len(segments),
        elapsed_ms,
    )

    return ScoringResponse(
        store_id=store_id,
        device_id=device_id,
        timestamp=request.timestamp,
        filename=request.filename,
        n_timesteps=len(data),
        n_features=cfg.n_features,
        threshold=threshold,
        n_anomalies=n_anomalies,
        anomaly_ratio=round(n_anomalies / len(data), 6),
        anomaly_segments=segments,
        dimension_means=dimension_means,
        per_timestep=per_timestep,
        scoring_mode=request.scoring_mode,
        threshold_method=scorer_state.get("method", "unknown"),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    logger.info("Starting TranAD server on port %d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
