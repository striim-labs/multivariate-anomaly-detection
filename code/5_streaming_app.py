"""
TranAD Anomaly Detection Server

Multivariate time series anomaly detection using TranAD.
Exposes FastAPI REST endpoints for on-demand scoring with
per-feature root cause attribution.

Usage:
    uv run uvicorn code.5_streaming_app:app --host 0.0.0.0 --port 8000
    # or
    uv run python code/5_streaming_app.py
"""

import logging
import os
import re
import sys
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import (
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
from src.registry import TranADRegistry
from src.scorer import build_segment_summaries, find_anomaly_segments, score_batch
from src.utils import auto_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# -- Global state --

MODEL_DIR = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "tranad"))
DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data" / "smd" / "processed"))
DEVICE = os.getenv("DEVICE", "cpu")
PRELOAD_MACHINES = os.getenv("PRELOAD_MACHINES", "")

registry = TranADRegistry(base_dir=MODEL_DIR)
device = auto_device(DEVICE)

_norm_params_cache: dict[str, np.ndarray] = {}
_scorer_state_cache: dict[str, dict] = {}

ROLLING_BUFFER_BATCHES = 20
_raw_data_buffer: dict[str, deque] = {}

_MACHINE_KEY_RE = re.compile(r"^machine-(\d+)-(\d+)$")


def _machine_key(store_id: int, device_id: int) -> str:
    return f"machine-{store_id}-{device_id}"


def _parse_machine_key(key: str) -> DeviceIdentifier | None:
    m = _MACHINE_KEY_RE.match(key)
    if m:
        return DeviceIdentifier(store_id=int(m.group(1)), device_id=int(m.group(2)))
    return None


def _list_devices() -> list[DeviceIdentifier]:
    devices = []
    for key in registry.list_machines():
        ident = _parse_machine_key(key)
        if ident:
            devices.append(ident)
    return devices


def _loaded_devices() -> list[DeviceIdentifier]:
    devices = []
    for key in _scorer_state_cache:
        ident = _parse_machine_key(key)
        if ident:
            devices.append(ident)
    return devices


def _load_machine_resources(machine_key: str) -> None:
    registry.get_model(machine_key, device)

    if machine_key not in _norm_params_cache:
        norm_params = registry.get_norm_params(machine_key, data_dir=DATA_DIR)
        if norm_params is None:
            raise FileNotFoundError(
                f"No norm_params found for '{machine_key}' in {DATA_DIR}"
            )
        _norm_params_cache[machine_key] = norm_params

    if machine_key not in _scorer_state_cache:
        scorer_state = registry.get_scorer_state(machine_key)
        if scorer_state is None:
            raise FileNotFoundError(
                f"No scorer_state.json found for '{machine_key}'"
            )
        _scorer_state_cache[machine_key] = scorer_state


# -- Lifespan --


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


# -- Endpoints --


@app.get("/health", response_model=HealthResponse)
async def health():
    loaded = _loaded_devices()
    return HealthResponse(
        status="ready" if loaded else "starting",
        detector="tranad",
        available_devices=_list_devices(),
        loaded_devices=loaded,
    )


@app.get("/models", response_model=ModelsResponse)
async def list_models():
    return ModelsResponse(devices=_list_devices())


@app.get("/config", response_model=ConfigResponse)
async def config():
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
    store_id = request.store_id
    device_id = request.device_id
    machine_key_str = _machine_key(store_id, device_id)
    t_start = time.monotonic()

    try:
        _load_machine_resources(machine_key_str)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    model, cfg = registry.get_model(machine_key_str, device)
    norm_params = _norm_params_cache[machine_key_str]
    scorer_state = _scorer_state_cache[machine_key_str]
    threshold = scorer_state["threshold"]
    baselines = np.array(scorer_state.get("feature_baselines", []))

    data = np.array(request.data, dtype=np.float64)
    if data.shape[1] != cfg.n_features:
        raise HTTPException(
            status_code=422,
            detail=f"Feature count mismatch: model expects {cfg.n_features}, got {data.shape[1]}",
        )

    min_vals = norm_params[0]
    max_vals = norm_params[1]
    normalized = (data - min_vals) / (max_vals - min_vals + 1e-4)

    scores = score_batch(
        model, normalized,
        window_size=cfg.window_size,
        device=str(device),
        scoring_mode=request.scoring_mode,
    )

    scores_1d = np.mean(scores, axis=1)
    predictions = (scores_1d > threshold).astype(int)
    n_anomalies = int(predictions.sum())

    buf = _raw_data_buffer.setdefault(
        machine_key_str, deque(maxlen=ROLLING_BUFFER_BATCHES * cfg.window_size)
    )
    history = np.array(buf) if len(buf) > 0 else None
    buf.extend(data)

    segments: list[AnomalySegment] = []
    if request.include_attribution and baselines.size > 0 and n_anomalies > 0:
        raw_summaries = build_segment_summaries(
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
        seg_boundaries = find_anomaly_segments(predictions)
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

    mean_src = history if history is not None else data
    dimension_means = np.mean(mean_src, axis=0).round(6).tolist()

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
        store_id, device_id, len(data), n_anomalies, len(segments), elapsed_ms,
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
