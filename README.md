# TranAD Multivariate Anomaly Detection

Real-time multivariate time series anomaly detection using [TranAD](https://arxiv.org/abs/2201.07284) (Tuli et al., VLDB 2022). The system ingests server telemetry (38 sensor features per device), scores incoming data with a trained transformer-based model, and returns per-feature root cause attribution identifying which dimensions drove each detected anomaly.

Trained and evaluated on the [Server Machine Dataset (SMD)](https://github.com/NetManAIOps/OmniAnomaly), achieving an average F1 of 0.925 across 4 reference machines. Each SMD machine maps to a store/device deployment.

For a deep dive into the model architecture, training pipeline, scoring system, and design decisions, see [TECHNICAL.md](TECHNICAL.md).

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (package manager)
- **Docker** and **Docker Compose** (for containerized deployment)

## Quick Start (Local)

```bash
# Clone and install
git clone <repo-url>
cd multivariate-anomaly-detection
uv sync

# Download the SMD dataset (~100 MB, 28 machines x 38 features)
uv run python scripts/download_smd.py

# Preprocess: normalize and generate per-machine artifacts
uv run python scripts/preprocess_smd.py
```

The server starts at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

## Quick Start (Docker)

```bash
# Build and run (single command)
docker compose -f docker-compose.rest.yml up --build
```

The Docker image is ~500 MB (Python 3.11-slim + PyTorch + FastAPI). It mounts `./models` and `./data` as read-only volumes — the container never writes to them.

## API Reference

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Readiness probe. Returns `"starting"` or `"ready"` and lists available/loaded devices. |
| `/models` | GET | Lists all devices that have a trained model checkpoint. |
| `/config` | GET | Returns loaded model details: thresholds, feature count, window size, device. |
| `/score` | POST | Score a batch of raw telemetry. Returns anomaly detection with root cause attribution. |

### POST /score

**Request:**

```json
{
    "store_id": 1,
    "device_id": 1,
    "data": [
        [0.075, 0.066, 0.070, ...],
        [0.073, 0.064, 0.068, ...],
        ...
    ],
    "include_per_timestep": false,
    "include_attribution": true,
    "scoring_mode": "phase2_only"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | int | yes | Store identifier (e.g., `3` for store 3) |
| `device_id` | int | yes | Device identifier within the store (e.g., `7` for device 7) |
| `data` | float[][] | yes | N timesteps x 38 features, raw/unnormalized sensor values. Minimum 10 timesteps. |
| `include_per_timestep` | bool | no | Include per-row scores in response (default: `false`) |
| `include_attribution` | bool | no | Include root cause feature attribution (default: `true`) |
| `scoring_mode` | string | no | `"phase2_only"` (default) or `"averaged"` |

**Response:**

```json
{
    "store_id": 1,
    "device_id": 1,
    "n_timesteps": 10,
    "n_features": 38,
    "threshold": 0.061975,
    "n_anomalies": 1,
    "anomaly_ratio": 0.1,
    "anomaly_segments": [
        {
            "segment_start": 6,
            "segment_end": 6,
            "segment_length": 1,
            "peak_score": 0.676304,
            "peak_timestamp": 6,
            "mean_score": 0.676304,
            "attributed_dimensions": [
                {
                    "dim": 9,
                    "label": "dim_9",
                    "mean_elevation": 77129.5,
                    "contribution": 0.687
                },
                {
                    "dim": 12,
                    "label": "dim_12",
                    "mean_elevation": 944.0,
                    "contribution": 0.199
                }
            ]
        }
    ],
    "per_timestep": null,
    "scoring_mode": "phase2_only",
    "threshold_method": "pot"
}
```

The `attributed_dimensions` array identifies which features caused the anomaly:
- **mean_elevation**: How many times above its training baseline this feature's error is (e.g., 77,129x normal)
- **contribution**: Fraction of total excess score this feature accounts for (0.0-1.0)

### Scoring Pipeline

```
POST /score
  1. Load model, norm_params, threshold (lazy, cached after first call)
  2. Validate input (min 10 timesteps, correct feature count)
  3. Normalize: (data - min) / (max - min + 1e-4) using training statistics
  4. TranAD two-phase inference → per-dimension MSE scores (N x 38)
  5. Aggregate to 1D (mean across features) and apply POT threshold
  6. Find contiguous anomaly segments, attribute root cause features
  7. Return structured JSON response
```

## Sample Requests

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Score telemetry data (sample contains a real anomaly from store 1, device 1)
curl -X POST http://localhost:8000/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request.json

```

## Project Structure

```
multivariate-anomaly-detection/
├── app/                                # FastAPI server + TranAD model
│   ├── main.py                         # FastAPI application, 4 endpoints
│   ├── schemas.py                      # Pydantic v2 request/response models
│   ├── tranad_model.py                 # TranAD transformer architecture (127K params)
│   ├── tranad_scorer.py                # Scoring, POT thresholding, attribution
│   ├── tranad_registry.py              # Per-device model loading + caching
│   ├── tranad_streaming_detector.py    # Stateful streaming wrapper (BaseDetector)
│   ├── tranad_utils.py                 # Sliding window, device selection
│   ├── spot.py                         # SPOT (Streaming Peaks-Over-Threshold)
│   ├── base_detector.py                # Abstract detector interface
│   ├── Dockerfile                      # Python 3.11-slim (~500 MB)
│   └── pyproject.toml                  # App dependencies (torch, fastapi, etc.)
├── scripts/                            # Data prep, training, evaluation
│   ├── download_smd.py                 # Download SMD dataset from GitHub
│   ├── preprocess_smd.py               # Normalize data, parse labels
│   ├── train_smd.py                    # Train single machine
│   ├── train_all_machines.py           # Train all 4 reference machines
│   ├── evaluate_smd.py                 # Score + threshold calibration + metrics
│   ├── sweep_smd.py                    # Hyperparameter grid search
│   └── plot_attribution.py             # Heatmap visualization (plotly)
├── configs/
│   └── default.yaml                    # Model/training/scoring defaults
├── models/tranad/                      # Trained checkpoints (per-device)
│   ├── machine-1-1/                    # store=1, device=1
│   │   ├── model.ckpt                  # PyTorch checkpoint
│   │   └── scorer_state.json           # Calibrated threshold + baselines
│   └── ...
├── data/smd/                           # Server Machine Dataset
│   ├── raw/                            # Original text files (28 machines)
│   └── processed/                      # Normalized .npy arrays
├── samples/
│   └── score_request.json              # Sample POST /score request
├── docker-compose.rest.yml             # REST-only deployment (1 service)
├── docker-compose.yml                  # Full streaming stack (Kafka, Spark)
├── TECHNICAL.md                        # Architecture deep-dive
├── pyproject.toml                      # Root workspace config (uv monorepo)
└── LICENSE                             # BSD-3-Clause
```

## Docker Architecture

**`docker-compose.rest.yml`** — REST-only deployment (recommended for integration):

```yaml
services:
  app:
    build: ./app
    ports: ["8000:8000"]
    volumes:
      - ./models:/app/models:ro    # Trained models (read-only)
      - ./data:/app/data:ro        # Normalization params (read-only)
```

The Dockerfile uses `uv` for fast dependency installation and copies only the Python modules needed for inference. 

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/tranad` | Path to model checkpoint directories |
| `DATA_DIR` | `data/smd/processed` | Path to preprocessed data (norm_params) |
| `DEVICE` | `cpu` | PyTorch device: `cpu`, `cuda`, `mps`, `auto` |
| `PRELOAD_MACHINES` | *(empty)* | Comma-separated machine keys to load at startup (e.g., `machine-1-1,machine-2-1`) |
| `PORT` | `8000` | Server port |

### Model Configuration (`configs/default.yaml`)

Key training parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 10 | Sliding window length (timesteps) |
| `n_features` | 38 | Input dimensions per timestep |
| `learning_rate` | 0.0001 | AdamW learning rate |
| `loss_weighting` | `epoch_inverse` | Loss schedule: `epoch_inverse` or `exponential_decay` |
| `epochs` | 5 | Training epochs (use 20 for production quality) |
| `scoring_mode` | `phase2_only` | Scoring: `phase2_only` or `averaged` |

## Training Pipeline

### Train a Single Device

```bash
uv run python scripts/train_smd.py \
    --machine machine-1-1 \
    --epochs 20 \
    --lr 0.0001 \
    --loss-weighting exponential_decay
```

### Train All Reference Devices

```bash
uv run python scripts/train_all_machines.py 
```

### Evaluate and Calibrate Threshold

```bash
uv run python scripts/evaluate_smd.py 
```

This generates `scorer_state.json` (the POT-calibrated threshold and per-feature baselines) which the API needs for inference.

### Device ID Mapping

The API uses `store_id` and `device_id` as separate integer fields. Internally, these map to filesystem directories using the pattern `machine-{store_id}-{device_id}`:

| API Fields | Internal Key | Model Directory |
|-----------|-------------|-----------------|
| `store_id=1, device_id=1` | `machine-1-1` | `models/tranad/machine-1-1/` |
| `store_id=2, device_id=1` | `machine-2-1` | `models/tranad/machine-2-1/` |
| `store_id=3, device_id=2` | `machine-3-2` | `models/tranad/machine-3-2/` |
| `store_id=3, device_id=7` | `machine-3-7` | `models/tranad/machine-3-7/` |

Training scripts use the `--machine machine-X-Y` format. The REST API accepts the split integer form.

## License

BSD-3-Clause. See [LICENSE](LICENSE).
