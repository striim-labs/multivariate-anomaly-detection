# TranAD Multivariate Anomaly Detection

Real-time multivariate time series anomaly detection using [TranAD](https://arxiv.org/abs/2201.07284) (Tuli et al., VLDB 2022) with per-feature root cause attribution. Trained on the [Server Machine Dataset (SMD)](https://github.com/NetManAIOps/OmniAnomaly), achieving an average F1 of 0.925 across 4 reference machines.

---

## Project Structure

```
multivariate-anomaly-detection/
│
├── code/                                  # Numbered workflow (start here)
│   ├── 0_verify_setup.py                  # Download data, verify environment
│   ├── 1_data_exploration.ipynb           # EDA: 38-feature telemetry, anomaly patterns
│   ├── 2_model_design.ipynb               # TranAD architecture walkthrough, training demo
│   ├── 3_train_model.py                   # Train TranAD on SMD machines
│   ├── 4_evaluate_model.py                # Score, threshold calibration, evaluation metrics
│   ├── 5_streaming_app.py                 # FastAPI real-time anomaly detection API
│   └── 6_optimize.py                      # Hyperparameter sweep
│
├── src/                                   # Reusable library code
│   ├── model.py                           # TranADConfig, TranADNet, transformer layers
│   ├── scorer.py                          # Scoring, POT thresholding, attribution
│   ├── preprocess.py                      # Data download, normalization, loading
│   ├── train.py                           # Training epoch runners, early stopping
│   ├── registry.py                        # Per-device model loading + caching
│   ├── spot.py                            # SPOT algorithm (extreme value theory)
│   ├── schemas.py                         # Pydantic v2 request/response models
│   └── utils.py                           # Sliding window, device selection
│
├── data/smd/                              # Server Machine Dataset
│   ├── raw/                               # Original text files (28 machines)
│   └── processed/                         # Normalized .npy arrays
│
├── models/tranad/                         # Pre-trained checkpoints (per-device)
│   ├── machine-1-1/                       # model.ckpt + scorer_state.json
│   ├── machine-2-1/
│   ├── machine-3-2/
│   └── machine-3-7/
│
├── configs/default.yaml                   # Model/training/scoring defaults
├── samples/                               # Sample API requests
├── docker-compose.rest.yml                # REST-only deployment
├── docker-compose.demo.yml                # Two-cluster demo deployment
├── Dockerfile                             # Production container
├── pyproject.toml                         # All dependencies
└── LICENSE                                # BSD-3-Clause
```

---

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** package manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- **Docker** (optional, for containerized deployment)

---

## Quick Start

```bash
# Install dependencies
uv sync

# Download data, preprocess, verify setup
uv run python code/0_verify_setup.py

# Open the notebooks (main learning path)
uv run jupyter notebook code/
```

Step through the two notebooks in order, they are the primary way to understand this project:

- **`1_data_exploration.ipynb`** — Visualize the 38-feature server telemetry, see where anomalies occur and which features cause them, understand why a multivariate approach is needed.
- **`2_model_design.ipynb`** — Walk through the TranAD architecture, see a forward pass, watch a short training demo, then use the pre-trained model to score data, calibrate thresholds, and attribute root causes.

Both notebooks use the pre-trained models in `models/` read-only. No scripts need to be run beyond `0_verify_setup.py`.

---

## Docker Demo

The repo includes a containerized FastAPI server (`code/5_streaming_app.py`) that serves the pre-trained models as a REST API. This demonstrates how TranAD would be deployed for real-time scoring — you POST raw sensor data and get back anomaly detections with per-feature root cause attribution.

### Start the API

```bash
docker compose -f docker-compose.rest.yml up --build
```

Wait for the log line `Starting TranAD Anomaly Detection Server`, then open a **new terminal window** to test:

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/models

# Score telemetry data (sample contains a real anomaly from store 1, device 1)
curl -s -X POST http://localhost:8000/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request.json | python -m json.tool
```

The response includes:
- `n_anomalies` — how many timesteps were flagged
- `anomaly_segments` — contiguous anomaly regions with start/end indices
- `attributed_dimensions` — which features drove each anomaly, ranked by `mean_elevation` (how many times above baseline)

Interactive API docs are available at `http://localhost:8000/docs`.

### Two-Cluster Demo

Stop the REST container first (`docker compose -f docker-compose.rest.yml down`), then launch two isolated API instances simulating a multi-store deployment:

```bash
docker compose -f docker-compose.demo.yml up --build
```

- **Port 8000** — machine-2-1 (store 2, device 1)
- **Port 8001** — machine-3-2 (store 3, device 2)

```bash
# Score against cluster 1 (port 8000 = machine-2-1)
curl -s -X POST http://localhost:8000/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request_machine-2-1.json | python -m json.tool

# Score against cluster 2 (port 8001 = machine-3-2)
curl -s -X POST http://localhost:8001/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request_machine-3-2.json | python -m json.tool
```

### Stop

```bash
docker compose -f docker-compose.rest.yml down
# or
docker compose -f docker-compose.demo.yml down
```

---

## Workflow

| # | File | What It Does |
|---|------|--------------|
| 0 | `0_verify_setup.py` | Downloads SMD dataset, preprocesses all 28 machines, verifies model checkpoints exist |
| 1 | `1_data_exploration.ipynb` | Visualize 38-feature server telemetry, anomaly patterns, feature correlations, why multivariate detection matters |
| 2 | `2_model_design.ipynb` | TranAD two-phase architecture, forward pass demo, scoring pipeline, POT thresholds, attribution mechanism |
| 3 | `3_train_model.py` | Full training pipeline with early stopping, loss weighting, adversarial mode |
| 4 | `4_evaluate_model.py` | Score data, calibrate POT thresholds, compute F1/precision/recall, root cause diagnosis metrics |
| 5 | `5_streaming_app.py` | FastAPI REST server for real-time anomaly detection with per-feature attribution |
| 6 | `6_optimize.py` | Hyperparameter grid search over learning rate, loss weighting, scoring mode, etc. |

> **Note:** The repo ships with pre-trained model checkpoints in `models/tranad/`. Scripts 3 and 6 will **overwrite** them if run. To view saved evaluation results without re-scoring: `uv run python code/4_evaluate_model.py --from-saved`

---

## License

BSD-3-Clause. See [LICENSE](LICENSE).
