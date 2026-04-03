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
├── models/tranad/                         # Trained checkpoints (per-device)
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

---

## Workflow

| File | What You Learn |
|------|---------------|
| `0_verify_setup.py` | Downloads SMD dataset, preprocesses all 28 machines, verifies model checkpoints |
| `1_data_exploration.ipynb` | Visualize 38-feature server telemetry, anomaly patterns, feature correlations, why multivariate detection matters |
| `2_model_design.ipynb` | TranAD two-phase architecture, forward pass walkthrough, training demo, scoring pipeline, POT thresholds, attribution mechanism |
| `3_train_model.py` | Full training pipeline with early stopping, loss weighting, adversarial mode. `--all` trains all 4 reference machines |
| `4_evaluate_model.py` | Score data, calibrate POT thresholds, compute F1/precision/recall, root cause diagnosis metrics |
| `5_streaming_app.py` | FastAPI REST server for real-time anomaly detection with per-feature attribution |
| `6_optimize.py` | Hyperparameter grid search over learning rate, loss weighting, scoring mode, etc. |

---

## Docker

```bash
# REST API (single container)
docker compose -f docker-compose.rest.yml up --build

# Two-cluster demo
docker compose -f docker-compose.demo.yml up --build
```

The API serves at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Training

```bash
# Train a single machine
uv run python code/3_train_model.py --machine machine-1-1 --epochs 20

# Train all 4 reference machines
uv run python code/3_train_model.py --all

# Evaluate
uv run python code/4_evaluate_model.py --machine machine-1-1
```

---

## License

BSD-3-Clause. See [LICENSE](LICENSE).
