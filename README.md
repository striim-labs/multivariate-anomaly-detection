# Striim AI Prototype: TranAD Multivariate Anomaly Detection

This repository contains a Striim AI Prototype for real-time multivariate time-series anomaly detection using [TranAD](https://arxiv.org/abs/2201.07284) (Tuli et al., VLDB 2022) with per-feature root cause attribution.

The prototype shows how a reconstruction-based anomaly detection workflow can move from offline model development into a streaming application that continuously scores incoming server telemetry and surfaces anomalous behavior with per-dimension attribution. It uses the [Server Machine Dataset (SMD)](https://github.com/NetManAIOps/OmniAnomaly) -- 38-feature server telemetry from 28 machines -- as a concrete example of complex multivariate structure, coordinated anomaly patterns, and threshold calibration via extreme value theory.

The repository includes method-oriented notebooks for learning the approach, reusable source code for the TranAD transformer and scoring pipeline, pre-trained model artifacts for running the demo immediately, and Dockerized services for the REST scoring API.

This project accompanies a forthcoming blog post about the prototype and its design decisions: **[Blog link coming soon]**

The modeling approach is based on: Tuli, S., Casale, G., & Jennings, N. R. (2022). "TranAD: Deep Transformer Networks for Anomaly Detection in Multivariate Time Series Data." *PVLDB*, 15(6), 1201-1214.

---

## Project Structure

```
multivariate-anomaly-detection/
│
├── code/                                  # Numbered scripts -- the canonical workflow
│   ├── 0_verify_setup.py                  # Download data, verify environment
│   ├── 1_train_model.py                   # Train baseline, save to models/tranad/initial/
│   ├── 2_evaluate_model.py                # Evaluate baseline or best, print metrics
│   ├── 3_streaming_app.py                 # FastAPI real-time scoring API (Docker)
│   └── 4_grid_sweep.py                    # Sweep hyperparams, retrain best to models/tranad/best/
│
├── notebooks/                             # Interactive walkthroughs (motivation + reasoning)
│   ├── data_exploration.ipynb             # EDA: 38-feature telemetry, anomaly patterns
│   └── model_design.ipynb                 # Architecture walkthrough, scoring, attribution
│
├── src/                                   # Reusable library code
│   ├── model.py                           # TranADConfig, TranADNet, transformer layers
│   ├── train.py                           # Shared training loop (used by 1_ and 4_)
│   ├── scorer.py                          # Scoring, POT thresholding, attribution
│   ├── preprocess.py                      # Data download, normalization, loading
│   ├── registry.py                        # Per-device model loading + caching
│   ├── spot.py                            # SPOT algorithm (extreme value theory)
│   ├── schemas.py                         # Pydantic v2 request/response models
│   └── utils.py                           # Sliding window, device selection
│
├── data/smd/                              # Server Machine Dataset
│   ├── raw/                               # Original text files (28 machines, gitignored)
│   └── processed/                         # Normalized .npy arrays (gitignored)
│
├── models/tranad/                         # Prebuilt reference (never overwritten)
│   ├── machine-1-1/                       # model.ckpt + scorer_state.json
│   ├── machine-2-1/
│   ├── machine-3-2/
│   ├── machine-3-7/
│   ├── initial/                           # User baseline from 1_train_model.py (gitignored)
│   └── best/                              # User retrained best from 4_grid_sweep.py (gitignored)
│
├── striim/                                # Striim integration: TQL, Open Processor, build scripts
├── docker-compose.rest.yml                # Single-node REST deployment
├── docker-compose.demo.yml                # Two-cluster demo deployment
├── Dockerfile                             # Production container
├── pyproject.toml                         # Python dependencies
├── STRIIM.md                              # Striim pipeline setup guide
└── TECHNICAL.md                           # Detailed technical reference
```

The scripts under `code/` are the **first-class** path: they reproduce the model end-to-end and are what you should run if you are trying to learn how training and evaluation work, or to adapt this to your own data. The notebooks under `notebooks/` are interactive **supporting material** -- they explain *why* the architecture is shaped the way it is, what the data looks like, and how the scoring methodology was chosen.

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager) -- install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Docker** (optional, for containerized deployment)

## Going through the code

### 1. Install dependencies

```bash
git clone <repo-url>
cd multivariate-anomaly-detection
uv sync
uv run python code/0_verify_setup.py
```

`0_verify_setup.py` downloads the SMD dataset, preprocesses all 28 machines, and verifies that the pre-trained checkpoints are present.

### 2. Train a baseline, then improve it via grid sweep

This is a four-command journey that tells the full reproduction story. None of these commands ever overwrite the prebuilt artifacts at `models/tranad/machine-*/` -- the user-trained models go to `models/tranad/initial/` and `models/tranad/best/`, both of which are gitignored.

> **Note:** `code/3_streaming_app.py` is intentionally skipped here. It is a FastAPI server for the Docker demo and is **not** meant to be run directly with `python`. See the [Docker demo](#docker-demo) section below.

#### 2a. Train the baseline

```bash
uv run python code/1_train_model.py --machine machine-1-1
```

This trains a baseline model (`d_feedforward=8`, `lr=0.001`, `epoch_inverse` loss weighting, 5 epochs) and writes the checkpoint to `models/tranad/initial/machine-1-1/`. The smaller feedforward network and moderate learning rate are a reasonable first attempt -- the grid sweep will discover that a larger architecture with a lower learning rate and `exponential_decay` weighting does better.

#### 2b. Evaluate the baseline

```bash
uv run python code/2_evaluate_model.py --machine machine-1-1
```

By default this reads `models/tranad/initial/` and prints F1, precision, recall, ROC AUC, and root cause diagnosis metrics.

#### 2c. Run the grid sweep to find a better configuration

```bash
uv run python code/4_grid_sweep.py --machine machine-1-1 --quick
```

The quick sweep runs 5 targeted configurations that each test a specific improvement over the baseline: better loss weighting, lower learning rate, larger feedforward network, and averaged scoring mode. Each trial trains with early stopping and evaluates with POT. After the sweep, the winning configuration is **retrained end-to-end** and saved to `models/tranad/best/`. Terminal output shows each trial's metrics, the winning config, and a clear banner when retraining begins.

#### 2d. Evaluate the best-config model

```bash
uv run python code/2_evaluate_model.py --machine machine-1-1 --model-dir models/tranad/best
```

Same evaluation script, pointed at the retrained best artifacts. You should see improved metrics compared to the baseline.

---

### 3. Read through the notebooks (optional, for context)

The notebooks are interactive walkthroughs of the methodology and motivation. They are **supporting material** -- read them when you want the *why* behind the architecture and the scoring methodology, not when you want to run things. They load the reference artifacts in `models/tranad/` so you can see everything end-to-end without waiting on training.

| Notebook | What you'll learn |
|----------|-------------------|
| **`data_exploration.ipynb`** | 38-feature server telemetry, anomaly patterns, feature correlations, why multivariate detection matters |
| **`model_design.ipynb`** | TranAD two-phase architecture, forward pass demo, scoring pipeline, POT thresholds, per-feature attribution |

```bash
uv run jupyter notebook notebooks/
```

## Docker demo

The repo includes a containerized FastAPI server (`code/3_streaming_app.py`) that serves the pre-trained models as a REST API. This demonstrates how TranAD would be deployed for real-time scoring -- you POST raw sensor data and get back anomaly detections with per-feature root cause attribution.

```bash
docker compose -f docker-compose.rest.yml up --build
```

Wait for the log line `Starting TranAD Anomaly Detection Server`, then open a **new terminal window** to test:

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

The response includes `n_anomalies`, contiguous `anomaly_segments` with start/end indices, and `attributed_dimensions` ranked by mean elevation above baseline. Interactive API docs are at `http://localhost:8000/docs`.

Stop the container when done:

```bash
docker compose -f docker-compose.rest.yml down
```

### Two-cluster demo

Simulates a multi-store deployment with two isolated API instances. Stop any running containers first, then launch:

```bash
docker compose -f docker-compose.rest.yml down
docker compose -f docker-compose.demo.yml up --build
```

- **Port 8000** -- machine-2-1 (store 2, device 1)
- **Port 8001** -- machine-3-2 (store 3, device 2)

Score against each cluster:

```bash
curl -s -X POST http://localhost:8000/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request_machine-2-1.json | python -m json.tool

curl -s -X POST http://localhost:8001/score \
    -H 'Content-Type: application/json' \
    -d @samples/score_request_machine-3-2.json | python -m json.tool
```

Stop:

```bash
docker compose -f docker-compose.demo.yml down
```

---

## Workflow

The numbered files in `code/` tell the full reproduction story:

| Step | File | Purpose |
|------|------|---------|
| 0 | `0_verify_setup.py` | Download SMD dataset, preprocess all 28 machines, verify model checkpoints |
| 1 | `1_train_model.py` | Train a baseline, save to `models/tranad/initial/` |
| 2 | `2_evaluate_model.py` | Evaluate any saved artifacts (default: `models/tranad/initial/`), print metrics |
| 3 | `3_streaming_app.py` | FastAPI REST server for real-time anomaly detection (Docker only) |
| 4 | `4_grid_sweep.py` | Sweep hyperparameters, retrain the winner, save to `models/tranad/best/` |

The reference artifacts at `models/tranad/machine-*/` are the prebuilt models -- the notebooks load them so you can read through the methodology without waiting on training, and **none of the scripts ever overwrite them**. Your trained models go to `models/tranad/initial/` and `models/tranad/best/`.

## Detection methodology

TranAD uses a transformer encoder-decoder with two-phase self-conditioning. Phase 1 produces a standard reconstruction of the input window. The squared error from Phase 1 becomes a "focus score" that is concatenated with the input for Phase 2, forcing the second decoder to attend more strongly to the regions Phase 1 struggled with. This two-phase design creates sharper anomaly separation: normal points are reconstructed well in both phases, while anomalous points accumulate compounding error.

The anomaly threshold is calibrated using POT (Peaks Over Threshold), an extreme value theory method that fits a Generalized Pareto Distribution to the tail of the training score distribution. This is more principled than a fixed percentile because it models the actual tail shape rather than assuming a fixed fraction of anomalies. Each machine gets its own POT parameters calibrated independently.

For root cause attribution, the per-dimension reconstruction errors are compared against training baselines using an elevation ratio (score / baseline). Features with high elevation ratios are the likely root causes -- they are being reconstructed much worse than normal, even after accounting for features that are inherently harder to reconstruct.

---

## License

BSD-3-Clause. See [LICENSE](LICENSE).
