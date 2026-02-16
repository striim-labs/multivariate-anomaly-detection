# TranAD Multivariate Anomaly Detection

Real-time multivariate time series anomaly detection using TranAD (Transformer-based Anomaly Detection) with Kafka + Spark Structured Streaming. The system ingests server telemetry (CPU, RAM, I/O metrics), scores incoming data with a trained TranAD model, provides root cause attribution per feature dimension, and exposes a FastAPI REST endpoint for on-demand scoring. Each SMD machine maps to a physical store deployment.

## Quick Start

TODO: Add docker compose up instructions

## Architecture

TODO: Add architecture overview and data flow diagram

## API

TODO: Document FastAPI /score endpoint

## Dataset

The SMD (Server Machine Dataset) consists of 28 server machines, each with 38 features.

Download with:
```bash
uv run python scripts/download_smd.py
```

## License

BSD-3-Clause. See [LICENSE](LICENSE).
