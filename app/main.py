"""
TranAD Anomaly Detection Server

Real-time multivariate anomaly detection on server telemetry using
Kafka streaming, Spark Structured Streaming, and TranAD detection.
Exposes a FastAPI REST endpoint for on-demand scoring.
"""

import logging
import os

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TranAD Anomaly Detection",
    description="Multivariate time series anomaly detection using TranAD",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "starting",
        "detector": "tranad",
        "spark_ready": False,
    }


@app.post("/score")
async def score():
    """Score a telemetry window for anomalies. TODO: implement in Phase 4."""
    raise NotImplementedError("Scoring endpoint not yet implemented")


@app.get("/config")
async def config():
    """Return model configuration. TODO: implement in Phase 4."""
    raise NotImplementedError("Config endpoint not yet implemented")


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting TranAD Anomaly Detection Server")
    uvicorn.run(app, host="0.0.0.0", port=8000)
