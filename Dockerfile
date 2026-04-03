FROM python:3.11-slim

WORKDIR /app

# Minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install only production dependencies (no jupyter/matplotlib in container)
RUN uv pip install --system \
    "numpy>=1.24.0" \
    "torch>=2.0.0" \
    "fastapi>=0.104.0" \
    "uvicorn>=0.24.0" \
    "pydantic>=2.5.0" \
    "scikit-learn>=1.3.0" \
    "scipy>=1.10.0"

# Copy application code
COPY src/ src/
COPY code/5_streaming_app.py code/5_streaming_app.py

# Create directories for model/data volumes
RUN mkdir -p models data

# Expose FastAPI port
EXPOSE 8000

# Default environment variables
ENV MODEL_PATH=models/tranad
ENV DATA_DIR=data/smd/processed
ENV DEVICE=cpu

# Run the application
CMD ["python", "-u", "code/5_streaming_app.py"]
