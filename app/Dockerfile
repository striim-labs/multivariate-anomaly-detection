FROM python:3.11-slim

WORKDIR /app

# Minimal system dependencies (no Java, no librdkafka needed for REST-only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy all application code
COPY *.py .

# Create directories for model/data volumes
RUN mkdir -p models data

# Expose FastAPI port
EXPOSE 8000

# Default environment variables
ENV MODEL_PATH=models/tranad
ENV DATA_DIR=data/smd/processed
ENV DEVICE=cpu

# Run the application
CMD ["python", "-u", "main.py"]
