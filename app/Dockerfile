FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Java for Spark, librdkafka for confluent-kafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    curl \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (detect architecture dynamically)
RUN JAVA_PATH=$(find /usr/lib/jvm -name "java-21-openjdk-*" -type d | head -1) && \
    echo "JAVA_HOME=$JAVA_PATH" >> /etc/environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy TranAD application code
COPY main.py .
COPY base_detector.py .
COPY schemas.py .
COPY tranad_model.py .
COPY tranad_scorer.py .
COPY tranad_registry.py .
COPY tranad_streaming_detector.py .

# Create models directory (will be mounted as volume)
RUN mkdir -p models

# Expose FastAPI port
EXPOSE 8000

# Default environment variables
ENV DETECTOR_TYPE=tranad
ENV WINDOW_SIZE=10
ENV MIN_SAMPLES=10
ENV MODEL_PATH=models/tranad
ENV DEVICE=cpu

# Run the application
CMD ["python", "-u", "main.py"]
