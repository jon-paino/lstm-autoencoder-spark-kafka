FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (Java for Spark, librdkafka for confluent-kafka)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    librdkafka-dev \
    curl \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME (detect architecture)
RUN JAVA_PATH=$(find /usr/lib/jvm -name "java-21-openjdk-*" -type d | head -1) && \
    echo "JAVA_HOME=$JAVA_PATH" >> /etc/environment
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Install only the dependencies the app needs (no jupyter/ipykernel)
RUN uv pip install --system \
    "pandas>=2.0.0" \
    "numpy>=1.24.0" \
    "torch>=2.0.0" \
    "confluent-kafka>=2.3.0" \
    "dash>=2.14.0" \
    "dash-bootstrap-components>=1.5.0" \
    "plotly>=5.18.0" \
    "scikit-learn>=1.3.0" \
    "pyspark==3.5.3" \
    "matplotlib>=3.10.8"

# Copy application code
COPY code/5_streaming_app.py ./main.py
COPY src/ ./src/

# Create models directory (will be mounted as volume)
RUN mkdir -p models

# Expose Dash port
EXPOSE 8050

# Default environment variables
ENV MODEL_PATH=models/lstm_model.pt
ENV SCALER_PATH=models/scaler.pkl
ENV SCORER_PATH=models/scorer.pkl

# Run the application
CMD ["python", "-u", "main.py"]
