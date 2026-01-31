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

# Copy pyproject.toml and install dependencies
COPY pyproject.toml .
RUN uv pip install --system -r pyproject.toml

# Copy application code
COPY main.py .
COPY isolation_forest_detector.py .

# Expose Dash port
EXPOSE 8050

# Run the application
CMD ["python", "-u", "main.py"]
