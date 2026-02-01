# LSTM Autoencoder Anomaly Detection with Spark Streaming

Real-time anomaly detection on NYC taxi demand data using Kafka streaming, Spark Structured Streaming, and a Dash visualization dashboard. Supports two detection modes: **LSTM Encoder-Decoder** and **Isolation Forest**.

## Architecture Overview

```
┌─────────────┐     ┌─────────┐     ┌───────────────┐     ┌──────────────┐
│  Producer   │────▶│  Kafka  │────▶│ Spark Stream  │────▶│     Dash     │
│ (NYC Taxi)  │     │         │     │  + Detector   │     │  Dashboard   │
└─────────────┘     └─────────┘     └───────────────┘     └──────────────┘
```

- **Producer**: Streams NYC taxi data to Kafka topic
- **Kafka**: Message broker for real-time data streaming
- **Spark**: Structured Streaming consumes from Kafka
- **Detector**: LSTM Encoder-Decoder or Isolation Forest anomaly detection
- **Dash**: Real-time visualization dashboard

## Prerequisites

- **Docker** and **Docker Compose** (for containerized deployment)
- **Python 3.11+** (for local development/training)
- **uv** (Python package manager) - [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

## Quick Start (Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/lstm-autoencoder-spark-kafka.git
cd lstm-autoencoder-spark-kafka
```

### 2. Download the Dataset

The NYC taxi dataset should be placed in `data/nyc_taxi.csv`. If not present, download it:

```bash
# The dataset is from the Numenta Anomaly Benchmark (NAB)
curl -o data/nyc_taxi.csv https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
```

### 3. Train the LSTM Model (Required for LSTM mode)

```bash
# Install dependencies locally
uv sync

# Run training script
python app/train.py
```

This creates the following files in `models/`:
- `lstm_model.pt` - Trained LSTM Encoder-Decoder model
- `scaler.pkl` - StandardScaler for data normalization
- `scorer.pkl` - Anomaly scorer with calibrated threshold
- `training_history.pkl` - Training loss history

### 4. Build and Run with Docker Compose

**For LSTM detection mode:**
```bash
MESSAGE_DELAY_SECONDS=0.01 DETECTOR_TYPE=lstm START_OFFSET=4992 LOOP_DATA=false docker compose up --build
```

**For Isolation Forest detection mode:**
```bash
DETECTOR_TYPE=isolation_forest docker compose up --build
```

### 5. View the Dashboard

Open your browser to: **http://localhost:8050**


### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f app
docker compose logs -f producer
docker compose logs -f kafka
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DETECTOR_TYPE` | `isolation_forest` | Detection mode: `lstm` or `isolation_forest` |
| `START_OFFSET` | `0` | Record index to start streaming from |
| `LOOP_DATA` | `true` | Whether to loop through data continuously |
| `MESSAGE_DELAY_SECONDS` | `0.1` | Delay between messages (simulates real-time) |
| `WAIT_FOR_APP` | `true` | Producer waits for Spark to be ready |
| `WINDOW_SIZE` | `200` | Sliding window size (Isolation Forest) |
| `CONTAMINATION` | `0.05` | Expected anomaly ratio (Isolation Forest) |

## Project Structure

```
lstm-autoencoder-spark-kafka/
├── app/                          # Main application
│   ├── main.py                   # Dash app + Spark streaming
│   ├── lstm_autoencoder.py       # LSTM Encoder-Decoder architecture
│   ├── streaming_detector.py     # LSTM streaming detector
│   ├── anomaly_scorer.py         # Mahalanobis distance scorer
│   ├── base_detector.py          # Abstract detector interface
│   ├── isolation_forest_detector.py  # Isolation Forest detector
│   ├── data_preprocessor.py      # Data preprocessing utilities
│   ├── train.py                  # Model training script
│   ├── evaluate.py               # Model evaluation script
│   ├── Dockerfile
│   └── pyproject.toml
├── producer/                     # Kafka producer
│   ├── producer.py               # Streams CSV data to Kafka
│   ├── Dockerfile
│   └── pyproject.toml
├── data/
│   └── nyc_taxi.csv              # NYC taxi demand dataset
├── models/                       # Trained model artifacts
│   ├── lstm_model.pt             # LSTM model weights
│   ├── scaler.pkl                # Data normalizer
│   ├── scorer.pkl                # Anomaly scorer
│   └── training_history.pkl      # Training metrics
├── docker-compose.yml            # Container orchestration
├── pyproject.toml                # Root Python dependencies
└── README.md
```

## Detection Modes

### LSTM Encoder-Decoder (EncDec-AD)

Based on [Malhotra et al. (2016)](https://arxiv.org/abs/1607.00148):

- **Architecture**: LSTM encoder compresses sequence, decoder reconstructs it
- **Scoring**: Mahalanobis distance with full covariance matrix
- **Windows**: Non-overlapping weekly windows (336 samples = 48/day × 7 days)
- **Threshold**: 95th percentile of validation reconstruction errors

### Isolation Forest

- **Architecture**: Ensemble of isolation trees
- **Scoring**: Path length-based anomaly score
- **Windows**: Sliding window analysis
- **Threshold**: Configurable contamination parameter

## Services & Ports

| Service | Port | Description |
|---------|------|-------------|
| Dash Dashboard | 8050 | http://localhost:8050 |
| Spark Master UI | 8080 | http://localhost:8080 |
| Kafka | 9092 | External access |
| Kafka (internal) | 29092 | Inter-container |
| Zookeeper | 2181 | Kafka coordination |

