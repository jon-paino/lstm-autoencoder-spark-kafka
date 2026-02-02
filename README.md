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

### 1. Clone the Repository and Install dependencies locally

```bash
git clone https://github.com/your-username/lstm-autoencoder-spark-kafka.git
cd lstm-autoencoder-spark-kafka
uv sync
```

### 2. Download the Dataset

The NYC taxi dataset should be placed in `data/nyc_taxi.csv`. If not present, download it:

```bash
# The dataset is from the Numenta Anomaly Benchmark (NAB)
curl -o data/nyc_taxi.csv https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/nyc_taxi.csv
```

### 3. Build and Run with Docker Compose

**For LSTM detection mode:**
```bash
MESSAGE_DELAY_SECONDS=0.01 DETECTOR_TYPE=lstm START_OFFSET=4992 LOOP_DATA=false docker compose up --build
```

**For Isolation Forest detection mode:**
```bash
DETECTOR_TYPE=isolation_forest docker compose up --build
```

### 4. View the Dashboard

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

#### Model Training Workflow

The pre-trained model in `models/` was created through the following optimization process:

**1. Data Split Optimization**

The NYC taxi dataset (29 complete weeks) was split into train/validation/threshold/test sets. We optimized the split configuration to maximize F1 score:

```bash
python app/optimize_split.py --output optimization_results/split_optimization.json
```

Best configuration: **8 train / 2 val / 4 threshold weeks** (15 weeks for testing)

**2. Hyperparameter Optimization**

Grid search over LSTM architecture parameters (hidden_dim, num_layers, dropout, learning_rate, threshold_percentile):

```bash
python app/optimize_hyperparams.py --mode grid --output optimization_results/hyperparam_grid.json
```

Best configuration: **hidden_dim=64, num_layers=1, dropout=0.2, lr=0.0005, threshold=99.99%**

**3. Model Training**

Train the final model with optimized parameters:

```bash
python app/train.py
```

**4. Evaluation**

Generate performance metrics and visualizations:

```bash
python app/evaluate.py 
```

#### Model Artifacts

The `models/` directory contains:
| File | Description |
|------|-------------|
| `lstm_model.pt` | Trained LSTM Encoder-Decoder weights |
| `scaler.pkl` | StandardScaler fitted on training data |
| `scorer.pkl` | Anomaly scorer with calibrated threshold |
| `training_history.pkl` | Training/validation loss curves |
| `preprocessor_config.pkl` | Data split configuration |

#### Evaluation Results

The `evaluation/` directory contains performance visualizations:

- `training_history.png` - Training and validation loss curves
- `score_distribution.png` - Anomaly score distributions (train vs test)
- `weekly_comparison.png` - Normal vs anomaly week reconstruction comparison
- `reconstruction_*.png` - Detailed reconstruction plots for each anomaly week


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

