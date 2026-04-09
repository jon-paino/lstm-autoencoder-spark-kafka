# Striim AI Prototype: LSTM Autoencoder Anomaly Detection

This repository contains a Striim AI Prototype for real-time time-series anomaly detection using an LSTM Encoder-Decoder model, Kafka streaming, Spark Structured Streaming, and a Dash visualization dashboard.

The prototype shows how a reconstruction-based anomaly detection workflow can move from offline model development into a streaming application that continuously scores incoming data and surfaces anomalous behavior in a live dashboard. It uses the NYC taxi demand dataset as a concrete example of recurring seasonal structure, localized disruptions, and thresholded anomaly detection.

The repository includes method-oriented notebooks for learning the approach, reusable source code for the LSTM encoder-decoder and anomaly scoring pipeline, pre-trained model artifacts for running the demo immediately, and Dockerized services for the streaming application and Kafka producer.

This project accompanies a forthcoming blog post about the prototype and its design decisions: **[Blog link coming soon]**

The modeling approach is based on [Malhotra et al. (2016)](https://arxiv.org/abs/1607.00148): "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

---

## Repository Guide

The repository is organized into a small number of components that map directly to the main ways you can use this prototype.

```
lstm-autoencoder-spark-kafka/
│
├── code/                                    # Numbered workflow (start here)
│   ├── 0_verify_setup.py                    # Optional environment / artifact check
│   ├── 1_data_exploration.ipynb             # EDA: patterns, seasonality, motivation
│   ├── 2_model_design.ipynb                 # Architecture walkthrough, train demo
│   ├── 3_train_model.py                     # Full training pipeline
│   ├── 4_evaluate_model.py                  # Evaluate model, generate plots
│   ├── 5_streaming_app.py                   # Real-time Dash + Spark + Kafka app
│   └── 6_optimize.py                        # Hyperparameter + split optimization
│
├── src/                                     # Reusable library code
│   ├── model.py                             # EncDecAD architecture (Malhotra et al.)
│   ├── scorer.py                            # Mahalanobis anomaly scoring
│   ├── preprocess.py                        # Data loading, segmentation, splits
│   └── synthetic.py                         # Synthetic anomaly generation
│
├── producer/                                # Kafka producer service
│   └── producer.py                          # Streams CSV data to Kafka
│
├── data/
│   └── nyc_taxi.csv                         # NYC taxi demand dataset (included)
│
├── models/                                  # Pre-trained model artifacts
│   ├── lstm_model.pt                        # LSTM Encoder-Decoder weights
│   ├── scaler.pkl                           # StandardScaler (train-fitted)
│   ├── scorer.pkl                           # Anomaly scorer with threshold
│   ├── training_history.pkl                 # Training loss curves
│   └── preprocessor_config.pkl              # Data split configuration
│
├── Dockerfile.app                           # App container definition
├── Dockerfile.producer                      # Producer container definition
├── docker-compose.yml                       # Full stack orchestration
├── pyproject.toml                           # Python dependencies
└── TECHNICAL.md                             # Detailed technical reference
```

The numbered files in `code/` form the main workflow for the project:

| Step | File | Purpose |
|------|------|---------|
| 0 | `0_verify_setup.py` | Optional troubleshooting: verify environment, data, and model artifacts |
| 1 | `1_data_exploration.ipynb` | Explore data patterns and motivate the approach |
| 2 | `2_model_design.ipynb` | Walk through the model architecture and anomaly scoring |
| 3 | `3_train_model.py` | Train the model from scratch |
| 4 | `4_evaluate_model.py` | Evaluate the trained model and generate plots |
| 5 | `5_streaming_app.py` | Run the real-time streaming application |
| 6 | `6_optimize.py` | Run hyperparameter and split optimization experiments |

If you are new to the project, start with steps 1 and 2.

## How To Use This Repo

You can approach this prototype in four different ways, depending on what you want to accomplish:

- **Learn the methods with notebooks**  
  Start with `code/1_data_exploration.ipynb` and `code/2_model_design.ipynb` to understand the dataset, the weekly-window framing, the LSTM encoder-decoder architecture, and the anomaly scoring methodology.

- **Run the pretrained streaming demo**  
  Use the model artifacts already included in `models/` and launch the streaming stack with Docker to see the dashboard and anomaly detection workflow in action without retraining the model first.

- **Train from scratch**  
  Run the training and evaluation scripts to regenerate the model artifacts yourself and reproduce the main modeling workflow end to end.

- **Optimize experimentally**  
  Use the optimization script to explore hyperparameter and split-search experiments beyond the default pretrained setup.

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Docker** and **Docker Compose** (for the streaming demo)

## Learn the Methods

To work through the notebooks, first install the Python dependencies:

```bash
git clone <repo-url>
cd lstm-autoencoder-spark-kafka
uv sync
```

Then open the notebooks:

```bash
uv run jupyter notebook code/
```

Work through the notebooks in order:

| Notebook | What you'll learn |
|----------|-------------------|
| **`1_data_exploration.ipynb`** | Dataset overview, periodicity analysis, why simple thresholds fail, motivation for reconstruction-based detection |
| **`2_model_design.ipynb`** | LSTM Encoder-Decoder architecture, training demo, anomaly scoring methodology, localization |

Both notebooks run cell-by-cell with no external setup beyond `uv sync`. All cells are self-contained — just press Shift+Enter through them.

> **Note:** A pre-trained model is included in `models/`. The model design notebook trains a small demo model (10 epochs) for illustration, but the pre-trained model was trained with full hyperparameter optimization.
>
> `0_verify_setup.py` is optional troubleshooting. You do not need to run it before the notebooks.

## Run the Pretrained Demo

The streaming demo uses the pre-trained artifacts already included in `models/`. You do not need to run training before starting the demo.

```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up --build -d
```

Open `http://localhost:8050` to view the live dashboard.

On subsequent runs (after images are built):
```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up -d
```

View logs:

```bash
docker compose logs -f app
docker compose logs -f producer
```

## Train From Scratch

If you want to reproduce the model artifacts yourself instead of using the included ones:

Run the scripts in order:

```bash
uv run python code/3_train_model.py
uv run python code/4_evaluate_model.py
```

This regenerates the model and evaluation outputs used by the workflow and updates the artifacts in `models/`.

## Optimize

If you want to explore alternative hyperparameters or split configurations:

```bash
uv run python code/6_optimize.py
```

This is optional and intended for experimentation beyond the default pretrained setup, including alternative hyperparameter and data split experiments.

## Architecture

```
┌─────────────┐     ┌─────────┐     ┌───────────────┐     ┌──────────────┐
│  Producer   │────▶│  Kafka  │────▶│ Spark Stream  │────▶│     Dash     │
│ (NYC Taxi)  │     │         │     │  + Detector   │     │  Dashboard   │
└─────────────┘     └─────────┘     └───────────────┘     └──────────────┘
```

- **Producer**: Streams NYC taxi data CSV to Kafka topic
- **Kafka**: Message broker for real-time data streaming
- **Spark**: Structured Streaming consumes micro-batches from Kafka
- **LSTM Detector**: Pre-trained Encoder-Decoder flags anomalous weekly windows
- **Dash**: Real-time visualization with anomaly markers and 6-hour localization
