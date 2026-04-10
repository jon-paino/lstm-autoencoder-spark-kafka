# Striim AI Prototype: LSTM Autoencoder Anomaly Detection

This repository contains a Striim AI Prototype for real-time time-series anomaly detection using an LSTM Encoder-Decoder model, Kafka streaming, Spark Structured Streaming, and a Dash visualization dashboard.

The prototype shows how a reconstruction-based anomaly detection workflow can move from offline model development into a streaming application that continuously scores incoming data and surfaces anomalous behavior in a live dashboard. It uses the NYC taxi demand dataset as a concrete example of recurring seasonal structure, localized disruptions, and thresholded anomaly detection.

The repository includes method-oriented notebooks for learning the approach, reusable source code for the LSTM encoder-decoder and anomaly scoring pipeline, pre-trained model artifacts for running the demo immediately, and Dockerized services for the streaming application and Kafka producer.

This project accompanies a forthcoming blog post about the prototype and its design decisions: **[Blog link coming soon]**

The modeling approach is based on [Malhotra et al. (2016)](https://arxiv.org/abs/1607.00148): "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection"

<img width="1768" height="1185" alt="image" src="https://github.com/user-attachments/assets/846ff700-0e3b-4b09-83b7-66631e9799f2" />

---

## Project Structure

```
lstm-autoencoder-spark-kafka/
│
├── code/                                    # Numbered scripts -- the canonical workflow
│   ├── 0_verify_setup.py                    # Optional environment / artifact check
│   ├── 1_train_model.py                     # Train baseline, save to models/initial/
│   ├── 2_evaluate_model.py                  # Evaluate baseline or best, generate plots
│   ├── 3_streaming_app.py                   # Real-time Dash + Spark + Kafka app (Docker)
│   └── 4_grid_sweep.py                      # Sweep hyperparams, retrain best to models/best/
│
├── notebooks/                               # Interactive walkthroughs (motivation + reasoning)
│   ├── data_exploration.ipynb               # EDA: patterns, seasonality, motivation
│   └── model_design.ipynb                   # Architecture walkthrough, scoring methodology
│
├── src/                                     # Reusable library code
│   ├── model.py                             # EncDecAD architecture (Malhotra et al.)
│   ├── training.py                          # Shared training loop (used by 1_ and 4_)
│   ├── scorer.py                            # Mahalanobis anomaly scoring
│   ├── preprocess.py                        # Data loading, segmentation, splits
│   └── synthetic.py                         # Synthetic anomaly generation
│
├── producer/                                # Kafka producer service
│   └── producer.py                          # Streams CSV data to Kafka
│
├── data/
│   ├── nyc_taxi.csv                         # NYC taxi demand dataset (included)
│   └── nyc_taxi_sunday_aligned.csv          # Pre-trimmed to Sunday start (for Striim)
│
├── models/                                  # Prebuilt reference (never overwritten)
│   ├── lstm_model.pt                        # LSTM Encoder-Decoder weights
│   ├── scaler.pkl                           # StandardScaler (train-fitted)
│   ├── scorer.pkl                           # Window-Mahalanobis scorer + threshold
│   ├── training_history.pkl                 # Training loss curves
│   ├── preprocessor_config.pkl              # Data split configuration
│   ├── initial/                             # User baseline output of 1_train_model.py (gitignored)
│   └── best/                                # User retrained best from 4_grid_sweep.py (gitignored)
│
├── striim/                                  # Striim Platform OP integration (see STRIIM.md)
│
├── Dockerfile.app                           # App container definition
├── Dockerfile.producer                      # Producer container definition
├── docker-compose.yml                       # Full stack orchestration
├── pyproject.toml                           # Python dependencies
├── STRIIM.md                                # Striim pipeline setup guide
└── TECHNICAL.md                             # Detailed technical reference
```

The scripts under `code/` are the **first-class** path: they reproduce the model end-to-end and are what you should run if you're trying to learn how training and evaluation work, or to adapt this to your own data. The notebooks under `notebooks/` are interactice **supporting material** -- they explain *why* the architecture is shaped the way it is, what the data looks like, and how the scoring methodology was chosen. 

## Prerequisites

- **Python 3.11+**
- **uv** (Python package manager) — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Docker** and **Docker Compose** (for the streaming demo)

## Going through the code

### 1. Install dependencies

```bash
git clone <repo-url>
cd lstm-autoencoder-spark-kafka
uv sync
```

> `0_verify_setup.py` is optional troubleshooting. You do not need to run it before the scripts.

### 2. Train a baseline, then improve it via grid sweep

This is a four-command journey that tells the full reproduction story. None of these commands ever overwrite the prebuilt artifacts at `models/lstm_model.pt`, `models/scaler.pkl`, or `models/scorer.pkl` -- the user-trained models go to `models/initial/` and `models/best/`, both of which are gitignored.

> **Note:** `code/3_streaming_app.py` is intentionally skipped here. It is the Docker entrypoint for the visual streaming demo and is **not** meant to be run directly with `python`. See the [Docker demo with visual application](#docker-demo-with-visual-application) section below.

#### 2a. Train the baseline

```bash
uv run python code/1_train_model.py
```

This trains a fast initial model (`hidden_dim=24`, `lr=2e-3`, `20` epochs, `6` training weeks) and writes the artifacts to `models/initial/`. The baseline catches all 5 labeled anomalies but trips on a couple of normal weeks:

> **Baseline metrics:** Precision = 71.43%, Recall = 100%, F1 = 83.33% over 14 scored test weeks. The model has enough capacity to learn the weekly shape, but with limited training data and an aggressive learning rate it ends up flagging two normal weeks (`2014-10-26` and `2014-12-07`) as false positives alongside the five real events.

#### 2b. Evaluate the baseline

```bash
uv run python code/2_evaluate_model.py
```

By default this reads `models/initial/` and reprints the baseline metrics, generates diagnostic plots in `evaluation/`, and shows the per-week scores. You can see exactly which weeks the baseline got wrong.

#### 2c. Run the grid sweep to find a better configuration

```bash
uv run python code/4_grid_sweep.py
```

The sweep explores ~14 hyperparameter combinations on a larger data split (8 train weeks, 2 val, 4 threshold) and ranks them by F1. It then **retrains the winning configuration end-to-end** and saves a fresh set of artifacts to `models/best/`. Terminal output makes it explicit what's happening: where the baseline lives, which configs were tested, what the winner is, and where the retrained model is saved.

The winning configuration on this dataset is `hidden_dim=64, num_layers=1, dropout=0.2, lr=5e-4, threshold_percentile=99.99` -- the same architecture as the prebuilt model. With those settings the model achieves:

> **Best-config metrics:** Precision = 100%, Recall = 100%, F1 = 100% over 14 scored test weeks (5 / 5 known anomalies detected, zero false positives).

#### 2d. Evaluate the best-config model

```bash
uv run python code/2_evaluate_model.py --model-dir models/best
```

Same evaluation script, pointed at the retrained best artifacts. You should see the per-week table line up with the prebuilt reference and the metrics jump from 83% F1 to 100% F1.

---
### 3. Read through the notebooks (optional, for context)

The notebooks are interactive walkthroughs of the methodology and motivation. They are **supporting material** -- read them when you want the *why* behind the architecture and the scoring methodology, not when you want to run things. They load the reference artifacts in `models/` so you can see everything end-to-end without waiting on training.

| Notebook | What you'll learn |
|----------|-------------------|
| **`data_exploration.ipynb`** | Dataset overview, periodicity analysis, why simple thresholds fail, motivation for reconstruction-based detection |
| **`model_design.ipynb`** | LSTM Encoder-Decoder architecture, training demo, window-Mahalanobis scoring, anomaly localization |

```bash
uv run jupyter notebook notebooks/
```

## Docker demo with visual application

`code/3_streaming_app.py` is the Docker entrypoint for the live Kafka -> Spark -> Dash demo. It loads a trained model, consumes the NYC taxi CSV through Kafka, scores each weekly window in Spark Structured Streaming, and renders the results in a live Dash dashboard. **Do not run it directly with `python`** -- launch the full stack with Docker Compose:

```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up --build -d
```

Open http://localhost:8050 to view the live dashboard.

On subsequent runs (after images are built):

```bash
MESSAGE_DELAY_SECONDS=0.005 START_OFFSET=4944 LOOP_DATA=false docker compose up -d
```

View logs:

```bash
docker compose logs -f app
docker compose logs -f producer
```

---

## Workflow

The numbered files in `code/` tell the full reproduction story:

| Step | File | Purpose |
|------|------|---------|
| 0 | `0_verify_setup.py` | Optional troubleshooting: verify environment, data, and model artifacts |
| 1 | `1_train_model.py` | Train a fast baseline, save to `models/initial/` (~83% F1) |
| 2 | `2_evaluate_model.py` | Evaluate any saved artifacts (default: `models/initial/`), generate plots |
| 3 | `3_streaming_app.py` | Real-time Kafka -> Spark -> Dash streaming demo (Docker only) |
| 4 | `4_grid_sweep.py` | Sweep hyperparameters, retrain the winner, save to `models/best/` (100% F1) |

The reference artifacts at `models/lstm_model.pt`, `models/scaler.pkl`, `models/scorer.pkl` are the prebuilt model -- the notebooks load them so you can read through the methodology without waiting on training, and **none of the scripts ever overwrite them**. Your trained models go to `models/initial/` and `models/best/`.

## Detection methodology

The detector treats each calendar week as a single sample of length 336 (one week sampled every 30 minutes). After training the LSTM Encoder-Decoder on normal weeks, we collect the per-timestep reconstruction error vectors on the validation set and fit a multivariate Gaussian to them: a mean vector `mu` of length 336 and a 336 x 336 covariance matrix `Sigma`.

A test week `w` with reconstruction error vector `e_w` is then scored with the Mahalanobis distance

```
a(w) = (e_w - mu)^T * Sigma^-1 * (e_w - mu)
```

and flagged as anomalous when `a(w)` exceeds a threshold set at the 99.99th percentile of validation Mahalanobis distances. This is the same scoring scheme demonstrated cell-by-cell in `notebooks/model_design.ipynb` and is what `code/1_train_model.py` and `code/2_evaluate_model.py` reproduce end-to-end.

## How we got 100%

100% on a held-out test set should always raise an eyebrow, so a quick note on why this result is honest rather than overfit. The NYC taxi dataset is a few months of demand at 30-minute resolution, dominated by clean daily and weekly cycles that the LSTM Encoder-Decoder learns easily, and the five labeled anomalies (NYC Marathon, Thanksgiving, Christmas, New Year's, January Blizzard) each break that weekly pattern in ways that are visually obvious in `notebooks/data_exploration.ipynb`.

The scorer is fit strictly on validation weeks, disjoint from the 16-week test set, so there is no leak from labeled anomalies into the threshold.

The win is the methodology, not model size. The architecture is the original Malhotra et al. 2016 LSTM Encoder-Decoder with 64 hidden units (~34k parameters) -- no transformer, no attention. What makes it work is scoring whole weeks rather than individual points, and using Mahalanobis distance over the full 336-step error vector instead of a per-point z-score. That captures the *shape* of how a week deviates from normal. Apply the same pipeline to noisier or higher-dimensional data and expect more nuanced numbers.

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

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `START_OFFSET` | `0` | Record index to start streaming from (4944 for test data) |
| `LOOP_DATA` | `true` | Whether to loop through data continuously |
| `MESSAGE_DELAY_SECONDS` | `0.1` | Delay between messages (0.005 for fast demo) |

## Detection Performance

The LSTM Encoder-Decoder detects all 5 known anomalies in the NYC taxi dataset:

- **NYC Marathon** (Nov 1-3, 2014) — demand spike
- **Thanksgiving** (Nov 25-29, 2014) — demand drop
- **Christmas** (Dec 23-27, 2014) — demand drop
- **New Year's** (Dec 29 - Jan 3, 2015) — pattern disruption
- **January Blizzard** (Jan 24-29, 2015) — demand drop

**Precision: 100% | Recall: 100% | Inference: <5ms per weekly window**

## Services & Ports

| Service | Port | URL |
|---------|------|-----|
| Dash Dashboard | 8050 | http://localhost:8050 |
| Spark Master UI | 8080 | http://localhost:8080 |
| Kafka | 9092 | External access |
| Zookeeper | 2181 | Kafka coordination |
</content>
</invoke>