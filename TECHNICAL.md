# LSTM Autoencoder Anomaly Detection — Technical Reference

Detailed architecture and internals of the LSTM Encoder-Decoder (EncDec-AD) anomaly detection system. For setup instructions and usage, see [README.md](README.md).

---

## Table of Contents

1. [Core Model Architecture](#1-core-model-architecture)
2. [Scoring System](#2-scoring-system)
3. [Data Preprocessing](#3-data-preprocessing)
4. [Streaming Detector](#4-streaming-detector)
5. [Isolation Forest Detector](#5-isolation-forest-detector)
6. [Training Pipeline](#6-training-pipeline)
7. [Evaluation](#7-evaluation)
8. [Streaming Pipeline](#8-streaming-pipeline)
9. [Docker Infrastructure](#9-docker-infrastructure)
10. [Configuration Reference](#10-configuration-reference)

---

## 1. Core Model Architecture

Implementation of the EncDec-AD architecture from [Malhotra et al. (2016)](https://arxiv.org/abs/1607.00148), "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection."

### 1.1 ModelConfig

All architecture hyperparameters are centralized in the `ModelConfig` dataclass (`app/lstm_autoencoder.py`):

```python
@dataclass
class ModelConfig:
    input_dim: int = 1          # Features per timestep (1 for univariate, 3 with DoW)
    hidden_dim: int = 64        # LSTM hidden state dimension
    num_layers: int = 1         # Stacked LSTM layers
    dropout: float = 0.2        # Dropout (applied only if num_layers > 1)
    sequence_length: int = 336  # Samples per window (one week = 48/day × 7)
```

### 1.2 Encoder

`LSTMEncoder` compresses the input sequence into a fixed-size latent representation:

```
Input: x (batch, 336, input_dim)
  │
  ▼
LSTM(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=True)
  │
  ▼
Output: (h_n, c_n)  —  both shape (num_layers, batch, 64)
```

The final hidden state `h_n` captures a compressed representation of the entire weekly sequence. Only `h_n` and `c_n` are passed to the decoder — intermediate outputs are discarded.

### 1.3 Decoder

`LSTMDecoder` reconstructs the sequence from the latent state. Key detail from the paper: **the decoder reconstructs in reverse order** `(x'(L), x'(L-1), ..., x'(1))`.

```
Decoder Input: reversed(x)  (batch, 336, input_dim)
Hidden Init:   (h_n, c_n) from encoder
  │
  ▼
LSTM(input_size=input_dim, hidden_size=64, batch_first=True)
  │
  ▼
Linear(64 → input_dim)
  │
  ▼
Output: reconstructed_reversed (batch, 336, input_dim)
```

The output is flipped back to original order before loss computation.

### 1.4 Full Model (EncDecAD)

The complete forward pass uses **teacher forcing** — the decoder receives the actual (reversed) input rather than its own predictions:

```
Forward Pass:
  1. h_n, c_n = encoder(x)
  2. decoder_input = reverse(x)
  3. reconstructed_reversed = decoder(decoder_input, (h_n, c_n))
  4. reconstructed = reverse(reconstructed_reversed)
  5. loss = MSE(x, reconstructed)
```

The model also provides a `generate()` method for autoregressive inference (without teacher forcing), though teacher forcing is used even during evaluation since the reconstruction error is what drives anomaly scoring.

### 1.5 Parameter Count

With default config (hidden_dim=64, num_layers=1, input_dim=1): **~33K parameters**.

---

## 2. Scoring System

Implemented in `app/anomaly_scorer.py`. Two scoring modes are available.

### 2.1 Point-Level Scoring (Default — Malhotra et al. 2016)

The primary scoring mode pools reconstruction errors across all timesteps and windows, then computes per-point anomaly scores:

**Error distribution fitting:**

Given reconstruction errors $e$ from normal validation data, compute per-dimension statistics:

$$\mu = \text{mean}(e), \quad \sigma = \text{var}(e)$$

**Per-point anomaly score:**

$$a^{(i)} = \frac{(e^{(i)} - \mu)^2}{\sigma}$$

**Threshold:** Set at a configurable percentile (default: 99.99th) of validation scores.

**Window decision (HardCriterion):** A window is flagged as anomalous if $k$ or more points exceed the point threshold $\tau$. Default $k = 5$.

### 2.2 Window-Level Scoring (Legacy)

Computes the mean error vector $\mu$ and full covariance matrix $\Sigma$ across the time axis, then uses **Mahalanobis distance**:

$$a(i) = (e(i) - \mu)^T \Sigma^{-1} (e(i) - \mu)$$

Threshold is determined from a percentile of validation set Mahalanobis distances.

### 2.3 Variance Floor

When training errors are very small ($\sigma \approx 0.0006$), the score formula amplifies by ~1500x, leading to unstable thresholds. A minimum variance floor (default: 0.01) prevents this:

```python
sigma_point = np.maximum(sigma_point, config.min_variance_floor)
```

### 2.4 Anomaly Localization

When a window is flagged, the scorer identifies the most anomalous sub-region:

1. Slide a 6-hour window (12 samples) across the 336-sample week
2. Find the position with the highest mean reconstruction error
3. Compute a **contrast ratio**: mean error inside the 6-hour window vs. outside
4. Return the top 3 highest-scoring individual points for dashboard display

---

## 3. Data Preprocessing

Implemented in `app/data_preprocessor.py`.

### 3.1 Dataset

NYC Taxi demand dataset from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB). Contains ~10,320 records of taxi demand at 30-minute intervals, spanning July 2014 – January 2015.

**Five known anomaly windows:**

| Event | Start | End |
|-------|-------|-----|
| NYC Marathon | Nov 1 2014 19:00 | Nov 3 2014 22:30 |
| Thanksgiving | Nov 25 2014 12:00 | Nov 29 2014 19:00 |
| Christmas | Dec 23 2014 11:30 | Dec 27 2014 18:30 |
| New Year's | Dec 29 2014 21:30 | Jan 3 2015 04:30 |
| Blizzard | Jan 24 2015 20:30 | Jan 29 2015 03:30 |

### 3.2 Week Segmentation

Data is segmented into non-overlapping weekly windows of 336 samples (48 samples/day × 7 days). A 5-day offset from the data start ensures Sunday–Saturday alignment and that all anomaly points (including the January blizzard) fall within complete weeks.

A week is labeled anomalous if ≥10% of its samples overlap with any known anomaly window (`min_anomaly_overlap=0.10`).

Result: ~29 complete weeks, of which ~5 contain anomalies.

### 3.3 Data Splits

Following Malhotra et al.:

| Split | Purpose | Default | Content |
|-------|---------|---------|---------|
| Train ($s_N$) | Model training | 9 weeks | Normal weeks only |
| Val ($v_{N1}$) | Early stopping + error distribution fitting | 3 weeks | Normal weeks only |
| Threshold ($v_{N2}$) | Threshold calibration | 2 weeks | Normal weeks only |
| Test | Evaluation | ~15 weeks | Remaining normal + all anomaly weeks |

### 3.4 Normalization

`StandardScaler` fitted **only on training data** to prevent data leakage. The scaler computes per-feature mean and standard deviation, then applies $x' = (x - \mu) / \sigma$ to all splits. The fitted scaler is serialized to `models/scaler.pkl` for inference.

---

## 4. Streaming Detector

Implemented in `app/streaming_detector.py`. `LSTMStreamingDetector` extends `BaseDetector` and wraps the trained model for real-time use.

### 4.1 Artifact Loading

On initialization, loads three artifacts from disk:
- `models/lstm_model.pt` — PyTorch checkpoint containing `ModelConfig` + `state_dict`
- `models/scaler.pkl` — Fitted `StandardScaler`
- `models/scorer.pkl` — `AnomalyScorer` with calibrated threshold

Uses lazy loading with error handling — if artifacts are missing, the detector reports `is_ready=False` and returns empty results.

### 4.2 Detection Pipeline

```
Raw values (336 samples)
  │
  ▼
StandardScaler.transform() → normalized values
  │
  ▼
[Optional] Add DoW features → (336, 3) if model expects 3-channel input
  │
  ▼
model.forward(x) → reconstruction
  │
  ▼
|x - reconstruction| → point-wise errors
  │
  ▼
Point scores: (e - μ)² / σ  → compare to τ
  │
  ▼
HardCriterion: k=5 points > τ → ANOMALY
  │
  ▼
Localize to 6-hour window → top 3 points for display
```

### 4.3 Day-of-Week Conditioning

When `ModelConfig.input_dim == 3`, the detector adds cyclical day-of-week features:

- Channel 0: Normalized transaction count
- Channel 1: $\sin(2\pi \cdot \text{dow} / 7)$
- Channel 2: $\cos(2\pi \cdot \text{dow} / 7)$

Scoring uses **channel 0 only** to prevent dilution from near-zero DoW reconstruction errors.

---

## 5. Isolation Forest Detector

Implemented in `app/isolation_forest_detector.py`. An alternative detection mode that requires no pre-training.

### 5.1 Configuration

```python
@dataclass
class DetectorConfig:
    window_size: int = 200        # Sliding window size
    min_samples: int = 50         # Minimum for detection
    contamination: float = 0.05   # Expected anomaly ratio
    n_estimators: int = 100       # Ensemble trees
```

### 5.2 Feature Engineering

From the raw value column, five features are computed:

| Feature | Description |
|---------|-------------|
| `value` | Raw data point |
| `rolling_mean` | 10-point rolling mean |
| `rolling_std` | 10-point rolling standard deviation |
| `deviation` | Absolute deviation from rolling mean |
| `diff` | First-order difference (rate of change) |

### 5.3 Detection

The model is **re-fitted on each detection cycle** using the current sliding window, then predicts on the same window. Points scored as outliers by the isolation forest (path length below threshold) are returned as anomalies.

---

## 6. Training Pipeline

Implemented in `app/train.py`.

### 6.1 Training Loop

| Parameter | Default | Optimized |
|-----------|---------|-----------|
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 5e-4 |
| Loss function | MSE | MSE |
| Early stopping patience | 10 epochs | 10 epochs |
| Min delta | 1e-6 | 1e-6 |
| Gradient clipping | max norm 1.0 | max norm 1.0 |
| Max epochs | 100 | 100 |

### 6.2 Scorer Fitting

After training, the error distribution is fitted on **validation data** (not training data). This prevents the scorer from learning an overly tight distribution from the training errors, which would inflate scores and destabilize thresholds.

### 6.3 Artifacts

| File | Contents |
|------|----------|
| `lstm_model.pt` | `ModelConfig` + `state_dict` (PyTorch checkpoint) |
| `scaler.pkl` | `StandardScaler` fitted on training data |
| `scorer.pkl` | `AnomalyScorer` with fitted error distribution and threshold |
| `training_history.pkl` | Per-epoch train/val loss curves |
| `preprocessor_config.pkl` | Data split configuration for reproducibility |

---

## 7. Evaluation

Implemented in `app/evaluate.py`.

### 7.1 Metrics

- **Confusion matrix:** TP, FP, FN, TN at both window and point level
- **Precision, Recall, F1-Score, Accuracy** at window level
- **F-beta score** at point level (configurable beta)

### 7.2 Two Evaluation Modes

**Window-level:** Each 336-sample week gets a single anomaly/normal label. Mahalanobis distance or HardCriterion decision.

**Point-level:** Each of the 336 samples in a window gets an individual score. Points exceeding the point threshold $\tau$ are anomalous. Window decision follows HardCriterion ($k$ points $> \tau$).

### 7.3 Visualizations

Generated in the `results/` directory:
- `training_history.png` — Train and validation loss curves
- `score_distribution.png` — Score distributions for normal vs. anomaly weeks
- `weekly_comparison.png` — Normal vs. anomaly week reconstruction comparison
- `reconstruction_*.png` — Per-anomaly-week original vs. reconstructed overlays

---

## 8. Streaming Pipeline

Implemented in `app/main.py`. Orchestrates Kafka consumption, Spark processing, and Dash visualization.

### 8.1 Data Flow

```
Producer ──► Kafka (anomaly_stream) ──► Spark Structured Streaming ──► Detector ──► Dash Dashboard
              ▲                                                            │
              │                                                            ▼
         Zookeeper                                                  http://localhost:8050
```

### 8.2 Kafka Message Schema

```json
{
  "timestamp": "2014-10-01T00:00:00",
  "value": 10229,
  "produced_at": "2024-01-01T12:00:00Z",
  "sequence_id": 0
}
```

Producer config: `acks=all`, `retries=3`, `linger.ms=5`, `batch.size=16384`.

### 8.3 Spark Consumer

- Reads from Kafka with `maxOffsetsPerTrigger=50`
- `startingOffsets=earliest` (replays from start)
- Processes micro-batches via `foreachBatch`

### 8.4 LSTM Streaming Mode

- Accumulates data into a buffer
- Processes **non-overlapping weekly windows** (336 samples each)
- Trims old data, keeping the last `VISUALIZATION_WEEKS=3` weeks for dashboard display
- Tracks: `weeks_processed`, `samples_since_last_window`, `samples_removed`

### 8.5 Isolation Forest Streaming Mode

- Maintains a sliding window (default 200 samples)
- Re-fits model on each detection cycle
- Returns individual anomalous points

### 8.6 Health Endpoint

```json
{"status": "ready", "spark_ready": true, "detector": "lstm"}
```

The producer polls `http://app:8050/health` and waits until `spark_ready=true` before streaming, ensuring correct startup ordering.

---

## 9. Docker Infrastructure

Defined in `docker-compose.yml`. Six services on a shared `streaming_network` bridge.

### 9.1 Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| Zookeeper | confluentinc/cp-zookeeper:7.5.0 | 2181 | Kafka coordination |
| Kafka | confluentinc/cp-kafka:7.5.0 | 9092, 29092 | Message broker |
| Spark Master | apache/spark:3.5.3-python3 | 8080, 7077 | Spark coordination |
| Spark Worker | apache/spark:3.5.3-python3 | — | Task execution (2GB, 2 cores) |
| Producer | Custom (./producer) | — | Streams CSV → Kafka |
| App | Custom (./app) | 8050 | Dash dashboard + detection |

### 9.2 Volume Mounts

- `./data:/app/data:ro` — Dataset (read-only, on producer)
- `./models:/app/models:ro` — Trained model artifacts (read-only, on app)

### 9.3 Startup Order

1. Zookeeper starts and passes healthcheck
2. Kafka starts (depends on Zookeeper healthy)
3. Spark Master + Worker start
4. App starts, initializes Spark streaming, exposes `/health`
5. Producer polls `/health`, waits for `spark_ready=true`, then begins streaming

---

## 10. Configuration Reference

### 10.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DETECTOR_TYPE` | `isolation_forest` | Detection mode: `lstm` or `isolation_forest` |
| `START_OFFSET` | `0` | Record index to start streaming from (set to 4992 for LSTM test data) |
| `LOOP_DATA` | `true` | Loop through data continuously |
| `MESSAGE_DELAY_SECONDS` | `0.1` | Delay between producer messages |
| `WAIT_FOR_APP` | `true` | Producer waits for Spark readiness |
| `WINDOW_SIZE` | `200` | Sliding window size (Isolation Forest) |
| `CONTAMINATION` | `0.05` | Expected anomaly ratio (Isolation Forest) |

### 10.2 Optimized LSTM Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `hidden_dim` | 64 | Grid search (`app/optimize_hyperparams.py`) |
| `num_layers` | 1 | Grid search |
| `dropout` | 0.2 | Grid search |
| `learning_rate` | 5e-4 | Grid search |
| `threshold_percentile` | 99.99% | Grid search |
| `train_weeks` | 8 | Split optimization (`app/optimize_split.py`) |
| `val_weeks` | 2 | Split optimization |
| `threshold_weeks` | 4 | Split optimization |
| `hard_criterion_k` | 5 | Manual tuning |

### 10.3 Base Detector Interface

All detectors implement `BaseDetector` (`app/base_detector.py`):

| Method | Returns | Description |
|--------|---------|-------------|
| `detect(df)` | `DataFrame` | Anomalous records with scores |
| `get_stats()` | `Dict` | Detector configuration and state |
| `get_name()` | `str` | Human-readable name |
| `is_ready` | `bool` | Whether detection can run |
| `min_samples_required` | `int` | Minimum data needed |

Factory function `create_detector(type, **kwargs)` instantiates the appropriate detector.
