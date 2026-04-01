"""
Step 6: Streaming Application
Real-time anomaly detection on NYC taxi demand data using
Kafka streaming, Spark Structured Streaming, and Dash visualization.

This is the Docker entrypoint for the real-time demo.
Run via: docker compose up --build
"""

import os
import pickle
import threading
import time
import logging
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# When running in Docker, src/ is at /app/src/
# When running locally, add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EncDecAD, ModelConfig
from src.scorer import AnomalyScorer, ScorerConfig

# Register old module paths so pickled artifacts (saved with old names) can be loaded
import src.model, src.scorer
sys.modules["lstm_autoencoder"] = src.model
sys.modules["anomaly_scorer"] = src.scorer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/lstm_model.pt")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")
SCORER_PATH = os.environ.get("SCORER_PATH", "models/scorer.pkl")
LSTM_WINDOW_SIZE = 336
VISUALIZATION_WEEKS = 3
STREAM_TIMEOUT = 10

# ---------------------------------------------------------------------------
# Model loading (replaces LSTMStreamingDetector class)
# ---------------------------------------------------------------------------

def load_model_artifacts(model_path, scaler_path, scorer_path, device=None):
    """
    Load pre-trained model, scaler, and scorer from disk.

    Returns:
        Tuple of (model, scaler, scorer, device)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = EncDecAD(config=checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load scaler
    logger.info(f"Loading scaler from {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # Load scorer
    logger.info(f"Loading scorer from {scorer_path}")
    scorer = AnomalyScorer.load(str(scorer_path))

    logger.info("All model artifacts loaded successfully")
    logger.info(f"  Model config: {model.get_config()}")
    logger.info(f"  Scorer threshold: {scorer.threshold}")
    logger.info(f"  Device: {device}")

    return model, scaler, scorer, device


def detect_anomalies(model, scaler, scorer, df, device):
    """
    Detect anomalies in a weekly window of data.

    Args:
        model: Trained EncDecAD model
        scaler: Fitted StandardScaler
        scorer: Fitted AnomalyScorer
        df: DataFrame with 'timestamp' and 'value' columns (336 rows)
        device: Torch device

    Returns:
        DataFrame of anomalous records (empty if normal)
    """
    values = df["value"].values.astype(float).reshape(-1, 1)
    values_normalized = scaler.transform(values).flatten()

    # Forward pass
    x = torch.FloatTensor(values_normalized).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        x_reconstructed = model(x)

    error = torch.abs(x - x_reconstructed).cpu().numpy().squeeze()

    # Check scoring mode
    use_point_level = scorer.mu_point is not None and scorer.point_threshold is not None

    if use_point_level:
        point_scores = ((error - scorer.mu_point[0]) ** 2) / scorer.sigma_point[0]
        point_predictions = point_scores > scorer.point_threshold
        k = scorer.config.hard_criterion_k
        is_anomaly = np.sum(point_predictions) >= k
        window_score = np.max(point_scores)
    else:
        window_score = scorer._mahalanobis_distance(error)
        is_anomaly = window_score > scorer.threshold
        point_scores = error

    logger.info(
        f"  Scoring: raw=[{values.min():.0f}, {values.max():.0f}], "
        f"norm=[{values_normalized.min():.4f}, {values_normalized.max():.4f}], "
        f"score={window_score:.2f} -> {'ANOMALY' if is_anomaly else 'NORMAL'}"
    )

    if is_anomaly:
        window_df = df.copy().reset_index(drop=True)
        timestamps_list = window_df["timestamp"].astype(str).tolist()
        localization = scorer.localize_anomaly(error, timestamps_list)

        loc_start_idx = localization["anomaly_start_idx"]
        loc_end_idx = localization["anomaly_end_idx"]

        localized_df = window_df.iloc[loc_start_idx:loc_end_idx + 1].copy()
        top_n = min(3, len(localized_df))
        localized_df["point_score"] = point_scores[loc_start_idx:loc_end_idx + 1]
        anomalous_points = localized_df.nlargest(top_n, "point_score").copy()
        anomalous_points["anomaly_score"] = window_score
        anomalous_points["localization_start"] = localization["anomaly_start"]
        anomalous_points["localization_end"] = localization["anomaly_end"]
        anomalous_points["scale_hours"] = localization["scale_hours"]
        anomalous_points["contrast_ratio"] = localization["contrast_ratio"]

        logger.info(
            f"  Localized to {localization['scale_hours']}h: "
            f"{localization['anomaly_start']} - {localization['anomaly_end']}"
        )

        return anomalous_points[["timestamp", "value", "anomaly_score",
                                  "localization_start", "localization_end",
                                  "scale_hours", "contrast_ratio"]]

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Thread-safe data store
# ---------------------------------------------------------------------------
data_lock = threading.Lock()
data_store = {
    "data": deque(),
    "anomalies": pd.DataFrame(),
    "total_received": 0,
    "total_anomalies": 0,
    "last_batch_size": 0,
    "last_update": None,
    "last_detection": None,
    "samples_since_last_window": 0,
    "flagged_windows": [],
    "weeks_processed": 0,
    "samples_removed": 0,
    "stream_ended": False,
    "spark_ready": False,
}


# ---------------------------------------------------------------------------
# Spark streaming
# ---------------------------------------------------------------------------

def wait_for_kafka(bootstrap_servers, max_retries=30, retry_interval=5):
    """Wait for Kafka to be available."""
    from confluent_kafka import Producer
    logger.info(f"Waiting for Kafka at {bootstrap_servers}...")
    for attempt in range(max_retries):
        try:
            producer = Producer({"bootstrap.servers": bootstrap_servers})
            metadata = producer.list_topics(timeout=10)
            logger.info(f"Connected to Kafka. Topics: {list(metadata.topics.keys())}")
            return True
        except Exception as e:
            logger.warning(f"Kafka not ready (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_interval)
    raise RuntimeError(f"Could not connect to Kafka after {max_retries} attempts")


def start_spark_streaming():
    """Start Spark Structured Streaming to consume from Kafka."""
    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")
    kafka_topic = os.environ.get("KAFKA_TOPIC", "anomaly_stream")
    spark_master = os.environ.get("SPARK_MASTER", "spark://spark-master:7077")

    wait_for_kafka(kafka_servers)
    logger.info(f"Starting Spark session connecting to {spark_master}")

    spark = SparkSession.builder \
        .appName("NYCTaxiAnomalyDetection") \
        .master(spark_master) \
        .config("spark.driver.host", "app") \
        .config("spark.driver.bindAddress", "0.0.0.0") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.3") \
        .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("value", IntegerType()),
        StructField("produced_at", StringType()),
        StructField("sequence_id", IntegerType()),
    ])

    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
        .option("subscribe", kafka_topic) \
        .option("startingOffsets", "earliest") \
        .option("maxOffsetsPerTrigger", 50) \
        .load()

    df = df.selectExpr("CAST(value AS STRING)")
    df_parsed = df.select(from_json(col("value"), schema).alias("data")).select("data.*")
    df_parsed = df_parsed.withColumn("timestamp", to_timestamp(col("timestamp")))

    def process_batch(batch_df, batch_id):
        try:
            pandas_df = batch_df.toPandas()
            if pandas_df.empty:
                return

            logger.info(
                f"Batch {batch_id}: {len(pandas_df)} records, "
                f"seq_id [{pandas_df['sequence_id'].iloc[0]}-{pandas_df['sequence_id'].iloc[-1]}]"
            )

            with data_lock:
                for _, row in pandas_df.iterrows():
                    data_store["data"].append({
                        "timestamp": str(row["timestamp"]),
                        "value": row["value"],
                        "produced_at": row["produced_at"],
                        "sequence_id": row["sequence_id"],
                    })
                data_store["total_received"] += len(pandas_df)
                data_store["last_batch_size"] = len(pandas_df)
                data_store["last_update"] = pd.Timestamp.now().isoformat()
                data_store["samples_since_last_window"] += len(pandas_df)

        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")

    query = df_parsed.writeStream \
        .trigger(processingTime="2 seconds") \
        .foreachBatch(process_batch) \
        .start()

    with data_lock:
        data_store["spark_ready"] = True
    logger.info("Spark Streaming started - ready for data")

    query.awaitTermination()


# ---------------------------------------------------------------------------
# Anomaly detection thread
# ---------------------------------------------------------------------------

def run_anomaly_detection(model, scaler, scorer, device):
    """Background thread: process non-overlapping weekly windows."""
    detection_interval = 2

    logger.info("Starting anomaly detection thread")
    logger.info(f"  Window size: {LSTM_WINDOW_SIZE}, interval: {detection_interval}s")

    while True:
        try:
            # Check stream timeout
            with data_lock:
                last_update = data_store["last_update"]
                if last_update and not data_store["stream_ended"]:
                    elapsed = (pd.Timestamp.now() - pd.Timestamp(last_update)).total_seconds()
                    if elapsed > STREAM_TIMEOUT:
                        data_store["stream_ended"] = True
                        logger.info(f"Stream ended: no new data for {STREAM_TIMEOUT}s")

            # Process all available weeks
            while True:
                with data_lock:
                    week_num = data_store["weeks_processed"] + 1
                    samples_since_last = data_store["samples_since_last_window"]
                    samples_removed = data_store["samples_removed"]
                    current_data_list = list(data_store["data"])

                if samples_since_last < LSTM_WINDOW_SIZE:
                    break

                absolute_start = (week_num - 1) * LSTM_WINDOW_SIZE
                absolute_end = week_num * LSTM_WINDOW_SIZE
                start_idx = absolute_start - samples_removed
                end_idx = absolute_end - samples_removed

                if start_idx < 0 or end_idx > len(current_data_list):
                    break

                df = pd.DataFrame(current_data_list[start_idx:end_idx])
                window_start_ts = df["timestamp"].iloc[0]
                window_end_ts = df["timestamp"].iloc[-1]

                logger.info("=" * 70)
                logger.info(f"Processing Week {week_num}: {window_start_ts} to {window_end_ts}")

                anomalies = detect_anomalies(model, scaler, scorer, df, device)

                with data_lock:
                    data_store["weeks_processed"] += 1
                    data_store["samples_since_last_window"] -= LSTM_WINDOW_SIZE

                    # Trim old data
                    max_keep = VISUALIZATION_WEEKS * LSTM_WINDOW_SIZE
                    if len(data_store["data"]) > max_keep + LSTM_WINDOW_SIZE:
                        trim_count = LSTM_WINDOW_SIZE
                        for _ in range(trim_count):
                            data_store["data"].popleft()
                        data_store["samples_removed"] += trim_count

                    if not anomalies.empty:
                        score = anomalies["anomaly_score"].iloc[0]
                        localization = {
                            "anomaly_start": anomalies["localization_start"].iloc[0] if "localization_start" in anomalies.columns else None,
                            "anomaly_end": anomalies["localization_end"].iloc[0] if "localization_end" in anomalies.columns else None,
                            "scale_hours": anomalies["scale_hours"].iloc[0] if "scale_hours" in anomalies.columns else None,
                            "contrast_ratio": anomalies["contrast_ratio"].iloc[0] if "contrast_ratio" in anomalies.columns else None,
                        }
                        data_store["flagged_windows"].append({
                            "week_num": week_num,
                            "start_time": str(window_start_ts),
                            "end_time": str(window_end_ts),
                            "score": score,
                            "high_error_points": anomalies.to_dict("records"),
                            "localization": localization,
                        })

                        all_points = []
                        for w in data_store["flagged_windows"]:
                            all_points.extend(w["high_error_points"])
                        data_store["anomalies"] = pd.DataFrame(all_points)
                        data_store["total_anomalies"] = len(data_store["flagged_windows"])

                        logger.info(f"  ANOMALY detected! Score: {score:.2f}")
                    else:
                        logger.info(f"  Normal (below threshold)")

                    data_store["last_detection"] = pd.Timestamp.now().isoformat()

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")

        time.sleep(detection_interval)


# ---------------------------------------------------------------------------
# Dash application
# ---------------------------------------------------------------------------
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ANOMALY_COLOR = "#e74c3c"
BACKGROUND_COLOR = "#ecf0f1"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "NYC Taxi Anomaly Detection"


@app.server.route("/health")
def health_check():
    from flask import jsonify
    with data_lock:
        spark_ready = data_store.get("spark_ready", False)
    return jsonify({"status": "ready" if spark_ready else "starting", "spark_ready": spark_ready, "detector": "lstm"})


app.layout = dbc.Container([
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand("NYC Taxi Demand - Real-Time Anomaly Detection",
                            style={"color": "white", "fontSize": "24px", "fontWeight": "bold"}),
        ]),
        color=PRIMARY_COLOR, dark=True, sticky="top", className="mb-4",
    ),

    dbc.Alert(
        [html.H4("Stream Complete", className="alert-heading"),
         html.P("All data has been processed.", className="mb-0")],
        id="stream-ended-alert", color="warning", is_open=False, className="mb-4",
    ),

    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Total Received", className="text-center"),
            html.H2(id="total-received", className="text-center text-success"),
        ])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Weeks Processed", className="text-center"),
            html.H2(id="window-size", className="text-center text-info"),
        ])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Flagged Windows", className="text-center"),
            html.H2(id="anomaly-count", className="text-center text-danger"),
        ])]), width=3),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H5("Detection Status", className="text-center"),
            html.P(id="detection-status", className="text-center text-muted", style={"fontSize": "14px"}),
        ])]), width=3),
    ], className="mb-4"),

    dbc.Card([
        dbc.CardHeader(html.H4("Taxi Demand with Anomaly Detection", className="mb-0"),
                       style={"backgroundColor": SECONDARY_COLOR, "color": "white"}),
        dbc.CardBody([dcc.Graph(id="stream-graph", style={"height": "400px"})]),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([dbc.Card([
            dbc.CardHeader(html.H4("Recent Records", className="mb-0"),
                           style={"backgroundColor": PRIMARY_COLOR, "color": "white"}),
            dbc.CardBody([dash_table.DataTable(
                id="data-table",
                columns=[{"name": "Timestamp", "id": "timestamp"}, {"name": "Value", "id": "value"},
                         {"name": "Seq ID", "id": "sequence_id"}],
                style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                style_header={"backgroundColor": PRIMARY_COLOR, "color": "white", "fontWeight": "bold"},
                page_size=10,
            )]),
        ])], width=6),
        dbc.Col([dbc.Card([
            dbc.CardHeader(html.H4("Detected Anomalies", className="mb-0"),
                           style={"backgroundColor": ANOMALY_COLOR, "color": "white"}),
            dbc.CardBody([dash_table.DataTable(
                id="anomaly-table",
                columns=[{"name": "Timestamp", "id": "timestamp"}, {"name": "Value", "id": "value"},
                         {"name": "Anomaly Score", "id": "anomaly_score"}],
                style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                style_header={"backgroundColor": ANOMALY_COLOR, "color": "white", "fontWeight": "bold"},
                style_data_conditional=[{"if": {"column_id": "anomaly_score"}, "color": ANOMALY_COLOR, "fontWeight": "bold"}],
                page_size=10,
            )]),
        ])], width=6),
    ], className="mb-4"),

    dcc.Interval(id="refresh", interval=2000, n_intervals=0),

    html.Footer(
        dbc.Container(html.P(
            "LSTM Encoder-Decoder Anomaly Detection | Non-Overlapping Weekly Windows",
            className="text-center mb-0", style={"color": "white", "padding": "10px"},
        )),
        style={"backgroundColor": PRIMARY_COLOR, "marginTop": "20px"},
    ),
], fluid=True, style={"backgroundColor": BACKGROUND_COLOR, "minHeight": "100vh"})


@app.callback(
    [Output("stream-graph", "figure"), Output("data-table", "data"), Output("anomaly-table", "data"),
     Output("total-received", "children"), Output("window-size", "children"),
     Output("anomaly-count", "children"), Output("detection-status", "children"),
     Output("stream-ended-alert", "is_open")],
    [Input("refresh", "n_intervals")],
)
def update_dashboard(n):
    with data_lock:
        data_list = list(data_store["data"])
        anomalies_df = data_store["anomalies"].copy() if not data_store["anomalies"].empty else pd.DataFrame()
        total = data_store["total_received"]
        weeks_processed = data_store["weeks_processed"]
        samples_since_last = data_store["samples_since_last_window"]
        flagged_windows = list(data_store["flagged_windows"])
        stream_ended = data_store["stream_ended"]

    fig = go.Figure()

    if data_list:
        max_display = VISUALIZATION_WEEKS * LSTM_WINDOW_SIZE
        display_data = data_list[-max_display:] if len(data_list) > max_display else data_list
        df = pd.DataFrame(display_data)

        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["value"], mode="lines", name="Taxi Demand",
            line=dict(color=SECONDARY_COLOR, width=2),
        ))

        if flagged_windows:
            visible_start = df["timestamp"].iloc[0]
            visible_end = df["timestamp"].iloc[-1]
            for window in flagged_windows:
                ws = window.get("start_time")
                we = window.get("end_time")
                if ws and ws >= visible_start and ws <= visible_end:
                    fig.add_vline(x=ws, line=dict(color=ANOMALY_COLOR, width=2, dash="dash"))
                if we and we >= visible_start and we <= visible_end:
                    fig.add_vline(x=we, line=dict(color=ANOMALY_COLOR, width=2, dash="dash"))

                loc = window.get("localization", {})
                if loc.get("anomaly_start") and loc.get("anomaly_end"):
                    ls, le = loc["anomaly_start"], loc["anomaly_end"]
                    if le >= visible_start and ls <= visible_end:
                        fig.add_vrect(
                            x0=ls, x1=le, fillcolor=ANOMALY_COLOR, opacity=0.15, layer="below",
                            line_width=1, line_color=ANOMALY_COLOR,
                            annotation_text=f"{loc.get('scale_hours', '?')}h",
                            annotation_position="top left",
                            annotation=dict(font_size=10, font_color=ANOMALY_COLOR),
                        )

        if not anomalies_df.empty:
            visible_start = df["timestamp"].iloc[0]
            visible_end = df["timestamp"].iloc[-1]
            vis = anomalies_df[(anomalies_df["timestamp"] >= visible_start) & (anomalies_df["timestamp"] <= visible_end)]
            if not vis.empty:
                fig.add_trace(go.Scatter(
                    x=vis["timestamp"], y=vis["value"], mode="markers", name="Anomalies",
                    marker=dict(color=ANOMALY_COLOR, size=12, symbol="x", line=dict(width=2, color="white")),
                ))

    fig.update_layout(
        xaxis=dict(title="Timestamp", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Taxi Demand", showgrid=True, gridcolor="#e0e0e0"),
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified", plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=50, r=50, t=30, b=50),
    )

    if not data_list:
        fig.add_annotation(text="Waiting for data from Kafka...", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=20, color="gray"))

    recent_data = list(reversed(data_list[-10:])) if data_list else []

    anomaly_data = []
    if not anomalies_df.empty:
        adf = anomalies_df.copy()
        adf["timestamp"] = adf["timestamp"].astype(str)
        adf["anomaly_score"] = adf["anomaly_score"].round(4)
        anomaly_data = adf[["timestamp", "value", "anomaly_score"]].to_dict("records")

    if weeks_processed == 0:
        status_text = f"Buffering: {samples_since_last}/{LSTM_WINDOW_SIZE}"
    else:
        status_text = f"Week {weeks_processed} | Next: {samples_since_last}/{LSTM_WINDOW_SIZE}"

    return (fig, recent_data, anomaly_data, f"{total:,}", f"{weeks_processed}",
            f"{len(flagged_windows)}", status_text, stream_ended)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NYC Taxi Anomaly Detection Dashboard")
    logger.info("=" * 60)

    # Load model artifacts
    model, scaler, scorer, device = load_model_artifacts(MODEL_PATH, SCALER_PATH, SCORER_PATH)

    logger.info(f"Model ready on {device}")
    logger.info(f"Scorer stats: {scorer.get_stats()}")

    # Start Spark streaming
    streaming_thread = threading.Thread(target=start_spark_streaming, daemon=True)
    streaming_thread.start()

    # Start anomaly detection
    detection_thread = threading.Thread(
        target=run_anomaly_detection, args=(model, scaler, scorer, device), daemon=True
    )
    detection_thread.start()

    # Run Dash app
    logger.info("Starting Dash server on http://0.0.0.0:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
