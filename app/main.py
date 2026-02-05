"""
NYC Taxi Anomaly Detection Dashboard

Real-time anomaly detection on NYC taxi demand data using
Kafka streaming, Spark Structured Streaming, Isolation Forest detection,
and Dash visualization.
"""

import os
import threading
import time
import logging
from collections import deque

import pandas as pd

import dash
from dash.dependencies import Output, Input
from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from base_detector import create_detector, BaseDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment
DETECTOR_TYPE = os.environ.get("DETECTOR_TYPE", "isolation_forest").lower()
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "200"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "50"))
CONTAMINATION = float(os.environ.get("CONTAMINATION", "0.05"))

# LSTM-specific configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "models/lstm_model.pt")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.pkl")
SCORER_PATH = os.environ.get("SCORER_PATH", "models/scorer.pkl")

# Thread-safe data storage
data_lock = threading.Lock()

# For LSTM: use dynamic buffer (no maxlen) that trims old processed weeks
# Keep VISUALIZATION_WEEKS of history after processing for the dashboard graph
LSTM_WINDOW_SIZE = 336
VISUALIZATION_WEEKS = 3  # Keep 3 weeks of history for visualization
# For Isolation Forest, use fixed buffer; for LSTM, None means unlimited growth (we'll trim manually)
BUFFER_SIZE = max(WINDOW_SIZE, 500) if DETECTOR_TYPE != "lstm" else None

data_store = {
    "data": deque() if BUFFER_SIZE is None else deque(maxlen=BUFFER_SIZE),
    "anomalies": pd.DataFrame(),  # All anomalous points from flagged windows
    "total_received": 0,
    "total_anomalies": 0,
    "last_batch_size": 0,
    "last_update": None,
    "last_detection": None,
    # LSTM-specific: track non-overlapping weekly windows
    "samples_since_last_window": 0,  # Count samples to know when a new week is complete
    "flagged_windows": [],  # List of {week_num, start_time, end_time, score, high_error_points}
    "weeks_processed": 0,  # Total weeks analyzed
    "samples_removed": 0,  # Track how many samples trimmed from buffer start (for index adjustment)
    # Stream status tracking
    "stream_ended": False,  # True when no new data for STREAM_TIMEOUT seconds
    # Readiness tracking for producer coordination
    "spark_ready": False,  # True when Spark streaming has started
}

# How long to wait before considering the stream ended (seconds)
STREAM_TIMEOUT = 10

# Initialize detector based on configuration
if DETECTOR_TYPE == "lstm":
    detector: BaseDetector = create_detector(
        detector_type="lstm",
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        scorer_path=SCORER_PATH,
        window_size=336,  # Weekly window for LSTM
        min_samples=336,
    )
else:
    detector: BaseDetector = create_detector(
        detector_type="isolation_forest",
        window_size=WINDOW_SIZE,
        min_samples=MIN_SAMPLES,
        contamination=CONTAMINATION,
        n_estimators=100,
    )


def wait_for_kafka(bootstrap_servers: str, max_retries: int = 30, retry_interval: int = 5):
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

    # Schema matching producer output
    schema = StructType([
        StructField("timestamp", StringType()),
        StructField("value", IntegerType()),
        StructField("produced_at", StringType()),
        StructField("sequence_id", IntegerType())
    ])

    logger.info(f"Subscribing to Kafka topic: {kafka_topic}")

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
        """Process each micro-batch from Spark Streaming."""
        try:
            pandas_df = batch_df.toPandas()

            if pandas_df.empty:
                logger.debug(f"Batch {batch_id}: empty")
                return

            # Log detailed batch info for debugging
            first_ts = pandas_df["timestamp"].iloc[0]
            last_ts = pandas_df["timestamp"].iloc[-1]
            first_seq = pandas_df["sequence_id"].iloc[0]
            last_seq = pandas_df["sequence_id"].iloc[-1]

            logger.info(
                f"Batch {batch_id}: {len(pandas_df)} records, "
                f"seq_id [{first_seq}-{last_seq}], "
                f"timestamps [{first_ts} to {last_ts}]"
            )

            with data_lock:
                for _, row in pandas_df.iterrows():
                    data_store["data"].append({
                        "timestamp": str(row["timestamp"]),
                        "value": row["value"],
                        "produced_at": row["produced_at"],
                        "sequence_id": row["sequence_id"]
                    })

                data_store["total_received"] += len(pandas_df)
                data_store["last_batch_size"] = len(pandas_df)
                data_store["last_update"] = pd.Timestamp.now().isoformat()
                # Track samples for LSTM non-overlapping window detection
                data_store["samples_since_last_window"] += len(pandas_df)

                # Always log buffer state after adding data (for debugging)
                logger.info(
                    f"  -> Buffer after batch: {len(data_store['data'])} samples, "
                    f"samples_since_last: {data_store['samples_since_last_window']}"
                )

        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")

    query = df_parsed.writeStream \
        .trigger(processingTime="2 seconds") \
        .foreachBatch(process_batch) \
        .start()

    # Signal that Spark is ready to receive data
    with data_lock:
        data_store["spark_ready"] = True
    logger.info("Spark Streaming started - ready for data")

    query.awaitTermination()


def check_stream_ended():
    """Check if stream has ended (no new data for STREAM_TIMEOUT seconds)."""
    with data_lock:
        last_update = data_store["last_update"]
        if last_update is None:
            return False

        last_update_time = pd.Timestamp(last_update)
        time_since_update = (pd.Timestamp.now() - last_update_time).total_seconds()

        if time_since_update > STREAM_TIMEOUT and not data_store["stream_ended"]:
            data_store["stream_ended"] = True
            logger.info(
                f"Stream ended: No new data for {STREAM_TIMEOUT} seconds. "
                f"Total received: {data_store['total_received']}"
            )
            return True

        return data_store["stream_ended"]


def run_anomaly_detection():
    """
    Background thread that periodically runs anomaly detection.

    For LSTM detector: Uses non-overlapping weekly windows.
    Detection only runs when a complete new 336-sample week has arrived.
    Each week is evaluated exactly once.

    For Isolation Forest: Uses sliding window (original behavior).
    """
    detection_interval = 2  # seconds (check frequency)

    logger.info("Starting anomaly detection thread")
    logger.info(f"Detection interval: {detection_interval}s")
    logger.info(f"Detector type: {DETECTOR_TYPE}")
    logger.info(f"Detector config: {detector.get_stats()}")

    while True:
        try:
            # Check if stream has ended
            check_stream_ended()

            with data_lock:
                data_list = list(data_store["data"])
                samples_since_last = data_store["samples_since_last_window"]

            # LSTM: Non-overlapping window detection
            # Process ALL available weeks in one iteration to avoid falling behind
            if DETECTOR_TYPE == "lstm":
                weeks_processed_this_iteration = 0

                while True:
                    # Get fresh state for each week check
                    with data_lock:
                        week_num = data_store["weeks_processed"] + 1
                        samples_since_last = data_store["samples_since_last_window"]
                        samples_removed = data_store["samples_removed"]
                        current_data_list = list(data_store["data"])

                    # Check if we have enough samples for another week
                    if samples_since_last < LSTM_WINDOW_SIZE:
                        if weeks_processed_this_iteration == 0:
                            # Log waiting status (but not too frequently)
                            if samples_since_last % 50 < 10 or samples_since_last > 300:
                                logger.info(
                                    f"LSTM: Waiting for week {week_num}: "
                                    f"{samples_since_last}/{LSTM_WINDOW_SIZE} samples, "
                                    f"buffer={len(current_data_list)}"
                                )
                        break  # No more weeks ready

                    # Calculate window range (adjusted for trimmed samples)
                    absolute_start = (week_num - 1) * LSTM_WINDOW_SIZE
                    absolute_end = week_num * LSTM_WINDOW_SIZE
                    # Adjust indices based on how much we've trimmed from buffer start
                    start_idx = absolute_start - samples_removed
                    end_idx = absolute_end - samples_removed

                    # Check if buffer has the data we need
                    if start_idx < 0 or end_idx > len(current_data_list):
                        logger.info(
                            f"LSTM: Buffer indices out of range for week {week_num}: "
                            f"adjusted [{start_idx}:{end_idx}], buffer={len(current_data_list)}, "
                            f"samples_removed={samples_removed}"
                        )
                        break  # Not enough data yet

                    # Process this week
                    df = pd.DataFrame(current_data_list[start_idx:end_idx])
                    window_start_ts = df["timestamp"].iloc[0]
                    window_end_ts = df["timestamp"].iloc[-1]
                    window_values = df["value"].values

                    logger.info("=" * 70)
                    logger.info(f"LSTM: Processing Week {week_num}")
                    logger.info(f"  Absolute indices: [{absolute_start}:{absolute_end}], adjusted: [{start_idx}:{end_idx}]")
                    logger.info(f"  Window timestamps: {window_start_ts} to {window_end_ts}")
                    logger.info(f"  Window size: {len(df)} samples")
                    logger.info(f"  Value range: [{window_values.min()}, {window_values.max()}]")
                    logger.info(f"  Buffer size: {len(current_data_list)}, samples_removed: {samples_removed}")

                    anomalies = detector.detect(df)

                    with data_lock:
                        data_store["weeks_processed"] += 1
                        data_store["samples_since_last_window"] -= LSTM_WINDOW_SIZE

                        # Trim old data from buffer, keeping VISUALIZATION_WEEKS of history
                        # This keeps memory bounded while preserving recent data for the dashboard
                        max_keep = VISUALIZATION_WEEKS * LSTM_WINDOW_SIZE
                        current_buffer_size = len(data_store["data"])
                        if current_buffer_size > max_keep + LSTM_WINDOW_SIZE:
                            # Remove oldest week from buffer
                            trim_count = LSTM_WINDOW_SIZE
                            for _ in range(trim_count):
                                data_store["data"].popleft()
                            data_store["samples_removed"] += trim_count
                            logger.info(
                                f"  Trimmed {trim_count} old samples from buffer, "
                                f"total removed: {data_store['samples_removed']}"
                            )

                        if not anomalies.empty:
                            score = anomalies["anomaly_score"].iloc[0]
                            # Extract localization metadata from the anomaly results
                            localization = {
                                "anomaly_start": anomalies["localization_start"].iloc[0] if "localization_start" in anomalies.columns else None,
                                "anomaly_end": anomalies["localization_end"].iloc[0] if "localization_end" in anomalies.columns else None,
                                "scale_hours": anomalies["scale_hours"].iloc[0] if "scale_hours" in anomalies.columns else None,
                                "contrast_ratio": anomalies["contrast_ratio"].iloc[0] if "contrast_ratio" in anomalies.columns else None,
                            }
                            window_info = {
                                "week_num": week_num,
                                "start_time": str(window_start_ts),
                                "end_time": str(window_end_ts),
                                "score": score,
                                "high_error_points": anomalies.to_dict("records"),
                                "localization": localization,
                            }
                            data_store["flagged_windows"].append(window_info)

                            logger.info(
                                f"  RESULT: ANOMALY! Score: {score:.2f} > threshold"
                            )
                            logger.info(
                                f"  Total flagged windows: {len(data_store['flagged_windows'])}"
                            )

                            all_anomaly_points = []
                            for window in data_store["flagged_windows"]:
                                all_anomaly_points.extend(window["high_error_points"])
                            data_store["anomalies"] = pd.DataFrame(all_anomaly_points)
                            data_store["total_anomalies"] = len(
                                data_store["flagged_windows"]
                            )
                        else:
                            logger.info(f"  RESULT: Normal (score below threshold)")

                        data_store["last_detection"] = pd.Timestamp.now().isoformat()

                    weeks_processed_this_iteration += 1

                # Sleep before next check
                time.sleep(detection_interval)
                continue

            # Isolation Forest: Sliding window detection (original behavior)
            else:
                if len(data_list) < detector.min_samples_required:
                    logger.debug(
                        f"Waiting for more data: "
                        f"{len(data_list)}/{detector.min_samples_required}"
                    )
                    time.sleep(detection_interval)
                    continue

                df = pd.DataFrame(data_list)
                anomalies = detector.detect(df)

                with data_lock:
                    if not anomalies.empty:
                        data_store["anomalies"] = anomalies.tail(100)
                        data_store["total_anomalies"] = len(anomalies)
                    else:
                        data_store["anomalies"] = pd.DataFrame()
                        data_store["total_anomalies"] = 0

                    data_store["last_detection"] = pd.Timestamp.now().isoformat()

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")

        time.sleep(detection_interval)


# Theme colors
PRIMARY_COLOR = "#2c3e50"
SECONDARY_COLOR = "#3498db"
ANOMALY_COLOR = "#e74c3c"
SUCCESS_COLOR = "#27ae60"
BACKGROUND_COLOR = "#ecf0f1"

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "NYC Taxi Anomaly Detection"

# Health endpoint for producer coordination
@app.server.route("/health")
def health_check():
    """Return health status including Spark readiness."""
    from flask import jsonify
    with data_lock:
        spark_ready = data_store.get("spark_ready", False)
    return jsonify({
        "status": "ready" if spark_ready else "starting",
        "spark_ready": spark_ready,
        "detector": DETECTOR_TYPE,
    })

app.layout = dbc.Container([
    # Header
    dbc.Navbar(
        dbc.Container([
            dbc.NavbarBrand(
                "NYC Taxi Demand - Real-Time Anomaly Detection",
                style={"color": "white", "fontSize": "24px", "fontWeight": "bold"}
            ),
        ]),
        color=PRIMARY_COLOR,
        dark=True,
        sticky="top",
        className="mb-4"
    ),

    # Stream ended alert banner (hidden by default)
    dbc.Alert(
        [
            html.H4("Stream Complete", className="alert-heading"),
            html.P(
                "All data has been processed. No more incoming data.",
                className="mb-0"
            ),
        ],
        id="stream-ended-alert",
        color="warning",
        is_open=False,
        className="mb-4"
    ),

    # Status cards row 1 - labels differ for LSTM vs Isolation Forest
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Received", className="text-center"),
                html.H2(id="total-received", className="text-center text-success")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5(
                    "Weeks Processed" if DETECTOR_TYPE == "lstm" else "Buffer Size",
                    className="text-center"
                ),
                html.H2(id="window-size", className="text-center text-info")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5(
                    "Flagged Windows" if DETECTOR_TYPE == "lstm" else "Anomalies Detected",
                    className="text-center"
                ),
                html.H2(id="anomaly-count", className="text-center text-danger")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Detection Status", className="text-center"),
                html.P(id="detection-status", className="text-center text-muted", style={"fontSize": "14px"})
            ])
        ]), width=3),
    ], className="mb-4"),

    # Main time series chart with anomalies
    dbc.Card([
        dbc.CardHeader(
            html.H4("Taxi Demand with Anomaly Detection", className="mb-0"),
            style={"backgroundColor": SECONDARY_COLOR, "color": "white"}
        ),
        dbc.CardBody([
            dcc.Graph(id="stream-graph", style={"height": "400px"})
        ])
    ], className="mb-4"),

    # Two-column layout: Recent data and Anomalies
    dbc.Row([
        # Recent records table
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Recent Records", className="mb-0"),
                    style={"backgroundColor": PRIMARY_COLOR, "color": "white"}
                ),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="data-table",
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Value", "id": "value"},
                            {"name": "Seq ID", "id": "sequence_id"},
                        ],
                        style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                        style_header={"backgroundColor": PRIMARY_COLOR, "color": "white", "fontWeight": "bold"},
                        page_size=10
                    )
                ])
            ])
        ], width=6),

        # Anomalies table
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H4("Detected Anomalies", className="mb-0"),
                    style={"backgroundColor": ANOMALY_COLOR, "color": "white"}
                ),
                dbc.CardBody([
                    dash_table.DataTable(
                        id="anomaly-table",
                        columns=[
                            {"name": "Timestamp", "id": "timestamp"},
                            {"name": "Value", "id": "value"},
                            {"name": "Anomaly Score", "id": "anomaly_score"},
                        ],
                        style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
                        style_cell={"textAlign": "left", "padding": "8px", "fontSize": "12px"},
                        style_header={"backgroundColor": ANOMALY_COLOR, "color": "white", "fontWeight": "bold"},
                        style_data_conditional=[
                            {
                                "if": {"column_id": "anomaly_score"},
                                "color": ANOMALY_COLOR,
                                "fontWeight": "bold"
                            }
                        ],
                        page_size=10
                    )
                ])
            ])
        ], width=6),
    ], className="mb-4"),

    # Auto-refresh interval
    dcc.Interval(id="refresh", interval=2000, n_intervals=0),

    # Footer
    html.Footer(
        dbc.Container(
            html.P(
                f"{detector.get_name()} Anomaly Detection | "
                f"{'Non-Overlapping Weekly Windows' if DETECTOR_TYPE == 'lstm' else 'Sliding Window Analysis'}",
                className="text-center mb-0",
                style={"color": "white", "padding": "10px"}
            )
        ),
        style={"backgroundColor": PRIMARY_COLOR, "marginTop": "20px"}
    )
], fluid=True, style={"backgroundColor": BACKGROUND_COLOR, "minHeight": "100vh"})


@app.callback(
    [
        Output("stream-graph", "figure"),
        Output("data-table", "data"),
        Output("anomaly-table", "data"),
        Output("total-received", "children"),
        Output("window-size", "children"),
        Output("anomaly-count", "children"),
        Output("detection-status", "children"),
        Output("stream-ended-alert", "is_open"),
    ],
    [Input("refresh", "n_intervals")]
)
def update_dashboard(n):
    """Update all dashboard components with latest data."""
    with data_lock:
        data_list = list(data_store["data"])
        anomalies_df = data_store["anomalies"].copy() if not data_store["anomalies"].empty else pd.DataFrame()
        total = data_store["total_received"]
        total_anomalies = data_store["total_anomalies"]
        last_detection = data_store["last_detection"] or "Waiting..."
        # LSTM-specific
        weeks_processed = data_store.get("weeks_processed", 0)
        samples_since_last = data_store.get("samples_since_last_window", 0)
        flagged_windows = data_store.get("flagged_windows", [])
        # Stream status
        stream_ended = data_store.get("stream_ended", False)

    # Create figure
    fig = go.Figure()

    if data_list:
        # Limit visualization to most recent 3 weeks (1008 samples) for LSTM mode
        max_display_samples = VISUALIZATION_WEEKS * LSTM_WINDOW_SIZE if DETECTOR_TYPE == "lstm" else len(data_list)
        display_data = data_list[-max_display_samples:] if len(data_list) > max_display_samples else data_list
        df = pd.DataFrame(display_data)

        # Main time series line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Taxi Demand",
            line=dict(color=SECONDARY_COLOR, width=2),
            hovertemplate="<b>Time:</b> %{x}<br><b>Demand:</b> %{y}<extra></extra>"
        ))

        # Add markers for flagged anomaly weeks and localized 6hr windows
        if DETECTOR_TYPE == "lstm" and flagged_windows:
            visible_start = df["timestamp"].iloc[0]
            visible_end = df["timestamp"].iloc[-1]
            for window in flagged_windows:
                week_start = window.get("start_time")
                week_end = window.get("end_time")

                # Add red vertical lines for the flagged week boundaries
                if week_start and week_start >= visible_start and week_start <= visible_end:
                    fig.add_vline(
                        x=week_start,
                        line=dict(color=ANOMALY_COLOR, width=2, dash="dash"),
                        layer="above"
                    )
                if week_end and week_end >= visible_start and week_end <= visible_end:
                    fig.add_vline(
                        x=week_end,
                        line=dict(color=ANOMALY_COLOR, width=2, dash="dash"),
                        layer="above"
                    )

                # Add shaded 6hr localized region
                loc = window.get("localization", {})
                if loc.get("anomaly_start") and loc.get("anomaly_end"):
                    loc_start = loc["anomaly_start"]
                    loc_end = loc["anomaly_end"]
                    # Only show if within visible range
                    if loc_end >= visible_start and loc_start <= visible_end:
                        scale_hours = loc.get("scale_hours", "?")
                        contrast = loc.get("contrast_ratio", 0)
                        fig.add_vrect(
                            x0=loc_start,
                            x1=loc_end,
                            fillcolor=ANOMALY_COLOR,
                            opacity=0.15,
                            layer="below",
                            line_width=1,
                            line_color=ANOMALY_COLOR,
                            annotation_text=f"{scale_hours}h",
                            annotation_position="top left",
                            annotation=dict(font_size=10, font_color=ANOMALY_COLOR)
                        )

        # Overlay anomalies as red markers (only those within the visible window)
        if not anomalies_df.empty:
            # Filter anomalies to only show those in the current display window
            visible_start = df["timestamp"].iloc[0]
            visible_end = df["timestamp"].iloc[-1]
            visible_anomalies = anomalies_df[
                (anomalies_df["timestamp"] >= visible_start) &
                (anomalies_df["timestamp"] <= visible_end)
            ]
            if not visible_anomalies.empty:
                fig.add_trace(go.Scatter(
                    x=visible_anomalies["timestamp"],
                    y=visible_anomalies["value"],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(
                        color=ANOMALY_COLOR,
                        size=12,
                        symbol="x",
                        line=dict(width=2, color="white")
                    ),
                    hovertemplate="<b>ANOMALY</b><br>Time: %{x}<br>Value: %{y}<extra></extra>"
                ))

    fig.update_layout(
        xaxis=dict(title="Timestamp", showgrid=True, gridcolor="#e0e0e0"),
        yaxis=dict(title="Taxi Demand", showgrid=True, gridcolor="#e0e0e0"),
        legend=dict(x=0, y=1.1, orientation="h"),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=50, r=50, t=30, b=50)
    )

    if not data_list:
        fig.add_annotation(
            text="Waiting for data from Kafka...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="gray")
        )

    # Prepare table data
    recent_data = list(reversed(data_list[-10:])) if data_list else []

    anomaly_data = []
    if not anomalies_df.empty:
        anomalies_df = anomalies_df.copy()
        anomalies_df["timestamp"] = anomalies_df["timestamp"].astype(str)
        anomalies_df["anomaly_score"] = anomalies_df["anomaly_score"].round(4)
        anomaly_data = anomalies_df[["timestamp", "value", "anomaly_score"]].to_dict("records")

    # Detection status text and card values differ by detector type
    if DETECTOR_TYPE == "lstm":
        # LSTM: Show weeks processed and flagged windows
        if weeks_processed == 0:
            status_text = f"Buffering: {samples_since_last}/{LSTM_WINDOW_SIZE}"
        else:
            status_text = f"Week {weeks_processed} | Next: {samples_since_last}/{LSTM_WINDOW_SIZE}"

        window_size_text = f"{weeks_processed}"
        anomaly_count_text = f"{len(flagged_windows)}"
    else:
        # Isolation Forest: Original behavior
        if last_detection == "Waiting...":
            status_text = "Waiting for enough data..."
        else:
            status_text = f"Last run: {last_detection[-8:]}"

        window_size_text = f"{len(data_list)}"
        anomaly_count_text = f"{total_anomalies}"

    return (
        fig,
        recent_data,
        anomaly_data,
        f"{total:,}",
        window_size_text,
        anomaly_count_text,
        status_text,
        stream_ended,
    )


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NYC Taxi Anomaly Detection Dashboard")
    logger.info("=" * 60)
    logger.info(f"Detector: {detector.get_name()}")
    logger.info(f"Detector Ready: {detector.is_ready}")
    logger.info(f"Min Samples Required: {detector.min_samples_required}")
    for key, value in detector.get_stats().items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)

    # Start Spark Streaming in background thread
    streaming_thread = threading.Thread(target=start_spark_streaming, daemon=True)
    streaming_thread.start()

    # Start anomaly detection in background thread
    detection_thread = threading.Thread(target=run_anomaly_detection, daemon=True)
    detection_thread.start()

    # Run Dash app
    logger.info("Starting Dash server on http://0.0.0.0:8050")
    app.run(debug=False, host="0.0.0.0", port=8050)
