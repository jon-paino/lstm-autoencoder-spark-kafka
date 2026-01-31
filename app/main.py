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

from isolation_forest_detector import IsolationForestDetector, DetectorConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Thread-safe data storage
data_lock = threading.Lock()
WINDOW_SIZE = 200

data_store = {
    "data": deque(maxlen=WINDOW_SIZE),
    "anomalies": pd.DataFrame(),
    "total_received": 0,
    "total_anomalies": 0,
    "last_batch_size": 0,
    "last_update": None,
    "last_detection": None,
}

# Initialize detector with configuration
detector_config = DetectorConfig(
    window_size=WINDOW_SIZE,
    min_samples=50,
    contamination=0.05,
    n_estimators=100,
)
detector = IsolationForestDetector(config=detector_config)


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
        .option("startingOffsets", "latest") \
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

            logger.info(f"Batch {batch_id}: received {len(pandas_df)} records")

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

        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")

    query = df_parsed.writeStream \
        .trigger(processingTime="2 seconds") \
        .foreachBatch(process_batch) \
        .start()

    logger.info("Spark Streaming started - waiting for data...")
    query.awaitTermination()


def run_anomaly_detection():
    """
    Background thread that periodically runs anomaly detection
    on the sliding window of data.
    """
    detection_interval = 5  # seconds

    logger.info("Starting anomaly detection thread")
    logger.info(f"Detection interval: {detection_interval}s")
    logger.info(f"Detector config: {detector.get_stats()}")

    while True:
        try:
            with data_lock:
                data_list = list(data_store["data"])

            if len(data_list) < detector_config.min_samples:
                logger.debug(
                    f"Waiting for more data: {len(data_list)}/{detector_config.min_samples}"
                )
                time.sleep(detection_interval)
                continue

            # Convert to DataFrame for detection
            df = pd.DataFrame(data_list)

            # Run detection
            anomalies = detector.detect(df)

            with data_lock:
                if not anomalies.empty:
                    # Keep only recent anomalies (last 100)
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

    # Status cards row 1
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Total Received", className="text-center"),
                html.H2(id="total-received", className="text-center text-success")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Window Size", className="text-center"),
                html.H2(id="window-size", className="text-center text-info")
            ])
        ]), width=3),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H5("Anomalies Detected", className="text-center"),
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
                "Isolation Forest Anomaly Detection | Sliding Window Analysis",
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

    # Create figure
    fig = go.Figure()

    if data_list:
        df = pd.DataFrame(data_list)

        # Main time series line
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Taxi Demand",
            line=dict(color=SECONDARY_COLOR, width=2),
            hovertemplate="<b>Time:</b> %{x}<br><b>Demand:</b> %{y}<extra></extra>"
        ))

        # Overlay anomalies as red markers
        if not anomalies_df.empty:
            fig.add_trace(go.Scatter(
                x=anomalies_df["timestamp"],
                y=anomalies_df["value"],
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

    # Detection status text
    if last_detection == "Waiting...":
        status_text = "Waiting for enough data..."
    else:
        status_text = f"Last run: {last_detection[-8:]}"

    return (
        fig,
        recent_data,
        anomaly_data,
        f"{total:,}",
        f"{len(data_list)}",
        f"{total_anomalies}",
        status_text
    )


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NYC Taxi Anomaly Detection Dashboard")
    logger.info("=" * 60)
    logger.info(f"Detector: Isolation Forest")
    logger.info(f"Window Size: {WINDOW_SIZE}")
    logger.info(f"Contamination: {detector_config.contamination}")
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
