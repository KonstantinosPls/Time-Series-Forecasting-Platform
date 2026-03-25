import os
import time
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine, text

DB_USER = os.environ.get("POSTGRES_USER", "admin")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "admin")
DB_HOST = os.environ.get("POSTGRES_HOST", "db")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "timeseries")
SERIES_NAME = os.environ.get("SERIES_NAME", "weather_Athens")
FORECAST_INTERVAL = int(os.environ.get("FORECAST_INTERVAL", 3600))
MIN_DATA_POINTS = int(os.environ.get("MIN_DATA_POINTS", 24))
HORIZON = int(os.environ.get("FORECAST_HORIZON", 72))

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def get_series_data():
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT timestamp, value FROM time_series_data "
                 "WHERE series_name = :name ORDER BY timestamp"),
            conn, params={"name": SERIES_NAME}
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset="timestamp", keep="first")
    return df

def run_forecast(df):
    from models import chronos_model, timesfm_model, lagllama_model
    from ensemble import weighted_average, equal_weights
    from anomaly import detect_anomalies

    # Aggregate to hourly for forecasting (raw data kept for anomaly detection)
    hourly = df.set_index("timestamp").resample("1h")["value"].mean().dropna().reset_index()
    print(f"  Aggregated to {len(hourly)} hourly data points.", flush=True)

    series = hourly["value"]
    timestamps = hourly["timestamp"]
    lagllama_threshold = HORIZON * 2

    # Run Chronos
    print("  Running Chronos-2 Bolt...", flush=True)
    chrono = chronos_model.load_model()
    chronos_forecast = chronos_model.forecast(chrono, series, HORIZON)

    # Run TimesFM
    print("  Running TimesFM...", flush=True)
    tfm = timesfm_model.load_model()
    timesfm_forecast = timesfm_model.forecast(tfm, series, HORIZON)

    forecasts = [chronos_forecast, timesfm_forecast]
    model_names = ["chronos", "timesfm"]

    # Run Lag-Llama only with enough data
    if len(series) >= lagllama_threshold:
        print(f"  Running Lag-Llama ({len(series)} >= {lagllama_threshold} threshold)...", flush=True)
        llama_kwargs = lagllama_model.load_model()
        lagllama_forecast = lagllama_model.forecast(llama_kwargs, series, timestamps, HORIZON)
        forecasts.append(lagllama_forecast)
        model_names.append("lagllama")
    else:
        print(f"  Skipping Lag-Llama ({len(series)} < {lagllama_threshold} threshold).", flush=True)

    # Equal weights ensemble
    weights = equal_weights(len(forecasts))
    print(f"  Weights: {dict(zip(model_names, [f'{w:.1%}' for w in weights]))}", flush=True)

    print("  Computing ensemble...", flush=True)
    ensemble = weighted_average(forecasts, weights)

    # Generate future timestamps
    freq = pd.infer_freq(timestamps) or "2min"
    future_ts = pd.date_range(start=timestamps.iloc[-1], periods=HORIZON + 1, freq=freq)[1:]

    # Save forecasts
    all_forecasts = [(name, f) for name, f in zip(model_names, forecasts)]
    all_forecasts.append(("ensemble", ensemble))

    run_time = datetime.now(timezone.utc)
    for model_name, (median, lower, upper) in all_forecasts:
        forecast_df = pd.DataFrame({
            "series_name": SERIES_NAME,
            "model_name": model_name,
            "timestamp": future_ts[:len(median)],
            "value": median,
            "lower_bound": lower,
            "upper_bound": upper,
            "created_at": run_time
        })
        forecast_df.to_sql("forecasts", engine, if_exists="append", index=False)

    # Anomaly detection (on raw data, not hourly)
    print("  Running anomaly detection...", flush=True)
    raw_series = df["value"]
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM anomalies WHERE series_name = :name"), {"name": SERIES_NAME})
    labels, scores = detect_anomalies(raw_series.values)
    anomaly_df = pd.DataFrame({
        "series_name": SERIES_NAME,
        "timestamp": df["timestamp"],
        "value": raw_series,
        "anomaly_score": scores,
        "is_anomaly": labels.astype(bool)
    })
    anomaly_df.to_sql("anomalies", engine, if_exists="append", index=False)

    anomaly_count = int(labels.sum())
    print(f"  Forecast complete. {anomaly_count} anomalies detected.", flush=True)

if __name__ == "__main__":
    print(f"Scheduler started. Forecasting '{SERIES_NAME}' every {FORECAST_INTERVAL}s.", flush=True)
    print(f"Minimum data points required: {MIN_DATA_POINTS}", flush=True)
    print(f"Forecast horizon: {HORIZON} steps", flush=True)

    while True:
        try:
            df = get_series_data()
            print(f"\n[{datetime.now(timezone.utc).isoformat()}] "
                  f"Found {len(df)} data points for '{SERIES_NAME}'.", flush=True)

            if len(df) >= MIN_DATA_POINTS:
                print("  Enough data. Running forecast pipeline...", flush=True)
                run_forecast(df)
            else:
                print(f"  Waiting for {MIN_DATA_POINTS - len(df)} more data points.", flush=True)

        except Exception as e:
            print(f"  Error: {e}", flush=True)

        time.sleep(FORECAST_INTERVAL)
