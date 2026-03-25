import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from validation import validate_csv, preprocess
from db import get_engine
from models import chronos_model, timesfm_model, lagllama_model
from ensemble import weighted_average, equal_weights

st.set_page_config(page_title="Time-Series Forecasting Platform", layout="wide")

st.title("Time-Series Forecasting Platform")
st.markdown("Upload time-series data, generate forecasts using three foundation models, "
            "detect anomalies, and evaluate model performance.")

# Initialize session state
for key in ["df", "series_name", "forecast", "anomaly", "backtest", "data_summary"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.divider()


def static_chart(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# Data Source
st.header("1. Load Data")
st.markdown("Upload a CSV file or load streamed data from the database.")

data_source = st.radio("Data source", ["Upload CSV", "Load from database"], horizontal=True)

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file:
        raw_df = pd.read_csv(uploaded_file)

        # Column selection for multi-column CSVs
        ts_col, val_col = None, None
        if raw_df.shape[1] > 2:
            st.markdown("**Select columns**")
            sel1, sel2 = st.columns(2)
            with sel1:
                ts_col = st.selectbox("Timestamp column", raw_df.columns)
            with sel2:
                val_col = st.selectbox("Value column", [c for c in raw_df.columns if c != ts_col])

        df, error = validate_csv(raw_df, ts_col, val_col)

        if error:
            st.error(error)
        else:
            df = preprocess(df)
            st.session_state.df = df
            st.session_state.series_name = uploaded_file.name.replace(".csv", "")

            freq = pd.infer_freq(df["timestamp"])
            stats = df["value"].describe()

            st.session_state.data_summary = {
                "series_name": st.session_state.series_name,
                "total_rows": len(df),
                "time_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "frequency": freq or "Could not detect",
                "columns": ["timestamp (datetime64[ns]), 0 nulls", "value (float64), 0 nulls"],
                "statistics": {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "25%": stats["25%"],
                    "50%": stats["50%"],
                    "75%": stats["75%"],
                    "max": stats["max"]
                }
            }

else:
    from sqlalchemy import text
    engine = get_engine()
    with engine.connect() as conn:
        series_list = conn.execute(
            text("SELECT DISTINCT series_name FROM time_series_data ORDER BY series_name")
        ).fetchall()

    series_names = [row[0] for row in series_list]

    if not series_names:
        st.warning("No data in the database yet. Start the Kafka streaming pipeline or upload a CSV first.")
    else:
        selected_series = st.selectbox("Select a data series", series_names)

        if st.button("Load data"):
            with engine.connect() as conn:
                raw_df = pd.read_sql(
                    text("SELECT timestamp, value FROM time_series_data WHERE series_name = :name ORDER BY timestamp"),
                    conn, params={"name": selected_series}
                )

            raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
            raw_df = raw_df.drop_duplicates(subset="timestamp", keep="first")
            df = raw_df.copy()
            st.session_state.df = df
            st.session_state.series_name = selected_series

            freq = pd.infer_freq(df["timestamp"])
            stats = df["value"].describe()

            st.session_state.data_summary = {
                "series_name": selected_series,
                "total_rows": len(df),
                "time_range": f"{df['timestamp'].min()} to {df['timestamp'].max()}",
                "frequency": freq or "Could not detect",
                "columns": ["timestamp (datetime64[ns]), 0 nulls", "value (float64), 0 nulls"],
                "statistics": {
                    "mean": stats["mean"],
                    "std": stats["std"],
                    "min": stats["min"],
                    "25%": stats["25%"],
                    "50%": stats["50%"],
                    "75%": stats["75%"],
                    "max": stats["max"]
                }
            }

# Display data summary if loaded
if st.session_state.df is not None and st.session_state.data_summary:
    df = st.session_state.df
    summary = st.session_state.data_summary

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Data preview**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.markdown("**Column info**")
        for info in summary["columns"]:
            st.text(info)
        st.metric("Detected frequency", summary["frequency"])
        st.metric("Total rows", summary["total_rows"])
    with col3:
        st.markdown("**Statistics**")
        for k, v in summary["statistics"].items():
            st.text(f"{k}: {v:.2f}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(df["timestamp"], df["value"], color="#1f77b4", linewidth=1)
    ax.set_title(f"Data: {summary['series_name']}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    static_chart(fig)

    series_name = st.text_input("Series name", value=st.session_state.series_name)
    st.session_state.series_name = series_name


# Only show remaining sections if data is loaded
if st.session_state.df is not None:
    df = st.session_state.df
    series_name = st.session_state.series_name

    st.divider()

    # Forecast
    st.header("2. Forecast and Model Evaluation")
    st.markdown("Run three foundation models on your data, combine their predictions "
                "into a weighted ensemble forecast, and evaluate model reliability.")

    freq = pd.infer_freq(df["timestamp"])
    freq_label = {"D": "days", "h": "hours", "min": "minutes", "W": "weeks", "MS": "months",
                  "M": "months", "H": "hours", "T": "minutes", "S": "seconds"}.get(freq, "steps")
    last_ts = df["timestamp"].iloc[-1]

    horizon = st.slider(f"Forecast horizon ({freq_label} ahead)", min_value=1, max_value=48, value=12,
                       help=f"Each step = 1 {freq_label.rstrip('s')}. Data frequency: {freq or 'unknown'}.")

    if freq:
        future_end = last_ts + pd.tseries.frequencies.to_offset(freq) * horizon
        st.markdown(f"**Forecasting from** {last_ts.strftime('%Y-%m-%d %H:%M')} **to** {future_end.strftime('%Y-%m-%d %H:%M')}")

    if st.button("Run forecast"):
        from backtest import train_test_split, evaluate_model

        series = df["value"]
        timestamps = df["timestamp"]
        progress = st.progress(0, text="Loading models...")

        # Forecast
        progress.progress(5, text="Running Chronos-2 Bolt...")
        chrono = chronos_model.load_model()
        chronos_forecast = chronos_model.forecast(chrono, series, horizon)

        progress.progress(20, text="Running TimesFM 2.0...")
        tfm = timesfm_model.load_model()
        timesfm_forecast = timesfm_model.forecast(tfm, series, horizon)

        progress.progress(35, text="Running Lag-Llama...")
        llama_kwargs = lagllama_model.load_model()
        lagllama_forecast = lagllama_model.forecast(
            llama_kwargs, series, timestamps, horizon
        )

        progress.progress(50, text="Computing ensemble...")

        forecasts = [chronos_forecast, timesfm_forecast, lagllama_forecast]
        weights = equal_weights(len(forecasts))
        ensemble = weighted_average(forecasts, weights)

        last_ts = timestamps.iloc[-1]
        freq = pd.infer_freq(timestamps)
        future_ts = pd.date_range(start=last_ts, periods=horizon + 1, freq=freq)[1:]

        # Save raw data and forecasts
        engine = get_engine()
        save_df = df.copy()
        save_df["series_name"] = series_name
        save_df.to_sql("time_series_data", engine, if_exists="append", index=False)

        all_forecasts = [
            ("chronos", chronos_forecast),
            ("timesfm", timesfm_forecast),
            ("lagllama", lagllama_forecast),
            ("ensemble", ensemble)
        ]

        for model_name, (median, lower, upper) in all_forecasts:
            forecast_df = pd.DataFrame({
                "series_name": series_name,
                "model_name": model_name,
                "timestamp": future_ts,
                "value": median,
                "lower_bound": lower,
                "upper_bound": upper
            })
            forecast_df.to_sql("forecasts", engine, if_exists="append", index=False)

        # Store forecast in session state
        st.session_state.forecast = {
            "horizon": horizon,
            "timestamps": future_ts.tolist(),
            "predictions": {n: m.tolist() for n, (m, _, _) in all_forecasts}
        }

        # Backtest
        progress.progress(55, text="Backtesting Chronos-2 Bolt...")
        train_vals, test_vals = train_test_split(df["value"].values)
        train_ts, test_ts = train_test_split(df["timestamp"].values)
        results = {}

        fn = lambda vals, h: chronos_model.forecast(chrono, pd.Series(vals), h)
        results["chronos"] = evaluate_model(fn, train_vals, test_vals)

        progress.progress(70, text="Backtesting TimesFM 2.0...")
        fn = lambda vals, h: timesfm_model.forecast(tfm, pd.Series(vals), h)
        results["timesfm"] = evaluate_model(fn, train_vals, test_vals)

        progress.progress(85, text="Backtesting Lag-Llama...")
        fn = lambda vals, h: lagllama_model.forecast(
            llama_kwargs, pd.Series(vals), pd.Series(train_ts), h
        )
        results["lagllama"] = evaluate_model(fn, train_vals, test_vals)

        best_model = min(results, key=lambda k: results[k]["mae"])
        pred_len = len(next(iter(results.values()))["predicted"])
        actual_aligned = test_vals[:pred_len]
        ts_aligned = test_ts[:pred_len]

        st.session_state.backtest = {
            "metrics": {n: {"mae": r["mae"], "mase": r["mase"]} for n, r in results.items()},
            "best_model": best_model,
            "actual": actual_aligned.tolist(),
            "test_ts": [str(t) for t in ts_aligned],
            "results": {n: {"predicted": r["predicted"].tolist()} for n, r in results.items()}
        }

        progress.progress(100, text="Done.")

    # Display forecast results if available
    if st.session_state.forecast:
        fc = st.session_state.forecast
        st.success("Forecasts generated and saved to database.")

        # Display summary for each model
        st.markdown("**Forecast summary (next step prediction)**")
        summary_cols = st.columns(4)
        for i, (name, vals) in enumerate(fc["predictions"].items()):
            with summary_cols[i]:
                st.metric(name.capitalize(), f"{vals[0]:.2f}")

        # Forecast chart
        st.markdown("**Forecast chart**")
        fig, ax = plt.subplots(figsize=(12, 4))
        for name, vals in fc["predictions"].items():
            ax.plot(range(len(vals)), vals, label=name.capitalize(), linewidth=1.5)
        ax.set_title(f"Forecast ({fc['horizon']} steps ahead)")
        ax.set_xlabel("Steps ahead")
        ax.set_ylabel("Predicted value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        static_chart(fig)

    # Display backtest results under forecast
    if st.session_state.backtest:
        bt = st.session_state.backtest

        with st.expander("Model reliability (backtest results)", expanded=True):
            st.markdown("**Model comparison** (sorted by MAE, lower is better)")
            metrics_df = pd.DataFrame(bt["metrics"]).T.sort_values("mae")
            metrics_df.columns = ["MAE", "MASE"]
            st.dataframe(metrics_df, use_container_width=True)

            st.markdown(f"**Best model: {bt['best_model'].capitalize()}** with MAE of "
                       f"{bt['metrics'][bt['best_model']]['mae']:.4f}")

            st.markdown("**What the metrics mean:**")
            st.markdown("- **MAE** (Mean Absolute Error): Average prediction error in data units. Lower is better.")
            st.markdown("- **MASE** (Mean Absolute Scaled Error): Compares against a naive forecast. "
                       "Below 1.0 means the model outperforms naive prediction.")

            # Backtest chart
            st.markdown("**Actual vs predicted**")
            fig, ax = plt.subplots(figsize=(12, 4))
            x = range(len(bt["actual"]))
            ax.plot(x, bt["actual"], label="Actual", color="#1f77b4", linewidth=2)
            colors = {"chronos": "#ff7f0e", "timesfm": "#2ca02c", "lagllama": "#d62728"}
            for name, r in bt["results"].items():
                ax.plot(x, r["predicted"], label=name.capitalize(),
                       color=colors.get(name, "gray"), linewidth=1, alpha=0.8)
            ax.set_title("Actual vs predicted (backtest)")
            ax.set_xlabel("Test steps")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            static_chart(fig)

    st.divider()

    # Anomaly Detection
    st.header("3. Anomaly Detection")
    st.markdown("Detect unusual data points using Isolation Forest. "
                "Anomalies are values that deviate significantly from the expected pattern.")

    if st.button("Detect anomalies"):
        from anomaly import detect_anomalies

        labels, scores = detect_anomalies(df["value"].values)

        engine = get_engine()
        anomaly_df = pd.DataFrame({
            "series_name": series_name,
            "timestamp": df["timestamp"],
            "value": df["value"],
            "anomaly_score": scores,
            "is_anomaly": labels.astype(bool)
        })
        anomaly_df.to_sql("anomalies", engine, if_exists="append", index=False)

        anomaly_count = int(labels.sum())

        st.session_state.anomaly = {
            "count": anomaly_count,
            "rate": anomaly_count / len(labels) * 100,
            "points": [
                {"timestamp": str(row["timestamp"]), "value": row["value"], "score": scores[idx]}
                for idx, (_, row) in enumerate(df.iterrows()) if labels[idx] == 1
            ],
            "labels": labels.tolist(),
            "scores": scores.tolist()
        }

    # Display anomaly results if available
    if st.session_state.anomaly:
        an = st.session_state.anomaly
        st.success("Anomaly detection complete and saved to database.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomalies found", an["count"])
        with col2:
            st.metric("Anomaly rate", f"{an['rate']:.1f}%")

        # Static chart with anomalies highlighted
        labels_arr = np.array(an["labels"])
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df["timestamp"], df["value"], color="#1f77b4", linewidth=1, label="Value")
        anomaly_mask = labels_arr == 1
        ax.scatter(
            df["timestamp"][anomaly_mask], df["value"][anomaly_mask],
            color="red", s=60, zorder=5, label="Anomaly"
        )
        ax.set_title("Data with anomalies highlighted")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        static_chart(fig)

        if an["points"]:
            st.markdown("**Anomalous data points**")
            st.dataframe(pd.DataFrame(an["points"]), use_container_width=True)

    st.divider()

    # LLM Analysis
    st.header("4. AI Analysis")
    st.markdown("Generate a written analysis of all results using a local language model (Qwen 2.5 3B).")

    if "analysis_text" not in st.session_state:
        st.session_state["analysis_text"] = None

    has_results = any([st.session_state.forecast, st.session_state.anomaly, st.session_state.backtest])

    if has_results:
        if st.button("Generate analysis"):
            from analysis import generate_analysis

            with st.spinner("Generating analysis with Qwen 2.5..."):
                text = generate_analysis(
                    data_summary=st.session_state.data_summary,
                    forecast_data=st.session_state.forecast,
                    anomaly_data=st.session_state.anomaly,
                    backtest_data=st.session_state.backtest
                )
                st.session_state["analysis_text"] = text

        if st.session_state["analysis_text"]:
            st.markdown(st.session_state["analysis_text"])
    else:
        st.info("Run at least one analysis (forecast, anomaly detection, or backtest) to generate an AI summary.")

    st.divider()

    # Download Report
    st.header("5. Download Report")
    st.markdown("Generate a PDF report containing all results and the AI analysis.")

    if has_results:
        if st.button("Generate PDF report"):
            from report import generate_pdf

            pdf_bytes = generate_pdf(
                data_summary=st.session_state.data_summary or {},
                forecast_data=st.session_state.forecast,
                anomaly_data=st.session_state.anomaly,
                backtest_data=st.session_state.backtest,
                analysis_text=st.session_state.get("analysis_text")
            )
            st.session_state["pdf_bytes"] = pdf_bytes

        if st.session_state.get("pdf_bytes"):
            st.download_button(
                label="Download PDF report",
                data=st.session_state["pdf_bytes"],
                file_name=f"{series_name}_report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("Run at least one analysis (forecast, anomaly detection, or backtest) to generate a report.")
