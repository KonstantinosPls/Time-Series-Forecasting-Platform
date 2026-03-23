import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
import numpy as np
import tempfile
import os


def _create_chart(x, y_dict, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, values in y_dict.items():
        ax.plot(x, values, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return tmp.name


def generate_pdf(data_summary, forecast_data, anomaly_data, backtest_data, analysis_text=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 40, "Time-Series Forecasting Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Series: {data_summary.get('series_name', 'N/A')}", ln=True, align="C")
    pdf.cell(0, 10, f"Total data points: {data_summary.get('total_rows', 'N/A')}", ln=True, align="C")
    pdf.cell(0, 10, f"Time range: {data_summary.get('time_range', 'N/A')}", ln=True, align="C")
    pdf.cell(0, 10, f"Detected frequency: {data_summary.get('frequency', 'N/A')}", ln=True, align="C")

    # Data summary section
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Data Summary", ln=True)
    pdf.ln(5)

    if "statistics" in data_summary:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(40, 8, "Statistic", border=1)
        pdf.cell(40, 8, "Value", border=1, ln=True)
        pdf.set_font("Helvetica", "", 11)
        for key, val in data_summary["statistics"].items():
            pdf.cell(40, 8, str(key), border=1)
            pdf.cell(40, 8, f"{val:.4f}" if isinstance(val, float) else str(val), border=1, ln=True)

    if "columns" in data_summary:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Columns:", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for col_info in data_summary["columns"]:
            pdf.cell(0, 7, f"  {col_info}", ln=True)

    # Forecast section
    if forecast_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "2. Forecast Results", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Horizon: {forecast_data['horizon']} steps", ln=True)
        pdf.ln(3)

        # Per-model first prediction
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Model", border=1)
        pdf.cell(50, 8, "First prediction", border=1, ln=True)
        pdf.set_font("Helvetica", "", 11)
        for name, vals in forecast_data["predictions"].items():
            pdf.cell(50, 8, name.capitalize(), border=1)
            pdf.cell(50, 8, f"{vals[0]:.2f}", border=1, ln=True)

        # Chart
        if "timestamps" in forecast_data:
            chart_path = _create_chart(
                range(len(forecast_data["timestamps"])),
                {n: v for n, v in forecast_data["predictions"].items()},
                "Forecast", "Steps ahead", "Value"
            )
            pdf.ln(5)
            pdf.image(chart_path, w=180)
            os.unlink(chart_path)

    # Anomaly section
    if anomaly_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "3. Anomaly Detection", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, f"Anomalies found: {anomaly_data['count']}", ln=True)
        pdf.cell(0, 8, f"Anomaly rate: {anomaly_data['rate']:.1f}%", ln=True)
        pdf.ln(3)

        if anomaly_data["points"]:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(60, 8, "Timestamp", border=1)
            pdf.cell(40, 8, "Value", border=1)
            pdf.cell(40, 8, "Score", border=1, ln=True)
            pdf.set_font("Helvetica", "", 10)
            for pt in anomaly_data["points"]:
                pdf.cell(60, 8, str(pt["timestamp"]), border=1)
                pdf.cell(40, 8, f"{pt['value']:.2f}", border=1)
                pdf.cell(40, 8, f"{pt['score']:.4f}", border=1, ln=True)

    # Backtest section
    if backtest_data:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "4. Model Evaluation (Backtest)", ln=True)
        pdf.ln(5)

        pdf.set_font("Helvetica", "", 11)
        pdf.cell(0, 8, "Data split: 80% train / 20% test", ln=True)
        pdf.cell(0, 8, f"Best model: {backtest_data['best_model'].capitalize()}", ln=True)
        pdf.ln(3)

        # Metrics table
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(50, 8, "Model", border=1)
        pdf.cell(40, 8, "MAE", border=1)
        pdf.cell(40, 8, "MASE", border=1, ln=True)
        pdf.set_font("Helvetica", "", 11)
        for name, metrics in backtest_data["metrics"].items():
            pdf.cell(50, 8, name.capitalize(), border=1)
            pdf.cell(40, 8, f"{metrics['mae']:.4f}", border=1)
            pdf.cell(40, 8, f"{metrics['mase']:.4f}", border=1, ln=True)

        pdf.ln(5)
        pdf.set_x(10)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Metric definitions:", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "MAE (Mean Absolute Error): Average prediction error in data units. Lower is better.")
        pdf.set_x(10)
        pdf.multi_cell(0, 6, "MASE (Mean Absolute Scaled Error): Compares against naive forecast. "
                       "Below 1.0 means the model outperforms naive prediction.")

        # Chart
        if "actual" in backtest_data:
            chart_data = {"actual": backtest_data["actual"]}
            chart_data.update({n: r["predicted"] for n, r in backtest_data["results"].items()})
            chart_path = _create_chart(
                range(len(backtest_data["actual"])),
                chart_data,
                "Actual vs Predicted", "Steps", "Value"
            )
            pdf.ln(5)
            pdf.image(chart_path, w=180)
            os.unlink(chart_path)

    # LLM Analysis section
    if analysis_text:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "5. AI Analysis", ln=True)
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_x(10)
        pdf.multi_cell(0, 6, analysis_text)

    return bytes(pdf.output())
