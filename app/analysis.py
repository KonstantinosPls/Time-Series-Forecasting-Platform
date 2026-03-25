import requests

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"

SYSTEM_PROMPT = (
    "You are a time-series analysis report writer. Rules:\n"
    "1. ONLY use numbers and facts explicitly provided in the input.\n"
    "2. Do NOT invent, estimate, or infer any numbers not given.\n"
    "3. Do NOT reference models or metrics not listed in the input.\n"
    "4. Do NOT answer questions or have conversations.\n"
    "5. If a section is missing from the input, skip it entirely.\n"
    "6. Write in a professional, concise tone.\n"
    "7. If no data is provided, respond only with: 'No data provided for analysis.'"
)


def _build_prompt(data_summary, forecast_data, anomaly_data, backtest_data):
    sections = []

    if data_summary:
        sections.append(
            f"DATA SUMMARY:\n"
            f"- Series name: {data_summary.get('series_name', 'N/A')}\n"
            f"- Total observations: {data_summary.get('total_rows', 'N/A')}\n"
            f"- Time range: {data_summary.get('time_range', 'N/A')}\n"
            f"- Detected frequency: {data_summary.get('frequency', 'N/A')}\n"
            f"- Mean: {data_summary['statistics']['mean']:.2f}\n"
            f"- Std: {data_summary['statistics']['std']:.2f}\n"
            f"- Min: {data_summary['statistics']['min']:.2f}\n"
            f"- Max: {data_summary['statistics']['max']:.2f}"
        )

    if forecast_data:
        horizon = forecast_data["horizon"]
        preds_detail = []
        for name, vals in forecast_data["predictions"].items():
            first = vals[0]
            last = vals[-1]
            avg = sum(vals) / len(vals)
            trend = "rising" if last > first else "falling" if last < first else "stable"
            preds_detail.append(
                f"  {name}: first={first:.2f}, last={last:.2f}, avg={avg:.2f}, trend={trend}"
            )
        preds_str = "\n".join(preds_detail)

        # Model agreement
        first_vals = [vals[0] for vals in forecast_data["predictions"].values()]
        spread = max(first_vals) - min(first_vals)

        sections.append(
            f"FORECAST RESULTS (horizon = {horizon} steps):\n"
            f"{preds_str}\n"
            f"- Model spread (max - min first prediction): {spread:.2f}\n"
            f"- A large spread means models disagree significantly"
        )

    if anomaly_data:
        top_points = "\n".join(
            f"  {p['timestamp']}: value={p['value']:.2f}, score={p['score']:.4f}"
            for p in anomaly_data["points"][:5]
        )
        sections.append(
            f"ANOMALY DETECTION:\n"
            f"- Total anomalies: {anomaly_data['count']}\n"
            f"- Anomaly rate: {anomaly_data['rate']:.1f}%\n"
            f"- Top 5 anomalous points:\n{top_points}"
        )

    if backtest_data:
        metrics = "\n".join(
            f"  {name}: MAE = {m['mae']:.4f}, MASE = {m['mase']:.4f}"
            for name, m in backtest_data["metrics"].items()
        )
        # Pre-interpret MASE for the model
        interpretations = "\n".join(
            f"  {name}: {'beats' if m['mase'] < 1.0 else 'does NOT beat'} naive forecast"
            for name, m in backtest_data["metrics"].items()
        )
        sections.append(
            f"BACKTEST RESULTS (80% train / 20% test):\n"
            f"- Best model: {backtest_data['best_model']}\n"
            f"- Metrics (lower is better):\n{metrics}\n"
            f"- MASE definition: MASE compares the model's error against a naive forecast. "
            f"A naive forecast predicts that the next value equals the last observed value "
            f"(NOT the mean). MASE < 1.0 means the model is better than naive. "
            f"MASE >= 1.0 means the model is equal to or worse than naive.\n"
            f"- MASE interpretation:\n{interpretations}"
        )

    context = "\n\n".join(sections)

    return (
        f"Write a 4-5 paragraph analysis based STRICTLY on the data below. "
        f"Structure: (1) Brief data overview, (2) DETAILED forecast interpretation - "
        f"describe what the predictions mean for the next {forecast_data['horizon'] if forecast_data else 'N'} steps, "
        f"whether values are expected to rise/fall/stay stable, how much models agree, "
        f"and what range the user should expect, (3) anomaly summary, "
        f"(4) model reliability based on backtest, (5) actionable recommendations. "
        f"FOCUS MOST on the forecast interpretation - this is what the user cares about. "
        f"Use ONLY the exact numbers provided. Do not add any numbers, "
        f"metrics, or model names that are not listed below.\n\n"
        f"--- BEGIN DATA ---\n{context}\n--- END DATA ---"
    )


def generate_analysis(data_summary, forecast_data, anomaly_data, backtest_data):
    prompt = _build_prompt(data_summary, forecast_data, anomaly_data, backtest_data)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "qwen2.5:3b",
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1200}
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama is running on your host machine."
    except Exception as e:
        return f"Error generating analysis: {str(e)}"
