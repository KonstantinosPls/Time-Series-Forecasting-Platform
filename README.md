# Multi-Model Time-Series Forecasting Platform

A forecasting platform that accepts univariate time-series CSV data, runs three foundation models, produces ensemble predictions with confidence intervals, detects anomalies, and generates AI analysis reports.


![Data Upload](Screenshots/Data_Upload.png)
![Forecast](Screenshots/Forecast.png)
![Model Evaluation](Screenshots/Model_Evaluation.png)
![Anomaly Detection](Screenshots/Anomaly_Detection.png)
![AI Analysis](Screenshots/AI_Analysis.png)

## What the platform does


1. Validates and preprocesses the data (gap filling, resampling, deduplication).
2. Runs three forecasting models and combines them into an ensemble prediction.
3. Detects anomalies using Isolation Forest.
4. Evaluates model reliability via backtesting (MAE, MASE metrics).
5. Generates a written analysis using a local language model (Qwen 2.5 3B via Ollama).
6. Exports a full PDF report with charts, metrics, and AI analysis.

## Forecasting Models

| Model | Type | Parameters | Description |
|-------|------|------------|-------------|
| Chronos-2 Bolt | Encoder-decoder transformer | ~70M | Amazon's pre-trained time-series model. Best overall accuracy in testing. |
| TimesFM 2.0 | Patched decoder transformer | 500M | Google's foundation model for time-series. Configured with a 128-step horizon. |
| Lag-Llama | Lag-based transformer | 2.4M | Lightweight probabilistic model using lagged features. Fast inference. |
| Ensemble | Equal-weight average | - | Combines all three models to reduce individual prediction errors. |

All models run zero-shot, meaning they produce forecasts on any dataset without training or fine-tuning.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Web UI | Streamlit |
| Database | PostgreSQL + TimescaleDB |
| Monitoring | Grafana |
| Anomaly detection | PyOD (Isolation Forest) |
| AI analysis | Qwen 2.5 3B via Ollama |
| PDF reports | fpdf2 + Matplotlib |
| Containerization | Docker Compose |

## Quick Start

Prerequisites: Docker, NVIDIA GPU with drivers, Ollama installed with `qwen2.5:3b` model.

```bash
git clone https://github.com/KonstantinosPls/Time-Series-Forecasting-Platform.git
cd Time-Series-Forecasting-Platform
cp .env.example .env
docker-compose up --build
```

Access the platform:
- Streamlit UI: http://localhost:8501
- Grafana dashboard: http://localhost:3000 (admin/admin)

## How It Works

The workflow is splitted into five sections:

**1. Upload Data** -- Upload a univariate time-series CSV. The platform auto-detects timestamp and value columns, shows data statistics, column types, and frequency detection. Supports multi-column CSVs with manual column selection.

**2. Forecast and Model Evaluation** -- Runs all three models on your data, computes an ensemble prediction, and automatically backtests each model using an 80/20 train/test split. Shows forecast charts, per-model metrics, and identifies the best-performing model.

**3. Anomaly Detection** -- Identifies unusual data points using Isolation Forest. Displays anomaly count, rate, and a chart with highlighted anomalous points.

**4. AI Analysis** -- Sends all computed results to a local Qwen 2.5 3B model via Ollama. The model generates a structured written analysis covering data characteristics, forecast interpretation, anomaly findings, model reliability, and actionable recommendations. The analysis is grounded in the computed results to minimize hallucinations.

**5. Download Report** -- Generates a comprehensive PDF report containing data summary, forecast results with charts, full anomaly table, backtest metrics with comparison chart, and the AI-generated analysis.

## Grafana

The platform includes a Grafana monitoring dashboard connected to TimescaleDB. While the current workflow is batch oriented via Streamlit, the Grafana layer is designed to support future real time streaming via Apache Kafka, enabling live forecast updates and anomaly alerts as new data arrives continuously.

## Limitations

**Forecast horizon:** TimesFM is configured with a horizon window of 128 steps. This value can be increased, though accuracy may decrease with longer horizons. Chronos has no hard limit but accuracy degrades after approximately 64 steps as prediction errors compound with each successive step. The forecast slider is capped at 48 steps for practical use.

**Model context:** Lag-Llama only considers the last 512 data points when making predictions. For datasets larger than 512 rows, earlier data is ignored. This value can be increased on machines with stronger GPU cards.

**Data format:** Only univariate time-series data is supported (one timestamp column and one value column). Multivariate forecasting (multiple value columns) could be implemented with the use of different models.

**Zero-shot accuracy:** All models run without fine-tuning on your specific data. Performance varies by dataset. Models work best on data with clear repeating patterns and struggle with highly volatile data (daily temperatures).

**Infrastructure:** Requires an NVIDIA GPU with CUDA support. The first forecast run is slow due to model weight downloads (~2.5 GB total). Ollama must be running separately from Docker.

**AI analysis:** The Qwen 2.5 3B model is constrained to 900 output tokens per analysis. On rare occasions it may produce slightly generic recommendations, though all numerical references are strictly grounded in the computed results.

## Future Improvements

- Apache Kafka integration for real-time data streaming and continuous forecasting
- Prefect orchestration for scheduled model retraining
- Adaptive ensemble weights based on per-dataset backtest performance
- Confidence interval visualization in forecast charts
- GitHub Actions CI/CD pipeline for automated testing and Docker builds
- Support for multivariate time-series forecasting

## Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- Ollama with `qwen2.5:3b` model installed
- ~13 GB disk space for Docker images and model weights
