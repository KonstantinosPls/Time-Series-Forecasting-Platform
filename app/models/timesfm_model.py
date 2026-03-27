import torch
import numpy as np
import timesfm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(horizon=128):
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            per_core_batch_size=1,
            horizon_len=horizon,
            num_layers=50,
            backend="gpu" if DEVICE == "cuda" else "cpu"
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        )
    )
    return model

def forecast(model, series, horizon):
    point_forecast, quantile_forecast = model.forecast(
        [series.values],
        freq=[0]
    )

    median = np.array(point_forecast[0][:horizon])

    # Use quantile forecasts if available, otherwise estimate from residuals
    if quantile_forecast is not None and len(quantile_forecast) > 0:
        q_data = np.array(quantile_forecast[0])
        lower = q_data[:horizon, 0] if q_data.ndim == 2 else median * 0.9
        upper = q_data[:horizon, -1] if q_data.ndim == 2 else median * 1.1
    else:
        std = np.std(series.values[-min(len(series), 50):])
        lower = median - 1.28 * std
        upper = median + 1.28 * std

    return median, lower, upper
