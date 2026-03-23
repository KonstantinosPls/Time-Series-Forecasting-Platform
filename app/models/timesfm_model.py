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
    # TimesFM expects a list of arrays
    point_forecast, _ = model.forecast(
        [series.values],
        freq=[0]
    )

    median = np.array(point_forecast[0][:horizon])
    lower = median * 0.9
    upper = median * 1.1

    return median, lower, upper
