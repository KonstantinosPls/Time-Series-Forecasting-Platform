import torch
import numpy as np
from chronos import ChronosBoltPipeline

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    return ChronosBoltPipeline.from_pretrained(
        "amazon/chronos-bolt-small",
        device_map=DEVICE,
        torch_dtype=torch.float32
    )

def forecast(model, series, horizon):
    context = torch.tensor(series.values, dtype=torch.float32).unsqueeze(0)

    # Predict returns quantile forecasts as tensor (batch, horizon, num_quantiles)
    raw = model.predict(context, prediction_length=horizon)
    preds = raw.detach().cpu().numpy()

    # Compute confidence intervals from prediction samples
    median = np.median(preds[0], axis=0) if preds.ndim == 3 else preds[0]
    lower = np.percentile(preds[0], 10, axis=0) if preds.ndim == 3 else preds[0] * 0.9
    upper = np.percentile(preds[0], 90, axis=0) if preds.ndim == 3 else preds[0] * 1.1

    return median, lower, upper
