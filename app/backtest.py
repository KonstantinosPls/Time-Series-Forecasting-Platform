import numpy as np
from eval_metrics import mae, mase

def train_test_split(series, test_ratio=0.2):
    split = int(len(series) * (1 - test_ratio))
    return series[:split], series[split:]

def evaluate_model(forecast_fn, train_values, test_values, max_horizon=128):
    horizon = min(len(test_values), max_horizon)
    predicted, lower, upper = forecast_fn(train_values, horizon)

    # Align lengths in case model returns fewer steps
    actual = test_values[:len(predicted)]

    return {
        "mae": mae(actual, predicted),
        "mase": mase(actual, predicted, train_values),
        "predicted": predicted,
        "lower": lower,
        "upper": upper
    }
