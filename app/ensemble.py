import numpy as np

def weighted_average(forecasts, weights):
    # forecasts: list of (median, lower, upper) tuples
    # weights: list of floats that sum to 1
    medians = np.array([f[0] for f in forecasts])
    lowers = np.array([f[1] for f in forecasts])
    uppers = np.array([f[2] for f in forecasts])

    weights = np.array(weights).reshape(-1, 1)

    ensemble_median = np.sum(medians * weights, axis=0)
    ensemble_lower = np.sum(lowers * weights, axis=0)
    ensemble_upper = np.sum(uppers * weights, axis=0)

    return ensemble_median, ensemble_lower, ensemble_upper


def equal_weights(n):
    return [1.0 / n] * n
