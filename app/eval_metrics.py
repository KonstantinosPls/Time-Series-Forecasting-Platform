import numpy as np

def mae(actual, predicted):
    # Mean Absolute Error
    return np.mean(np.abs(actual - predicted))

def mase(actual, predicted, training_series):
    # Mean Absolute Scaled Error
    naive_errors = np.abs(np.diff(training_series))
    scale = np.mean(naive_errors)
    return mae(actual, predicted) / scale if scale != 0 else np.inf

def crps(actual, samples):
    # Continuous Ranked Probability Score
    num_samples = samples.shape[0]
    abs_diff = np.mean(np.abs(samples - actual), axis=0)
    spread = np.mean(np.abs(samples[:, None] - samples[None, :]), axis=(0, 1)) / (2 * num_samples)
    return np.mean(abs_diff - spread)
