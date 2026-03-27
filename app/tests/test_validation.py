import numpy as np
import pytest
from ensemble import weighted_average, equal_weights
from eval_metrics import mae, mase
from anomaly import detect_anomalies
from validation import validate_csv
import pandas as pd


def test_equal_weights():
    assert equal_weights(3) == pytest.approx([1/3, 1/3, 1/3])
    assert equal_weights(2) == pytest.approx([0.5, 0.5])
    assert sum(equal_weights(5)) == pytest.approx(1.0)


def test_weighted_average():
    f1 = (np.array([10, 20]), np.array([8, 16]), np.array([12, 24]))
    f2 = (np.array([30, 40]), np.array([28, 36]), np.array([32, 44]))
    weights = [0.5, 0.5]
    median, lower, upper = weighted_average([f1, f2], weights)
    assert median == pytest.approx([20, 30])
    assert lower == pytest.approx([18, 26])
    assert upper == pytest.approx([22, 34])


def test_mae():
    actual = np.array([10, 20, 30])
    predicted = np.array([12, 18, 33])
    assert mae(actual, predicted) == pytest.approx(7.0 / 3.0)


def test_mae_perfect():
    actual = np.array([10, 20, 30])
    assert mae(actual, actual) == pytest.approx(0.0)


def test_mase():
    actual = np.array([12, 18, 33])
    predicted = np.array([10, 20, 30])
    training = np.array([5, 10, 15, 20, 25])
    result = mase(actual, predicted, training)
    assert result > 0


def test_anomaly_detect():
    normal = np.array([10, 11, 10, 11, 10, 11, 10, 11, 10, 11] * 10)
    values = np.append(normal, [50, 55])
    labels, scores = detect_anomalies(values)
    assert len(labels) == len(values)
    assert labels[-1] == 1  # outlier detected
    assert labels[-2] == 1  # outlier detected


def test_validate_csv_valid():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=5, freq="h"),
        "value": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    validated, error = validate_csv(df)
    assert error is None
    assert len(validated) == 5


def test_validate_csv_missing_columns():
    df = pd.DataFrame({"name": ["foo", "bar"], "label": ["x", "y"]})
    _, error = validate_csv(df)
    assert error is not None


def test_validate_csv_non_numeric_values():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
        "value": ["abc", "def", "ghi"]
    })
    _, error = validate_csv(df)
    assert error is not None


def test_validate_csv_manual_column_selection():
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=3, freq="h"),
        "temp": [10.0, 11.0, 12.0],
        "humidity": [50, 60, 70]
    })
    result, error = validate_csv(df, ts_col="date", val_col="temp")
    assert error is None
    assert len(result) == 3


def test_mae_single_value():
    assert mae(np.array([5]), np.array([8])) == pytest.approx(3.0)


def test_mase_constant_training():
    actual = np.array([10, 10, 10])
    predicted = np.array([11, 11, 11])
    training = np.array([5, 5, 5, 5])
    result = mase(actual, predicted, training)
    assert result == np.inf


def test_weighted_average_single_model():
    f1 = (np.array([10, 20, 30]), np.array([8, 16, 24]), np.array([12, 24, 36]))
    median, _, _ = weighted_average([f1], [1.0])
    assert median == pytest.approx([10, 20, 30])


def test_anomaly_all_identical():
    values = np.array([10.0] * 100)
    labels, scores = detect_anomalies(values)
    assert len(labels) == 100


def test_preprocess_removes_duplicates():
    from validation import preprocess
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
        "value": [10.0, 11.0, 12.0]
    })
    result = preprocess(df)
    assert len(result) == 2


def test_preprocess_sorts_chronologically():
    from validation import preprocess
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]),
        "value": [30.0, 10.0, 20.0]
    })
    result = preprocess(df)
    assert result["value"].iloc[0] == 10.0
    assert result["value"].iloc[-1] == 30.0
