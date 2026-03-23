import pandas as pd


def validate_csv(df, ts_col=None, val_col=None):
    # Auto-detect columns if not specified
    if ts_col and val_col:
        df = df[[ts_col, val_col]].copy()
        df.columns = ["timestamp", "value"]
    elif df.shape[1] == 2:
        df.columns = ["timestamp", "value"]
    else:
        # Try to auto-detect timestamp and numeric columns
        ts_candidates = [c for c in df.columns if _is_timestamp_col(df[c])]
        num_candidates = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if not ts_candidates:
            return None, "No timestamp column detected. Please select columns manually."
        if not num_candidates:
            return None, "No numeric column detected. Please select columns manually."

        df = df[[ts_candidates[0], num_candidates[0]]].copy()
        df.columns = ["timestamp", "value"]

    # Parse timestamps
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        return None, "Selected timestamp column contains invalid dates."

    # Parse values
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().all():
        return None, "Selected value column contains no numeric data."

    return df, None


def _is_timestamp_col(series):
    try:
        pd.to_datetime(series.head(20))
        return True
    except Exception:
        return False


def preprocess(df):
    # Drop rows with missing values
    df = df.dropna(subset=["timestamp", "value"])

    # Sort chronologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicate timestamps, keep last
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    # Infer frequency and resample to fill gaps
    freq = pd.infer_freq(df["timestamp"])
    if freq:
        df = df.set_index("timestamp").resample(freq).mean().interpolate().reset_index()

    return df
