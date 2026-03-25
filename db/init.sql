CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS time_series_data (
    id SERIAL,
    series_name TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('time_series_data', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL,
    series_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    lower_bound DOUBLE PRECISION,
    upper_bound DOUBLE PRECISION,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('forecasts', 'timestamp', if_not_exists => TRUE);

CREATE TABLE IF NOT EXISTS anomalies (
    id SERIAL,
    series_name TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    anomaly_score DOUBLE PRECISION NOT NULL,
    is_anomaly BOOLEAN NOT NULL DEFAULT FALSE
);

SELECT create_hypertable('anomalies', 'timestamp', if_not_exists => TRUE);
