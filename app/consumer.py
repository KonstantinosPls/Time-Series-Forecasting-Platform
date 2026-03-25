import os
import json
import time
from datetime import datetime, timezone
from collections import defaultdict
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

BROKER = os.environ.get("KAFKA_BROKER", "kafka:29092")
TOPIC = "weather"
DB_USER = os.environ.get("POSTGRES_USER", "admin")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "admin")
DB_HOST = os.environ.get("POSTGRES_HOST", "db")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "timeseries")

engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def create_consumer():
    for _ in range(30):
        try:
            return KafkaConsumer(
                TOPIC,
                bootstrap_servers=BROKER,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest"
            )
        except Exception:
            time.sleep(2)
    raise ConnectionError("Could not connect to Kafka")

def save_reading(timestamp, city, temperature):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO time_series_data (series_name, timestamp, value) VALUES (:name, :ts, :val)"),
            {"name": f"weather_{city}", "ts": timestamp, "val": temperature}
        )

if __name__ == "__main__":
    consumer = create_consumer()
    hourly_buffer = defaultdict(list)
    print("Consumer started. Listening for weather data.")

    for message in consumer:
        data = message.value
        dt = datetime.fromtimestamp(data["timestamp"], tz=timezone.utc)
        hour_key = dt.strftime("%Y-%m-%d %H:00:00")

        hourly_buffer[hour_key].append(data["temperature"])
        print(f"Received: {data['temperature']}C at {dt.isoformat()}")

        # Save raw reading
        save_reading(dt, data["city"], data["temperature"])

        # When hour is complete (30+ readings at 2-min intervals), save hourly average
        if len(hourly_buffer[hour_key]) >= 30:
            avg_temp = sum(hourly_buffer[hour_key]) / len(hourly_buffer[hour_key])
            hourly_ts = datetime.strptime(hour_key, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            save_reading(hourly_ts, f"weather_{data['city']}_hourly", avg_temp)
            print(f"Hourly average: {avg_temp:.1f}C for {hour_key}")
            del hourly_buffer[hour_key]
