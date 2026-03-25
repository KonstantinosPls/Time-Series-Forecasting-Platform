import os
import json
import time
import requests
from kafka import KafkaProducer

API_KEY = os.environ["OPENWEATHERMAP_API_KEY"]
CITY = os.environ.get("WEATHER_CITY", "Athens")
INTERVAL = int(os.environ.get("FETCH_INTERVAL", 120))
BROKER = os.environ.get("KAFKA_BROKER", "kafka:29092")
TOPIC = "weather"

def create_producer():
    for _ in range(30):
        try:
            return KafkaProducer(
                bootstrap_servers=BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8")
            )
        except Exception:
            time.sleep(2)
    raise ConnectionError("Could not connect to Kafka")

def fetch_weather():
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return {
        "timestamp": data["dt"],
        "city": CITY,
        "temperature": data["main"]["temp"]
    }

if __name__ == "__main__":
    producer = create_producer()
    print(f"Producer started. Fetching {CITY} weather every {INTERVAL}s.")

    while True:
        try:
            weather = fetch_weather()
            producer.send(TOPIC, weather)
            print(f"Sent: {weather['temperature']}C at {weather['timestamp']}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(INTERVAL)
