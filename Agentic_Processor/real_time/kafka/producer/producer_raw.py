import time
from kafka import KafkaProducer
import json
import os
import sys
import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def send_raw(data):
    producer.send("raw_input", data)
    producer.flush()
    print("[KafkaProducer] Sent raw data to 'raw_input' topic")


# Example: simulate real-time data stream
if __name__ == "__main__":
    sample_data = [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        #"Ceci est un texte d'exemple avec des erreurs pour décider quels agents exécuter.",
        #"Este es un texto de ejemplo con errores para decidir qué agentes ejecutjr."
    ]


    for d in sample_data:

        send_raw(d)
        time.sleep(2)  # simulate streaming delay
