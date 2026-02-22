import os
import sys
import json
from kafka import KafkaConsumer, KafkaProducer

# ❤️ FIX PATH ISSUE
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

    print("PROJECT_ROOT =", PROJECT_ROOT)

# Now this import will work
from orchestrator.main_orchestrator import MainOrchestrator

consumer = KafkaConsumer(
    "raw_input",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

orch = MainOrchestrator()

for message in consumer:
    data = message.value  # incoming text, e.g., "Hello World"
    
    # Wrap text in a dictionary
    if isinstance(data, str):
        data = {"text": data}
    else:
        data = data  # already a dict
    
    dtype = orch.route(data)
    
    if dtype != "unknown":
        producer.send(dtype, data)
        producer.flush()
        print(f"[Router] Sent data to topic {dtype}")
