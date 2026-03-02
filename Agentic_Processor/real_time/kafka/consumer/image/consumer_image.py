""" from kafka import KafkaConsumer
import json
from agents.image.agentic_image_processor import AgenticImageProcessor

processor = AgenticImageProcessor()

consumer = KafkaConsumer(
    "detected_image",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

for msg in consumer:
    data = msg.value
    output = processor.run(data["image"])
    print("[IMAGE ORCH]", output) """
