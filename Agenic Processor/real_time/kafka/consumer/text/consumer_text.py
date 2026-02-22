import os
import sys
import json
from kafka import KafkaConsumer
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from orchestrator.text.orchestrator import AgenticTextProcessor


processor = AgenticTextProcessor()

consumer = KafkaConsumer(
    "detected_text",
    bootstrap_servers="localhost:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8"))
)

for msg in consumer:
    data = msg.value
    
    # Wrap string data into a dict if necessary
    if isinstance(data, str):
        data = {"text": data}
    
    # Now it's safe to access data["text"]
    output = processor.run(data["text"])

    # Save output
    output_file = "cleaned_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print("[TEXT ORCH]", output)