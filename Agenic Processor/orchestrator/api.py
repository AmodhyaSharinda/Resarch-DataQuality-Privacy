import os
import sys
import json
import time
from threading import Thread
from uuid import uuid4
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from kafka import KafkaConsumer
from real_time.kafka.producer.producer_raw import send_raw

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory queue to store processed results
processed_results = []


def normalize_message(data: dict):
    """Ensure all expected fields exist for the frontend."""
    return {
        "id": data.get("id", str(uuid4())),
        "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
        "entity_detections": data.get("entity_detections", ""),  # originalText in React
        "clean_text": data.get("clean_text", ""),                # cleanedText in React
        "cleaning_status": data.get("cleaning_status", "processing"),
        "language": data.get("language", "unknown"),
        "agents_run": data.get("agents_run", []),
        "confidence": data.get("confidence", 0.0),
        "errors": data.get("errors", []),
        "llm_plan": data.get("llm_plan", [])
    }


def consume_processed_results():
    """Background thread to listen for processed results from Kafka."""
    try:
        consumer = KafkaConsumer(
            "processed_text",
            bootstrap_servers="localhost:9092",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            group_id="api_consumer_group"
        )
        print("[API] Listening for processed results from Kafka...")
        for msg in consumer:
            data = msg.value
            normalized = normalize_message(data)
            processed_results.append(normalized)
            # Keep only last 100 results
            if len(processed_results) > 100:
                processed_results.pop(0)
    except Exception as e:
        print(f"[API] Error consuming results: {e}")


@app.on_event("startup")
def startup_event():
    """Start background consumer thread on API startup."""
    consumer_thread = Thread(target=consume_processed_results, daemon=True)
    consumer_thread.start()


@app.post("/ingest")
async def ingest(text: str = Form(None), file: UploadFile = File(None)):
    """Accept text or file and publish to Kafka via existing producer."""
    try:
        if file is not None:
            content = await file.read()
            payload = {"image": content.hex(), "filename": file.filename}
            send_raw(payload)
            return JSONResponse({"status": "sent", "type": "image", "message": "Image sent for processing"})

        if text is not None:
            send_raw(text)
            return JSONResponse({"status": "sent", "type": "text", "message": "Text sent for processing"})

        return JSONResponse({"status": "error", "message": "no_input_provided"}, status_code=400)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/results")
def get_results():
    """Get all processed results (latest 100)."""
    return JSONResponse({"results": processed_results})


@app.get("/events")
def events():
    """Server-Sent Events stream for real-time updates."""
    def event_generator():
        last_index = 0
        while True:
            if last_index < len(processed_results):
                payload = processed_results[last_index]
                last_index += 1
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(0.2)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
def health():
    """Health check endpoint."""
    return JSONResponse({"status": "healthy"})
