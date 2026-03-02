"""
This FastAPI server acts as a bridge between your Kafka-based backend 
and the Streamlit frontend
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from kafka import KafkaProducer, KafkaConsumer
import json
import asyncio
from typing import List, Dict
import threading
from datetime import datetime
import redis
import base64

app = FastAPI(title="Agentic AI Pipeline API")

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Redis for caching logs and status
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Kafka Producer Configuration
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/ingest/text")
async def ingest_text(data: dict):
    """
    Endpoint to receive text data from Streamlit and send to Kafka
    """
    try:
        message = {
            'type': 'text',
            'content': data['text'],
            'timestamp': datetime.now().isoformat(),
            'source': 'dashboard'
        }
        
        # Send to Kafka topic (your producer_raw.py equivalent)
        producer.send('raw_data_topic', value=message)
        producer.flush()
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            'event': 'data_ingested',
            'data': message
        })
        
        return {"status": "success", "message": "Text data sent to pipeline"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Endpoint to receive file uploads from Streamlit and send to Kafka
    """
    try:
        content = await file.read()
        
        # Determine file type
        if file.content_type.startswith('image/'):
            file_type = 'image'
            # Encode image as base64 for Kafka
            content_encoded = base64.b64encode(content).decode('utf-8')
        else:
            file_type = 'text'
            content_encoded = content.decode('utf-8')
        
        message = {
            'type': file_type,
            'filename': file.filename,
            'content': content_encoded,
            'content_type': file.content_type,
            'timestamp': datetime.now().isoformat(),
            'source': 'dashboard'
        }
        
        # Send to Kafka topic
        producer.send('raw_data_topic', value=message)
        producer.flush()
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            'event': 'file_ingested',
            'data': {'filename': file.filename, 'type': file_type}
        })
        
        return {"status": "success", "message": f"File {file.filename} sent to pipeline"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.websocket("/ws/pipeline")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time pipeline updates
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and send any updates
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/stats")
async def get_stats():
    """
    Get current pipeline statistics
    """
    stats = {
        'total_processed': int(redis_client.get('total_processed') or 0),
        'text_processed': int(redis_client.get('text_processed') or 0),
        'image_processed': int(redis_client.get('image_processed') or 0),
        'active_agents': int(redis_client.get('active_agents') or 0),
        'avg_processing_time': float(redis_client.get('avg_processing_time') or 0)
    }
    return stats

@app.get("/api/logs")
async def get_logs(limit: int = 50):
    """
    Get recent logs
    """
    logs = redis_client.lrange('pipeline_logs', 0, limit - 1)
    return [json.loads(log) for log in logs]

# ============================================================================
# Background Kafka Consumer Thread
# ============================================================================

def kafka_consumer_thread():
    """
    Background thread to consume Kafka messages and broadcast to WebSocket clients
    """
    consumer = KafkaConsumer(
        'pipeline_logs_topic',
        'pipeline_status_topic',
        bootstrap_servers=['localhost:9092'],
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    for message in consumer:
        data = message.value
        
        # Store in Redis
        if message.topic == 'pipeline_logs_topic':
            redis_client.lpush('pipeline_logs', json.dumps(data))
            redis_client.ltrim('pipeline_logs', 0, 999)  # Keep last 1000 logs
        
        # Update stats
        if message.topic == 'pipeline_status_topic':
            if 'stats' in data:
                for key, value in data['stats'].items():
                    redis_client.set(key, value)
        
        # Broadcast to WebSocket clients
        asyncio.run(manager.broadcast({
            'event': message.topic,
            'data': data
        }))

# Start background consumer thread
threading.Thread(target=kafka_consumer_thread, daemon=True).start()