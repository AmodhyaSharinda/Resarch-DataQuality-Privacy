import streamlit as st
import requests
import json
from datetime import datetime
import pandas as pd
import asyncio
import websockets
import threading

# API Configuration
API_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/pipeline"

st.set_page_config(
    page_title="Agentic AI Pipeline - Connected",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'logs' not in st.session_state:
    st.session_state.logs = []
if 'stats' not in st.session_state:
    st.session_state.stats = {
        'total_processed': 0,
        'text_processed': 0,
        'image_processed': 0,
        'active_agents': 0,
        'avg_processing_time': 0.0
    }
if 'pipeline_status' not in st.session_state:
    st.session_state.pipeline_status = {
        'producer': 'idle',
        'detector': 'idle',
        'consumer': 'idle',
        'orchestrator': 'idle',
        'agents': []
    }

# Fetch stats from API
def fetch_stats():
    try:
        response = requests.get(f"{API_URL}/api/stats")
        if response.status_code == 200:
            st.session_state.stats = response.json()
    except:
        pass

# Fetch logs from API
def fetch_logs():
    try:
        response = requests.get(f"{API_URL}/api/logs")
        if response.status_code == 200:
            st.session_state.logs = response.json()
    except:
        pass

# Send text data to backend
def send_text_data(text):
    try:
        response = requests.post(
            f"{API_URL}/api/ingest/text",
            json={"text": text}
        )
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Send file data to backend
def send_file_data(file):
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_URL}/api/ingest/file", files=files)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Fetch initial data
fetch_stats()
fetch_logs()

# Header
st.title("🤖 Agentic AI Data Preprocessing Pipeline")
st.markdown("Real-time monitoring with backend connection")
st.markdown("---")

# Statistics
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Processed", st.session_state.stats['total_processed'])
with col2:
    st.metric("Text Files", st.session_state.stats['text_processed'])
with col3:
    st.metric("Images", st.session_state.stats['image_processed'])
with col4:
    st.metric("Active Agents", st.session_state.stats['active_agents'])
with col5:
    st.metric("Avg Time (s)", f"{st.session_state.stats['avg_processing_time']:.1f}")

st.markdown("---")

# Main layout
left_col, middle_col, right_col = st.columns([1, 1.2, 1])

with left_col:
    st.markdown("### 📝 Data Input")
    
    # Text input
    with st.expander("✍️ Text Input", expanded=True):
        text_input = st.text_area("Enter text:", height=150)
        if st.button("🚀 Process Text", use_container_width=True):
            if text_input.strip():
                result = send_text_data(text_input)
                if result['status'] == 'success':
                    st.success(result['message'])
                else:
                    st.error(result['message'])
            else:
                st.warning("Please enter some text!")
    
    # File upload
    with st.expander("📁 File Upload", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload file",
            type=['csv', 'txt', 'png', 'jpg', 'jpeg']
        )
        if uploaded_file:
            st.info(f"📄 {uploaded_file.name}")
            if st.button("🚀 Process File", use_container_width=True):
                result = send_file_data(uploaded_file)
                if result['status'] == 'success':
                    st.success(result['message'])
                else:
                    st.error(result['message'])

with middle_col:
    st.markdown("### 🔄 Pipeline Flow")
    # Pipeline visualization (same as before)
    st.info("Pipeline visualization here")

with right_col:
    st.markdown("### 📊 System Logs")
    if st.button("🔄 Refresh Logs"):
        fetch_logs()
        fetch_stats()
        st.rerun()
    
    for log in st.session_state.logs[:20]:
        with st.container():
            st.text(f"[{log['timestamp']}] {log['type'].upper()}")
            st.text(log['message'])

# Auto-refresh every 2 seconds
if st.button("Enable Auto-Refresh"):
    st.rerun()