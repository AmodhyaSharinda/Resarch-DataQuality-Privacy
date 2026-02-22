import streamlit as st
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import random

# Page configuration
st.set_page_config(
    page_title="AI Pipeline Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for engaging UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
    }
    
    /* Cyber Header */
    .cyber-header {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(45deg, #00f5ff, #ff00ff, #00f5ff);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .cyber-subtitle {
        font-family: 'Rajdhani', sans-serif;
        text-align: center;
        color: #00f5ff;
        font-size: 1.2rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    
    /* Neon Cards */
    .neon-card {
        background: rgba(10, 14, 39, 0.8);
        border: 2px solid #00f5ff;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3), inset 0 0 20px rgba(0, 245, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .neon-card:hover {
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.6), inset 0 0 30px rgba(0, 245, 255, 0.2);
        transform: translateY(-5px);
    }
    
    /* Holographic Stats */
    .holo-stat {
        background: linear-gradient(135deg, rgba(0, 245, 255, 0.1) 0%, rgba(255, 0, 255, 0.1) 100%);
        border: 2px solid transparent;
        border-image: linear-gradient(45deg, #00f5ff, #ff00ff) 1;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .holo-stat::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .stat-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        color: #00f5ff;
        text-shadow: 0 0 10px rgba(0, 245, 255, 0.8);
        position: relative;
        z-index: 1;
    }
    
    .stat-label {
        font-family: 'Rajdhani', sans-serif;
        color: #a0aec0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    /* Pipeline Node */
    .pipeline-node {
        background: linear-gradient(135deg, #1a1f3a 0%, #0a0e27 100%);
        border: 2px solid #00f5ff;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        position: relative;
        box-shadow: 0 0 15px rgba(0, 245, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .pipeline-node.active {
        border-color: #00ff41;
        box-shadow: 0 0 30px rgba(0, 255, 65, 0.6);
        animation: pulse-green 1.5s infinite;
    }
    
    .pipeline-node.complete {
        border-color: #00ff41;
        background: linear-gradient(135deg, rgba(0, 255, 65, 0.1) 0%, rgba(0, 245, 255, 0.1) 100%);
    }
    
    @keyframes pulse-green {
        0%, 100% { box-shadow: 0 0 30px rgba(0, 255, 65, 0.6); }
        50% { box-shadow: 0 0 50px rgba(0, 255, 65, 0.9); }
    }
    
    .node-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #00f5ff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .node-desc {
        font-family: 'Rajdhani', sans-serif;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }
    
    /* Agent Card */
    .agent-card {
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid #ff00ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0 0.5rem 2rem;
        box-shadow: 0 0 10px rgba(255, 0, 255, 0.2);
    }
    
    .agent-card.active {
        animation: agent-pulse 1s infinite;
        border-color: #00ff41;
        box-shadow: 0 0 20px rgba(0, 255, 65, 0.5);
    }
    
    @keyframes agent-pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    /* Terminal Log */
    .terminal-log {
        background: #0a0e27;
        border: 1px solid #00f5ff;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        box-shadow: inset 0 0 10px rgba(0, 245, 255, 0.1);
    }
    
    .log-timestamp {
        color: #a0aec0;
        margin-right: 0.5rem;
    }
    
    .log-info { color: #00f5ff; }
    .log-success { color: #00ff41; }
    .log-error { color: #ff0055; }
    .log-warning { color: #ffaa00; }
    
    /* Glowing Button */
    .glow-button {
        background: linear-gradient(45deg, #00f5ff, #ff00ff);
        border: none;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 700;
        color: #000;
        text-transform: uppercase;
        letter-spacing: 2px;
        cursor: pointer;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
        transition: all 0.3s ease;
    }
    
    .glow-button:hover {
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.8);
        transform: scale(1.05);
    }
    
    /* Input Fields */
    .stTextArea textarea, .stTextInput input {
        background: rgba(10, 14, 39, 0.8) !important;
        border: 2px solid #00f5ff !important;
        border-radius: 8px !important;
        color: #00f5ff !important;
        font-family: 'Rajdhani', sans-serif !important;
        box-shadow: inset 0 0 10px rgba(0, 245, 255, 0.2) !important;
    }
    
    /* Connection Line */
    .connection-line {
        text-align: center;
        color: #00f5ff;
        font-size: 2rem;
        margin: 0.5rem 0;
        animation: flow 2s infinite;
    }
    
    @keyframes flow {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
    
    /* Status Indicator */
    .status-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-idle { background: #64748b; }
    .status-active {
        background: #00ff41;
        box-shadow: 0 0 10px #00ff41;
        animation: blink 1s infinite;
    }
    .status-complete {
        background: #00f5ff;
        box-shadow: 0 0 10px #00f5ff;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #1a1f3a 100%);
        border-right: 2px solid #00f5ff;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processing_history' not in st.session_state:
    st.session_state.processing_history = []

def add_log(message, log_type='info'):
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    st.session_state.logs.insert(0, {
        'timestamp': timestamp,
        'message': message,
        'type': log_type
    })
    st.session_state.logs = st.session_state.logs[:100]

def simulate_processing(data, data_type):
    st.session_state.processing = True
    
    # Producer
    st.session_state.pipeline_status['producer'] = 'active'
    add_log(f'🚀 Producer: Data ingestion initiated ({data_type})', 'info')
    time.sleep(0.8)
    st.rerun()
    
    st.session_state.pipeline_status['producer'] = 'complete'
    st.session_state.pipeline_status['detector'] = 'active'
    add_log('🔍 Consumer Detect: Analyzing data signature...', 'info')
    time.sleep(1.0)
    st.rerun()
    
    add_log(f'✅ Main Orchestrator: Type detected → {data_type.upper()}', 'success')
    add_log(f'📡 Main Orchestrator: Routing to {data_type} topic', 'info')
    
    st.session_state.pipeline_status['detector'] = 'complete'
    st.session_state.pipeline_status['consumer'] = 'active'
    time.sleep(0.8)
    st.rerun()
    
    add_log(f'📥 Consumer: Data received from {data_type} topic', 'info')
    add_log('⚡ Consumer: Forwarding to Orchestrator', 'info')
    
    st.session_state.pipeline_status['consumer'] = 'complete'
    st.session_state.pipeline_status['orchestrator'] = 'active'
    time.sleep(1.0)
    st.rerun()
    
    add_log('🧠 Orchestrator: Initializing Ollama model...', 'info')
    time.sleep(1.5)
    st.rerun()
    
    agents = ['TextCleaningAgent', 'SentimentAnalysisAgent', 'EntityExtractionAgent'] if data_type == 'text' else ['ImageResizeAgent', 'ObjectDetectionAgent', 'MetadataExtractorAgent']
    
    add_log(f'🎯 Ollama Model: Agent sequence computed → {" → ".join(agents)}', 'success')
    st.session_state.pipeline_status['agents'] = [{'name': a, 'status': 'pending'} for a in agents]
    st.rerun()
    
    for i, agent in enumerate(agents):
        time.sleep(1.2)
        st.session_state.pipeline_status['agents'][i]['status'] = 'active'
        add_log(f'⚙️ Agent [{i+1}/{len(agents)}]: Executing {agent}...', 'info')
        st.rerun()
        
        time.sleep(1.5)
        st.session_state.pipeline_status['agents'][i]['status'] = 'complete'
        add_log(f'✔️ Agent [{i+1}/{len(agents)}]: {agent} completed successfully', 'success')
        st.rerun()
    
    st.session_state.pipeline_status['orchestrator'] = 'complete'
    add_log('🎉 Pipeline: Processing complete! All systems nominal.', 'success')
    
    # Update stats
    st.session_state.stats['total_processed'] += 1
    if data_type == 'text':
        st.session_state.stats['text_processed'] += 1
    else:
        st.session_state.stats['image_processed'] += 1
    st.session_state.stats['active_agents'] = len(agents)
    st.session_state.stats['avg_processing_time'] = round(random.uniform(4.5, 6.2), 1)
    
    # Add to history
    st.session_state.processing_history.insert(0, {
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'type': data_type,
        'duration': st.session_state.stats['avg_processing_time']
    })
    st.session_state.processing_history = st.session_state.processing_history[:10]
    
    time.sleep(2)
    st.session_state.pipeline_status = {
        'producer': 'idle',
        'detector': 'idle',
        'consumer': 'idle',
        'orchestrator': 'idle',
        'agents': []
    }
    st.session_state.processing = False
    st.rerun()

def get_status_html(status):
    status_class = {
        'idle': 'status-idle',
        'active': 'status-active',
        'complete': 'status-complete'
    }.get(status, 'status-idle')
    return f'<span class="status-dot {status_class}"></span>'

# Sidebar
with st.sidebar:
    st.markdown('<div style="font-family: Orbitron; font-size: 1.5rem; color: #00f5ff; text-align: center; margin-bottom: 2rem;">⚙️ CONTROL PANEL</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('### 🎮 System Controls')
    
    if st.button('🔄 RESET SYSTEM', use_container_width=True):
        st.session_state.logs = []
        st.session_state.stats = {
            'total_processed': 0,
            'text_processed': 0,
            'image_processed': 0,
            'active_agents': 0,
            'avg_processing_time': 0.0
        }
        st.session_state.processing_history = []
        st.rerun()
    
    if st.button('🗑️ CLEAR LOGS', use_container_width=True):
        st.session_state.logs = []
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('---')
    
    # Processing History
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('### 📜 Recent Activity')
    if st.session_state.processing_history:
        for item in st.session_state.processing_history[:5]:
            st.markdown(f"""
            <div style="padding: 0.5rem; margin: 0.3rem 0; background: rgba(0, 245, 255, 0.1); border-radius: 5px; border-left: 3px solid #00f5ff;">
                <div style="font-family: Rajdhani; color: #00f5ff; font-size: 0.9rem;">{item['timestamp']}</div>
                <div style="font-family: Rajdhani; color: #a0aec0; font-size: 0.8rem;">{item['type'].upper()} • {item['duration']}s</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #64748b; text-align: center; padding: 1rem;">No activity yet</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main Header
st.markdown('<div class="cyber-header">⚡ AI PIPELINE COMMAND CENTER ⚡</div>', unsafe_allow_html=True)
st.markdown('<div class="cyber-subtitle">◈ Real-Time Agentic Data Processing System ◈</div>', unsafe_allow_html=True)

# Stats Dashboard with Charts
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="holo-stat">
        <div class="stat-value">{st.session_state.stats['total_processed']}</div>
        <div class="stat-label">⚡ TOTAL PROCESSED</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="holo-stat">
        <div class="stat-value">{st.session_state.stats['text_processed']}</div>
        <div class="stat-label">📝 TEXT FILES</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="holo-stat">
        <div class="stat-value">{st.session_state.stats['image_processed']}</div>
        <div class="stat-label">🖼️ IMAGES</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="holo-stat">
        <div class="stat-value">{st.session_state.stats['active_agents']}</div>
        <div class="stat-label">🤖 ACTIVE AGENTS</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="holo-stat">
        <div class="stat-value">{st.session_state.stats['avg_processing_time']:.1f}s</div>
        <div class="stat-label">⏱️ AVG TIME</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<br>', unsafe_allow_html=True)

# Main Content Area
left_col, middle_col, right_col = st.columns([1.2, 1.5, 1.3])

# Left Column - Input
with left_col:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('### 📡 DATA INGESTION')
    
    tab1, tab2 = st.tabs(['✍️ TEXT INPUT', '📁 FILE UPLOAD'])
    
    with tab1:
        text_input = st.text_area('Data Payload', height=200, placeholder='Enter your data payload here...', label_visibility='collapsed')
        if st.button('🚀 LAUNCH PROCESSING', disabled=st.session_state.processing, use_container_width=True, key='text_btn'):
            if text_input.strip():
                simulate_processing(text_input, 'text')
            else:
                st.warning('⚠️ Input required')
    
    with tab2:
        uploaded_file = st.file_uploader('Upload File', type=['csv', 'txt', 'png', 'jpg', 'jpeg'], label_visibility='collapsed', disabled=st.session_state.processing)
        
        if uploaded_file:
            st.markdown(f"""
            <div style="background: rgba(0, 245, 255, 0.1); padding: 1rem; border-radius: 8px; border: 1px solid #00f5ff; margin: 1rem 0;">
                <div style="font-family: Rajdhani; color: #00f5ff; font-weight: 700;">📄 {uploaded_file.name}</div>
                <div style="font-family: Rajdhani; color: #a0aec0; font-size: 0.85rem; margin-top: 0.3rem;">Size: {uploaded_file.size / 1024:.2f} KB</div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button('🚀 LAUNCH PROCESSING', disabled=st.session_state.processing, use_container_width=True, key='file_btn'):
                file_type = 'image' if uploaded_file.type.startswith('image/') else 'text'
                simulate_processing(uploaded_file.name, file_type)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Middle Column - Pipeline
with middle_col:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('### 🔄 PIPELINE FLOW VISUALIZATION')
    
    # Producer
    status = st.session_state.pipeline_status['producer']
    st.markdown(f"""
    <div class="pipeline-node {status}">
        {get_status_html(status)}
        <span class="node-title">🗄️ PRODUCER</span>
        <div class="node-desc">producer_raw.py • Data Ingestion Layer</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="connection-line">⬇</div>', unsafe_allow_html=True)
    
    # Detector
    status = st.session_state.pipeline_status['detector']
    st.markdown(f"""
    <div class="pipeline-node {status}">
        {get_status_html(status)}
        <span class="node-title">🔍 DETECTOR</span>
        <div class="node-desc">consumer_detect.py • Type Detection & Routing</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="connection-line">⬇</div>', unsafe_allow_html=True)
    
    # Consumer
    status = st.session_state.pipeline_status['consumer']
    st.markdown(f"""
    <div class="pipeline-node {status}">
        {get_status_html(status)}
        <span class="node-title">📥 CONSUMER</span>
        <div class="node-desc">consumer.py • Text Data Consumer</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="connection-line">⬇</div>', unsafe_allow_html=True)
    
    # Orchestrator
    status = st.session_state.pipeline_status['orchestrator']
    st.markdown(f"""
    <div class="pipeline-node {status}">
        {get_status_html(status)}
        <span class="node-title">🎭 ORCHESTRATOR</span>
        <div class="node-desc">orchestrator.py • Agent Coordination via Ollama</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Agents
    if st.session_state.pipeline_status['agents']:
        st.markdown('<div class="connection-line">⬇</div>', unsafe_allow_html=True)
        for i, agent in enumerate(st.session_state.pipeline_status['agents']):
            status_icon = '🔄' if agent['status'] == 'active' else '✅' if agent['status'] == 'complete' else '⏳'
            active_class = 'active' if agent['status'] == 'active' else ''
            st.markdown(f"""
            <div class="agent-card {active_class}">
                {status_icon} <strong style="font-family: Rajdhani; color: #ff00ff;">{agent['name']}</strong>
                <span style="color: #64748b; font-size: 0.8rem; margin-left: 0.5rem;">Agent #{i+1}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Right Column - Logs
with right_col:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    st.markdown('### 📊 SYSTEM LOGS')
    
    log_container = st.container()
    with log_container:
        if not st.session_state.logs:
            st.markdown('<div style="text-align: center; color: #64748b; padding: 2rem; font-family: Rajdhani;">◈ AWAITING DATA INPUT ◈</div>', unsafe_allow_html=True)
        else:
            for log in st.session_state.logs[:25]:
                log_class = f"log-{log['type']}"
                st.markdown(f"""
                <div class="terminal-log">
                    <span class="log-timestamp">[{log['timestamp']}]</span>
                    <span class="{log_class}"><strong>{log['type'].upper()}</strong></span>
                    <div class="{log_class}" style="margin-top: 0.3rem;">{log['message']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<br><br>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-family: Rajdhani; color: #64748b; font-size: 0.9rem; border-top: 1px solid #00f5ff; padding-top: 1rem; margin-top: 2rem;">
    <span style="color: #00f5ff;">◈</span> AGENTIC AI PIPELINE COMMAND CENTER <span style="color: #00f5ff;">◈</span><br>
    <span style="font-size: 0.8rem;">Powered by Kafka • Ollama • Streamlit</span>
</div>
""", unsafe_allow_html=True)

# Auto-refresh during processing
if st.session_state.processing:
    time.sleep(0.5)
    st.rerun()
