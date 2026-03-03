from rich import _console
import streamlit as st
import time
from datetime import datetime
import pandas as pd
from io import StringIO

from real_time.kafka.producer import producer_raw
from orchestrator.text.orchestrator import AgenticTextProcessor
from agents.text.ollama_model import OllamaPlanner

# Page configuration
st.set_page_config(
    page_title="Agentic AI Data Preprocessing Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #3b82f6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .stat-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #475569;
        text-align: center;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #3b82f6;
    }
    .stat-label {
        color: #94a3b8;
        font-size: 0.875rem;
    }
    .pipeline-stage {
        background: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-bottom: 0.5rem;
    }
    .pipeline-stage.active {
        border-left-color: #10b981;
        animation: pulse 2s infinite;
    }
    .pipeline-stage.complete {
        border-left-color: #10b981;
    }
    .agent-item {
        background: #0f172a;
        padding: 0.75rem;
        border-radius: 0.375rem;
        margin-left: 2rem;
        margin-bottom: 0.5rem;
        border-left: 2px solid #3b82f6;
    }
    .log-entry {
        background: #0f172a;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-bottom: 0.25rem;
        font-family: monospace;
        font-size: 0.85rem;
    }
    .log-info { color: #94a3b8; }
    .log-success { color: #10b981; }
    .log-error { color: #ef4444; }
    .log-warning { color: #f59e0b; }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
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

def add_log(message, log_type='info'):
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.logs.insert(0, {
        'timestamp': timestamp,
        'message': message,
        'type': log_type
    })
    st.session_state.logs = st.session_state.logs[:100]

def simulate_processing(data, data_type):
    """Simulate the entire pipeline processing"""
    st.session_state.processing = True
    print(data)
    
    # Producer Stage
    st.session_state.pipeline_status['producer'] = 'active'
    add_log(f'Producer: Data ingestion started ({data_type})', 'info')
    add_log(f'Producer: Data ingestion started ({data})', 'info')

    try:
        producer_raw.send_raw(data)
        add_log("Producer: Data sent successfully", "success")
    except Exception as e:
        _console.print(e)
        add_log(f"Producer ERROR: {e}", "error")

    add_log('Producer: Data sent to Kafka topic "raw_input"', 'success')
    
    # Detector Stage
    st.session_state.pipeline_status['producer'] = 'complete'
    st.session_state.pipeline_status['detector'] = 'active'
    add_log('Consumer Detect: Analyzing data type...', 'info')
    add_log(f'Main Orchestrator: Detected type - {data_type}', 'success')
    add_log(f'Main Orchestrator: Routing to {data_type} topic', 'info')
    
    # Consumer Stage
    st.session_state.pipeline_status['detector'] = 'complete'
    st.session_state.pipeline_status['consumer'] = 'active'
    add_log(f'Consumer: Received {data_type} data from topic', 'info')
    add_log('Consumer: Forwarding to Orchestrator', 'info')
    
    # Orchestrator Stage
    st.session_state.pipeline_status['consumer'] = 'complete'
    st.session_state.pipeline_status['orchestrator'] = 'active'
    add_log('Orchestrator: Processing request with Ollama model...', 'info')
    
    # # Ollama Model Decision
    # agents = ['TextCleaningAgent', 'SentimentAnalysisAgent', 'EntityExtractionAgent'] if data_type == 'text' else ['ImageResizeAgent', 'ObjectDetectionAgent', 'MetadataExtractorAgent']
    
    # add_log(f'Ollama Model: Predicted agent sequence: {" → ".join(agents)}', 'success')
    # st.session_state.pipeline_status['agents'] = [{'name': a, 'status': 'pending'} for a in agents]
    
    # # Execute Agents
    # for i, agent in enumerate(agents):
    #     st.session_state.pipeline_status['agents'][i]['status'] = 'active'
    #     add_log(f'Agent [{i+1}/{len(agents)}]: Running {agent}...', 'info')
        
    #     st.session_state.pipeline_status['agents'][i]['status'] = 'complete'
    #     add_log(f'Agent [{i+1}/{len(agents)}]: {agent} completed', 'success')
    
    
    planner = OllamaPlanner()
    try:
        # Call plan with your record (input data)
        model_output = planner.plan(record=data)  # returns a list of agent names
        if isinstance(model_output, list):
            agents = model_output
        else:
            # fallback if it returns a string
            agents = [a.strip() for a in model_output.split("→")]

        # Log the output to Streamlit system logs
        add_log(f"OllamaPlanner: Predicted agent sequence → {' → '.join(agents)}", "success")

    except Exception as e:
        add_log(f"Model ERROR: {e}", "error")
        # fallback agent sequence
        agents = ['TextCleaningAgent', 'SentimentAnalysisAgent', 'EntityExtractionAgent']

    # Now 'agents' is ready to use in your Streamlit logs and pipeline
    st.session_state.pipeline_status['agents'] = [{'name': a, 'status': 'pending'} for a in agents]

    # Complete
    st.session_state.pipeline_status['orchestrator'] = 'complete'
    add_log('Pipeline: Processing complete! ✓', 'success')
    
    # Update stats
    st.session_state.stats['total_processed'] += 1
    if data_type == 'text':
        st.session_state.stats['text_processed'] += 1
    else:
        st.session_state.stats['image_processed'] += 1
    st.session_state.stats['active_agents'] = len(agents)
    st.session_state.stats['avg_processing_time'] = 5.2
    
    # Reset pipeline status after delay
    st.session_state.pipeline_status = {
        'producer': 'idle',
        'detector': 'idle',
        'consumer': 'idle',
        'orchestrator': 'idle',
        'agents': []
    }
    st.session_state.processing = False

# Header
st.markdown('<div class="main-header">🤖 Agentic AI Data Preprocessing Pipeline</div>', unsafe_allow_html=True)
st.markdown('Real-time monitoring and data ingestion dashboard')
st.markdown('---')

# Statistics Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Processed</div>
        <div class="stat-value">{st.session_state.stats['total_processed']}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Text Files</div>
        <div class="stat-value">{st.session_state.stats['text_processed']}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Images</div>
        <div class="stat-value">{st.session_state.stats['image_processed']}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Active Agents</div>
        <div class="stat-value">{st.session_state.stats['active_agents']}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Avg Time (s)</div>
        <div class="stat-value">{st.session_state.stats['avg_processing_time']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('---')

# Main Layout
left_col, middle_col, right_col = st.columns([1, 1.2, 1])

# Left Column - Data Input
with left_col:
    st.markdown('### 📝 Data Input')

    with st.expander('✍️ Text Input', expanded=True):
        # Use a form to reliably capture input
        with st.form("text_form"):
            # Default value from session state
            text_value = st.session_state.get("text_input", "")
            user_input = st.text_area(
                "Enter your text data:",
                value=text_value,
                height=150,
                placeholder="Type or paste your text here..."
            )

            
            submitted = st.form_submit_button("🚀 Process Text", disabled=st.session_state.processing)

            if submitted:
                if user_input.strip():
                    # Save in session state
                    st.session_state.text_input = user_input
                    st.session_state.debug_input = user_input
                    add_log(f'New text input received: "{user_input[:50]}..."', 'info')
                    
                    print(st.session_state.text_input)

                    # Send to your pipeline
                    simulate_processing(user_input, "text")

                    # Clear input after processing
                    st.session_state.text_input = ""
                else:
                    st.warning("Please enter some text first!")
    
    # File Upload
    with st.expander('📁 File Upload', expanded=True):
        uploaded_file = st.file_uploader(
            'Upload CSV, TXT, or Image',
            type=['csv', 'txt', 'png', 'jpg', 'jpeg'],
            disabled=st.session_state.processing
        )

        if uploaded_file is not None:
            st.info(f'📄 {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)')

            if st.button('🚀 Process File', disabled=st.session_state.processing, use_container_width=True):

                mime_type = uploaded_file.type

                # -------- TEXT FILES --------
                if mime_type.startswith("text/") or uploaded_file.name.endswith((".csv", ".txt")):
                    try:
                        file_content = uploaded_file.read().decode("utf-8")
                    except UnicodeDecodeError:
                        # fallback encoding
                        file_content = uploaded_file.read().decode("latin-1")

                    file_type = "text"

                # -------- IMAGE FILES --------
                elif mime_type.startswith("image/"):
                    file_content = uploaded_file.read()  # KEEP AS BYTES
                    file_type = "image"

                else:
                    st.error("Unsupported file type")
                    st.stop()

                add_log(
                    f"File uploaded: {uploaded_file.name} ({uploaded_file.size/1024:.2f} KB)",
                    "info"
                )

                simulate_processing(file_content, file_type)


# Middle Column - Pipeline Visualization
with middle_col:
    st.markdown('### 🔄 Pipeline Flow')
    
    pipeline_container = st.container()
    
    with pipeline_container:
        # Producer
        status_class = 'active' if st.session_state.pipeline_status['producer'] == 'active' else 'complete' if st.session_state.pipeline_status['producer'] == 'complete' else ''
        st.markdown(f"""
        <div class="pipeline-stage {status_class}">
            <strong>🗄️ Producer (producer_raw.py)</strong><br>
            <small>Data Ingestion Layer</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;⬇️')
        
        # Detector
        status_class = 'active' if st.session_state.pipeline_status['detector'] == 'active' else 'complete' if st.session_state.pipeline_status['detector'] == 'complete' else ''
        st.markdown(f"""
        <div class="pipeline-stage {status_class}">
            <strong>🔍 Detector (consumer_detect.py)</strong><br>
            <small>Type Detection & Routing</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;⬇️')
        
        # Consumer
        status_class = 'active' if st.session_state.pipeline_status['consumer'] == 'active' else 'complete' if st.session_state.pipeline_status['consumer'] == 'complete' else ''
        st.markdown(f"""
        <div class="pipeline-stage {status_class}">
            <strong>📥 Consumer (consumer.py)</strong><br>
            <small>Text Data Consumer</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;⬇️')
        
        # Orchestrator
        status_class = 'active' if st.session_state.pipeline_status['orchestrator'] == 'active' else 'complete' if st.session_state.pipeline_status['orchestrator'] == 'complete' else ''
        st.markdown(f"""
        <div class="pipeline-stage {status_class}">
            <strong>🎭 Orchestrator (orchestrator.py)</strong><br>
            <small>Agent Coordination with Ollama</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Agents
        if st.session_state.pipeline_status['agents']:
            st.markdown('&nbsp;&nbsp;&nbsp;&nbsp;⬇️')
            for i, agent in enumerate(st.session_state.pipeline_status['agents']):
                status_icon = '🔄' if agent['status'] == 'active' else '✅' if agent['status'] == 'complete' else '⏳'
                st.markdown(f"""
                <div class="agent-item">
                    {status_icon} <strong>{agent['name']}</strong><br>
                    <small>Agent {i+1}</small>
                </div>
                """, unsafe_allow_html=True)

# Right Column - System Logs
with right_col:
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown('### 📊 System Logs')
    with col_header2:
        if st.button('🗑️ Clear', use_container_width=True):
            st.session_state.logs = []
            st.rerun()
    
    log_container = st.container()
    
    with log_container:
        if not st.session_state.logs:
            st.info('No logs yet. Start processing data!')
        else:
            # Display logs in scrollable area
            for log in st.session_state.logs[:50]:  # Show last 50 logs
                log_class = f"log-{log['type']}"
                st.markdown(f"""
                <div class="log-entry">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                        <span style="color: #64748b;">{log['timestamp']}</span>
                        <span class="{log_class}"><strong>{log['type'].upper()}</strong></span>
                    </div>
                    <div class="{log_class}">{log['message']}</div>
                </div>
                """, unsafe_allow_html=True)

# Footer
st.markdown('---')
st.write("LOG COUNT:", len(st.session_state.logs))
st.write(st.session_state.logs)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.875rem;">
    <p>Agentic AI Data Preprocessing Pipeline Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
