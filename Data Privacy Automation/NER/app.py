import streamlit as st
import pandas as pd

from data_pipeline import find_sensitivity
from policy_engine_ui import show_policy_ui
from file_loader import extract_text_from_file

# ======================================================
# Demo Encryption (UI only – NOT real encryption)
# ======================================================
def demo_encrypt(value, entity_type, method):
    return f"<{entity_type}:{method}>"


# ======================================================
# Page Routing
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "main"


# ======================================================
# Policy Rules Page
# ======================================================
if st.session_state.page == "policy":
    show_policy_ui()
    st.stop()


# ======================================================
# Page Config
# ======================================================
st.set_page_config(
    page_title="Privacy Preservation Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ======================================================
# Enhanced Custom CSS
# ======================================================
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background: #f8f9fa;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
        margin: 0;
        font-weight: 300;
    }
    
    /* Card Styling */
    .pro-card {
        background: #06306F;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .pro-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f0f0;
    }
    
    .card-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a202c;
        margin: 0;
    }
    
    /* Step Indicator */
    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7c87ba 0%, #293097 100%);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    
    /* Process Flow */
    .process-flow {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 2rem 0;
        color: white;
        font-weight: 500;
    }
    
    .flow-arrow {
        font-size: 1.5rem;
        opacity: 0.8;
    }
    
    /* Enhanced Table */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg,#7c87ba 0%, #293097 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 8px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Policy Button */
    .policy-button {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }
    
    /* Text Area Enhancement */
        /* ===== Text Area Text Color Fix ===== */
    .stTextArea textarea {
        color: #ffffff !important;          /* White text */
        background-color: #2d3748 !important; /* Dark background */
        caret-color: #ffffff !important;    /* Cursor color */
    }

    /* Placeholder text color */
    .stTextArea textarea::placeholder {
        color: #cbd5e0 !important;
    }

    /* When focused */
    .stTextArea textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.25) !important;
    }

    
    /* Output Code Block */
    .stCodeBlock {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        background: #f7fafc;
    }
    
    /* Sensitivity Badge */
    .sensitivity-high {
        color: #e53e3e;
        font-weight: 700;
    }
    
    .sensitivity-medium {
        color: #dd6b20;
        font-weight: 700;
    }
    
    .sensitivity-low {
        color: #38a169;
        font-weight: 700;
    }
    
    /* Info Banner */
    .info-banner {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1rem 1.5rem;
        border-radius: 8px;
        color: #1a202c;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    /* Success Message */
    /* Success Message */
    .success-box {
        background: #c6f6d5;
        border-left: 4px solid #38a169;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        color: #1a202c !important;   /* ✅ Black text */
        font-weight: 600;
    }

    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.8rem;
        }
        
        .process-flow {
            flex-direction: column;
            gap: 1rem;
        }
        
        .flow-arrow {
            transform: rotate(90deg);
        }
    }
</style>
""", unsafe_allow_html=True)


# ======================================================
# Header with Modern Design
# ======================================================
st.markdown("""
<div class="main-header">
    <h1> Privacy Preservation Platform</h1>
    <p>Advanced data protection through intelligent entity detection and encryption</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col2:
    if st.button("📜 Privacy Policy Rules", use_container_width=True):
        st.session_state.page = "policy"
        st.rerun()

# ======================================================
# Process Flow Visualization
# ======================================================
# st.markdown("""
# <div class="process-flow">
#     <span>🔍 Detect</span>
#     <span class="flow-arrow">→</span>
#     <span>🏷️ Classify</span>
#     <span class="flow-arrow">→</span>
#     <span>📋 Apply Policy</span>
#     <span class="flow-arrow">→</span>
#     <span>🔒 Encrypt</span>
#     <span class="flow-arrow">→</span>
#     <span>✅ Secure</span>
# </div>
# """, unsafe_allow_html=True)


# ======================================================
# 1️⃣ INPUT SECTION
# ======================================================
st.markdown("""
<div class="pro-card">
    <span class="step-badge">Step 1</span>
    <div class="card-header">
        <span class="card-icon"></span>
        <h3 class="card-title">Input Text</h3>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="margin-top: -1rem;">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a file containing sensitive data",
    type=["txt", "json"]
)

input_text = None

if uploaded_file:
    try:
        input_text = extract_text_from_file(uploaded_file)

        st.success(f"📄 File loaded: {uploaded_file.name}")

        with st.expander("🔍 Preview extracted content"):
            st.code(input_text[:3000], language="text")

    except Exception as e:
        st.error(str(e))
        st.stop()


col1, col2, col3 = st.columns([1, 1, 2])
with col1:
    process_clicked = st.button("Process Text", use_container_width=True)
with col2:
    if st.button(" Clear", use_container_width=True):
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# Processing Logic
# ======================================================
if process_clicked:

    if not input_text.strip():
        st.error("⚠️ Please enter some text to process.")
        st.stop()

    with st.spinner("🔍 Analyzing text and detecting entities..."):
        entities = find_sensitivity(input_text)
        print(entities)
    if not entities:
        st.info("ℹ️ No sensitive entities detected in the provided text.")
        st.stop()

    # ==================================================
    # 2️⃣ ENTITY DETECTION SECTION
    # ==================================================
    st.markdown("""
    <div class="pro-card">
        <span class="step-badge">Step 2</span>
        <div class="card-header">
            <span class="card-icon"></span>
            <h3 class="card-title">Detected Entities</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    unique_entities = {}
    for ent in entities:
        unique_entities.setdefault(
            ent["entity"],
            ent.get("sensitivity_level", "Unknown")
        )

    df_entities = pd.DataFrame(
        [
            {"Entity Type": k, "Sensitivity Level": v}
            for k, v in unique_entities.items()
        ]
    )

    st.markdown('<div style="margin-top: -1rem;">', unsafe_allow_html=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entities", len(unique_entities), delta=None)
    with col2:
        high_count = sum(1 for v in unique_entities.values() if v == "High")
        st.metric("High Sensitivity", high_count, delta=None)
    with col3:
        st.metric("Total Detections", len(entities), delta=None)
    
    st.dataframe(df_entities, use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================================
    # 3️⃣ ENCRYPTION SELECTION SECTION
    # ==================================================
    st.markdown("""
    <div class="pro-card">
        <span class="step-badge">Step 3</span>
        <div class="card-header">
            <span class="card-icon"></span>
            <h3 class="card-title">Encryption Configuration</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: -1rem;">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-banner">
        💡 Select appropriate encryption methods for each entity type based on sensitivity level
    </div>
    """, unsafe_allow_html=True)

    ENCRYPTION_OPTIONS = ["Masking", "Hashing", "AES", "Tokenization"]
    encryption_choices = {}

    cols = st.columns(2)
    for idx, (_, row) in enumerate(df_entities.iterrows()):
        with cols[idx % 2]:
            default_index = 2 if row["Sensitivity Level"] == "High" else 1
            
            sensitivity_class = f"sensitivity-{row['Sensitivity Level'].lower()}"
            st.markdown(f"""
            <div style="margin-bottom: 0.5rem;">
                <strong>{row['Entity Type']}</strong>
                <span class="{sensitivity_class}"> • {row['Sensitivity Level']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            encryption_choices[row["Entity Type"]] = st.selectbox(
                f"Encryption method for {row['Entity Type']}",
                ENCRYPTION_OPTIONS,
                index=default_index,
                key=f"enc_{row['Entity Type']}",
                label_visibility="collapsed"
            )

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================================
    # 4️⃣ OUTPUT SECTION
    # ==================================================
    st.markdown("""
    <div class="pro-card">
        <span class="step-badge">Step 4</span>
        <div class="card-header">
            <span class="card-icon"></span>
            <h3 class="card-title">Privacy-Protected Output</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top: -1rem;">', unsafe_allow_html=True)

    safe_text = input_text

    # Replace longer values first (prevents partial overlap)
    for ent in sorted(entities, key=lambda x: len(x["value"]), reverse=True):
        method = encryption_choices.get(ent["entity"], "Masking")
        safe_text = safe_text.replace(
            ent["value"],
            demo_encrypt(ent["value"], ent["entity"], method)
        )

    st.code(safe_text, language="text")
    
    st.markdown("""
    <div class="success-box">
        ✅ Text successfully processed with privacy preservation applied
    </div>
    """, unsafe_allow_html=True)
    
    # # Download button
    # col1, col2 = st.columns([1, 3])
    # with col1:
    #     st.download_button(
    #         label="📥 Download Protected Text",
    #         data=safe_text,
    #         file_name="privacy_protected_output.txt",
    #         mime="text/plain",
    #         use_container_width=True
    #     )
    
    st.markdown('</div>', unsafe_allow_html=True)