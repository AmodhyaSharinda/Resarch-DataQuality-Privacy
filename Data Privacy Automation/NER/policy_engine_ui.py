# policy_ui.py
import json
import streamlit as st
import pandas as pd
import os
from datetime import datetime


POLICY_FILE = "policy_engine.json"
RESOURCE_DIR = "../RAG/privacy-regulation-resources"

os.makedirs(RESOURCE_DIR, exist_ok=True)


def load_policies():
    with open(POLICY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def show_policy_ui():
    st.set_page_config(page_title="Privacy Policy Rules", layout="wide")

    st.title(" Privacy Policy Rules & Regulatory Resources")
    st.caption("Policy definitions and reference regulations")

    # -------------------------
    # Show Existing Policy Rules
    # -------------------------
    policy_data = load_policies()
    policies = policy_data.get("policies", [])

    df = pd.DataFrame(policies)[
        ["entity", "sensitivity_level", "reason", "regulation"]
    ]

    df.columns = [
        "Entity",
        "Sensitivity Level",
        "Reason",
        "Regulation"
    ]

    st.subheader("Policy Rules")
    st.dataframe(df, use_container_width=True)

    st.divider()

    # -------------------------
    # Upload Regulation Resource
    # -------------------------
    st.subheader("📎 Upload Regulatory Resource (PDF)")

    uploaded_pdf = st.file_uploader(
        "Upload regulation document",
        type=["pdf"]
    )

    if st.button("Upload to Resources"):
        if not uploaded_pdf:
            st.warning("Please upload a PDF file.")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                RESOURCE_DIR, f"{timestamp}_{uploaded_pdf.name}"
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            st.success(f"✅ File saved to `{RESOURCE_DIR}/`")
            st.rerun()

    st.divider()

    # # -------------------------
    # # List Uploaded Resources
    # # -------------------------
    # st.subheader("📂 Uploaded Regulatory Resources")

    # files = os.listdir(RESOURCE_DIR)

    # if files:
    #     for f in files:
    #         st.markdown(f"• {f}")
    # else:
    #     st.info("No regulatory documents uploaded yet.")

    if st.button("⬅ Back to Privacy Pipeline"):
        st.session_state.page = "main"
        st.rerun()
