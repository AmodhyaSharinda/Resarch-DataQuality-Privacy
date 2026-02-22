import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
import os
import json
import re
import warnings
from io import StringIO, BytesIO

# ML & XAI
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import chi2_contingency, spearmanr

import shap
from lime.lime_tabular import LimeTabularExplainer

# For PDF reading
try:
    from pypdf import PdfReader
except ImportError:
    st.error("Install pypdf: pip install pypdf")
    st.stop()

# For Groq
try:
    from groq import Groq
except ImportError:
    st.error("Install groq: pip install groq")
    st.stop()

warnings.filterwarnings('ignore')

# Streamlit config
st.set_page_config(page_title="Structured Data Cleaning Tool", layout="wide")

# Dark mode CSS with enhanced styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 50%, #16213e 100%);
        color: #fafafa;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background: linear-gradient(145deg, #262730, #1e1e2e) !important;
        color: #fafafa !important;
        border: 1px solid #4f4f4f !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
    }
    .stButton > button {
        background: linear-gradient(145deg, #4f46e5, #7c3aed) !important;
        color: #fafafa !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4) !important;
        transition: all 0.3s ease !important;
        font-weight: 600 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(145deg, #7c3aed, #4f46e5) !important;
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.6) !important;
        transform: translateY(-2px) !important;
    }
    .stDataFrame, .stTable {
        background: linear-gradient(145deg, #262730, #1e1e2e) !important;
        color: #fafafa !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }
    .stMarkdown, .stText {
        color: #fafafa !important;
    }
    .stHeader, .stSubheader {
        color: #fafafa !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.5) !important;
    }
    .stExpander {
        background: linear-gradient(145deg, #1e1e2e, #262730) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }
    .stMetric {
        background: linear-gradient(145deg, #262730, #1e1e2e) !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
        padding: 10px !important;
    }
    .stSuccess {
        background: linear-gradient(145deg, #10b981, #059669) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stError {
        background: linear-gradient(145deg, #ef4444, #dc2626) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stInfo {
        background: linear-gradient(145deg, #3b82f6, #2563eb) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stWarning {
        background: linear-gradient(145deg, #f59e0b, #d97706) !important;
        color: white !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'rules_df' not in st.session_state:
    st.session_state.rules_df = None
if 'id_columns' not in st.session_state:
    st.session_state.id_columns = []
if 'column_dependencies' not in st.session_state:
    st.session_state.column_dependencies = {}
if 'issues_df' not in st.session_state:
    st.session_state.issues_df = None
if 'quality_report' not in st.session_state:
    st.session_state.quality_report = None
if 'column_models' not in st.session_state:
    st.session_state.column_models = {}
if 'fixes_df' not in st.session_state:
    st.session_state.fixes_df = None
if 'user_selections' not in st.session_state:
    st.session_state.user_selections = {}
if 'df_final' not in st.session_state:
    st.session_state.df_final = None

def main():
    # Fancy title with gradient
    st.markdown("""
    <h1 style="text-align: center; background: linear-gradient(45deg, #4f46e5, #7c3aed, #ec4899);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                background-clip: text; font-size: 3em; font-weight: bold;
                text-shadow: 0 4px 8px rgba(0,0,0,0.5); margin-bottom: 20px;">
        ðŸ” Structured Data Cleaning Tool
    </h1>
    <p style="text-align: center; color: #a0aec0; font-size: 1.2em; margin-top: -10px;">
        ðŸš€ Automated data quality assessment and fixing with AI-powered insights
    </p>
    """, unsafe_allow_html=True)

    # Hardcoded Groq API key
    import os
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        client = Groq(api_key=groq_api_key)
        groq_model = choose_groq_model(client)
        st.success(f"Groq model: {groq_model}")
    else:
        st.error("Groq API key not set")
        return

    # Step 1: Upload Dataset
    st.header("1. Upload Dataset")
    dataset_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
    if dataset_file:
        if dataset_file.name.endswith('.csv'):
            df_raw = pd.read_csv(dataset_file)
        else:
            df_raw = pd.read_excel(dataset_file)
        st.session_state.df_raw = df_raw
        st.success(f"Loaded dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
        st.dataframe(df_raw.head())

    if st.session_state.df_raw is None:
        return

    df_raw = st.session_state.df_raw

    # Step 2: Upload Rules Document
    st.header("2. Upload Rules Document (Optional)")
    rules_file = st.file_uploader("Upload PDF/CSV/XLSX/TXT for rules", type=["pdf", "csv", "xlsx", "xls", "txt"])
    if rules_file:
        rules_df = extract_rules(rules_file, df_raw.columns.tolist(), client, groq_model)
        st.session_state.rules_df = rules_df
        if rules_df is not None:
            st.success("Rules extracted")
            st.dataframe(rules_df)

    # Automatically process background steps
    if st.button("Process Data and Detect Issues"):
        # Identify ID columns
        id_columns = identify_id_columns(df_raw)
        st.session_state.id_columns = id_columns

        # Analyze dependencies
        non_id_columns = [col for col in df_raw.columns if col not in id_columns]
        column_dependencies = analyze_column_dependencies(df_raw, non_id_columns)
        st.session_state.column_dependencies = column_dependencies

        # Detect issues
        issues_df = detect_issues_rules_only(df_raw, st.session_state.rules_df)
        st.session_state.issues_df = issues_df

        # Calculate quality score
        quality_report = calculate_quality_score(df_raw, issues_df, st.session_state.rules_df)
        st.session_state.quality_report = quality_report

        # Train models
        column_models = train_models_with_dependencies(df_raw, column_dependencies, id_columns)
        st.session_state.column_models = column_models

        # Generate fixes
        fixes_df = build_fixes_dataframe(df_raw, issues_df, column_models, st.session_state.rules_df)
        st.session_state.fixes_df = fixes_df

        st.success("Processing complete!")

    # Step 3: Issues Detected
    if st.session_state.issues_df is not None:
        st.header("3. Issues Detected")
        st.write(f"Found {len(st.session_state.issues_df)} issues")
        
        # Summary metrics with colored cards
        missing_count = len(st.session_state.issues_df[st.session_state.issues_df['issue_type'] == 'missing'])
        outlier_count = len(st.session_state.issues_df[st.session_state.issues_df['issue_type'] == 'outlier'])
        invalid_count = len(st.session_state.issues_df[st.session_state.issues_df['issue_type'] == 'invalid'])
        rule_violations = outlier_count + invalid_count
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                        border-radius: 10px; padding: 20px; color: white; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;">
                <h3 style="margin: 0 0 10px 0;">â“ Missing Values</h3>
                <div style="font-size: 36px; font-weight: bold;">{missing_count}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                        border-radius: 10px; padding: 20px; color: white; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); text-align: center;">
                <h3 style="margin: 0 0 10px 0;">ðŸš¨ Rule Violations</h3>
                <div style="font-size: 36px; font-weight: bold;">{rule_violations}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Display all issues
        st.dataframe(st.session_state.issues_df)

        # Quality score
        if st.session_state.quality_report:
            display_quality_report_streamlit(st.session_state.quality_report)

    # Step 4: Select Fixes
    if st.session_state.fixes_df is not None:
        st.header("4. Select Fixes")
        fixes_df = st.session_state.fixes_df
        
        # Add some creative elements
        st.markdown("ðŸŽ¯ **Review and select the best fix for each data quality issue below.**")
        st.markdown("ðŸ’¡ *Click on each issue to expand and see details, then choose your preferred fix.*")
        
        for idx, fix in fixes_df.iterrows():
            issue_type = fix['issue_type']
            issue_icon = "ðŸš¨" if issue_type in ["outlier", "invalid"] else "â“"
            
            with st.expander(f"{issue_icon} Issue {idx+1}: Row {fix['row']} | Column '{fix['column']}' | {issue_type.title()}"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown(f"**Current Value:** `{fix['current_value']}`")
                    if fix.get('rule_violated'):
                        st.markdown(f"**Rule Violated:** {fix['rule_violated']}")
                    st.markdown(f"**Issue Type:** {issue_type.title()}")
                
                with col2:
                    options = []
                    if fix['ml_suggestion'] is not None:
                        options.append(f"ðŸ¤– ML Suggested - {fix['ml_suggestion']}")
                    
                    for i, alt in enumerate(fix['alternatives'], 1):
                        options.append(f"ðŸ“Š Alternative {i}: {alt['method'].title()} - {alt['value']}")
                    
                    if fix.get('cap_suggestion') is not None:
                        options.append(f"ðŸ”§ Cap to Boundary - {fix['cap_suggestion']}")
                    
                    options.append(f"â¸ï¸ Keep Current - {fix['current_value']}")
                    
                    choice = st.selectbox(
                        "Choose your fix:",
                        options,
                        key=f"fix_{idx}",
                        help="Select the best option for this data issue"
                    )
                    
                    st.session_state.user_selections[(fix['row'], fix['column'])] = choice.split(" - ")[1] if " - " in choice else choice
                    
                    # Show explanation button
                    if st.button(f"ðŸ” Why this suggestion? (SHAP & LIME)", key=f"exp_{idx}"):
                        st.markdown(fix['xai_explanation'])

    # Step 5: Apply Fixes
    st.header("5. Apply Fixes")
    if st.button("Apply Fixes"):
        df_final = apply_fixes(df_raw, st.session_state.user_selections)
        st.session_state.df_final = df_final
        st.success("Fixes applied")
        st.dataframe(df_final.head())

        # Quality improvement
        final_issues = detect_issues_rules_only(df_final, st.session_state.rules_df)
        final_quality = calculate_quality_score(df_final, final_issues, st.session_state.rules_df)
        st.write("Quality improvement:")
        if st.session_state.quality_report:
            st.write(f"Before: {st.session_state.quality_report['total_score']}")
        st.write(f"After: {final_quality['total_score']}")

    # Step 6: Download Cleaned Dataset
    if st.session_state.df_final is not None:
        st.header("6. Download Cleaned Dataset")
        csv = st.session_state.df_final.to_csv(index=False)
        st.download_button("Download Cleaned Dataset", csv, "cleaned_dataset.csv", "text/csv")

# Helper functions (adapted from notebook)

def choose_groq_model(client):
    candidates = ["llama-3.3-70b-versatile", "llama-3.2-90b-vision-preview", "llama-3.1-8b-instant"]
    try:
        available = [m["id"] for m in client.models.list()]
        for c in candidates:
            if c in available:
                return c
        return available[0] if available else candidates[0]
    except:
        return candidates[0]

def decode_bytes(raw_bytes):
    """Decode bytes with fallback encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for enc in encodings:
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    # If all fail, use 'latin-1' as it can decode any byte
    return raw_bytes.decode('latin-1', errors='replace')

def build_extraction_prompt(doc_text, columns):
    return f"""Extract data validation rules from this document and output ONLY a CSV table with one row per column.

Dataset columns: {', '.join(columns)}

Document:
{doc_text[:3000]}

Output CSV format (no other text, no headers, just data):
Column,dtype,min,max,allowed_values,constraints

Instructions:
- Column: exact column name from the dataset columns list above
- dtype: 'integer', 'float', 'categorical', 'text', or 'boolean'
- min: numeric minimum value for integer/float columns (leave blank for categorical/text/boolean)
- max: numeric maximum value for integer/float columns (leave blank for categorical/text/boolean)
- allowed_values: pipe-separated list of valid values for categorical columns only, like "Male|Female|Other" (leave blank for numeric columns)
- constraints: additional constraints for numeric columns, like "positive", "negative", "non-negative", "non-positive" (leave blank if none)

Examples:
- For a numeric column "Age": Age,integer,0,120,,positive
- For a categorical column "Gender": Gender,categorical,,,,Male|Female|Other
- For a text column with no rules: Description,text,,,,,

Only include columns that have rules mentioned in the document. Leave fields blank if not specified.
"""

def extract_rules(rules_file, columns, client, groq_model):
    # First, try to parse as direct CSV file
    try:
        raw_bytes = rules_file.read()
        doc_text = decode_bytes(raw_bytes)
        from io import StringIO
        rules_df = pd.read_csv(StringIO(doc_text), on_bad_lines='skip')
        # Check if it has the expected columns
        expected_cols = ['Column', 'dtype', 'min', 'max', 'allowed_values', 'constraints']
        if len(rules_df.columns) >= 6 and all(col in rules_df.columns for col in expected_cols[:len(rules_df.columns)]):
            return rules_df
        # If not, continue to LLM extraction
    except:
        pass  # Continue to LLM extraction
    
    # Reset file pointer and extract text for LLM
    rules_file.seek(0)
    if rules_file.type == "application/pdf":
        doc_text = read_pdf_text(rules_file)
    else:
        raw_bytes = rules_file.read()
        doc_text = decode_bytes(raw_bytes)

    if doc_text:
        prompt = build_extraction_prompt(doc_text, columns)
        try:
            resp = client.chat.completions.create(
                model=groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            csv_text = resp.choices[0].message.content.strip()

            # Parse CSV response robustly
            from io import StringIO
            import csv
            
            # Try pandas first
            try:
                rules_df = pd.read_csv(StringIO(csv_text), on_bad_lines='skip')
            except Exception:
                # Fallback to manual parsing
                lines = csv_text.split('\n')
                data = []
                for line in lines:
                    if line.strip():
                        # Split by comma, but handle quoted fields
                        reader = csv.reader([line], delimiter=',')
                        try:
                            row = next(reader)
                            if len(row) >= 6:  # At least Column,dtype,min,max,allowed_values,constraints
                                data.append(row[:6])  # Take first 6 fields
                        except:
                            continue
                if data:
                    rules_df = pd.DataFrame(data, columns=['Column', 'dtype', 'min', 'max', 'allowed_values', 'constraints'])
                else:
                    raise ValueError("Could not parse CSV response")
            
            # Check if columns are correct, if not, assume no header
            expected_cols = ['Column', 'dtype', 'min', 'max', 'allowed_values', 'constraints']
            if not all(col in rules_df.columns for col in expected_cols):
                # Try reading without header
                try:
                    rules_df = pd.read_csv(StringIO(csv_text), header=None, names=expected_cols, on_bad_lines='skip')
                except:
                    # Manual parsing without header
                    lines = csv_text.split('\n')
                    data = []
                    for line in lines:
                        if line.strip():
                            reader = csv.reader([line], delimiter=',')
                            try:
                                row = next(reader)
                                data.append(row[:6] + [''] * (6 - len(row)))  # Pad to 6 fields
                            except:
                                continue
                    rules_df = pd.DataFrame(data, columns=expected_cols)

            return rules_df
        except Exception as e:
            st.error(f"Rule extraction failed: {e}")
    return None

def read_pdf_text(file):
    reader = PdfReader(BytesIO(file.read()))
    pages = []
    for p in reader.pages:
        t = p.extract_text()
        if t:
            pages.append(t)
    return "\n".join(pages).strip()

def build_extraction_prompt(doc_text, columns):
    return f"""Extract data validation rules from this document and output ONLY a CSV table with one row per column.

Dataset columns: {', '.join(columns)}

Document:
{doc_text[:3000]}

Output CSV format (no other text, no headers, just data):
Column,dtype,min,max,allowed_values

Instructions:
- Column: exact column name from the dataset columns list above
- dtype: 'integer', 'float', 'categorical', 'text', or 'boolean'
- min: numeric minimum value for integer/float columns (leave blank for categorical/text/boolean)
- max: numeric maximum value for integer/float columns (leave blank for categorical/text/boolean)
- allowed_values: pipe-separated list of valid values for categorical columns only, like "Male|Female|Other" (leave blank for numeric columns)

Examples:
- For a numeric column "Age": Age,integer,0,120,
- For a categorical column "Gender": Gender,categorical,,,Male|Female|Other
- For a text column with no rules: Description,text,,,

Only include columns that have rules mentioned in the document. Leave fields blank if not specified.
"""

def identify_id_columns(df):
    id_columns = []
    for col in df.columns:
        if 'id' in col.lower():
            id_columns.append(col)
            continue
        if len(df) > 0:
            uniqueness = df[col].nunique() / len(df)
            if uniqueness > 0.95:
                id_columns.append(col)
                continue
        if df[col].dtype in [np.int64, np.int32]:
            non_null = df[col].dropna()
            if len(non_null) > 1:
                diffs = non_null.diff().dropna()
                if len(diffs) > 0 and (diffs == 1).sum() / len(diffs) > 0.9:
                    id_columns.append(col)
    return id_columns

def analyze_column_dependencies(df, non_id_cols, threshold=0.1):
    dependencies = {}
    for target_col in non_id_cols:
        dependent_cols = []
        target_data = df[target_col].dropna()
        if len(target_data) < 10:
            dependencies[target_col] = []
            continue
        for predictor_col in non_id_cols:
            if predictor_col == target_col:
                continue
            valid_idx = df[[target_col, predictor_col]].dropna().index
            if len(valid_idx) < 10:
                continue
            X = df.loc[valid_idx, predictor_col]
            y = df.loc[valid_idx, target_col]
            try:
                if df[target_col].dtype == 'object':
                    if df[predictor_col].dtype == 'object':
                        contingency = pd.crosstab(X, y)
                        chi2, p_value, _, _ = chi2_contingency(contingency)
                        score = 1 - p_value
                    else:
                        X_encoded = LabelEncoder().fit_transform(y)
                        score = mutual_info_regression(X.values.reshape(-1, 1), X_encoded)[0]
                else:
                    if df[predictor_col].dtype == 'object':
                        X_encoded = LabelEncoder().fit_transform(X)
                        score = mutual_info_regression(X_encoded.reshape(-1, 1), y)[0]
                    else:
                        score = abs(spearmanr(X, y)[0])
                if score > threshold:
                    dependent_cols.append((predictor_col, score))
            except:
                pass
        dependent_cols.sort(key=lambda x: x[1], reverse=True)
        dependencies[target_col] = [col for col, score in dependent_cols]
    return dependencies

def detect_issues_rules_only(df, rules_df=None):
    issues = []
    for col in df.columns:
        missing_idx = df[df[col].isna()].index.tolist()
        for idx in missing_idx:
            issues.append({
                'row': idx,
                'column': col,
                'issue_type': 'missing',
                'current_value': None,
                'rule_violated': None
            })
    if rules_df is not None and 'Column' in rules_df.columns:
        for _, rule in rules_df.iterrows():
            col = rule['Column']
            if col not in df.columns:
                continue
            # Check min
            if pd.notna(rule.get('min')):
                try:
                    min_val = float(rule['min'])
                    # Ensure numeric comparison
                    numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
                    violating_idx = df[(df[col].notna()) & numeric_mask & (pd.to_numeric(df[col], errors='coerce') < min_val)].index.tolist()
                    for idx in violating_idx:
                        issues.append({
                            'row': idx,
                            'column': col,
                            'issue_type': 'outlier',
                            'current_value': df.loc[idx, col],
                            'rule_violated': f'Below minimum: {min_val}'
                        })
                except:
                    pass
            # Check max
            if pd.notna(rule.get('max')):
                try:
                    max_val = float(rule['max'])
                    # Ensure numeric comparison
                    numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
                    violating_idx = df[(df[col].notna()) & numeric_mask & (pd.to_numeric(df[col], errors='coerce') > max_val)].index.tolist()
                    for idx in violating_idx:
                        issues.append({
                            'row': idx,
                            'column': col,
                            'issue_type': 'outlier',
                            'current_value': df.loc[idx, col],
                            'rule_violated': f'Above maximum: {max_val}'
                        })
                except:
                    pass
            # Check allowed values
            if pd.notna(rule.get('allowed_values')):
                allowed = [v.strip() for v in str(rule['allowed_values']).split('|')]
                violating_idx = df[(df[col].notna()) & (~df[col].isin(allowed))].index.tolist()
                for idx in violating_idx:
                    issues.append({
                        'row': idx,
                        'column': col,
                        'issue_type': 'invalid',
                        'current_value': df.loc[idx, col],
                        'rule_violated': f'Not in allowed values: {", ".join(allowed)}'
                    })
    # Additional statistical outlier detection for numeric columns without min/max rules
    for col in df.columns:
        if col in (rules_df['Column'].tolist() if rules_df is not None and 'Column' in rules_df.columns else []):
            rule = rules_df[rules_df['Column'] == col].iloc[0] if len(rules_df[rules_df['Column'] == col]) > 0 else None
            has_min_max = rule is not None and (pd.notna(rule.get('min')) or pd.notna(rule.get('max')))
        else:
            has_min_max = False
        if not has_min_max and df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                if len(numeric_data) > 10:  # Only if enough data
                    Q1 = numeric_data.quantile(0.25)
                    Q3 = numeric_data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outlier_idx = df[(df[col].notna()) & (pd.to_numeric(df[col], errors='coerce') < lower_bound) | (pd.to_numeric(df[col], errors='coerce') > upper_bound)].index.tolist()
                    for idx in outlier_idx:
                        issues.append({
                            'row': idx,
                            'column': col,
                            'issue_type': 'outlier',
                            'current_value': df.loc[idx, col],
                            'rule_violated': f'Statistical outlier (IQR method): outside [{lower_bound:.2f}, {upper_bound:.2f}]'
                        })
            except:
                pass
    return pd.DataFrame(issues) if issues else pd.DataFrame(columns=['row', 'column', 'issue_type', 'current_value', 'rule_violated'])

def calculate_quality_score(df, issues_df, rules_df=None):
    total_cells = df.shape[0] * df.shape[1]
    total_rows = df.shape[0]
    
    missing_issues = issues_df[issues_df['issue_type'] == 'missing']
    missing_count = len(missing_issues)
    missing_percentage = (missing_count / total_cells) * 100
    
    if missing_percentage == 0:
        completeness_score = 40
    elif missing_percentage < 1:
        completeness_score = 35
    elif missing_percentage < 5:
        completeness_score = 30
    elif missing_percentage < 10:
        completeness_score = 20
    elif missing_percentage < 20:
        completeness_score = 10
    else:
        completeness_score = max(0, 40 - (missing_percentage * 2))
    
    outlier_issues = issues_df[issues_df['issue_type'].isin(['outlier', 'invalid'])]
    outlier_count = len(outlier_issues)
    outlier_percentage = (outlier_count / total_cells) * 100
    
    if outlier_percentage == 0:
        validity_score = 40
    elif outlier_percentage < 1:
        validity_score = 35
    elif outlier_percentage < 5:
        validity_score = 25
    elif outlier_percentage < 10:
        validity_score = 15
    elif outlier_percentage < 20:
        validity_score = 5
    else:
        validity_score = max(0, 40 - (outlier_percentage * 2))
    
    duplicate_rows = df.duplicated().sum()
    duplicate_percentage = (duplicate_rows / total_rows) * 100
    
    low_variance_cols = 0
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() == 1 and len(df[col].dropna()) > 0:
            low_variance_cols += 1
    
    consistency_score = 20
    consistency_score -= min(10, duplicate_percentage * 2)
    consistency_score -= min(5, low_variance_cols * 2)
    consistency_score = max(0, consistency_score)
    
    total_score = completeness_score + validity_score + consistency_score
    
    if total_score >= 90:
        grade = "A (Excellent)"
        color = "#27ae60"
    elif total_score >= 75:
        grade = "B (Good)"
        color = "#2ecc71"
    elif total_score >= 60:
        grade = "C (Fair)"
        color = "#f39c12"
    elif total_score >= 40:
        grade = "D (Poor)"
        color = "#e67e22"
    else:
        grade = "F (Critical)"
        color = "#e74c3c"
    
    report = {
        'total_score': round(total_score, 1),
        'grade': grade,
        'color': color,
        'completeness': {
            'score': round(completeness_score, 1),
            'percentage': round(100 - missing_percentage, 1),
            'missing_count': missing_count,
            'missing_percentage': round(missing_percentage, 2)
        },
        'validity': {
            'score': round(validity_score, 1),
            'percentage': round(100 - outlier_percentage, 1),
            'outlier_count': outlier_count,
            'outlier_percentage': round(outlier_percentage, 2)
        },
        'consistency': {
            'score': round(consistency_score, 1),
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': round(duplicate_percentage, 2),
            'low_variance_columns': low_variance_cols
        },
        'total_cells': total_cells,
        'total_rows': total_rows,
        'total_issues': len(issues_df)
    }
    return report

def display_quality_report_streamlit(report):
    # Card-style container for quality report
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; padding: 25px; margin: 20px 0; color: white; 
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);">
        <h2 style="margin: 0 0 20px 0; text-align: center;">ðŸ“Š Dataset Quality Score</h2>
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="font-size: 48px; font-weight: bold;">{}</div>
            <div style="font-size: 24px;">{}</div>
        </div>
    </div>
    """.format(report['total_score'], report['grade']), unsafe_allow_html=True)
    
    # Component cards in a grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comp = report['completeness']
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 20px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-top: 4px solid #3498db; 
                    text-align: center;">
            <h4 style="margin: 0 0 10px 0; color: #3498db;">ðŸ“ Completeness</h4>
            <div style="font-size: 32px; font-weight: bold; color: #2c3e50;">{comp['score']}/{comp.get('max', 40)}</div>
            <div style="background: #ecf0f1; border-radius: 10px; height: 12px; margin: 10px 0;">
                <div style="background: #3498db; height: 100%; width: {comp['percentage']}%; border-radius: 10px;"></div>
            </div>
            <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
                âœ“ {comp['percentage']}% complete<br>
                âœ— {comp['missing_count']:,} missing
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        val = report['validity']
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 20px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-top: 4px solid #9b59b6; 
                    text-align: center;">
            <h4 style="margin: 0 0 10px 0; color: #9b59b6;">âœ… Validity</h4>
            <div style="font-size: 32px; font-weight: bold; color: #2c3e50;">{val['score']}/{val.get('max', 40)}</div>
            <div style="background: #ecf0f1; border-radius: 10px; height: 12px; margin: 10px 0;">
                <div style="background: #9b59b6; height: 100%; width: {val['percentage']}%; border-radius: 10px;"></div>
            </div>
            <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
                âœ“ {val['percentage']}% valid<br>
                âœ— {val['outlier_count']:,} violations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cons = report['consistency']
        cons_percentage = (cons['score'] / cons.get('max', 20)) * 100
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 20px; 
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-top: 4px solid #e67e22; 
                    text-align: center;">
            <h4 style="margin: 0 0 10px 0; color: #e67e22;">ðŸ”„ Consistency</h4>
            <div style="font-size: 32px; font-weight: bold; color: #2c3e50;">{cons['score']}/{cons.get('max', 20)}</div>
            <div style="background: #ecf0f1; border-radius: 10px; height: 12px; margin: 10px 0;">
                <div style="background: #e67e22; height: 100%; width: {cons_percentage}%; border-radius: 10px;"></div>
            </div>
            <p style="margin: 5px 0; font-size: 12px; color: #7f8c8d;">
                ðŸ” {cons['duplicate_rows']} duplicates<br>
                ðŸ“Š {cons['low_variance_columns']} constant cols
            </p>
        </div>
        """, unsafe_allow_html=True)

def train_models_with_dependencies(df, dependencies, id_cols):
    column_models = {}
    MIN_TRAIN = 10
    for target_col in df.columns:
        if target_col in id_cols:
            continue
        predictor_cols = dependencies.get(target_col, [])
        predictor_cols = [c for c in predictor_cols if c not in id_cols]
        if not predictor_cols:
            continue
        train_idx = df[target_col].dropna().index
        if len(train_idx) < MIN_TRAIN:
            continue
        X = df[predictor_cols]
        y = df.loc[train_idx, target_col]
        cat_cols = [c for c in predictor_cols if df[c].dtype == 'object']
        num_cols = [c for c in predictor_cols if df[c].dtype != 'object']
        transformers = []
        if num_cols:
            transformers.append(('num', SimpleImputer(strategy='median'), num_cols))
        if cat_cols:
            transformers.append(('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_cols))
        if not transformers:
            continue
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        label_encoder = None
        if df[target_col].dtype == 'object':
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y)
            model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        else:
            y_encoded = y
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        try:
            pipeline.fit(X.loc[train_idx], y_encoded)
            column_models[target_col] = {
                'pipeline': pipeline,
                'label_encoder': label_encoder,
                'predictor_cols': predictor_cols
            }
        except:
            pass
    return column_models

def ml_predict_with_xai(df, row, col, column_models, rules_df=None, xai_method='both'):
    if col not in column_models:
        return None, "No ML model available"
    model_info = column_models[col]
    pipeline = model_info['pipeline']
    label_encoder = model_info['label_encoder']
    predictor_cols = model_info['predictor_cols']
    X_row = df.loc[[row], predictor_cols]
    try:
        pred = pipeline.predict(X_row)[0]
        if label_encoder is not None:
            pred = label_encoder.inverse_transform([int(pred)])[0]
        else:
            # Apply constraints for numeric predictions
            if rules_df is not None and 'constraints' in rules_df.columns:
                rule = rules_df[rules_df['Column'] == col]
                if len(rule) > 0:
                    constraints = str(rule.iloc[0].get('constraints', '')).lower()
                    if isinstance(pred, (int, float)):
                        if 'positive' in constraints and pred <= 0:
                            pred = abs(pred) + 1 if pred != 0 else 1
                        elif 'negative' in constraints and pred >= 0:
                            pred = -abs(pred) - 1 if pred != 0 else -1
                        elif 'non-negative' in constraints and pred < 0:
                            pred = 0
                        elif 'non-positive' in constraints and pred > 0:
                            pred = 0
        # Align prediction with column data type from rules or dataframe
        pred = cast_to_dtype_from_rules(pred, col, rules_df, df)
        xai_text = generate_combined_xai_explanation(df, row, col, X_row, pipeline, predictor_cols, label_encoder, xai_method)
        return pred, xai_text
    except:
        return None, "Prediction failed"

def generate_combined_xai_explanation(df, row, col, X_row, pipeline, predictor_cols, label_encoder, xai_method='both'):
    explanations = []
    current_values = {}
    for pc in predictor_cols:
        val = df.loc[X_row.index[0], pc]
        current_values[pc] = val
    header = f"**ML Prediction based on {len(predictor_cols)} dependent columns:**\n"
    header += "\n".join([f"- `{c}` = {current_values[c]}" for c in predictor_cols[:5]])
    if len(predictor_cols) > 5:
        header += f"\n- ... and {len(predictor_cols)-5} more"
    explanations.append(header)
    if xai_method in ['shap', 'both']:
        shap_exp = generate_shap_explanation(X_row, pipeline, predictor_cols)
        if shap_exp:
            explanations.append("\n---\n### SHAP Analysis")
            explanations.append(shap_exp)
    if xai_method in ['lime', 'both']:
        lime_exp = generate_lime_explanation(df, X_row, pipeline, predictor_cols, label_encoder)
        if lime_exp:
            explanations.append("\n---\n### LIME Analysis")
            explanations.append(lime_exp)
    return "\n".join(explanations)

def generate_shap_explanation(X_row, pipeline, predictor_cols):
    try:
        X_transformed = pipeline.named_steps['preprocessor'].transform(X_row)
        model = pipeline.named_steps['model']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        feature_names = []
        for name, trans, cols in pipeline.named_steps['preprocessor'].transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                ohe = trans.named_steps['onehot']
                feature_names.extend(ohe.get_feature_names_out(cols))
        if isinstance(shap_values, list):
            shap_vals = shap_values[0][0]
        else:
            shap_vals = shap_values[0]
        top_indices = np.argsort(np.abs(shap_vals))[-5:][::-1]
        explanation = "**SHAP shows feature importance:**\n"
        for i, idx in enumerate(top_indices, 1):
            if idx < len(feature_names):
                feat_name = feature_names[idx]
                impact = "increases" if shap_vals[idx] > 0 else "decreases"
                strength = abs(shap_vals[idx])
                explanation += f"{i}. `{feat_name}` {impact} likelihood (impact: {strength:.3f})\n"
        return explanation
    except:
        return None

def generate_lime_explanation(df, X_row, pipeline, predictor_cols, label_encoder):
    try:
        X_train = df[predictor_cols].dropna()
        if len(X_train) > 100:
            X_train = X_train.sample(100, random_state=42)
        X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
        feature_names = []
        for name, trans, cols in pipeline.named_steps['preprocessor'].transformers_:
            if name == 'num':
                feature_names.extend(cols)
            elif name == 'cat':
                ohe = trans.named_steps['onehot']
                feature_names.extend(ohe.get_feature_names_out(cols))
        mode = 'classification' if label_encoder is not None else 'regression'
        explainer = LimeTabularExplainer(X_train_transformed, mode=mode, feature_names=feature_names, discretize_continuous=True)
        model = pipeline.named_steps['model']
        if label_encoder is not None:
            predict_fn = model.predict_proba
        else:
            predict_fn = model.predict
        X_instance = pipeline.named_steps['preprocessor'].transform(X_row)
        exp = explainer.explain_instance(X_instance[0], predict_fn, num_features=5)
        explanation = "**LIME shows local feature contributions:**\n"
        for feat, weight in exp.as_list()[:5]:
            direction = "supports" if weight > 0 else "opposes"
            explanation += f"- {feat} {direction} this prediction (weight: {weight:.3f})\n"
        return explanation
    except:
        return None

def generate_alternatives_from_rules(df, row, col, rules_df):
    alternatives = []
    if rules_df is not None and 'Column' in rules_df.columns:
        rule = rules_df[rules_df['Column'] == col]
        if len(rule) > 0:
            rule = rule.iloc[0]
            if pd.notna(rule.get('allowed_values')):
                allowed = [v.strip() for v in str(rule['allowed_values']).split('|')]
                value_counts = df[col].value_counts()
                for val in value_counts.index:
                    if val in allowed:
                        alternatives.append({'value': cast_to_dtype_from_rules(val, col, rules_df, df), 'method': 'allowed_value'})
                    if len(alternatives) >= 3:
                        break
                if len(alternatives) < 3:
                    for val in allowed:
                        if val not in alternatives:
                            alternatives.append({'value': cast_to_dtype_from_rules(val, col, rules_df, df), 'method': 'allowed_value'})
                        if len(alternatives) >= 3:
                            break
                if alternatives:
                    return alternatives[:3]
            if df[col].dtype != 'object':
                min_val = None
                max_val = None
                if pd.notna(rule.get('min')):
                    min_val = float(rule.get('min'))
                if pd.notna(rule.get('max')):
                    max_val = float(rule.get('max'))
                if min_val is not None or max_val is not None:
                    valid_data = df[col].dropna()
                    if min_val is not None:
                        valid_data = valid_data[valid_data >= min_val]
                    if max_val is not None:
                        valid_data = valid_data[valid_data <= max_val]
                    if len(valid_data) > 0:
                        alternatives.append({'value': cast_to_dtype_from_rules(valid_data.median(), col, rules_df, df), 'method': 'median'})
                        q25 = valid_data.quantile(0.25)
                        q75 = valid_data.quantile(0.75)
                        if q25 not in [a['value'] for a in alternatives]:
                            alternatives.append({'value': cast_to_dtype_from_rules(q25, col, rules_df, df), 'method': 'q25'})
                        if q75 not in [a['value'] for a in alternatives]:
                            alternatives.append({'value': cast_to_dtype_from_rules(q75, col, rules_df, df), 'method': 'q75'})
                        return alternatives[:3]
    if df[col].dtype == 'object':
        mode_vals = df[col].mode()
        for mv in mode_vals.tolist()[:3]:
            alternatives.append({'value': cast_to_dtype_from_rules(mv, col, rules_df, df), 'method': 'mode'})
    else:
        median_val = df[col].median()
        mean_val = df[col].mean()
        mode_vals = df[col].mode()
        if pd.notna(median_val):
            alternatives.append({'value': cast_to_dtype_from_rules(median_val, col, rules_df, df), 'method': 'median'})
        if pd.notna(mean_val) and mean_val not in [a['value'] for a in alternatives]:
            alternatives.append({'value': cast_to_dtype_from_rules(mean_val, col, rules_df, df), 'method': 'mean'})
        for mv in mode_vals.tolist():
            if mv not in [a['value'] for a in alternatives]:
                alternatives.append({'value': cast_to_dtype_from_rules(mv, col, rules_df, df), 'method': 'mode'})
            if len(alternatives) >= 3:
                break
    # Apply constraints if available
    if rules_df is not None and 'constraints' in rules_df.columns:
        rule = rules_df[rules_df['Column'] == col]
        if len(rule) > 0:
            constraints = str(rule.iloc[0].get('constraints', '')).lower()
            if df[col].dtype != 'object':  # Only for numeric
                if 'positive' in constraints:
                    alternatives = [a for a in alternatives if isinstance(a['value'], (int, float)) and a['value'] > 0]
                elif 'negative' in constraints:
                    alternatives = [a for a in alternatives if isinstance(a['value'], (int, float)) and a['value'] < 0]
                elif 'non-negative' in constraints:
                    alternatives = [a for a in alternatives if isinstance(a['value'], (int, float)) and a['value'] >= 0]
                elif 'non-positive' in constraints:
                    alternatives = [a for a in alternatives if isinstance(a['value'], (int, float)) and a['value'] <= 0]
    return alternatives[:3]

def cast_to_dtype_from_rules(value, col, rules_df, df):
    """Cast value to data type based on rules document or dataframe dtype"""
    try:
        # Check if rules specify dtype
        if rules_df is not None and 'Column' in rules_df.columns and 'dtype' in rules_df.columns:
            rule = rules_df[rules_df['Column'] == col]
            if len(rule) > 0:
                rule_dtype = rule.iloc[0]['dtype']
                if rule_dtype == 'integer':
                    return int(round(float(value)))
                elif rule_dtype == 'float':
                    return float(value)
                else:  # categorical, text, boolean
                    return str(value)
        
        # Fallback to dataframe dtype
        if df[col].dtype in ['int64', 'int32']:
            return int(round(float(value)))
        elif df[col].dtype in ['float64', 'float32']:
            return float(value)
        else:
            return str(value)
    except:
        return value

def build_fixes_dataframe(df, issues_df, column_models, rules_df):
    fixes = []
    for _, issue in issues_df.iterrows():
        row = issue['row']
        col = issue['column']
        issue_type = issue['issue_type']
        current_val = issue['current_value']
        rule_violated = issue.get('rule_violated', None)
        ml_suggestion, xai_explanation = ml_predict_with_xai(df, row, col, column_models, rules_df, xai_method='both')
        alternatives = generate_alternatives_from_rules(df, row, col, rules_df)
        alternatives = [v for v in alternatives if v != ml_suggestion]
        
        # Add cap to boundary suggestion for numerical outliers
        cap_suggestion = None
        if issue_type == 'outlier' and rules_df is not None:
            rule = rules_df[rules_df['Column'] == col]
            if len(rule) > 0:
                rule = rule.iloc[0]
                if pd.notna(rule.get('min')) and pd.notna(current_val) and current_val < float(rule['min']):
                    cap_suggestion = cast_to_dtype_from_rules(float(rule['min']), col, rules_df, df)
                elif pd.notna(rule.get('max')) and pd.notna(current_val) and current_val > float(rule['max']):
                    cap_suggestion = cast_to_dtype_from_rules(float(rule['max']), col, rules_df, df)
        
        fixes.append({
            'row': row,
            'column': col,
            'issue_type': issue_type,
            'current_value': current_val,
            'rule_violated': rule_violated,
            'ml_suggestion': ml_suggestion,
            'alternatives': alternatives,
            'cap_suggestion': cap_suggestion,
            'xai_explanation': xai_explanation
        })
    return pd.DataFrame(fixes)

def apply_fixes(df, user_selections):
    df_fixed = df.copy()
    for (row, col), value in user_selections.items():
        df_fixed.loc[row, col] = value
    return df_fixed

if __name__ == "__main__":
    main()