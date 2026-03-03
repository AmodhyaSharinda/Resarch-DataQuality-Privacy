from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
import pandas as pd
import json
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Optional

from data_pipeline import find_sensitivity
from file_loader import extract_text_from_file
from data_protection import (
    select_protection_method,
    apply_selected_protection,
    calculate_risk_score
)

app = FastAPI()

class SuggestEntityItem(BaseModel):
    value: str
    entity: str
    sensitivity_level: str

class ApplyEntityItem(BaseModel):
    value: str
    entity: str
    sensitivity_level: str
    selected_method: str

class RiskScoreRequest(BaseModel):
    sensitivity_level: str
    data_usage_context: str

class EntityItem(BaseModel):
    entity: str
    risk_score: float

class SuggestRequest(BaseModel):
    entities: List[EntityItem]

class ApplyRequest(BaseModel):
    entities: List[ApplyEntityItem]
    original_content: str
    data_usage_context: str

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CONTEXT_OPTIONS = {
    "logging": "logging",
    "storage": "storage",
    "analytics": "analytics",
    "external_transfer": "external_transfer",
}


# ======================================================
# File Text Extraction (like your Streamlit logic)
# ======================================================

def extract_text_from_file(file: UploadFile):
    file_type = file.filename.split(".")[-1].lower()

    try:
        content = file.file.read()

        # TXT
        if file_type == "txt":
            return content.decode("utf-8")

        # JSON
        elif file_type == "json":
            data = json.loads(content.decode("utf-8"))
            return json.dumps(data, indent=2)

        # CSV
        elif file_type == "csv":
            df = pd.read_csv(pd.io.common.BytesIO(content))
            return df.to_string()

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# Upload Endpoint
# ======================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    data_usage_context: str = Form(...)
):
    # Validate context
    if data_usage_context not in CONTEXT_OPTIONS:
        raise HTTPException(status_code=400, detail="Invalid data usage context")

    # Extract file text
    extracted_text = extract_text_from_file(file)

    # 🔥 CALL YOUR SENSITIVITY PIPELINE
    sensitive_entities = find_sensitivity(extracted_text)

    return {
        "filename": file.filename,
        "data_usage_context": data_usage_context,
        "original_content": extracted_text,   # 👈 ADD THIS
        "content_preview": extracted_text[:1000],
        "sensitive_entities": sensitive_entities
    }


@app.get("/policy-rules")
async def get_policy_rules():
    import json
    with open("policy_engine.json", "r") as f:
        policy = json.load(f)
    return policy


@app.post("/assign-sensitivity")
async def assign_sensitivity(entities: list[dict]):
    with open("policy_engine.json", "r") as f:
        policy = json.load(f)

    results = []

    for entity in entities:
        label = entity.get("entity") or entity.get("entity_group")
        sensitivity = policy.get(label, {}).get("sensitivity_level", "LOW")

        results.append({
            **entity,
            "sensitivity": sensitivity
        })

    return results

@app.post("/calculate-risk-score")
def calculate_risk(request: RiskScoreRequest):

    score = calculate_risk_score(
        sensitivity_level=request.sensitivity_level,
        data_usage_context=request.data_usage_context
    )

    return {
        "risk_score": round(score, 2)
    }


@app.post("/suggest-protection")
def suggest_protection(request: SuggestRequest):
    suggestions = []

    for item in request.entities:
        suggestion = select_protection_method(
            entity_type=item.entity,
            risk_score=item.risk_score
        )

        suggestions.append({
            "entity": item.entity,
            "risk_score": item.risk_score,
            **suggestion
        })

    return {
        "suggestions": suggestions
    }


from fastapi import APIRouter
import json
import os

router = APIRouter()

CONFIG_DIR = "protection_configs"
os.makedirs(CONFIG_DIR, exist_ok=True)


@router.post("/save-protection-settings")
async def save_protection_settings(payload: dict):
    filename = payload.get("filename")
    settings = payload.get("settings")

    if not filename:
        raise HTTPException(status_code=400, detail="Filename required")

    if not settings:
        raise HTTPException(status_code=400, detail="Settings data required")

    safe_filename = filename.replace(" ", "_") + ".json"
    file_path = os.path.join(CONFIG_DIR, safe_filename)

    with open(file_path, "w") as f:
        json.dump(settings, f, indent=4)

    return {"message": "Settings saved successfully", "file": safe_filename}

# get available configs and their settings
@app.get("/protection-configs")
async def get_protection_configs():
    """
    Fetch all saved protection configuration files
    and return their contents as selectable options.
    """

    configs = []

    try:
        files = os.listdir(CONFIG_DIR)

        for file_name in files:
            if file_name.endswith(".json"):
                file_path = os.path.join(CONFIG_DIR, file_name)

                with open(file_path, "r") as f:
                    data = json.load(f)

                configs.append({
                    "config_name": file_name.replace(".json", ""),
                    "file_name": file_name,
                    "settings": data
                })

        return {
            "available_configs": configs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/apply-protection")
def apply_protection_endpoint(request: ApplyRequest):

    protected_content = request.original_content
    protection_results = []

    # 🔐 Sort by length (longer values first)
    sorted_entities = sorted(
        request.entities,
        key=lambda x: len(x.value),
        reverse=True
    )

    for item in sorted_entities:

        protected_value = apply_selected_protection(
            value=item.value,
            method=item.selected_method
        )

        protection_results.append({
            "entity_type": item.entity,
            "method_used": item.selected_method,
            "protected_value": protected_value
        })

        # Replace safely
        protected_content = protected_content.replace(
            item.value,
            str(protected_value)
        )

    return {
        "protected_content": protected_content,
        "entity_protection_details": protection_results
    }

app.include_router(router)